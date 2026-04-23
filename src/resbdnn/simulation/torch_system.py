from functools import lru_cache
import math

import numpy as np
import torch

from resbdnn.config import SystemConfig
from resbdnn.simulation.candidates import candidate_group_arrays


def _candidate_groups(config: SystemConfig, device: torch.device):
    groups = []
    for na, candidate_indices, combos in candidate_group_arrays(config.n_t, config.s):
        groups.append(
            (
                na,
                torch.as_tensor(candidate_indices, device=device, dtype=torch.long),
                torch.as_tensor(combos, device=device, dtype=torch.long),
            )
        )
    return groups


def _complex_gaussian(shape, device: torch.device) -> torch.Tensor:
    return (
        torch.randn(shape, device=device, dtype=torch.float32)
        + 1j * torch.randn(shape, device=device, dtype=torch.float32)
    ) / math.sqrt(2.0)


def _resolve_csi_error_var(
    config: SystemConfig,
    snr: torch.Tensor,
    csi_error_var: float | torch.Tensor | None,
) -> torch.Tensor:
    if csi_error_var is None:
        base_var = torch.full_like(snr, float(config.csi_error_var), dtype=torch.float32)
    else:
        base_var = torch.as_tensor(csi_error_var, device=snr.device, dtype=torch.float32)
        if base_var.ndim == 0:
            base_var = torch.full_like(snr, float(base_var.item()), dtype=torch.float32)
        else:
            base_var = base_var.to(device=snr.device, dtype=torch.float32)
    if config.csi_error_snr_coupled:
        snr_scale = torch.pow(10.0, (config.csi_error_snr_ref_db - snr) / 10.0)
        base_var = base_var * snr_scale
    if config.csi_error_model == "normalized":
        return base_var.clamp_(0.0, 1.0)
    return base_var.clamp_min_(0.0)


def _apply_csi_error(
    h: torch.Tensor,
    g: torch.Tensor,
    *,
    csi_error_var: float | torch.Tensor,
    csi_error_model: str,
    csi_error_target: str,
    csi_outlier_prob: float,
    csi_outlier_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    error_var = torch.as_tensor(csi_error_var, device=h.device, dtype=torch.float32)
    if error_var.ndim == 0:
        error_var = error_var.expand(h.shape[0])
    if float(error_var.max().item()) <= 0.0:
        return h, g

    h_noise = _complex_gaussian(h.shape, h.device)
    g_noise = _complex_gaussian(g.shape, g.device)
    h_scale = error_var[:, None, None]
    g_scale = error_var[:, None]

    if csi_error_model == "additive":
        h_hat = h + torch.sqrt(h_scale) * h_noise
        g_hat = g + torch.sqrt(g_scale) * g_noise
    elif csi_error_model == "normalized":
        alpha_h = 1.0 - h_scale / 2.0
        beta_h = torch.sqrt(h_scale * (1.0 - h_scale / 4.0))
        alpha_g = 1.0 - g_scale / 2.0
        beta_g = torch.sqrt(g_scale * (1.0 - g_scale / 4.0))
        h_hat = alpha_h * h + beta_h * h_noise
        g_hat = alpha_g * g + beta_g * g_noise
    else:
        raise ValueError(f"Unknown CSI error model: {csi_error_model}")

    if csi_outlier_prob > 0.0 and csi_outlier_scale > 0.0:
        outlier_mask = (torch.rand(h.shape[0], device=h.device) < csi_outlier_prob).to(torch.float32)
        h_hat = h_hat + outlier_mask[:, None, None] * csi_outlier_scale * _complex_gaussian(h.shape, h.device)
        g_hat = g_hat + outlier_mask[:, None] * csi_outlier_scale * _complex_gaussian(g.shape, g.device)

    if csi_error_target == "dual_link":
        return h_hat.to(torch.complex64), g_hat.to(torch.complex64)
    if csi_error_target == "h_only":
        return h_hat.to(torch.complex64), g.to(torch.complex64)
    if csi_error_target == "g_only":
        return h.to(torch.complex64), g_hat.to(torch.complex64)
    raise ValueError(f"Unknown CSI error target: {csi_error_target}")


def quantize_ris_phase(phi: torch.Tensor, phase_bits: int) -> torch.Tensor:
    step = (2.0 * math.pi) / float(2**phase_bits)
    phi_mod = torch.remainder(phi, 2.0 * math.pi)
    return torch.remainder(torch.round(phi_mod / step) * step, 2.0 * math.pi)


def ris_amplitude_from_phase(phi: torch.Tensor, config: SystemConfig) -> torch.Tensor:
    if not config.enable_amplitude_coupling:
        return torch.ones_like(phi, dtype=torch.float32)
    amplitude = config.ris_amplitude_bias + config.ris_amplitude_scale * torch.cos(phi)
    return amplitude.to(torch.float32)


@lru_cache(maxsize=None)
def _coupling_matrix_array(n_ris: int, decay: float) -> np.ndarray:
    offsets = np.abs(np.subtract.outer(np.arange(n_ris), np.arange(n_ris)))
    return np.power(decay, offsets).astype(np.float32)


def ris_coupling_matrix(config: SystemConfig, device: torch.device) -> torch.Tensor:
    matrix = _coupling_matrix_array(config.n_ris, float(config.ris_coupling_decay))
    return torch.as_tensor(matrix, device=device, dtype=torch.float32)


def apply_ris_coupling(g_batch: torch.Tensor, config: SystemConfig) -> torch.Tensor:
    if not config.enable_mutual_coupling:
        return g_batch.to(torch.complex64)
    coupling = ris_coupling_matrix(config, g_batch.device).to(torch.complex64)
    return (g_batch @ coupling.transpose(0, 1)).to(torch.complex64)


def candidate_phase_table_torch(
    h_batch: torch.Tensor,
    g_batch: torch.Tensor,
    config: SystemConfig,
    *,
    quantized: bool,
) -> torch.Tensor:
    """Calculate RIS phase table for amplitude coupling.

    For each antenna m and RIS element i, the ideal phase is:
    phi_m,i = angle(H_m,i) + angle(G_i)

    This function returns the phase per candidate and RIS element,
    which is the sum of phases from all antennas in the candidate.
    Used for amplitude coupling calculation.

    Returns: phase_table of shape (batch, num_candidates, n_ris)
    """
    phase_table = torch.empty(
        h_batch.size(0),
        config.num_candidates,
        config.n_ris,
        device=h_batch.device,
        dtype=torch.float32,
    )
    for active_count, candidate_indices, combo_tensor in _candidate_groups(config, h_batch.device):
        for i, combo in enumerate(combo_tensor.unbind(0)):
            cand_idx = candidate_indices[i].item()
            # Sum ideal phases from all antennas
            phase_sum = torch.zeros(h_batch.size(0), config.n_ris, device=h_batch.device)
            for m in combo.tolist():
                phase_m = torch.angle(h_batch[:, m, :]) + torch.angle(g_batch)
                phase_sum = phase_sum + phase_m
            if quantized and config.enable_phase_quantization:
                phase_sum = quantize_ris_phase(phase_sum, config.ris_phase_bits)
            phase_table[:, cand_idx, :] = phase_sum
    return phase_table


def candidate_expected_signals_torch(h_batch: torch.Tensor, g_batch: torch.Tensor, config: SystemConfig) -> torch.Tensor:
    """Calculate expected signals for each candidate.

    Paper model: y = sqrt(E) * sum over antennas m of (sum over RIS elements i of |H_m,i| * |G_i|)
    This is equivalent to each antenna being perfectly phase-compensated independently.
    """
    expected = torch.empty(
        h_batch.size(0),
        config.num_candidates,
        device=h_batch.device,
        dtype=torch.complex64,
    )
    sqrt_energy = math.sqrt(float(config.signal_energy))
    for active_count, candidate_indices, combo_tensor in _candidate_groups(config, h_batch.device):
        # combo_tensor shape: (num_candidates_in_group, active_count)
        # Each row is one antenna combination
        batch_size = h_batch.size(0)
        num_candidates_in_group = len(candidate_indices)

        # Calculate signal for each antenna in each combination
        # Result is real (positive) for perfect phase compensation
        candidate_signals = torch.zeros(batch_size, num_candidates_in_group, device=h_batch.device, dtype=torch.complex64)
        for i, combo in enumerate(combo_tensor.unbind(0)):
            # combo: (active_count,) - indices of active antennas
            # Calculate sum over RIS elements: sum_i |H_m,i| * |G_i| for each antenna m, then sum
            antenna_sum = torch.zeros(batch_size, device=h_batch.device, dtype=torch.complex64)
            for m in combo.tolist():
                # Signal contribution from antenna m: real positive
                signal_m = sqrt_energy * (torch.abs(h_batch[:, m, :]) * torch.abs(g_batch)).sum(dim=1)
                antenna_sum = antenna_sum + signal_m.to(torch.complex64)
            candidate_signals[:, i] = antenna_sum

        expected[:, candidate_indices] = candidate_signals
    return expected


def candidate_expected_signals_torch_true(
    h_true: torch.Tensor,
    g_true: torch.Tensor,
    config: SystemConfig,
    *,
    phase_h: torch.Tensor | None = None,
    phase_g: torch.Tensor | None = None,
    phase_table: torch.Tensor | None = None,
) -> torch.Tensor:
    return candidate_expected_signals_torch_nonideal(
        h_true,
        g_true,
        config,
        phase_h=phase_h,
        phase_g=phase_g,
        phase_table=phase_table,
    )


def candidate_expected_signals_torch_practical_baseline(
    h_hat: torch.Tensor,
    g_hat: torch.Tensor,
    config: SystemConfig,
    *,
    phase_table: torch.Tensor | None = None,
) -> torch.Tensor:
    return candidate_expected_signals_torch_nonideal(
        h_hat,
        g_hat,
        config,
        phase_h=h_hat,
        phase_g=g_hat,
        phase_table=phase_table,
    )


def candidate_expected_signals_torch_practical_oracle(
    h_hat: torch.Tensor,
    g_hat: torch.Tensor,
    config: SystemConfig,
    *,
    phase_table: torch.Tensor | None = None,
) -> torch.Tensor:
    return candidate_expected_signals_torch_practical_baseline(
        h_hat,
        g_hat,
        config,
        phase_table=phase_table,
    )


def candidate_expected_signals_torch_nonideal(
    h_true: torch.Tensor,
    g_true: torch.Tensor,
    config: SystemConfig,
    *,
    phase_h: torch.Tensor | None = None,
    phase_g: torch.Tensor | None = None,
    phase_table: torch.Tensor | None = None,
) -> torch.Tensor:
    """Calculate expected signals with non-ideal effects.

    For non-ideal case with quantized phase and amplitude coupling:
    - Phase quantization is applied to the total phase for each antenna
    - Signal = sqrt(E) * sum_m (sum_i |H_m,i| * |G_i| * A_m,i * exp(j * (phi_q,m,i - theta_m,i - psi_m,i)))
    where phi_q,m,i is the quantized phase for antenna m.
    """
    phase_h = h_true if phase_h is None else phase_h
    phase_g = g_true if phase_g is None else phase_g

    # Get phase for each antenna (before quantization for ideal case)
    amplitudes = None
    if config.enable_phase_quantization or config.enable_amplitude_coupling:
        # Compute quantized phase per antenna
        phase_table = (
            candidate_phase_table_torch(phase_h, phase_g, config, quantized=True)
            if phase_table is None
            else phase_table
        )
        amplitudes = ris_amplitude_from_phase(phase_table, config)

    g_eff = apply_ris_coupling(g_true, config)  # (batch, n_ris)

    expected = torch.empty(
        h_true.size(0),
        config.num_candidates,
        device=h_true.device,
        dtype=torch.complex64,
    )
    sqrt_energy = math.sqrt(float(config.signal_energy))

    for active_count, candidate_indices, combo_tensor in _candidate_groups(config, h_true.device):
        batch_size = h_true.size(0)
        num_candidates_in_group = len(candidate_indices)

        candidate_signals = torch.zeros(batch_size, num_candidates_in_group, device=h_true.device, dtype=torch.complex64)
        for i, combo in enumerate(combo_tensor.unbind(0)):
            antenna_sum = torch.zeros(batch_size, device=h_true.device, dtype=torch.complex64)
            for m in combo.tolist():
                h_m = h_true[:, m, :]  # (batch, n_ris)
                g_eff_m = g_eff  # (batch, n_ris)

                # Ideal phase for this antenna
                ideal_phase = torch.angle(h_m) + torch.angle(g_eff_m)

                if amplitudes is not None:
                    # Get quantized phase and amplitude for this candidate/antenna
                    cand_idx = candidate_indices[i].item()
                    phi_q = phase_table[:, cand_idx, :]  # quantized phase
                    amp = amplitudes[:, cand_idx, :]

                    # Steering: exp(j * (phi_q - ideal_phase))
                    steering = torch.polar(torch.ones_like(phi_q), phi_q - ideal_phase)
                else:
                    # No quantization or amplitude coupling - perfect compensation
                    steering = torch.ones_like(ideal_phase)
                    amp = torch.ones_like(ideal_phase)

                # Signal contribution: |H| * |G| * A * steering
                signal_m = sqrt_energy * torch.abs(h_m) * torch.abs(g_eff_m) * amp * steering
                antenna_sum = antenna_sum + signal_m.sum(dim=1)

            candidate_signals[:, i] = antenna_sum

        expected[:, candidate_indices] = candidate_signals
    return expected


def candidate_distances_from_centers(y_batch: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    return (y_batch[:, None] - centers).abs().square().to(torch.float32)


def candidate_distances_torch(y_batch: torch.Tensor, h_batch: torch.Tensor, g_batch: torch.Tensor, config: SystemConfig) -> torch.Tensor:
    return candidate_distances_from_centers(y_batch, candidate_expected_signals_torch(h_batch, g_batch, config))


def observation_noise_std_from_snr_db(
    snr_db: torch.Tensor | float,
    *,
    signal_power: torch.Tensor | float,
) -> torch.Tensor:
    snr_tensor = torch.as_tensor(snr_db, dtype=torch.float32)
    signal_power_tensor = torch.as_tensor(signal_power, dtype=torch.float32, device=snr_tensor.device)
    return torch.sqrt(signal_power_tensor * torch.pow(10.0, -snr_tensor / 10.0))


def random_tmc_batch(
    config: SystemConfig,
    batch_size: int,
    device: torch.device,
    *,
    snr_db: float | None = None,
    csi_error_var: float | None = None,
) -> dict[str, torch.Tensor]:
    h_true = _complex_gaussian((batch_size, config.n_t, config.n_ris), device).to(torch.complex64)
    g_true = _complex_gaussian((batch_size, config.n_ris), device).to(torch.complex64)

    labels = torch.randint(0, config.num_candidates, (batch_size,), device=device, dtype=torch.long)
    na_labels = labels // config.s_classes
    s_labels = labels % config.s_classes

    if snr_db is None:
        snr_values = torch.as_tensor(config.snr_range, device=device, dtype=torch.float32)
        # Focus on SNR >= 10 dB with linearly increasing weights (higher SNR = higher weight)
        valid_snr = snr_values[snr_values >= 10.0]
        if len(valid_snr) == 0:
            valid_snr = snr_values
        # Create weighted sampling: higher SNR has higher probability
        # Weight proportional to (snr - min_snr) / range
        min_snr = valid_snr.min().item()
        max_snr = valid_snr.max().item()
        snr_range = max_snr - min_snr
        weights = (valid_snr - min_snr) / snr_range if snr_range > 0 else torch.ones_like(valid_snr, dtype=torch.float32)
        weights = weights / weights.sum()  # Normalize to probability
        indices = torch.multinomial(weights, batch_size, replacement=True)
        snr = valid_snr[indices]
    else:
        snr = torch.full((batch_size,), float(snr_db), device=device, dtype=torch.float32)

    resolved_csi_error_var = _resolve_csi_error_var(config, snr, csi_error_var)
    h_hat, g_hat = _apply_csi_error(
        h_true,
        g_true,
        csi_error_var=resolved_csi_error_var,
        csi_error_model=config.csi_error_model,
        csi_error_target=config.csi_error_target,
        csi_outlier_prob=config.csi_outlier_prob,
        csi_outlier_scale=config.csi_outlier_scale,
    )

    mu_ideal = candidate_expected_signals_torch(h_hat, g_hat, config)
    phi_config = candidate_phase_table_torch(h_hat, g_hat, config, quantized=True)
    mu_practical = candidate_expected_signals_torch_practical_baseline(
        h_hat,
        g_hat,
        config,
        phase_table=phi_config,
    )
    # True center: uses estimated phase config (from h_hat/g_hat) but true channel for signal
    # This represents the signal we'd get if RIS is configured based on estimated CSI but actual channel
    mu_true = candidate_expected_signals_torch_true(
        h_true,
        g_true,
        config,
        phase_h=h_hat,  # Use estimated channel phase for RIS config
        phase_g=g_hat,  # Use estimated channel phase for RIS config
        phase_table=phi_config,  # Use the same phase table as practical
    )
    signal_power_ref = mu_true.abs().square().mean(dim=1).clamp_min(1e-12)
    sigma_n = observation_noise_std_from_snr_db(snr, signal_power=signal_power_ref).to(device)
    y_clean = mu_true.gather(1, labels[:, None]).squeeze(1)
    y_noisy = (y_clean + sigma_n * _complex_gaussian((batch_size,), device)).to(torch.complex64)

    return {
        "h_hat": h_hat,
        "g_hat": g_hat,
        "h_true": h_true,
        "g_true": g_true,
        "y": y_noisy,
        "labels": labels,
        "na_labels": na_labels,
        "s_labels": s_labels,
        "snr": snr,
        "snr_db": snr.unsqueeze(-1),
        "sigma_n": sigma_n.unsqueeze(-1),
        "csi_error_var": resolved_csi_error_var,
        "phi_config": phi_config,
        "mu_practical": mu_practical,
        "mu_practical_oracle": mu_practical,
        "mu_ideal": mu_ideal,
        "mu_true": mu_true,
        "signal_power_ref": signal_power_ref.unsqueeze(-1),
        "delta_target": (mu_true - mu_practical).to(torch.complex64),
    }


def bit_errors_from_joint(
    pred: torch.Tensor,
    target: torch.Tensor,
    config: SystemConfig,
) -> tuple[int, int]:
    xor = pred.long() ^ target.long()
    bit_errors = torch.zeros_like(xor)
    temp = xor.clone()
    for _ in range(config.bits_per_symbol):
        bit_errors += temp & 1
        temp = temp >> 1
    return int(bit_errors.sum().item()), int(target.numel() * config.bits_per_symbol)
