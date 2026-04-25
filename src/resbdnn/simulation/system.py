from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from resbdnn.config import SystemConfig
from resbdnn.simulation.candidates import candidate_combo_groups


@dataclass
class ChannelState:
    h: np.ndarray
    g: np.ndarray

    @classmethod
    def generate(cls, config: SystemConfig) -> "ChannelState":
        h = (
            np.random.randn(config.n_t, config.n_ris)
            + 1j * np.random.randn(config.n_t, config.n_ris)
        ) / np.sqrt(2)
        g = (
            np.random.randn(config.n_ris)
            + 1j * np.random.randn(config.n_ris)
        ) / np.sqrt(2)
        return cls(h=h, g=g)


class BitMapper:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.active_map = {
            format(idx, f"0{config.bits_for_active_count}b"): idx + 1
            for idx in range(config.na_classes)
        }
        self.active_map_inv = {value: key for key, value in self.active_map.items()}
        self.combination_map: Dict[int, Dict[int, Tuple[int, ...]]] = {}
        self.combination_to_bits: Dict[int, Dict[Tuple[int, ...], str]] = {}
        self._build_default_mapping()

    def _build_default_mapping(self) -> None:
        for na, combos in candidate_combo_groups(self.config.n_t, self.config.s):
            self.combination_map[na] = {}
            self.combination_to_bits[na] = {}
            for idx, combo in enumerate(combos):
                self.combination_map[na][idx] = combo
                self.combination_to_bits[na][combo] = format(idx, f"0{self.config.bits_for_combination}b")

    def bits_to_antennas(self, bits: str) -> Tuple[int, Tuple[int, ...]]:
        bits_na = bits[: self.config.bits_for_active_count]
        bits_s = bits[self.config.bits_for_active_count :]
        na = self.active_map[bits_na]
        s_idx = int(bits_s, 2) % len(self.combination_map[na])
        return na, self.combination_map[na][s_idx]

    def antennas_to_bits(self, na: int, combo: Tuple[int, ...]) -> str:
        return self.active_map_inv[na] + self.combination_to_bits[na][combo]


def calculate_expected_signal(h: np.ndarray, g: np.ndarray, combo: Tuple[int, ...], signal_energy: float = 1.0) -> float:
    # Eq. (4): optimal RIS phase alignment — y = √E · Σᵢ |h_combined_i| · |gᵢ|
    na = len(combo)
    h_subset = h[list(combo), :]
    h_combined = np.sum(h_subset, axis=0) / np.sqrt(na)
    return np.sqrt(signal_energy) * np.sum(np.abs(h_combined) * np.abs(g))


def candidate_expected_signals_batch(
    h_batch: np.ndarray,
    g_batch: np.ndarray,
    config: SystemConfig,
    *,
    phase_h_batch: np.ndarray | None = None,
    phase_g_batch: np.ndarray | None = None,
) -> np.ndarray:
    """Expected complex candidate signals under a configurable RIS phase source.

    When ``phase_h_batch``/``phase_g_batch`` are omitted, the RIS phases are
    aligned with the true channels and the result reduces to the paper's
    perfect-CSI mean signal.  Providing estimated channels reproduces the
    paper-text path where RIS phase design uses imperfect CSI while the actual
    propagation still follows the true channel.
    """
    batch_size = len(h_batch)
    expected = np.empty(
        (batch_size, config.na_classes * config.s_classes),
        dtype=np.complex64,
    )
    phase_h_batch = h_batch if phase_h_batch is None else phase_h_batch
    phase_g_batch = g_batch if phase_g_batch is None else phase_g_batch
    sqrt_E = np.sqrt(config.signal_energy)

    for na_label, (na, combos) in enumerate(candidate_combo_groups(config.n_t, config.s)):
        for s_idx, combo in enumerate(combos):
            candidate_idx = na_label * config.s_classes + s_idx
            h_true = h_batch[:, combo, :]
            h_phase = phase_h_batch[:, combo, :]
            h_combined_true = np.sum(h_true, axis=1) / np.sqrt(na)
            h_combined_phase = np.sum(h_phase, axis=1) / np.sqrt(na)
            phase = -(np.angle(h_combined_phase) + np.angle(phase_g_batch))
            expected[:, candidate_idx] = sqrt_E * np.sum(
                h_combined_true * g_batch * np.exp(1j * phase),
                axis=1,
            )

    return expected


def compute_phase_configured_signals_batch(
    h_batch: np.ndarray,
    g_batch: np.ndarray,
    na_labels: np.ndarray,
    s_labels: np.ndarray,
    config: SystemConfig,
    *,
    phase_h_batch: np.ndarray | None = None,
    phase_g_batch: np.ndarray | None = None,
) -> np.ndarray:
    """Select the transmitted signal for each sample under configurable phases."""
    expected = candidate_expected_signals_batch(
        h_batch,
        g_batch,
        config,
        phase_h_batch=phase_h_batch,
        phase_g_batch=phase_g_batch,
    )
    joint_labels = na_labels.astype(np.int64) * config.s_classes + s_labels.astype(np.int64)
    return expected[np.arange(len(joint_labels)), joint_labels]


def compute_clean_signals_batch(
    h_batch: np.ndarray,
    g_batch: np.ndarray,
    na_labels: np.ndarray,
    s_labels: np.ndarray,
    config: SystemConfig,
) -> np.ndarray:
    """Vectorized: groups samples by (na_label, s_label) and processes each
    group with batched numpy ops instead of a per-sample Python loop."""
    batch_size = len(na_labels)
    y = np.zeros(batch_size, dtype=np.complex64)
    abs_g = np.abs(g_batch)
    sqrt_E = np.sqrt(config.signal_energy)

    for na_label, (na, combos) in enumerate(
        candidate_combo_groups(config.n_t, config.s)
    ):
        for s_idx, combo in enumerate(combos):
            mask = (na_labels == na_label) & (s_labels == s_idx)
            if not np.any(mask):
                continue
            h_sub = h_batch[mask][:, combo, :]           # (count, na, n_ris)
            h_combined = np.sum(h_sub, axis=1) / np.sqrt(na)  # (count, n_ris)
            y[mask] = sqrt_E * np.sum(np.abs(h_combined) * abs_g[mask], axis=1)

    return y


def compute_ber_vectorized(
    na_pred: np.ndarray,
    s_pred: np.ndarray,
    na_true: np.ndarray,
    s_true: np.ndarray,
    config: SystemConfig,
) -> Tuple[int, int]:
    """Vectorized BER: returns (total_bit_errors, total_bits)."""
    bits_s = config.bits_for_combination
    bps = config.bits_per_symbol
    pred_int = (na_pred.astype(np.int64) << bits_s) | s_pred.astype(np.int64)
    true_int = (na_true.astype(np.int64) << bits_s) | s_true.astype(np.int64)
    xor = pred_int ^ true_int
    bit_errors = np.zeros(len(xor), dtype=np.int64)
    temp = xor.copy()
    for _ in range(bps):
        bit_errors += temp & 1
        temp >>= 1
    return int(bit_errors.sum()), len(xor) * bps


def add_csi_error_batch(
    h_batch: np.ndarray,
    g_batch: np.ndarray,
    sigma_e_sq: float,
    rng: np.random.Generator | None = None,
    error_model: str = "normalized",
    error_target: str = "dual_link",
) -> tuple:
    """Add estimation error to channel matrices.

    Supported models:
    - ``additive``: ĥ = h + Δh, ĝ = g + Δg with Δ ~ CN(0, σ²_e)
    - ``normalized``: a power-preserving NMSE model where E|ĥ|² = E|h|² and
      E|ĥ-h|² = σ²_e for unit-variance Rayleigh entries.

    Returns (h_hat, g_hat) with the same dtype as inputs.
    """
    if sigma_e_sq <= 0:
        return h_batch, g_batch
    rng = rng or np.random.default_rng()
    sqrt2 = np.float32(np.sqrt(2.0))
    dtype = h_batch.dtype

    def _cn01(shape):
        return (
            rng.standard_normal(shape).astype(np.float32)
            + 1j * rng.standard_normal(shape).astype(np.float32)
        ) / sqrt2

    if error_model == "additive":
        std = np.float32(np.sqrt(sigma_e_sq))
        h_hat = h_batch + _cn01(h_batch.shape) * std
        g_hat = g_batch + _cn01(g_batch.shape) * std
    elif error_model == "normalized":
        if not 0.0 <= sigma_e_sq <= 1.0:
            raise ValueError(
                "normalized CSI error requires 0 <= sigma_e_sq <= 1, "
                f"got {sigma_e_sq}"
            )
        alpha = np.float32(1.0 - sigma_e_sq / 2.0)
        beta = np.float32(np.sqrt(sigma_e_sq * (1.0 - sigma_e_sq / 4.0)))
        h_hat = alpha * h_batch + beta * _cn01(h_batch.shape)
        g_hat = alpha * g_batch + beta * _cn01(g_batch.shape)
    else:
        raise ValueError(f"Unknown CSI error model: {error_model}")

    if error_target == "dual_link":
        pass
    elif error_target == "h_only":
        g_hat = g_batch
    elif error_target == "g_only":
        h_hat = h_batch
    else:
        raise ValueError(f"Unknown CSI error target: {error_target}")

    return h_hat.astype(dtype), g_hat.astype(dtype)


class RISTFSSKTransmitter:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.mapper = BitMapper(config)

    def transmit(self, bits: str, channel: ChannelState) -> Tuple[complex, int, Tuple[int, ...]]:
        na, combo = self.mapper.bits_to_antennas(bits)
        y = calculate_expected_signal(channel.h, channel.g, combo, self.config.signal_energy)
        return y, na, combo


class RISTFSSKReceiver:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.mapper = BitMapper(config)

    def ml_detect(self, y: complex, channel: ChannelState) -> Tuple[str, int, Tuple[int, ...]]:
        best_na = 1
        best_combo = self.mapper.combination_map[1][0]
        best_dist = float("inf")

        for na, combo_map in self.mapper.combination_map.items():
            for combo in combo_map.values():
                y_hat = calculate_expected_signal(channel.h, channel.g, combo, self.config.signal_energy)
                dist = abs(y - y_hat) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_na = na
                    best_combo = combo

        return self.mapper.antennas_to_bits(best_na, best_combo), best_na, best_combo
