from dataclasses import dataclass, field
import math

import numpy as np


PAPER_SYSTEM_PRESETS = {
    "fig3-2b": {"description": "Paper Figure 3, 2 bits/s/Hz", "n_t": 4, "n_ris": 64, "s": 2},
    "fig3-3b": {"description": "Paper Figure 3, 3 bits/s/Hz", "n_t": 4, "n_ris": 64, "s": 4},
    "fig4-4b": {"description": "Paper Figure 4, 4 bits/s/Hz", "n_t": 8, "n_ris": 64, "s": 4},
    "fig4-5b": {"description": "Paper Figure 4, 5 bits/s/Hz", "n_t": 8, "n_ris": 64, "s": 8},
}

CSI_ERROR_MODELS = ("normalized", "additive")
CSI_ERROR_TARGETS = ("dual_link", "h_only", "g_only")


@dataclass
class SystemConfig:
    n_t: int = 8
    n_ris: int = 64
    s: int = 8
    signal_energy: float = 1.0
    csi_error_var: float = 0.5
    csi_error_model: str = "normalized"
    csi_error_target: str = "dual_link"
    csi_error_snr_coupled: bool = False
    csi_error_snr_ref_db: float = 10.0
    csi_outlier_prob: float = 0.0
    csi_outlier_scale: float = 0.0
    ris_phase_bits: int = 2
    ris_amplitude_bias: float = 0.8
    ris_amplitude_scale: float = 0.2
    ris_coupling_decay: float = 0.15
    enable_phase_quantization: bool = True
    enable_amplitude_coupling: bool = True
    enable_mutual_coupling: bool = True
    snr_range: np.ndarray = field(default_factory=lambda: np.arange(0, 31, 2))

    def __post_init__(self):
        self.snr_range = np.asarray(self.snr_range, dtype=np.int64)
        if self.n_t < 2 or self.n_t % 2 != 0 or self.n_t & (self.n_t - 1):
            raise ValueError(f"n_t must be an even power of two, got {self.n_t}")
        if self.s < 1 or self.s & (self.s - 1) or self.s > self.n_t:
            raise ValueError(f"s must be a power of two in [1, n_t], got s={self.s}, n_t={self.n_t}")
        if self.n_ris < 1:
            raise ValueError(f"n_ris must be >= 1, got {self.n_ris}")
        if self.signal_energy <= 0.0:
            raise ValueError("signal_energy must be > 0")
        if self.snr_range.ndim != 1 or self.snr_range.size == 0:
            raise ValueError("snr_range must be a non-empty 1D array")
        if self.csi_error_model not in CSI_ERROR_MODELS:
            raise ValueError(f"Unknown csi_error_model: {self.csi_error_model}")
        if self.csi_error_target not in CSI_ERROR_TARGETS:
            raise ValueError(f"Unknown csi_error_target: {self.csi_error_target}")
        if self.csi_error_model == "normalized" and not 0.0 <= self.csi_error_var <= 1.0:
            raise ValueError("normalized CSI error requires 0 <= csi_error_var <= 1")
        if not math.isfinite(self.csi_error_snr_ref_db):
            raise ValueError("csi_error_snr_ref_db must be finite")
        if not 0.0 <= self.csi_outlier_prob <= 1.0:
            raise ValueError("csi_outlier_prob must be in [0, 1]")
        if self.csi_outlier_scale < 0.0:
            raise ValueError("csi_outlier_scale must be >= 0")
        if self.ris_phase_bits < 1:
            raise ValueError("ris_phase_bits must be >= 1")
        if self.ris_amplitude_bias <= 0.0:
            raise ValueError("ris_amplitude_bias must be > 0")
        if self.ris_amplitude_scale < 0.0:
            raise ValueError("ris_amplitude_scale must be >= 0")
        if self.ris_amplitude_bias - self.ris_amplitude_scale < 0.0:
            raise ValueError("ris_amplitude_bias - ris_amplitude_scale must be >= 0")
        if not 0.0 <= self.ris_coupling_decay < 1.0:
            raise ValueError("ris_coupling_decay must be in [0, 1)")

    @property
    def bits_for_active_count(self) -> int:
        return int(np.log2(self.n_t // 2))

    @property
    def bits_for_combination(self) -> int:
        return int(np.log2(self.s))

    @property
    def bits_per_symbol(self) -> int:
        return self.bits_for_active_count + self.bits_for_combination

    @property
    def na_classes(self) -> int:
        return self.n_t // 2

    @property
    def s_classes(self) -> int:
        return self.s

    @property
    def num_candidates(self) -> int:
        return self.na_classes * self.s_classes

    @property
    def spectral_efficiency(self) -> int:
        return self.bits_per_symbol


def build_snr_range(snr_start: int = 0, snr_stop: int = 30, snr_step: int = 2) -> np.ndarray:
    if snr_step <= 0:
        raise ValueError(f"snr_step must be positive, got {snr_step}")
    if snr_stop < snr_start:
        raise ValueError(f"snr_stop must be >= snr_start, got start={snr_start}, stop={snr_stop}")
    return np.arange(snr_start, snr_stop + 1, snr_step)


def build_system_config(
    *,
    paper_preset: str | None = None,
    n_t: int | None = None,
    n_ris: int | None = None,
    s: int | None = None,
    signal_energy: float = 1.0,
    csi_error_var: float = 0.5,
    csi_error_model: str = "normalized",
    csi_error_target: str = "dual_link",
    csi_error_snr_coupled: bool = False,
    csi_error_snr_ref_db: float = 10.0,
    csi_outlier_prob: float = 0.0,
    csi_outlier_scale: float = 0.0,
    ris_phase_bits: int = 2,
    ris_amplitude_bias: float = 0.8,
    ris_amplitude_scale: float = 0.2,
    ris_coupling_decay: float = 0.15,
    enable_phase_quantization: bool = True,
    enable_amplitude_coupling: bool = True,
    enable_mutual_coupling: bool = True,
    snr_start: int = 0,
    snr_stop: int = 30,
    snr_step: int = 2,
) -> SystemConfig:
    if paper_preset is not None:
        if paper_preset not in PAPER_SYSTEM_PRESETS:
            valid = ", ".join(sorted(PAPER_SYSTEM_PRESETS))
            raise ValueError(f"Unknown paper preset '{paper_preset}'. Valid values: {valid}")
        preset = PAPER_SYSTEM_PRESETS[paper_preset]
        n_t = preset["n_t"] if n_t is None else n_t
        n_ris = preset["n_ris"] if n_ris is None else n_ris
        s = preset["s"] if s is None else s

    return SystemConfig(
        n_t=8 if n_t is None else n_t,
        n_ris=64 if n_ris is None else n_ris,
        s=8 if s is None else s,
        signal_energy=signal_energy,
        csi_error_var=csi_error_var,
        csi_error_model=csi_error_model,
        csi_error_target=csi_error_target,
        csi_error_snr_coupled=csi_error_snr_coupled,
        csi_error_snr_ref_db=csi_error_snr_ref_db,
        csi_outlier_prob=csi_outlier_prob,
        csi_outlier_scale=csi_outlier_scale,
        ris_phase_bits=ris_phase_bits,
        ris_amplitude_bias=ris_amplitude_bias,
        ris_amplitude_scale=ris_amplitude_scale,
        ris_coupling_decay=ris_coupling_decay,
        enable_phase_quantization=enable_phase_quantization,
        enable_amplitude_coupling=enable_amplitude_coupling,
        enable_mutual_coupling=enable_mutual_coupling,
        snr_range=build_snr_range(snr_start, snr_stop, snr_step),
    )


def paper_preset_description(paper_preset: str | None) -> str | None:
    if paper_preset is None:
        return None
    preset = PAPER_SYSTEM_PRESETS[paper_preset]
    return (
        f"{preset['description']} "
        f"(n_t={preset['n_t']}, n_ris={preset['n_ris']}, s={preset['s']}, "
        f"r={int(math.log2(preset['n_t'] // 2) + math.log2(preset['s']))})"
    )
