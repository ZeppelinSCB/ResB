"""Reproduce the RIS-TFSSK paper's original-text ML baseline.

This follows the wording in Section II of the paper:
1. RIS phase design uses estimated CSI when imperfect CSI is enabled.
2. The receiver still uses the perfect-CSI ML rule from Eq. (5)-(6).

It is intentionally separate from the repo's detector-side mismatch experiments.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from resbdnn.config import (
    CSI_ERROR_MODELS,
    CSI_ERROR_TARGETS,
    PAPER_SYSTEM_PRESETS,
    build_system_config,
    paper_preset_description,
)
from resbdnn.simulation import (
    add_csi_error_batch,
    candidate_expected_signals_batch,
    compute_ber_vectorized,
    compute_phase_configured_signals_batch,
)
from resbdnn.utils import ensure_dir, save_json, set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce the paper-text RIS-TFSSK ML baseline.")
    parser.add_argument(
        "--paper-preset",
        type=str,
        default="fig3-3b",
        choices=sorted(PAPER_SYSTEM_PRESETS),
    )
    parser.add_argument("--n-t", type=int, default=None)
    parser.add_argument("--n-ris", type=int, default=None)
    parser.add_argument("--s", type=int, default=None)
    parser.add_argument("--snr-start", type=int, default=0)
    parser.add_argument("--snr-stop", type=int, default=30)
    parser.add_argument("--snr-step", type=int, default=2)
    parser.add_argument("--num-bits", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=4000)
    parser.add_argument("--csi-error-var", type=float, default=0.5)
    parser.add_argument(
        "--csi-error-model",
        type=str,
        default="normalized",
        choices=CSI_ERROR_MODELS,
    )
    parser.add_argument(
        "--csi-error-target",
        type=str,
        default="dual_link",
        choices=CSI_ERROR_TARGETS,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-path", type=str, default=None)
    parser.add_argument("--figure-path", type=str, default=None)
    return parser.parse_args()


def _bit_error_rate(pred, na_true, s_true, config):
    pred_na = (pred // config.s_classes).astype(np.int64)
    pred_s = (pred % config.s_classes).astype(np.int64)
    return compute_ber_vectorized(pred_na, pred_s, na_true, s_true, config)


def evaluate(config, num_bits, batch_size, csi_error_var, seed):
    bps = config.bits_per_symbol
    num_symbols = num_bits // bps
    snr_range = config.snr_range
    perfect_ber = []
    imperfect_ber = []

    rng = np.random.default_rng(seed)
    for snr_db in snr_range:
        perfect_err = perfect_bits = 0
        imperfect_err = imperfect_bits = 0
        remaining = num_symbols
        while remaining > 0:
            n = min(batch_size, remaining)
            h = (
                rng.standard_normal((n, config.n_t, config.n_ris)).astype(np.float32)
                + 1j * rng.standard_normal((n, config.n_t, config.n_ris)).astype(np.float32)
            ) / np.sqrt(2)
            g = (
                rng.standard_normal((n, config.n_ris)).astype(np.float32)
                + 1j * rng.standard_normal((n, config.n_ris)).astype(np.float32)
            ) / np.sqrt(2)
            bits = rng.integers(0, 2 ** bps, size=n)
            na_true = (bits >> config.bits_for_combination).astype(np.int64)
            s_true = (bits & ((1 << config.bits_for_combination) - 1)).astype(np.int64)

            perfect_expected = candidate_expected_signals_batch(h, g, config)
            perfect_clean = compute_phase_configured_signals_batch(
                h,
                g,
                na_true,
                s_true,
                config,
            )

            if csi_error_var > 0.0:
                h_hat, g_hat = add_csi_error_batch(
                    h,
                    g,
                    csi_error_var,
                    rng,
                    error_model=config.csi_error_model,
                    error_target=config.csi_error_target,
                )
                imperfect_clean = compute_phase_configured_signals_batch(
                    h,
                    g,
                    na_true,
                    s_true,
                    config,
                    phase_h_batch=h_hat,
                    phase_g_batch=g_hat,
                )
            else:
                imperfect_clean = perfect_clean

            snr_lin = 10 ** (snr_db / 10)
            noise = (
                rng.standard_normal(n).astype(np.float32)
                + 1j * rng.standard_normal(n).astype(np.float32)
            ) * np.float32(np.sqrt(1.0 / snr_lin / 2))

            perfect_rx = perfect_clean + noise
            imperfect_rx = imperfect_clean + noise

            perfect_pred = np.abs(perfect_rx[:, None] - perfect_expected).argmin(axis=1)
            imperfect_pred = np.abs(imperfect_rx[:, None] - perfect_expected).argmin(axis=1)

            e, b = _bit_error_rate(perfect_pred, na_true, s_true, config)
            perfect_err += e
            perfect_bits += b
            e, b = _bit_error_rate(imperfect_pred, na_true, s_true, config)
            imperfect_err += e
            imperfect_bits += b
            remaining -= n

        perfect_ber.append(perfect_err / perfect_bits)
        imperfect_ber.append(imperfect_err / imperfect_bits)

    return {
        "perfect_ml": {"ber": perfect_ber},
        "paper_text_imperfect_ml": {"ber": imperfect_ber},
    }


def main():
    args = parse_args()
    set_random_seed(args.seed)
    config = build_system_config(
        paper_preset=args.paper_preset,
        n_t=args.n_t,
        n_ris=args.n_ris,
        s=args.s,
        csi_error_var=args.csi_error_var,
        csi_error_model=args.csi_error_model,
        csi_error_target=args.csi_error_target,
        snr_start=args.snr_start,
        snr_stop=args.snr_stop,
        snr_step=args.snr_step,
    )
    preset_desc = paper_preset_description(args.paper_preset)
    print(
        "Reproducing paper-text ML baseline "
        f"({args.num_bits} bits, σ²_e={args.csi_error_var}, "
        f"n_t={config.n_t}, n_ris={config.n_ris}, s={config.s}, "
        f"r={config.spectral_efficiency}"
        + (f", preset={preset_desc}" if preset_desc else "")
        + ")..."
    )
    t0 = time.perf_counter()
    results = evaluate(config, args.num_bits, args.batch_size, args.csi_error_var, args.seed)
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Perfect ML avg BER={np.mean(results['perfect_ml']['ber']):.5f}")
    print(f"  Paper-text imperfect ML avg BER={np.mean(results['paper_text_imperfect_ml']['ber']):.5f}")

    print(f"\n{'SNR':>4}  {'Perfect ML':>12}  {'Paper-text imperfect':>21}  {'Gap':>9}")
    for idx, snr_db in enumerate(config.snr_range):
        perfect = results["perfect_ml"]["ber"][idx]
        imperfect = results["paper_text_imperfect_ml"]["ber"][idx]
        print(f"{snr_db:4d}  {perfect:12.5f}  {imperfect:21.5f}  {imperfect - perfect:+9.5f}")

    report = {
        "seed": args.seed,
        "num_bits": args.num_bits,
        "csi_error_var": args.csi_error_var,
        "csi_error_model": config.csi_error_model,
        "csi_error_target": config.csi_error_target,
        "paper_preset": args.paper_preset,
        "spectral_efficiency": config.spectral_efficiency,
        "n_t": config.n_t,
        "n_ris": config.n_ris,
        "s": config.s,
        "snr_db": config.snr_range.tolist(),
        "model": {
            "transmit_phase_design": "estimated h and g when σ²_e > 0",
            "receiver_detection": "perfect-CSI ML from the paper text",
            "channel_error": f"{config.csi_error_model} CSI estimation error on h and g",
        },
        **results,
    }
    reports_dir = ensure_dir(ROOT / "outputs" / "reports")
    report_path = Path(args.report_path) if args.report_path else reports_dir / f"paper_text_{args.paper_preset}.json"
    save_json(report_path, report)

    try:
        import os

        os.environ.setdefault("MPLCONFIGDIR", str(ensure_dir(ROOT / "outputs" / ".matplotlib")))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.semilogy(
            config.snr_range,
            np.maximum(results["perfect_ml"]["ber"], 1e-7),
            "gs-",
            label="Perfect ML",
        )
        plt.semilogy(
            config.snr_range,
            np.maximum(results["paper_text_imperfect_ml"]["ber"], 1e-7),
            "ro-",
            label="Paper-text imperfect ML",
        )
        plt.xlabel("SNR (dB)")
        plt.ylabel("BER")
        plt.title(f"Paper-text RIS-TFSSK reproduction (σ²_e={args.csi_error_var})")
        plt.grid(True, which="both", linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        figures_dir = ensure_dir(ROOT / "outputs" / "figures")
        figure_path = (
            Path(args.figure_path)
            if args.figure_path
            else figures_dir / f"paper_text_{args.paper_preset}.png"
        )
        plt.savefig(figure_path, dpi=300)
        print(f"\nSaved figure to {figure_path}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
