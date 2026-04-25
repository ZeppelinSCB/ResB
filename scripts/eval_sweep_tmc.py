"""Evaluate one TMC checkpoint across multiple CSI error levels."""

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from infer_tmc import build_config_from_checkpoint, evaluate, load_model  # noqa: E402
from resbdnn.config import CSI_ERROR_MODELS, CSI_ERROR_TARGETS, paper_preset_description  # noqa: E402
from resbdnn.utils import save_json, set_random_seed  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep TMC evaluation over CSI error levels.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--csi-error-vars", type=float, nargs="+", required=True)
    parser.add_argument("--csi-error-model", type=str, default=None, choices=CSI_ERROR_MODELS)
    parser.add_argument("--csi-error-target", type=str, default=None, choices=CSI_ERROR_TARGETS)
    parser.add_argument("--csi-error-snr-coupled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--csi-error-snr-ref-db", type=float, default=None)
    parser.add_argument("--csi-outlier-prob", type=float, default=None)
    parser.add_argument("--csi-outlier-scale", type=float, default=None)
    parser.add_argument("--ris-phase-bits", type=int, default=None)
    parser.add_argument("--ris-amplitude-bias", type=float, default=None)
    parser.add_argument("--ris-amplitude-scale", type=float, default=None)
    parser.add_argument("--ris-coupling-decay", type=float, default=None)
    parser.add_argument("--enable-phase-quantization", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-amplitude-coupling", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-mutual-coupling", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-bits", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=4000)
    parser.add_argument("--snr-start", type=int, default=0)
    parser.add_argument("--snr-stop", type=int, default=30)
    parser.add_argument("--snr-step", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-path", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = torch.device(args.device)
    model, checkpoint = load_model(ROOT / args.checkpoint, device)
    config_dict = checkpoint["system_config"]
    model_class = checkpoint.get("model_class", "TMCNet")
    snr_range = np.arange(args.snr_start, args.snr_stop + 1, args.snr_step)
    preset_desc = paper_preset_description(config_dict.get("paper_preset"))

    print(
        f"Sweeping {model_class} CSI robustness ({args.num_bits} bits, n_t={config_dict['n_t']}, "
        f"n_ris={config_dict['n_ris']}, s={config_dict['s']}"
        + (f", preset={preset_desc}" if preset_desc else "")
        + ")...",
        flush=True,
    )

    sweep = []
    for csi_error_var in args.csi_error_vars:
        eval_args = SimpleNamespace(
            device=args.device,
            num_bits=args.num_bits,
            batch_size=args.batch_size,
            csi_error_var=float(csi_error_var),
            csi_error_model=args.csi_error_model,
            csi_error_target=args.csi_error_target,
            csi_error_snr_coupled=args.csi_error_snr_coupled,
            csi_error_snr_ref_db=args.csi_error_snr_ref_db,
            csi_outlier_prob=args.csi_outlier_prob,
            csi_outlier_scale=args.csi_outlier_scale,
            ris_phase_bits=args.ris_phase_bits,
            ris_amplitude_bias=args.ris_amplitude_bias,
            ris_amplitude_scale=args.ris_amplitude_scale,
            ris_coupling_decay=args.ris_coupling_decay,
            enable_phase_quantization=args.enable_phase_quantization,
            enable_amplitude_coupling=args.enable_amplitude_coupling,
            enable_mutual_coupling=args.enable_mutual_coupling,
        )
        config = build_config_from_checkpoint(config_dict, eval_args, snr_range)
        t0 = time.perf_counter()
        results = evaluate(model, config, eval_args, snr_range)
        elapsed = time.perf_counter() - t0
        summary = {
            "csi_error_var": float(csi_error_var),
            "tmc_corrected_avg_ber": float(np.mean(results["tmc_corrected"]["ber"])),
            "ideal_ml_avg_ber": float(np.mean(results["ideal_ml"]["ber"])),
            "practical_baseline_avg_ber": float(np.mean(results["practical_baseline"]["ber"])),
            "practical_oracle_avg_ber": float(np.mean(results["practical_baseline"]["ber"])),
            "true_center_oracle_avg_ber": float(np.mean(results["true_center_oracle"]["ber"])),
            "center_gain_db": float(np.mean(results["calibration"]["center_gain_db"])),
            "center_gain_vs_practical_db": float(np.mean(results["calibration"]["center_gain_vs_practical_db"])),
            "seconds": elapsed,
            "results": results,
        }
        sweep.append(summary)
        print(
            f"  sigma_e^2={csi_error_var:g} | tmc={summary['tmc_corrected_avg_ber']:.5f} "
            f"ideal={summary['ideal_ml_avg_ber']:.5f} baseline={summary['practical_baseline_avg_ber']:.5f} "
            f"true_center={summary['true_center_oracle_avg_ber']:.5f} "
            f"gain_i={summary['center_gain_db']:.2f}dB "
            f"gain_p={summary['center_gain_vs_practical_db']:.2f}dB ({elapsed:.1f}s)",
            flush=True,
        )

    report = {
        "checkpoint": str(ROOT / args.checkpoint),
        "seed": args.seed,
        "paper_preset": config_dict.get("paper_preset"),
        "model_class": model_class,
        "spectral_efficiency": config_dict.get("spectral_efficiency"),
        "n_t": config_dict["n_t"],
        "n_ris": config_dict["n_ris"],
        "s": config_dict["s"],
        "signal_energy": config_dict["signal_energy"],
        "snr_db": snr_range.tolist(),
        "sweep": sweep,
    }
    report_path = Path(args.report_path) if args.report_path else ROOT / "outputs" / "reports" / "tmc_csi_sweep.json"
    save_json(report_path, report)


if __name__ == "__main__":
    main()
