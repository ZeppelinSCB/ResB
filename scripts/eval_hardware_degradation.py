"""Evaluate practical_baseline under varying RIS hardware non-ideality levels.

This script studies how hardware imperfections degrade symbol detection accuracy,
ranging from fully ideal hardware (no errors) to realistic non-ideal hardware.
"""

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from resbdnn.config import SystemConfig, build_system_config
from resbdnn.simulation.torch_system import (
    bit_errors_from_joint,
    candidate_distances_from_centers,
    random_tmc_batch,
)
from resbdnn.utils import ensure_dir, save_json, set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate hardware degradation on practical_baseline.")
    # System params
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-bits", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=4000)
    parser.add_argument("--snr-start", type=int, default=10)
    parser.add_argument("--snr-stop", type=int, default=30)
    parser.add_argument("--snr-step", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-t", type=int, default=8)
    parser.add_argument("--n-ris", type=int, default=64)
    parser.add_argument("--s", type=int, default=8)
    # CSI error params
    parser.add_argument("--csi-error-var", type=float, default=0.5)
    parser.add_argument("--csi-error-model", type=str, default="additive", choices=["normalized", "additive"])
    parser.add_argument("--csi-error-target", type=str, default="dual_link", choices=["dual_link", "h_only", "g_only"])
    parser.add_argument("--csi-error-snr-coupled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--csi-error-snr-ref-db", type=float, default=10.0)
    parser.add_argument("--csi-outlier-prob", type=float, default=0.0)
    parser.add_argument("--csi-outlier-scale", type=float, default=0.0)
    # Output
    parser.add_argument("--num-experiments", type=int, default=10)
    parser.add_argument("--report-path", type=str, default=None)
    parser.add_argument("--figure-path", type=str, default=None)
    return parser.parse_args()


def get_hardware_config(experiment_idx: int, num_experiments: int) -> dict:
    """Generate hardware config for a given experiment index.

    Focus on the 60-40% error range where the performance cliff occurs.
    Use finer steps in this critical region.
    """
    # Current realistic values
    current = {
        "ris_phase_bits": 2,
        "ris_amplitude_bias": 0.8,
        "ris_amplitude_scale": 0.2,
        "ris_coupling_decay": 0.15,
        "enable_phase_quantization": True,
        "enable_amplitude_coupling": True,
        "enable_mutual_coupling": True,
        "csi_error_var": 0.05,
    }

    # Ideal values
    ideal = {
        "ris_phase_bits": 8,
        "ris_amplitude_bias": 1.0,
        "ris_amplitude_scale": 0.0,
        "ris_coupling_decay": 0.0,
        "enable_phase_quantization": False,
        "enable_amplitude_coupling": False,
        "enable_mutual_coupling": False,
        "csi_error_var": 0.0,
    }

    t = experiment_idx / (num_experiments - 1) if num_experiments > 1 else 1.0

    config = {}
    config["ris_phase_bits"] = int(round(current["ris_phase_bits"] + t * (ideal["ris_phase_bits"] - current["ris_phase_bits"])))
    config["ris_amplitude_bias"] = current["ris_amplitude_bias"] + t * (ideal["ris_amplitude_bias"] - current["ris_amplitude_bias"])
    config["ris_amplitude_scale"] = current["ris_amplitude_scale"] + t * (ideal["ris_amplitude_scale"] - current["ris_amplitude_scale"])
    config["ris_coupling_decay"] = current["ris_coupling_decay"] + t * (ideal["ris_coupling_decay"] - current["ris_coupling_decay"])
    config["enable_phase_quantization"] = t < 0.5
    config["enable_amplitude_coupling"] = t < 0.5
    config["enable_mutual_coupling"] = t < 0.5
    config["csi_error_var"] = current["csi_error_var"] + t * (ideal["csi_error_var"] - current["csi_error_var"])

    error_level = 1.0 - t
    return config, error_level


def get_single_param_configs(args) -> list[dict]:
    """Generate configs to isolate each parameter's effect at the transition point."""
    base_realistic = {
        "ris_phase_bits": 2,
        "ris_amplitude_bias": 0.8,
        "ris_amplitude_scale": 0.2,
        "ris_coupling_decay": 0.15,
        "enable_phase_quantization": True,
        "enable_amplitude_coupling": True,
        "enable_mutual_coupling": True,
        "csi_error_var": args.csi_error_var,
        "csi_error_model": args.csi_error_model,
        "csi_error_target": args.csi_error_target,
        "csi_error_snr_coupled": args.csi_error_snr_coupled,
        "csi_error_snr_ref_db": args.csi_error_snr_ref_db,
        "csi_outlier_prob": args.csi_outlier_prob,
        "csi_outlier_scale": args.csi_outlier_scale,
    }

    configs = []

    # Baseline: full realistic
    configs.append(("Baseline", base_realistic.copy(), 1.0))

    # Study phase_bits transition (2 → 4 → 6 → 8)
    for bits in [3, 4, 5, 6]:
        c = base_realistic.copy()
        c["ris_phase_bits"] = bits
        configs.append((f"Phase={bits}b", c, None))

    # Study amplitude coupling toggle
    for enable in [False]:
        c = base_realistic.copy()
        c["enable_amplitude_coupling"] = enable
        c["ris_amplitude_bias"] = 1.0
        c["ris_amplitude_scale"] = 0.0
        configs.append(("Amp Ideal", c, None))

    # Study mutual coupling toggle
    for enable in [False]:
        c = base_realistic.copy()
        c["enable_mutual_coupling"] = enable
        c["ris_coupling_decay"] = 0.0
        configs.append(("Coupling=0", c, None))

    # Study phase quantization toggle (ONLY this change)
    for enable in [False]:
        c = base_realistic.copy()
        c["enable_phase_quantization"] = enable
        c["ris_phase_bits"] = 8
        configs.append(("PhaseQ OFF", c, None))

    # Combined: disable all hardware errors
    c = base_realistic.copy()
    c["ris_phase_bits"] = 8
    c["enable_phase_quantization"] = False
    c["enable_amplitude_coupling"] = False
    c["enable_mutual_coupling"] = False
    c["ris_amplitude_bias"] = 1.0
    c["ris_amplitude_scale"] = 0.0
    c["ris_coupling_decay"] = 0.0
    configs.append(("All HW Ideal", c, None))

    return configs


def get_error_summary(config: dict) -> str:
    """Generate a short summary string for the hardware config."""
    errors = []
    if config["enable_phase_quantization"]:
        errors.append(f"Q{config['ris_phase_bits']}b")
    if config["enable_amplitude_coupling"]:
        errors.append(f"A{config['ris_amplitude_bias']:.1f}+{config['ris_amplitude_scale']:.2f}cos")
    if config["enable_mutual_coupling"]:
        errors.append(f"C{config['ris_coupling_decay']:.2f}")
    if config["csi_error_var"] > 0.01:
        errors.append(f"CSI{config['csi_error_var']:.2f}")
    return "-".join(errors) if errors else "IDEAL"


@torch.inference_mode()
def evaluate_hardware(config: SystemConfig, args, snr_range):
    """Evaluate practical_baseline at given SNR values."""
    device = torch.device(args.device)
    num_symbols = args.num_bits // config.bits_per_symbol

    results = {
        "ber": [],
        "symbol_acc": [],
        "oracle_ber": [],
        "oracle_acc": [],
    }

    for snr_db in snr_range:
        err = 0
        bits = 0
        correct = 0
        oracle_err = 0
        oracle_bits = 0
        oracle_correct = 0
        total = 0

        remaining = num_symbols
        while remaining > 0:
            n = min(args.batch_size, remaining)
            batch = random_tmc_batch(config, n, device, snr_db=float(snr_db))

            # Practical baseline: uses estimated channels and quantized RIS
            practical_pred = candidate_distances_from_centers(batch["y"], batch["mu_practical"]).argmin(dim=1)

            # True-center oracle: uses mu_true (known channel) for reference
            oracle_pred = candidate_distances_from_centers(batch["y"], batch["mu_true"]).argmin(dim=1)

            e, b = bit_errors_from_joint(practical_pred, batch["labels"], config)
            err += e
            bits += b
            correct += int((practical_pred == batch["labels"]).sum().item())

            e, b = bit_errors_from_joint(oracle_pred, batch["labels"], config)
            oracle_err += e
            oracle_bits += b
            oracle_correct += int((oracle_pred == batch["labels"]).sum().item())

            total += n
            remaining -= n

        results["ber"].append(err / bits)
        results["symbol_acc"].append(correct / total)
        results["oracle_ber"].append(oracle_err / oracle_bits)
        results["oracle_acc"].append(oracle_correct / total)

    return results


def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = torch.device(args.device)
    snr_range = np.arange(args.snr_start, args.snr_stop + 1, args.snr_step)

    # Base system config (common parameters)
    base_config = {
        "n_t": args.n_t,
        "n_ris": args.n_ris,
        "s": args.s,
        "signal_energy": 1.0,
    }

    print(f"Hardware Degradation Study")
    print(f"System: n_t={args.n_t}, n_ris={args.n_ris}, s={args.s}")
    print(f"CSI: var={args.csi_error_var}, model={args.csi_error_model}, target={args.csi_error_target}")
    print(f"      snr_coupled={args.csi_error_snr_coupled}, snr_ref={args.csi_error_snr_ref_db}dB")
    print(f"      outlier_prob={args.csi_outlier_prob}, outlier_scale={args.csi_outlier_scale}")
    print(f"SNR range: {args.snr_start}-{args.snr_stop} dB, step={args.snr_step}")
    print(f"Symbols per point: {args.num_bits} bits")
    print("-" * 70)

    all_results = []

    # Use single-param analysis configs
    param_configs = get_single_param_configs(args)

    for exp_idx, (name, hw_config, error_level) in enumerate(param_configs):
        config_dict = {**base_config, **hw_config}
        config = SystemConfig(**config_dict, snr_range=snr_range)

        print(
            f"[{exp_idx+1:2d}/{len(param_configs)}] {name:<20} | "
            f"phase_bits={config.ris_phase_bits}, amp_bias={config.ris_amplitude_bias:.2f}, "
            f"coupling={config.ris_coupling_decay:.2f}, CSI={config.csi_error_var:.3f}",
            flush=True,
        )

        t0 = time.perf_counter()
        results = evaluate_hardware(config, args, snr_range)
        elapsed = time.perf_counter() - t0

        mean_ber = np.mean(results["ber"])
        mean_acc = np.mean(results["symbol_acc"])
        mean_oracle_ber = np.mean(results["oracle_ber"])
        mean_oracle_acc = np.mean(results["oracle_acc"])

        print(f"           -> BER={mean_ber:.5f}, Acc={mean_acc:.4f} | Oracle: BER={mean_oracle_ber:.5f}, Acc={mean_oracle_acc:.4f} ({elapsed:.1f}s)")

        all_results.append({
            "name": name,
            "experiment_idx": exp_idx,
            "config": hw_config,
            "snr_range": snr_range.tolist(),
            "ber": results["ber"],
            "symbol_acc": results["symbol_acc"],
            "oracle_ber": results["oracle_ber"],
            "oracle_acc": results["oracle_acc"],
            "mean_ber": mean_ber,
            "mean_acc": mean_acc,
            "mean_oracle_ber": mean_oracle_ber,
            "mean_oracle_acc": mean_oracle_acc,
        })

    # Save results
    report = {
        "snr_db": snr_range.tolist(),
        "seed": args.seed,
        "n_t": args.n_t,
        "n_ris": args.n_ris,
        "s": args.s,
        "csi_error_var": args.csi_error_var,
        "csi_error_model": args.csi_error_model,
        "csi_error_target": args.csi_error_target,
        "csi_error_snr_coupled": args.csi_error_snr_coupled,
        "csi_error_snr_ref_db": args.csi_error_snr_ref_db,
        "csi_outlier_prob": args.csi_outlier_prob,
        "csi_outlier_scale": args.csi_outlier_scale,
        "num_experiments": len(param_configs),
        "results": all_results,
    }
    report_path = Path(args.report_path) if args.report_path else ROOT / "outputs" / "reports" / "hardware_degradation.json"
    ensure_dir(report_path.parent)
    save_json(report_path, report)

    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY: Parameter-by-Parameter Analysis")
    print("=" * 100)
    print(f"{'Name':<22}  {'Phase':>5}  {'AmpBias':>7}  {'AmpScale':>8}  {'Coup':>5}  {'CSI':>5}  {'BER':>8}  {'Acc':>6}  {'Oracle BER':>10}  {'Oracle Acc':>10}")
    print("-" * 100)
    for r in all_results:
        c = r["config"]
        print(
            f"{r['name']:<22}  "
            f"{c['ris_phase_bits']:5d}  {c['ris_amplitude_bias']:7.2f}  "
            f"{c['ris_amplitude_scale']:8.2f}  "
            f"{c['ris_coupling_decay']:5.2f}  "
            f"{c['csi_error_var']:5.2f}  "
            f"{r['mean_ber']:8.5f}  {r['mean_acc']:6.4f}  "
            f"{r['mean_oracle_ber']:10.5f}  {r['mean_oracle_acc']:10.4f}"
        )

    # Plot
    try:
        import os
        os.environ.setdefault("MPLCONFIGDIR", str(ensure_dir(ROOT / "outputs" / ".matplotlib")))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # Plot 1: BER curves for all experiments (baseline and oracle)
        ax1 = axes[0]
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
        linestyles = ["-", "--"]  # solid for baseline, dashed for oracle
        for idx, r in enumerate(all_results):
            ax1.semilogy(snr_range, np.maximum(r["ber"], 1e-8), "o-",
                        color=colors[idx], linestyle="-",
                        label=f"{r['name']}", linewidth=2, markersize=5)
            ax1.semilogy(snr_range, np.maximum(r["oracle_ber"], 1e-8), "s--",
                        color=colors[idx], linestyle="--",
                        label=f"{r['name']} (Oracle)", linewidth=1.5, markersize=4, alpha=0.7)

        ax1.set_xlabel("SNR (dB)")
        ax1.set_ylabel("BER")
        ax1.set_title("Hardware Degradation: Baseline vs Oracle")
        ax1.grid(True, which="both", linestyle="--", alpha=0.6)
        ax1.legend(title="Config", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, ncol=1)

        # Plot 2: Bar chart comparison (baseline and oracle)
        ax2 = axes[1]
        names = [r["name"] for r in all_results]
        bers = [r["mean_ber"] for r in all_results]
        oracle_bers = [r["mean_oracle_ber"] for r in all_results]
        bar_colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(names)))

        x = np.arange(len(names))
        width = 0.35
        bars1 = ax2.bar(x - width/2, bers, width, color=bar_colors, edgecolor='black', linewidth=0.5, label='Baseline')
        bars2 = ax2.bar(x + width/2, oracle_bers, width, color=bar_colors, edgecolor='black', linewidth=0.5, alpha=0.6, hatch='//', label='Oracle')

        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel("Mean BER (SNR 10-30 dB)")
        ax2.set_title("Baseline vs Oracle BER Comparison")
        ax2.grid(True, axis='y', linestyle='--', alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        figure_path = Path(args.figure_path) if args.figure_path else ROOT / "outputs" / "figures" / "hardware_degradation.png"
        ensure_dir(figure_path.parent)
        plt.savefig(figure_path, dpi=300, bbox_inches="tight")
        print(f"\nFigure saved to: {figure_path}")

    except ImportError as e:
        print(f"\nMatplotlib not available, skipping plots: {e}")


if __name__ == "__main__":
    main()
