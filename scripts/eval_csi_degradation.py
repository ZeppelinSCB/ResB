"""Evaluate practical_baseline under varying CSI error levels.

This script studies how CSI estimation errors degrade symbol detection accuracy,
keeping RIS hardware parameters fixed while sweeping CSI-related parameters.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from resbdnn.config import SystemConfig
from resbdnn.simulation.torch_system import (
    bit_errors_from_joint,
    candidate_distances_from_centers,
    random_tmc_batch,
)
from resbdnn.utils import ensure_dir, save_json, set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CSI error degradation on practical_baseline.")
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
    # RIS params (fixed across experiments)
    parser.add_argument("--ris-phase-bits", type=int, default=2)
    parser.add_argument("--ris-amplitude-bias", type=float, default=0.8)
    parser.add_argument("--ris-amplitude-scale", type=float, default=0.2)
    parser.add_argument("--ris-coupling-decay", type=float, default=0.15)
    parser.add_argument("--enable-phase-quantization", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--enable-amplitude-coupling", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--enable-mutual-coupling", type=lambda x: x.lower() == "true", default=True)
    # Output
    parser.add_argument("--report-path", type=str, default=None)
    parser.add_argument("--figure-path", type=str, default=None)
    return parser.parse_args()


def get_csi_variation_configs(include_snr_coupled: bool = False) -> list[tuple[str, dict]]:
    """Generate configs to vary CSI parameters."""
    configs = []

    # Baseline CSI error vars (fixed)
    csi_vars = [0.5, 0.3, 0.1, 0.05, 0.01, 0.0]

    for csi_var in csi_vars:
        c = {
            "csi_error_var": csi_var,
            "csi_error_model": "normalized",
            "csi_error_target": "dual_link",
            "csi_error_snr_coupled": False,
            "csi_error_snr_ref_db": 10.0,
            "csi_outlier_prob": 0.0,
            "csi_outlier_scale": 0.0,
        }
        name = f"CSI={csi_var}"
        if csi_var == 0.0:
            name = "CSI=0 (Ideal)"
        configs.append((name, c))

    # SNR-coupled CSI error
    if include_snr_coupled:
        for csi_var in [0.5, 0.3, 0.1]:
            c = {
                "csi_error_var": csi_var,
                "csi_error_model": "normalized",
                "csi_error_target": "dual_link",
                "csi_error_snr_coupled": True,
                "csi_error_snr_ref_db": 10.0,
                "csi_outlier_prob": 0.0,
                "csi_outlier_scale": 0.0,
            }
            name = f"CSI={csi_var} (SNR-coupled)"
            configs.append((name, c))

    return configs


def get_model_target_configs() -> list[tuple[str, dict]]:
    """Generate configs to study different CSI models and targets."""
    configs = []

    base = {
        "csi_error_var": 0.5,
        "csi_error_model": "normalized",
        "csi_error_target": "dual_link",
        "csi_error_snr_coupled": False,
        "csi_error_snr_ref_db": 10.0,
        "csi_outlier_prob": 0.0,
        "csi_outlier_scale": 0.0,
    }

    # Study error models
    for model in ["normalized", "additive"]:
        c = base.copy()
        c["csi_error_model"] = model
        configs.append((f"Model={model}", c))

    # Study error targets
    for target in ["dual_link", "h_only", "g_only"]:
        c = base.copy()
        c["csi_error_target"] = target
        configs.append((f"Target={target}", c))

    return configs


def get_outlier_configs() -> list[tuple[str, dict]]:
    """Generate configs to study CSI outliers."""
    configs = []

    base = {
        "csi_error_var": 0.3,
        "csi_error_model": "normalized",
        "csi_error_target": "dual_link",
        "csi_error_snr_coupled": False,
        "csi_error_snr_ref_db": 10.0,
        "csi_outlier_prob": 0.0,
        "csi_outlier_scale": 0.0,
    }

    # Study outlier probability
    for prob in [0.01, 0.05, 0.1, 0.2]:
        c = base.copy()
        c["csi_outlier_prob"] = prob
        c["csi_outlier_scale"] = 2.0  # 2x error scale for outliers
        configs.append((f"Outlier_prob={prob}", c))

    return configs


@torch.inference_mode()
def evaluate_csi(config: SystemConfig, args, snr_range):
    """Evaluate practical_baseline at given SNR values with CSI parameters."""
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

            # Practical baseline
            practical_pred = candidate_distances_from_centers(batch["y"], batch["mu_practical"]).argmin(dim=1)

            # True-center oracle
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

    # Fixed RIS config
    ris_config = {
        "ris_phase_bits": args.ris_phase_bits,
        "ris_amplitude_bias": args.ris_amplitude_bias,
        "ris_amplitude_scale": args.ris_amplitude_scale,
        "ris_coupling_decay": args.ris_coupling_decay,
        "enable_phase_quantization": args.enable_phase_quantization,
        "enable_amplitude_coupling": args.enable_amplitude_coupling,
        "enable_mutual_coupling": args.enable_mutual_coupling,
    }

    # Base system config
    base_config = {
        "n_t": args.n_t,
        "n_ris": args.n_ris,
        "s": args.s,
        "signal_energy": 1.0,
    }

    print("=" * 80)
    print("CSI Error Degradation Study")
    print("=" * 80)
    print(f"System: n_t={args.n_t}, n_ris={args.n_ris}, s={args.s}")
    print(f"Fixed RIS: phase_bits={args.ris_phase_bits}, amp_bias={args.ris_amplitude_bias:.2f}")
    print(f"           coupling={args.ris_coupling_decay:.2f}, hw=({int(args.enable_phase_quantization)},"
          f"{int(args.enable_amplitude_coupling)},{int(args.enable_mutual_coupling)})")
    print(f"SNR range: {args.snr_start}-{args.snr_stop} dB, step={args.snr_step}")
    print(f"Symbols per point: {args.num_bits} bits")
    print("-" * 80)

    all_results = []

    # Part 1: Vary CSI error variance
    print("\n[Part 1] Varying CSI error variance (fixed vs SNR-coupled)")
    print("-" * 50)
    csi_var_configs = get_csi_variation_configs(include_snr_coupled=True)

    for exp_idx, (name, csi_config) in enumerate(csi_var_configs):
        config_dict = {**base_config, **ris_config, **csi_config}
        config = SystemConfig(**config_dict, snr_range=snr_range)

        print(
            f"[{exp_idx+1:2d}/{len(csi_var_configs)}] {name:<20} | "
            f"csi_var={config.csi_error_var:.3f}, model={config.csi_error_model}",
            flush=True,
        )

        t0 = time.perf_counter()
        results = evaluate_csi(config, args, snr_range)
        elapsed = time.perf_counter() - t0

        mean_ber = np.mean(results["ber"])
        mean_acc = np.mean(results["symbol_acc"])
        mean_oracle_ber = np.mean(results["oracle_ber"])
        mean_oracle_acc = np.mean(results["oracle_acc"])

        print(f"           -> BER={mean_ber:.5f}, Acc={mean_acc:.4f} | Oracle: BER={mean_oracle_ber:.5f}, Acc={mean_oracle_acc:.4f} ({elapsed:.1f}s)")

        all_results.append({
            "category": "csi_variance",
            "name": name,
            "config": {**ris_config, **csi_config},
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

    # Part 2: Vary CSI models and targets
    print("\n[Part 2] Varying CSI models and targets")
    print("-" * 50)
    model_target_configs = get_model_target_configs()

    for exp_idx, (name, csi_config) in enumerate(model_target_configs):
        config_dict = {**base_config, **ris_config, **csi_config}
        config = SystemConfig(**config_dict, snr_range=snr_range)

        print(
            f"[{exp_idx+1:2d}/{len(model_target_configs)}] {name:<20} | "
            f"model={config.csi_error_model}, target={config.csi_error_target}",
            flush=True,
        )

        t0 = time.perf_counter()
        results = evaluate_csi(config, args, snr_range)
        elapsed = time.perf_counter() - t0

        mean_ber = np.mean(results["ber"])
        mean_acc = np.mean(results["symbol_acc"])
        mean_oracle_ber = np.mean(results["oracle_ber"])
        mean_oracle_acc = np.mean(results["oracle_acc"])

        print(f"           -> BER={mean_ber:.5f}, Acc={mean_acc:.4f} | Oracle: BER={mean_oracle_ber:.5f}, Acc={mean_oracle_acc:.4f} ({elapsed:.1f}s)")

        all_results.append({
            "category": "model_target",
            "name": name,
            "config": {**ris_config, **csi_config},
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

    # Part 3: Vary CSI outliers
    print("\n[Part 3] Varying CSI outliers")
    print("-" * 50)
    outlier_configs = get_outlier_configs()

    for exp_idx, (name, csi_config) in enumerate(outlier_configs):
        config_dict = {**base_config, **ris_config, **csi_config}
        config = SystemConfig(**config_dict, snr_range=snr_range)

        print(
            f"[{exp_idx+1:2d}/{len(outlier_configs)}] {name:<20} | "
            f"outlier_prob={config.csi_outlier_prob:.2f}, scale={config.csi_outlier_scale:.2f}",
            flush=True,
        )

        t0 = time.perf_counter()
        results = evaluate_csi(config, args, snr_range)
        elapsed = time.perf_counter() - t0

        mean_ber = np.mean(results["ber"])
        mean_acc = np.mean(results["symbol_acc"])
        mean_oracle_ber = np.mean(results["oracle_ber"])
        mean_oracle_acc = np.mean(results["oracle_acc"])

        print(f"           -> BER={mean_ber:.5f}, Acc={mean_acc:.4f} | Oracle: BER={mean_oracle_ber:.5f}, Acc={mean_oracle_acc:.4f} ({elapsed:.1f}s)")

        all_results.append({
            "category": "outliers",
            "name": name,
            "config": {**ris_config, **csi_config},
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
        "ris_config": ris_config,
        "results": all_results,
    }
    report_path = Path(args.report_path) if args.report_path else ROOT / "outputs" / "reports" / "csi_degradation.json"
    ensure_dir(report_path.parent)
    save_json(report_path, report)

    # Print summary tables
    print("\n" + "=" * 110)
    print("SUMMARY: CSI Error Variance Analysis (Fixed vs SNR-coupled)")
    print("=" * 110)
    print(f"{'Name':<25}  {'CSI_var':>8}  {'SNR_coupled':>12}  {'BER':>10}  {'Oracle BER':>10}  {'Delta':>10}")
    print("-" * 110)

    baseline_ber = {}
    for r in all_results:
        if r["category"] == "csi_variance":
            c = r["config"]
            # Calculate delta from ideal baseline
            delta = r['mean_ber'] - 0.0  # Compare to ideal
            coupled_str = "Yes" if c["csi_error_snr_coupled"] else "No"
            print(
                f"{r['name']:<25}  "
                f"{c['csi_error_var']:8.3f}  "
                f"{coupled_str:>12}  "
                f"{r['mean_ber']:10.5f}  "
                f"{r['mean_oracle_ber']:10.5f}  "
                f"{delta:10.5f}"
            )
            # Store for comparison
            var_key = f"{c['csi_error_var']}"
            if not c["csi_error_snr_coupled"]:
                baseline_ber[var_key] = r['mean_ber']

    # Print SNR-coupled comparison
    print("\n" + "-" * 110)
    print("SNR-coupled vs Fixed CSI Error:")
    print(f"{'CSI_var':<10}  {'Fixed BER':>12}  {'SNR-coupled BER':>15}  {'Improvement':>12}")
    print("-" * 50)
    for r in all_results:
        if r["category"] == "csi_variance" and r["config"]["csi_error_snr_coupled"]:
            var_key = f"{r['config']['csi_error_var']}"
            if var_key in baseline_ber:
                fixed_ber = baseline_ber[var_key]
                coupled_ber = r['mean_ber']
                improvement = fixed_ber - coupled_ber
                improvement_pct = (improvement / fixed_ber * 100) if fixed_ber > 0 else 0
                print(
                    f"{var_key:<10}  "
                    f"{fixed_ber:>12.5f}  "
                    f"{coupled_ber:>15.5f}  "
                    f"{improvement:>10.5f} ({improvement_pct:.1f}%)"
                )

    print("\n" + "=" * 90)
    print("SUMMARY: CSI Model/Target Analysis")
    print("=" * 90)
    print(f"{'Name':<20}  {'CSI_var':>8}  {'Model':>10}  {'Target':>10}  {'BER':>10}  {'Oracle BER':>10}")
    print("-" * 90)

    for r in all_results:
        if r["category"] == "model_target":
            c = r["config"]
            print(
                f"{r['name']:<20}  "
                f"{c['csi_error_var']:8.3f}  "
                f"{c['csi_error_model']:>10}  "
                f"{c['csi_error_target']:>10}  "
                f"{r['mean_ber']:10.5f}  "
                f"{r['mean_oracle_ber']:10.5f}"
            )

    print("\n" + "=" * 90)
    print("SUMMARY: CSI Outlier Analysis")
    print("=" * 90)
    print(f"{'Name':<20}  {'Outlier_prob':>12}  {'Outlier_scale':>13}  {'BER':>10}  {'Oracle BER':>10}")
    print("-" * 90)

    for r in all_results:
        if r["category"] == "outliers":
            c = r["config"]
            print(
                f"{r['name']:<20}  "
                f"{c['csi_outlier_prob']:12.2f}  "
                f"{c['csi_outlier_scale']:13.2f}  "
                f"{r['mean_ber']:10.5f}  "
                f"{r['mean_oracle_ber']:10.5f}"
            )

    # Plot
    try:
        import os
        os.environ.setdefault("MPLCONFIGDIR", str(ensure_dir(ROOT / "outputs" / ".matplotlib")))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(18, 14))

        # Plot 1: CSI variance BER curves (fixed vs SNR-coupled)
        ax1 = axes[0, 0]
        variance_results = [r for r in all_results if r["category"] == "csi_variance"]
        # Separate fixed and SNR-coupled
        fixed_results = [r for r in variance_results if not r["config"]["csi_error_snr_coupled"]]
        coupled_results = [r for r in variance_results if r["config"]["csi_error_snr_coupled"]]

        colors_fixed = plt.cm.Blues(np.linspace(0.4, 0.9, len(fixed_results)))
        colors_coupled = plt.cm.Oranges(np.linspace(0.4, 0.9, len(coupled_results)))

        for idx, r in enumerate(fixed_results):
            ax1.semilogy(snr_range, np.maximum(r["ber"], 1e-8), "o-",
                        color=colors_fixed[idx], label=f"{r['name']} (Fixed)", linewidth=2, markersize=5)

        for idx, r in enumerate(coupled_results):
            ax1.semilogy(snr_range, np.maximum(r["ber"], 1e-8), "s--",
                        color=colors_coupled[idx], label=f"{r['name']}", linewidth=2, markersize=5)

        ax1.set_xlabel("SNR (dB)")
        ax1.set_ylabel("BER")
        ax1.set_title("CSI Variance: Fixed vs SNR-coupled (solid=fixed, dashed=SNR-coupled)")
        ax1.grid(True, which="both", linestyle="--", alpha=0.6)
        ax1.legend(fontsize=7, loc='lower left')

        # Plot 2: Direct comparison of fixed vs SNR-coupled
        ax2 = axes[0, 1]
        # Pair fixed and coupled for same var
        comparison_data = []
        for r in fixed_results:
            var = r["config"]["csi_error_var"]
            coupled = next((c for c in coupled_results if c["config"]["csi_error_var"] == var), None)
            if coupled:
                comparison_data.append((f"CSI={var}", r, coupled))

        names = [d[0] for d in comparison_data]
        x = np.arange(len(names))
        width = 0.35
        fixed_bers = [d[1]["mean_ber"] for d in comparison_data]
        coupled_bers = [d[2]["mean_ber"] for d in comparison_data]

        ax2.bar(x - width/2, fixed_bers, width, label="Fixed CSI", color="steelblue", edgecolor='black')
        ax2.bar(x + width/2, coupled_bers, width, label="SNR-coupled CSI", color="darkorange", edgecolor='black')
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=0, fontsize=10)
        ax2.set_ylabel("Mean BER")
        ax2.set_title("Fixed vs SNR-coupled CSI Error")
        ax2.grid(True, axis='y', linestyle='--', alpha=0.6)
        ax2.legend()

        # Add improvement annotations
        for i, (fixed, coupled) in enumerate(zip(fixed_bers, coupled_bers)):
            if fixed > 0:
                improvement = (fixed - coupled) / fixed * 100
                ax2.annotate(f"-{improvement:.1f}%", xy=(i + width/2, coupled),
                           xytext=(i + width/2, coupled - 0.01),
                           ha='center', fontsize=8, color='green')

        # Plot 3: SNR-coupled vs Fixed BER across SNR range (for selected vars)
        ax3 = axes[1, 0]
        selected_vars = [0.5, 0.3, 0.1]
        colors_select = plt.cm.tab10(np.linspace(0, 1, len(selected_vars) * 2))
        marker_idx = 0
        for var in selected_vars:
            fixed_r = next((r for r in fixed_results if r["config"]["csi_error_var"] == var), None)
            coupled_r = next((r for r in coupled_results if r["config"]["csi_error_var"] == var), None)
            if fixed_r and coupled_r:
                ax3.semilogy(snr_range, np.maximum(fixed_r["ber"], 1e-8), "o-",
                           color=colors_select[marker_idx], label=f"Fixed CSI={var}", linewidth=2)
                ax3.semilogy(snr_range, np.maximum(coupled_r["ber"], 1e-8), "s--",
                           color=colors_select[marker_idx+1], label=f"SNR-coupled CSI={var}", linewidth=2)
                marker_idx += 2

        ax3.set_xlabel("SNR (dB)")
        ax3.set_ylabel("BER")
        ax3.set_title("Fixed vs SNR-coupled: BER vs SNR")
        ax3.grid(True, which="both", linestyle="--", alpha=0.6)
        ax3.legend(fontsize=8, ncol=2)

        # Plot 4: Model/Target comparison
        ax4 = axes[1, 1]
        model_results = [r for r in all_results if r["category"] == "model_target"]
        colors = plt.cm.Set2(np.linspace(0, 1, len(model_results)))
        for idx, r in enumerate(model_results):
            ax4.semilogy(snr_range, np.maximum(r["ber"], 1e-8), "o-",
                        color=colors[idx], label=f"{r['name']}", linewidth=2, markersize=5)
        ax4.set_xlabel("SNR (dB)")
        ax4.set_ylabel("BER")
        ax4.set_title("CSI Model/Target Comparison")
        ax4.grid(True, which="both", linestyle="--", alpha=0.6)
        ax4.legend(fontsize=8)
        ax2.legend()

        # Plot 3: Model/Target comparison
        ax3 = axes[1, 0]
        model_results = [r for r in all_results if r["category"] == "model_target"]
        colors = plt.cm.Set2(np.linspace(0, 1, len(model_results)))
        for idx, r in enumerate(model_results):
            ax3.semilogy(snr_range, np.maximum(r["ber"], 1e-8), "o-",
                        color=colors[idx], label=f"{r['name']}", linewidth=2, markersize=5)
        ax3.set_xlabel("SNR (dB)")
        ax3.set_ylabel("BER")
        ax3.set_title("CSI Model/Target Comparison")
        ax3.grid(True, which="both", linestyle="--", alpha=0.6)
        ax3.legend(fontsize=8)

        # Plot 4: Outlier comparison
        ax4 = axes[1, 1]
        outlier_results = [r for r in all_results if r["category"] == "outliers"]
        colors = plt.cm.Set1(np.linspace(0, 1, len(outlier_results)))
        for idx, r in enumerate(outlier_results):
            ax4.semilogy(snr_range, np.maximum(r["ber"], 1e-8), "o-",
                        color=colors[idx], label=f"{r['name']}", linewidth=2, markersize=5)
        ax4.set_xlabel("SNR (dB)")
        ax4.set_ylabel("BER")
        ax4.set_title("CSI Outlier Analysis")
        ax4.grid(True, which="both", linestyle="--", alpha=0.6)
        ax4.legend(fontsize=8)

        plt.tight_layout()
        figure_path = Path(args.figure_path) if args.figure_path else ROOT / "outputs" / "figures" / "csi_degradation.png"
        ensure_dir(figure_path.parent)
        plt.savefig(figure_path, dpi=300, bbox_inches="tight")
        print(f"\nFigure saved to: {figure_path}")

    except ImportError as e:
        print(f"\nMatplotlib not available, skipping plots: {e}")


if __name__ == "__main__":
    main()
