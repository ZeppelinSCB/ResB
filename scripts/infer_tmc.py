"""Evaluate TMC-Net against ideal ML, the practical baseline, and the locked true-center oracle."""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from resbdnn.config import CSI_ERROR_MODELS, CSI_ERROR_TARGETS, SystemConfig, paper_preset_description  # noqa: E402
from resbdnn.modeling import TMCNet  # noqa: E402
from resbdnn.simulation.torch_system import (  # noqa: E402
    bit_errors_from_joint,
    candidate_distances_from_centers,
    random_tmc_batch,
)
from resbdnn.utils import ensure_dir, save_json, set_random_seed  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TMC-Net.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-bits", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=4000)
    parser.add_argument("--snr-start", type=int, default=0)
    parser.add_argument("--snr-stop", type=int, default=40)
    parser.add_argument("--snr-step", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csi-error-var", type=float, default=None)
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
    parser.add_argument("--report-path", type=str, default=None)
    parser.add_argument("--figure-path", type=str, default=None)
    return parser.parse_args()


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _complex_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(torch.view_as_real(pred), torch.view_as_real(target))


def _complex_mse_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.view_as_real(pred - target).square().mean(dim=(1, 2)).to(torch.float32)


def _mean_top2_gap(distances: torch.Tensor) -> float:
    top2 = distances.topk(k=2, dim=1, largest=False).values
    return float((top2[:, 1] - top2[:, 0]).mean().item())


def _new_na_group_totals(config) -> dict[int, dict[str, float]]:
    return {
        na: {
            "count": 0,
            "practical_correct": 0,
            "corrected_correct": 0,
            "oracle_correct": 0,
            "recovered": 0,
            "coord_loss_sum": 0.0,
            "center_mse_practical_sum": 0.0,
            "center_mse_corrected_sum": 0.0,
        }
        for na in range(1, config.n_t // 2 + 1)
    }


def _update_na_group_totals(
    totals: dict[int, dict[str, float]],
    batch: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    practical_pred: torch.Tensor,
    corrected_pred: torch.Tensor,
    oracle_pred: torch.Tensor,
) -> None:
    labels = batch["labels"]
    practical_correct = practical_pred == labels
    corrected_correct = corrected_pred == labels
    oracle_correct = oracle_pred == labels
    recovered = corrected_correct & ~practical_correct
    practical_mse = _complex_mse_per_sample(batch["mu_practical"], batch["mu_true"])
    corrected_mse = _complex_mse_per_sample(outputs["mu_corrected"], batch["mu_true"])

    for na_label in batch["na_labels"].unique().tolist():
        mask = batch["na_labels"] == na_label
        na = int(na_label) + 1
        count = int(mask.sum().item())
        row = totals[na]
        row["count"] += count
        row["practical_correct"] += int(practical_correct[mask].sum().item())
        row["corrected_correct"] += int(corrected_correct[mask].sum().item())
        row["oracle_correct"] += int(oracle_correct[mask].sum().item())
        row["recovered"] += int(recovered[mask].sum().item())
        row["coord_loss_sum"] += float(corrected_mse[mask].sum().item())
        row["center_mse_practical_sum"] += float(practical_mse[mask].sum().item())
        row["center_mse_corrected_sum"] += float(corrected_mse[mask].sum().item())


def _finalize_na_group_totals(totals: dict[int, dict[str, float]]) -> list[dict[str, float]]:
    total_count = sum(row["count"] for row in totals.values())
    table = []
    for na, row in totals.items():
        count = max(int(row["count"]), 0)
        denom = max(count, 1)
        practical_acc = row["practical_correct"] / denom
        corrected_acc = row["corrected_correct"] / denom
        oracle_acc = row["oracle_correct"] / denom
        practical_mse = row["center_mse_practical_sum"] / denom
        corrected_mse = row["center_mse_corrected_sum"] / denom
        table.append(
            {
                "na": na,
                "sample_ratio": count / max(total_count, 1),
                "samples": count,
                "practical_acc": practical_acc,
                "corrected_acc": corrected_acc,
                "oracle_acc": oracle_acc,
                "oracle_gap": oracle_acc - corrected_acc,
                "recovered_samples": int(row["recovered"]),
                "recovered_ratio": row["recovered"] / denom,
                "coord_loss": row["coord_loss_sum"] / denom,
                "center_gain_db": 10.0 * math.log10(max(practical_mse, 1e-12) / max(corrected_mse, 1e-12)),
            }
        )
    return table


def _print_na_group_table(table: list[dict[str, float]]) -> None:
    print("\nNA group diagnostics:")
    print(
        f"{'na group':<8} {'sample ratio':>12} {'practical acc':>14} "
        f"{'corrected acc':>14} {'oracle acc':>10} {'oracle gap':>10} "
        f"{'center gain dB':>14} {'recovered':>10}"
    )
    print("-" * 100)
    for row in table:
        print(
            f"na={row['na']:<5} {row['sample_ratio']:>12.4f} "
            f"{row['practical_acc']:>14.4f} {row['corrected_acc']:>14.4f} "
            f"{row['oracle_acc']:>10.4f} {row['oracle_gap']:>10.4f} "
            f"{row['center_gain_db']:>14.2f} {row['recovered_samples']:>10d}"
        )


def _resolve(override, checkpoint_value, default):
    if override is not None:
        return override
    if checkpoint_value is not None:
        return checkpoint_value
    return default


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TMCNet(**checkpoint["model_args"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def build_config_from_checkpoint(config_dict, args, snr_range):
    return SystemConfig(
        n_t=config_dict["n_t"],
        n_ris=config_dict["n_ris"],
        s=config_dict["s"],
        signal_energy=config_dict["signal_energy"],
        csi_error_var=float(_resolve(args.csi_error_var, config_dict.get("csi_error_var"), 0.0)),
        csi_error_model=_resolve(args.csi_error_model, config_dict.get("csi_error_model"), "normalized"),
        csi_error_target=_resolve(args.csi_error_target, config_dict.get("csi_error_target"), "dual_link"),
        csi_error_snr_coupled=bool(
            _resolve(args.csi_error_snr_coupled, config_dict.get("csi_error_snr_coupled"), False)
        ),
        csi_error_snr_ref_db=float(_resolve(args.csi_error_snr_ref_db, config_dict.get("csi_error_snr_ref_db"), 10.0)),
        csi_outlier_prob=float(_resolve(args.csi_outlier_prob, config_dict.get("csi_outlier_prob"), 0.0)),
        csi_outlier_scale=float(_resolve(args.csi_outlier_scale, config_dict.get("csi_outlier_scale"), 0.0)),
        ris_phase_bits=int(_resolve(args.ris_phase_bits, config_dict.get("ris_phase_bits"), 2)),
        ris_amplitude_bias=float(_resolve(args.ris_amplitude_bias, config_dict.get("ris_amplitude_bias"), 0.8)),
        ris_amplitude_scale=float(_resolve(args.ris_amplitude_scale, config_dict.get("ris_amplitude_scale"), 0.2)),
        ris_coupling_decay=float(_resolve(args.ris_coupling_decay, config_dict.get("ris_coupling_decay"), 0.15)),
        enable_phase_quantization=bool(
            _resolve(args.enable_phase_quantization, config_dict.get("enable_phase_quantization"), True)
        ),
        enable_amplitude_coupling=bool(
            _resolve(args.enable_amplitude_coupling, config_dict.get("enable_amplitude_coupling"), True)
        ),
        enable_mutual_coupling=bool(
            _resolve(args.enable_mutual_coupling, config_dict.get("enable_mutual_coupling"), True)
        ),
        candidate_strategy=config_dict.get("candidate_strategy", "prefix"),
        snr_range=snr_range,
    )


@torch.inference_mode()
def evaluate(model, config, args, snr_range):
    device = torch.device(args.device)
    num_symbols = args.num_bits // config.bits_per_symbol
    results = {
        "tmc_corrected": {"ber": [], "symbol_acc": []},
        "practical_baseline": {"ber": [], "symbol_acc": []},
        "shrinkage_posterior": {"ber": [], "symbol_acc": []},
        "true_center_oracle": {"ber": [], "symbol_acc": []},
        "calibration": {
            "center_mse_practical": [],
            "center_mse_corrected": [],
            "center_gain_vs_practical_db": [],
            "delta_abs": [],
            "corrected_gap_mean": [],
        },
        "timing": {
            "tmc_seconds_per_symbol": [],
        },
    }
    na_group_totals = _new_na_group_totals(config)

    for snr_db in snr_range:
        err = {key: 0 for key in ("tmc_corrected", "practical_baseline", "shrinkage_posterior", "true_center_oracle")}
        bits = {key: 0 for key in ("tmc_corrected", "practical_baseline", "shrinkage_posterior", "true_center_oracle")}
        correct = {key: 0 for key in ("tmc_corrected", "practical_baseline", "shrinkage_posterior", "true_center_oracle")}
        total = 0
        tmc_seconds = 0.0
        center_mse_practical = center_mse_corrected = 0.0
        delta_abs = 0.0
        corrected_gap = 0.0
        remaining = num_symbols

        while remaining > 0:
            n = min(args.batch_size, remaining)
            batch = random_tmc_batch(config, n, device, snr_db=float(snr_db))

            _sync_device(device)
            start = time.perf_counter()
            outputs = model.forward_parts(
                batch["h_hat"],
                batch["g_hat"],
                batch["sigma_n"],
                batch["phi_config"],
                batch["mu_ideal"],
                batch["mu_practical"],
                csi_error_var=batch.get("csi_error_var") if getattr(model, "csi_conditioned", False) else None,
            )
            corrected_dist = candidate_distances_from_centers(batch["y"], outputs["mu_corrected"])
            corrected_pred = corrected_dist.argmin(dim=1)
            _sync_device(device)
            tmc_seconds += time.perf_counter() - start

            practical_baseline_pred = candidate_distances_from_centers(batch["y"], batch["mu_practical"]).argmin(dim=1)
            shrinkage_posterior_pred = candidate_distances_from_centers(
                batch["y"],
                batch["mu_shrinkage_posterior"],
            ).argmin(dim=1)
            true_center_oracle_pred = candidate_distances_from_centers(batch["y"], batch["mu_true"]).argmin(dim=1)
            _update_na_group_totals(
                na_group_totals,
                batch,
                outputs,
                practical_baseline_pred,
                corrected_pred,
                true_center_oracle_pred,
            )
            preds = {
                "tmc_corrected": corrected_pred,
                "practical_baseline": practical_baseline_pred,
                "shrinkage_posterior": shrinkage_posterior_pred,
                "true_center_oracle": true_center_oracle_pred,
            }
            for key, pred in preds.items():
                e, b = bit_errors_from_joint(pred, batch["labels"], config)
                err[key] += e
                bits[key] += b
                correct[key] += int((pred == batch["labels"]).sum().item())

            center_mse_practical += float(_complex_mse(batch["mu_practical"], batch["mu_true"]).item()) * n
            center_mse_corrected += float(_complex_mse(outputs["mu_corrected"], batch["mu_true"]).item()) * n
            delta_abs += float(outputs["delta_mu"].abs().mean().item()) * n
            corrected_gap += _mean_top2_gap(corrected_dist) * n

            total += n
            remaining -= n

        for key in err:
            results[key]["ber"].append(err[key] / bits[key])
            results[key]["symbol_acc"].append(correct[key] / total)

        mean_center_mse_practical = center_mse_practical / total
        mean_center_mse_corrected = center_mse_corrected / total
        results["calibration"]["center_mse_practical"].append(mean_center_mse_practical)
        results["calibration"]["center_mse_corrected"].append(mean_center_mse_corrected)
        results["calibration"]["center_gain_vs_practical_db"].append(
            10.0 * math.log10(max(mean_center_mse_practical, 1e-12) / max(mean_center_mse_corrected, 1e-12))
        )
        results["calibration"]["delta_abs"].append(delta_abs / total)
        results["calibration"]["corrected_gap_mean"].append(corrected_gap / total)
        results["timing"]["tmc_seconds_per_symbol"].append(tmc_seconds / total)

    results["practical_oracle"] = {
        "ber": list(results["practical_baseline"]["ber"]),
        "symbol_acc": list(results["practical_baseline"]["symbol_acc"]),
    }
    results["na_group_diagnostics"] = _finalize_na_group_totals(na_group_totals)
    results["feasibility"] = _compute_feasibility_metrics(results)
    return results


def _compute_feasibility_metrics(results: dict) -> dict:
    """Per-SNR feasibility decomposition relative to update.md definitions.

    G_learnable  = BER_practical - BER_posterior          (gap closable from observation)
    G_hidden     = BER_posterior - BER_true_oracle        (gap not observable from h_hat,g_hat)
    R_tmc        = (BER_practical - BER_TMC) / (BER_practical - BER_posterior)
    R_observable = G_learnable / (BER_practical - BER_true_oracle)
    """
    eps = 1e-9
    practical = np.asarray(results["practical_baseline"]["ber"], dtype=np.float64)
    tmc = np.asarray(results["tmc_corrected"]["ber"], dtype=np.float64)
    posterior = np.asarray(results["shrinkage_posterior"]["ber"], dtype=np.float64)
    true_oracle = np.asarray(results["true_center_oracle"]["ber"], dtype=np.float64)

    g_learnable = practical - posterior
    g_hidden = posterior - true_oracle
    g_total = practical - true_oracle

    denom_learnable = np.where(np.abs(g_learnable) < eps, np.nan, g_learnable)
    denom_total = np.where(np.abs(g_total) < eps, np.nan, g_total)
    r_tmc = (practical - tmc) / denom_learnable
    r_observable = g_learnable / denom_total

    sum_num = float((practical - tmc).sum())
    sum_den = float(g_learnable.sum())
    r_tmc_weighted = sum_num / sum_den if abs(sum_den) > eps else None

    valid_mask = g_learnable > 0.01
    if valid_mask.any():
        num_v = float((practical[valid_mask] - tmc[valid_mask]).sum())
        den_v = float(g_learnable[valid_mask].sum())
        r_tmc_valid = num_v / den_v if abs(den_v) > eps else None
    else:
        r_tmc_valid = None

    return {
        "G_learnable": g_learnable.tolist(),
        "G_hidden": g_hidden.tolist(),
        "R_tmc": [None if np.isnan(v) else float(v) for v in r_tmc],
        "R_observable": [None if np.isnan(v) else float(v) for v in r_observable],
        "R_tmc_mean": float(np.nanmean(r_tmc)) if np.any(~np.isnan(r_tmc)) else None,
        "R_observable_mean": float(np.nanmean(r_observable)) if np.any(~np.isnan(r_observable)) else None,
        "R_tmc_weighted": r_tmc_weighted,
        "R_tmc_valid": r_tmc_valid,
        "valid_snr_mask": valid_mask.tolist(),
        "G_learnable_mean": float(np.mean(g_learnable)),
        "G_hidden_mean": float(np.mean(g_hidden)),
    }


def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = torch.device(args.device)
    model, checkpoint = load_model(ROOT / args.checkpoint, device)
    config_dict = checkpoint["system_config"]
    model_class = checkpoint.get("model_class", "TMCNet")
    snr_range = np.arange(args.snr_start, args.snr_stop + 1, args.snr_step)
    config = build_config_from_checkpoint(config_dict, args, snr_range)
    preset_desc = paper_preset_description(config_dict.get("paper_preset"))

    print(
        f"Evaluating {model_class} ({args.num_bits} bits, n_t={config.n_t}, n_ris={config.n_ris}, s={config.s}"
        + (f", preset={preset_desc}" if preset_desc else "")
        + f", sigma_e^2={config.csi_error_var:g}, csi={config.csi_error_model}/{config.csi_error_target}"
        + f", phase_bits={config.ris_phase_bits}, amp={config.ris_amplitude_bias:g}+{config.ris_amplitude_scale:g}cos(phi)"
        + f", coupling_decay={config.ris_coupling_decay:g}, hw=({int(config.enable_phase_quantization)},{int(config.enable_amplitude_coupling)},{int(config.enable_mutual_coupling)}))...",
        flush=True,
    )
    t0 = time.perf_counter()
    results = evaluate(model, config, args, snr_range)
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s")
    print(f"  TMC corrected avg BER={np.mean(results['tmc_corrected']['ber']):.5f}")
    print(f"  Practical baseline avg BER={np.mean(results['practical_baseline']['ber']):.5f}")
    print(f"  Shrinkage posterior avg BER={np.mean(results['shrinkage_posterior']['ber']):.5f}")
    print(f"  Locked true-center oracle avg BER={np.mean(results['true_center_oracle']['ber']):.5f}")
    print(f"  Center gain vs practical avg={np.mean(results['calibration']['center_gain_vs_practical_db']):.2f} dB")
    print(f"  Delta magnitude avg={np.mean(results['calibration']['delta_abs']):.4f}")
    feas = results["feasibility"]
    print(
        "  R_tmc_weighted={rw}  R_tmc_valid={rv}  R_tmc_mean(per-SNR)={rm}  "
        "R_observable avg={ro}  G_learnable avg={gl:.5f}  G_hidden avg={gh:.5f}".format(
            rw=("nan" if feas["R_tmc_weighted"] is None else f"{feas['R_tmc_weighted']:.4f}"),
            rv=("nan" if feas["R_tmc_valid"] is None else f"{feas['R_tmc_valid']:.4f}"),
            rm=("nan" if feas["R_tmc_mean"] is None else f"{feas['R_tmc_mean']:.4f}"),
            ro=("nan" if feas["R_observable_mean"] is None else f"{feas['R_observable_mean']:.4f}"),
            gl=feas["G_learnable_mean"],
            gh=feas["G_hidden_mean"],
        )
    )
    _print_na_group_table(results["na_group_diagnostics"])

    print(f"\n{'SNR':>4}  {'TMC':>10}  {'Baseline':>10}  {'Shrink':>10}  {'TrueCtr':>10}  {'R_tmc':>8}  {'R_obs':>8}")
    for idx, snr_db in enumerate(snr_range):
        r_t = feas["R_tmc"][idx]
        r_o = feas["R_observable"][idx]
        print(
            f"{snr_db:4d}  {results['tmc_corrected']['ber'][idx]:10.5f}  "
            f"{results['practical_baseline']['ber'][idx]:10.5f}  "
            f"{results['shrinkage_posterior']['ber'][idx]:10.5f}  "
            f"{results['true_center_oracle']['ber'][idx]:10.5f}  "
            f"{'nan' if r_t is None else f'{r_t:8.4f}':>8}  "
            f"{'nan' if r_o is None else f'{r_o:8.4f}':>8}"
        )

    report = {
        "snr_db": snr_range.tolist(),
        "seed": args.seed,
        "paper_preset": config_dict.get("paper_preset"),
        "model_class": model_class,
        "spectral_efficiency": config.spectral_efficiency,
        "n_t": config.n_t,
        "n_ris": config.n_ris,
        "s": config.s,
        "csi_error_var": config.csi_error_var,
        "csi_error_model": config.csi_error_model,
        "csi_error_target": config.csi_error_target,
        "csi_error_snr_coupled": config.csi_error_snr_coupled,
        "csi_error_snr_ref_db": config.csi_error_snr_ref_db,
        "csi_outlier_prob": config.csi_outlier_prob,
        "csi_outlier_scale": config.csi_outlier_scale,
        "ris_phase_bits": config.ris_phase_bits,
        "ris_amplitude_bias": config.ris_amplitude_bias,
        "ris_amplitude_scale": config.ris_amplitude_scale,
        "ris_coupling_decay": config.ris_coupling_decay,
        "enable_phase_quantization": config.enable_phase_quantization,
        "enable_amplitude_coupling": config.enable_amplitude_coupling,
        "enable_mutual_coupling": config.enable_mutual_coupling,
        "checkpoint": str(ROOT / args.checkpoint),
        **results,
    }
    report_path = Path(args.report_path) if args.report_path else ROOT / "outputs" / "reports" / "tmc_infer.json"
    save_json(report_path, report)

    try:
        import os

        os.environ.setdefault("MPLCONFIGDIR", str(ensure_dir(ROOT / "outputs" / ".matplotlib")))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.semilogy(snr_range, np.maximum(results["tmc_corrected"]["ber"], 1e-7), "ro-", label="BER_TMC")
        plt.semilogy(
            snr_range,
            np.maximum(results["practical_baseline"]["ber"], 1e-7),
            "b^-.",
            label="BER_practical",
        )
        plt.semilogy(
            snr_range,
            np.maximum(results["shrinkage_posterior"]["ber"], 1e-7),
            "gs--",
            label="BER_posterior (shrinkage)",
        )
        plt.semilogy(
            snr_range,
            np.maximum(results["true_center_oracle"]["ber"], 1e-7),
            "m--",
            label="BER_true_oracle",
        )
        plt.xlabel("SNR (dB)")
        plt.ylabel("BER")
        plt.title(f"{model_class} evaluation")
        plt.grid(True, which="both", linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        figure_path = Path(args.figure_path) if args.figure_path else ROOT / "outputs" / "figures" / "tmc_ber.png"
        ensure_dir(figure_path.parent)
        plt.savefig(figure_path, dpi=300)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
