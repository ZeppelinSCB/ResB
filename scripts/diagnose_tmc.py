"""Diagnose TMC checkpoints with confusion, alignment, and template-vector plots."""

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

from infer_tmc import build_config_from_checkpoint, evaluate, load_model  # noqa: E402
from resbdnn.config import CSI_ERROR_MODELS, CSI_ERROR_TARGETS, paper_preset_description  # noqa: E402
from resbdnn.simulation.torch_system import candidate_distances_from_centers, random_tmc_batch  # noqa: E402
from resbdnn.utils import ensure_dir, save_json, set_random_seed  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose TMC behavior beyond BER curves.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-bits", type=int, default=40000)
    parser.add_argument("--batch-size", type=int, default=4000)
    parser.add_argument("--snr-start", type=int, default=0)
    parser.add_argument("--snr-stop", type=int, default=30)
    parser.add_argument("--snr-step", type=int, default=2)
    parser.add_argument("--diagnostic-snr-db", type=float, default=20.0)
    parser.add_argument("--sample-batch-size", type=int, default=1024)
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
    parser.add_argument("--figure-dir", type=str, default=None)
    return parser.parse_args()


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _update_confusion(matrix: np.ndarray, labels: torch.Tensor, preds: torch.Tensor, num_classes: int) -> None:
    flat = labels * num_classes + preds
    counts = torch.bincount(flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    matrix += counts.cpu().numpy()


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    return matrix / np.maximum(row_sums, 1)


@torch.inference_mode()
def collect_diagnostics(model, config, args, snr_range):
    device = torch.device(args.device)
    num_symbols = args.num_bits // config.bits_per_symbol
    num_classes = config.num_candidates
    corrected_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    ideal_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    practical_baseline_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    delta_cosine_sum = 0.0
    delta_ratio_sum = 0.0
    delta_count = 0

    for snr_db in snr_range:
        remaining = num_symbols
        while remaining > 0:
            n = min(args.batch_size, remaining)
            batch = random_tmc_batch(config, n, device, snr_db=float(snr_db))
            outputs = model.forward_parts(
                batch["h_hat"],
                batch["g_hat"],
                batch["sigma_n"],
                batch["phi_config"],
                batch["mu_ideal"],
                batch["mu_practical"],
            )

            corrected_pred = candidate_distances_from_centers(batch["y"], outputs["mu_corrected"]).argmin(dim=1)
            ideal_pred = candidate_distances_from_centers(batch["y"], batch["mu_ideal"]).argmin(dim=1)
            practical_baseline_pred = candidate_distances_from_centers(batch["y"], batch["mu_practical"]).argmin(dim=1)

            _update_confusion(corrected_confusion, batch["labels"], corrected_pred, num_classes)
            _update_confusion(ideal_confusion, batch["labels"], ideal_pred, num_classes)
            _update_confusion(practical_baseline_confusion, batch["labels"], practical_baseline_pred, num_classes)

            pred_vec = torch.view_as_real(outputs["delta_mu"]).reshape(-1, 2)
            target_vec = torch.view_as_real(batch["delta_target"]).reshape(-1, 2)
            pred_norm = pred_vec.norm(dim=1)
            target_norm = target_vec.norm(dim=1)
            valid = (pred_norm > 1e-6) & (target_norm > 1e-6)
            if valid.any():
                cos = (pred_vec[valid] * target_vec[valid]).sum(dim=1) / (pred_norm[valid] * target_norm[valid])
                ratio = pred_norm[valid] / target_norm[valid]
                delta_cosine_sum += float(cos.sum().item())
                delta_ratio_sum += float(ratio.sum().item())
                delta_count += int(valid.sum().item())
            remaining -= n

    sample_batch = random_tmc_batch(config, args.sample_batch_size, device, snr_db=float(args.diagnostic_snr_db))
    outputs = model.forward_parts(
        sample_batch["h_hat"],
        sample_batch["g_hat"],
        sample_batch["sigma_n"],
        sample_batch["phi_config"],
        sample_batch["mu_ideal"],
        sample_batch["mu_practical"],
    )
    target_strength = sample_batch["delta_target"].abs().mean(dim=1)
    sample_idx = int(target_strength.argmax().item())
    corrected_pred = candidate_distances_from_centers(
        sample_batch["y"][sample_idx : sample_idx + 1],
        outputs["mu_corrected"][sample_idx : sample_idx + 1],
    ).argmin(dim=1)
    ideal_pred = candidate_distances_from_centers(
        sample_batch["y"][sample_idx : sample_idx + 1],
        sample_batch["mu_ideal"][sample_idx : sample_idx + 1],
    ).argmin(dim=1)
    practical_baseline_pred = candidate_distances_from_centers(
        sample_batch["y"][sample_idx : sample_idx + 1],
        sample_batch["mu_practical"][sample_idx : sample_idx + 1],
    ).argmin(dim=1)
    true_center_oracle_pred = candidate_distances_from_centers(
        sample_batch["y"][sample_idx : sample_idx + 1],
        sample_batch["mu_true"][sample_idx : sample_idx + 1],
    ).argmin(dim=1)

    sample = {
        "snr_db": float(sample_batch["snr"][sample_idx].item()),
        "label": int(sample_batch["labels"][sample_idx].item()),
        "ideal_pred": int(ideal_pred.item()),
        "corrected_pred": int(corrected_pred.item()),
        "practical_baseline_pred": int(practical_baseline_pred.item()),
        "practical_oracle_pred": int(practical_baseline_pred.item()),
        "true_center_oracle_pred": int(true_center_oracle_pred.item()),
        "mu_ideal": [[float(z.real), float(z.imag)] for z in sample_batch["mu_ideal"][sample_idx].cpu().tolist()],
        "mu_practical": [[float(z.real), float(z.imag)] for z in sample_batch["mu_practical"][sample_idx].cpu().tolist()],
        "mu_practical_oracle": [[float(z.real), float(z.imag)] for z in sample_batch["mu_practical"][sample_idx].cpu().tolist()],
        "mu_true": [[float(z.real), float(z.imag)] for z in sample_batch["mu_true"][sample_idx].cpu().tolist()],
        "mu_corrected": [[float(z.real), float(z.imag)] for z in outputs["mu_corrected"][sample_idx].cpu().tolist()],
    }

    return {
        "alignment": {
            "delta_direction_cosine": delta_cosine_sum / max(delta_count, 1),
            "delta_magnitude_ratio": delta_ratio_sum / max(delta_count, 1),
            "count": delta_count,
        },
        "confusion": {
            "corrected": corrected_confusion.tolist(),
            "ideal": ideal_confusion.tolist(),
            "practical_baseline": practical_baseline_confusion.tolist(),
            "oracle": practical_baseline_confusion.tolist(),
            "corrected_row_normalized": _row_normalize(corrected_confusion).tolist(),
            "ideal_row_normalized": _row_normalize(ideal_confusion).tolist(),
            "practical_baseline_row_normalized": _row_normalize(practical_baseline_confusion).tolist(),
            "oracle_row_normalized": _row_normalize(practical_baseline_confusion).tolist(),
        },
        "sample": sample,
    }


def _save_figures(report, figure_dir: Path) -> dict[str, str]:
    paths = {}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return paths

    figure_dir = ensure_dir(figure_dir)
    corrected = np.asarray(report["confusion"]["corrected_row_normalized"], dtype=np.float32)
    ideal = np.asarray(report["confusion"]["ideal_row_normalized"], dtype=np.float32)
    practical_baseline = np.asarray(report["confusion"]["practical_baseline_row_normalized"], dtype=np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, matrix, title in zip(
        axes,
        (ideal, practical_baseline, corrected),
        ("Ideal ML confusion", "Practical baseline confusion", "TMC confusion"),
    ):
        im = ax.imshow(matrix, vmin=0.0, vmax=max(matrix.max(), 1e-6), cmap="magma")
        ax.set_title(title)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    confusion_path = figure_dir / "tmc_confusion.png"
    fig.savefig(confusion_path, dpi=240)
    plt.close(fig)
    paths["confusion"] = str(confusion_path)

    sample = report["sample"]
    mu_ideal = np.asarray(sample["mu_ideal"], dtype=np.float32)
    mu_practical = np.asarray(sample["mu_practical"], dtype=np.float32)
    mu_true = np.asarray(sample["mu_true"], dtype=np.float32)
    mu_corrected = np.asarray(sample["mu_corrected"], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(mu_ideal[:, 0], mu_ideal[:, 1], c="tab:blue", label="mu_ideal")
    ax.scatter(mu_practical[:, 0], mu_practical[:, 1], c="tab:orange", label="mu_practical")
    ax.scatter(mu_true[:, 0], mu_true[:, 1], c="tab:red", label="mu_true")
    ax.scatter(mu_corrected[:, 0], mu_corrected[:, 1], c="tab:green", label="mu_corrected")
    for start, end in zip(mu_ideal, mu_practical):
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "tab:orange", "alpha": 0.3})
    for start, end in zip(mu_ideal, mu_true):
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "tab:blue", "alpha": 0.35})
    for start, end in zip(mu_ideal, mu_corrected):
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "tab:green", "alpha": 0.45})
    ax.set_title(
        f"Template shifts at {sample['snr_db']:.1f} dB "
        f"(label={sample['label']}, ideal={sample['ideal_pred']}, corrected={sample['corrected_pred']}, "
        f"baseline={sample['practical_baseline_pred']}, true_center={sample['true_center_oracle_pred']})"
    )
    ax.set_xlabel("Real")
    ax.set_ylabel("Imag")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    ax.axis("equal")
    fig.tight_layout()
    vectors_path = figure_dir / "tmc_vectors.png"
    fig.savefig(vectors_path, dpi=240)
    plt.close(fig)
    paths["vectors"] = str(vectors_path)
    return paths


def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = torch.device(args.device)
    model, checkpoint = load_model(ROOT / args.checkpoint, device)
    config_dict = checkpoint["system_config"]
    snr_range = np.arange(args.snr_start, args.snr_stop + 1, args.snr_step)
    config = build_config_from_checkpoint(config_dict, args, snr_range)
    preset_desc = paper_preset_description(config_dict.get("paper_preset"))

    print(
        f"Diagnosing TMC ({args.num_bits} bits, n_t={config.n_t}, n_ris={config.n_ris}, s={config.s}"
        + (f", preset={preset_desc}" if preset_desc else "")
        + f", diagnostic_snr={args.diagnostic_snr_db:g}dB)...",
        flush=True,
    )
    t0 = time.perf_counter()
    eval_results = evaluate(model, config, args, snr_range)
    diagnostics = collect_diagnostics(model, config, args, snr_range)
    elapsed = time.perf_counter() - t0

    practical_baseline = np.asarray(eval_results["practical_baseline"]["symbol_acc"], dtype=np.float32)
    true_center_oracle = np.asarray(eval_results["true_center_oracle"]["symbol_acc"], dtype=np.float32)
    corrected = np.asarray(eval_results["tmc_corrected"]["symbol_acc"], dtype=np.float32)
    ideal = np.asarray(eval_results["ideal_ml"]["symbol_acc"], dtype=np.float32)
    snr_gap = (practical_baseline - corrected).tolist()

    report = {
        "checkpoint": str(ROOT / args.checkpoint),
        "seed": args.seed,
        "paper_preset": config_dict.get("paper_preset"),
        "snr_db": snr_range.tolist(),
        "corrected_symbol_acc": corrected.tolist(),
        "ideal_symbol_acc": ideal.tolist(),
        "practical_baseline_symbol_acc": practical_baseline.tolist(),
        "practical_oracle_symbol_acc": practical_baseline.tolist(),
        "true_center_oracle_symbol_acc": true_center_oracle.tolist(),
        "practical_baseline_gap_by_snr": snr_gap,
        "practical_oracle_gap_by_snr": snr_gap,
        "largest_practical_baseline_gap_snr": int(snr_range[int(np.argmax(snr_gap))]),
        "largest_practical_baseline_gap": float(np.max(snr_gap)),
        "largest_practical_oracle_gap_snr": int(snr_range[int(np.argmax(snr_gap))]),
        "largest_practical_oracle_gap": float(np.max(snr_gap)),
        "elapsed_seconds": elapsed,
        **diagnostics,
    }

    figure_dir = Path(args.figure_dir) if args.figure_dir else ROOT / "outputs" / "figures" / "tmc_diagnostics"
    report["figures"] = _save_figures(report, figure_dir)
    report_path = Path(args.report_path) if args.report_path else ROOT / "outputs" / "reports" / "tmc_diagnostics.json"
    save_json(report_path, report)

    print(f"  Done in {elapsed:.1f}s")
    print(
        f"  Largest practical-baseline gap at SNR={report['largest_practical_baseline_gap_snr']} dB: "
        f"{report['largest_practical_baseline_gap']:.4f}"
    )
    print(f"  Delta direction cosine={report['alignment']['delta_direction_cosine']:.4f}")
    print(f"  Delta magnitude ratio={report['alignment']['delta_magnitude_ratio']:.4f}")


if __name__ == "__main__":
    main()
