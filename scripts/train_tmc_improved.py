"""Train TMC-Net with improved loss functions for high-SNR performance.

Implements improvements from update.md:
1. NLL Loss with true sigma_n (instead of heuristic temperature)
2. SNR-adaptive coordinate loss weighting
3. Error pattern features (mu_ideal - mu_practical)
"""

import argparse
import copy
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

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
from resbdnn.modeling import TMCNet
from resbdnn.simulation.torch_system import (
    candidate_distances_from_centers,
    random_tmc_batch,
)
from resbdnn.utils import ensure_dir, save_json, save_torch_checkpoint, set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train TMC-Net with improved losses")
    parser.add_argument("--samples-per-snr", type=int, default=20000)
    parser.add_argument("--steps-per-epoch", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--val-batches", type=int, default=10)
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--rank-margin", type=float, default=0.25)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    # NEW: NLL Loss with true sigma_n
    parser.add_argument("--nll-weight", type=float, default=1.0)
    # NEW: SNR-adaptive coordinate loss
    parser.add_argument("--coord-weight", type=float, default=1.0)
    parser.add_argument("--snr-coord-start", type=float, default=20.0,
                        help="SNR above which coordinate weight increases")
    parser.add_argument("--snr-coord-max-mult", type=float, default=3.0,
                        help="Maximum multiplier for coordinate loss at high SNR")
    # NEW: Hard negative mining
    parser.add_argument("--hard-neg-k", type=int, default=3,
                        help="Number of hard negatives to use in ranking loss")
    parser.add_argument("--csi-error-var", type=float, default=0.1)
    parser.add_argument("--csi-error-model", type=str, default="normalized", choices=CSI_ERROR_MODELS)
    parser.add_argument("--csi-error-target", type=str, default="dual_link", choices=CSI_ERROR_TARGETS)
    parser.add_argument("--csi-error-snr-coupled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--csi-error-snr-ref-db", type=float, default=10.0)
    parser.add_argument("--csi-outlier-prob", type=float, default=0.0)
    parser.add_argument("--csi-outlier-scale", type=float, default=0.0)
    parser.add_argument("--ris-phase-bits", type=int, default=4)
    parser.add_argument("--ris-amplitude-bias", type=float, default=0.9)
    parser.add_argument("--ris-amplitude-scale", type=float, default=0.05)
    parser.add_argument("--ris-coupling-decay", type=float, default=0.05)
    parser.add_argument("--enable-phase-quantization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-amplitude-coupling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-mutual-coupling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--paper-preset", type=str, default="fig3-3b", choices=sorted(PAPER_SYSTEM_PRESETS))
    parser.add_argument("--n-t", type=int, default=None)
    parser.add_argument("--n-ris", type=int, default=None)
    parser.add_argument("--s", type=int, default=None)
    parser.add_argument("--signal-energy", type=float, default=2.0)
    parser.add_argument("--snr-start", type=int, default=0)
    parser.add_argument("--snr-stop", type=int, default=40)
    parser.add_argument("--snr-step", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/checkpoints_improved")
    parser.add_argument("--report-path", type=str, default=None)
    return parser.parse_args()


def _complex_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(torch.view_as_real(pred), torch.view_as_real(target))


def _ranking_loss(distances: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    """Standard ranking loss (margin-based)."""
    positive = distances.gather(1, labels[:, None])
    margin_terms = torch.relu(positive - distances + margin)
    label_mask = torch.zeros_like(margin_terms, dtype=torch.bool)
    label_mask.scatter_(1, labels[:, None], True)
    margin_terms = margin_terms.masked_fill(label_mask, 0.0)
    return (margin_terms.sum(dim=1) / max(distances.size(1) - 1, 1)).mean()


def _hard_ranking_loss(distances: torch.Tensor, labels: torch.Tensor, margin: float, k: int = 3) -> torch.Tensor:
    """Hard negative mining ranking loss - only penalize top-k hardest negatives."""
    positive = distances.gather(1, labels[:, None])
    # Mask out positive samples
    masked_dist = distances.clone()
    masked_dist.scatter_(1, labels[:, None], float('inf'))
    # Get top-k hardest negatives (smallest distances among wrong candidates)
    hard_negatives = masked_dist.topk(k=min(k, distances.size(1) - 1), dim=1, largest=False).values
    margin_terms = torch.relu(positive - hard_negatives + margin)
    return margin_terms.mean()


def _nll_loss(mu_corrected: torch.Tensor, y: torch.Tensor, labels: torch.Tensor, sigma_n: torch.Tensor) -> torch.Tensor:
    """NLL loss using true noise variance - key for high SNR gradient amplification."""
    # dist_sq: (B, C)
    dist_sq = (y[:, None] - mu_corrected).abs().square()
    # Use true sigma_n^2 - at high SNR this will amplify small coordinate errors
    var = sigma_n.square().clamp_min(1e-6)  # (B, 1)
    log_likelihood = -dist_sq / (2.0 * var)
    return F.cross_entropy(log_likelihood, labels)


def _snr_adaptive_coord_weight(snr: torch.Tensor, start: float = 20.0, max_mult: float = 3.0) -> torch.Tensor:
    """Compute SNR-adaptive weight for coordinate loss.
    Higher SNR gets higher weight to encourage precise coordinate fitting."""
    # Linear increase from 1.0 to max_mult as SNR goes from start to 40 dB
    base_weight = 1.0
    weight = base_weight + (max_mult - base_weight) * ((snr - start) / (40.0 - start)).clamp(0.0, 1.0)
    return weight.clamp_min(base_weight)


def _loss_and_metrics(model, batch, args, epoch: int | None = None):
    outputs = model.forward_parts(
        batch["h_hat"],
        batch["g_hat"],
        batch["sigma_n"],
        batch["phi_config"],
        batch["mu_ideal"],
        batch["mu_practical"],  # Use practical as base
    )
    corrected_dist = candidate_distances_from_centers(batch["y"], outputs["mu_corrected"])
    practical_baseline_dist = candidate_distances_from_centers(batch["y"], batch["mu_practical"])

    # Ranking loss with optional hard negative mining
    if args.hard_neg_k > 0:
        rank_loss = _hard_ranking_loss(corrected_dist, batch["labels"], args.rank_margin, args.hard_neg_k)
    else:
        rank_loss = _ranking_loss(corrected_dist, batch["labels"], args.rank_margin)

    # NEW: NLL loss with true sigma_n
    nll_loss = _nll_loss(outputs["mu_corrected"], batch["y"], batch["labels"], batch["sigma_n"])

    # Coordinate loss with SNR-adaptive weighting
    coord_loss = _complex_mse(outputs["mu_corrected"], batch["mu_true"])
    snr_coord_weight = _snr_adaptive_coord_weight(
        batch["snr"], args.snr_coord_start, args.snr_coord_max_mult
    )
    weighted_coord_loss = (coord_loss * snr_coord_weight).mean()

    # Total loss
    loss = (
        args.rank_weight * rank_loss
        + args.nll_weight * nll_loss
        + args.coord_weight * weighted_coord_loss
    )

    # Metrics
    center_mse_practical = _complex_mse(batch["mu_practical"], batch["mu_true"])
    center_mse_corrected = _complex_mse(outputs["mu_corrected"], batch["mu_true"])

    corrected_pred = corrected_dist.argmin(dim=1)
    practical_pred = practical_baseline_dist.argmin(dim=1)
    oracle_pred = candidate_distances_from_centers(batch["y"], batch["mu_true"]).argmin(dim=1)

    return {
        "loss": loss,
        "rank_loss": rank_loss,
        "nll_loss": nll_loss,
        "coord_loss": coord_loss,
        "weighted_coord_loss": weighted_coord_loss,
        "center_mse_practical": center_mse_practical,
        "center_mse_corrected": center_mse_corrected,
        "corrected_correct": int((corrected_pred == batch["labels"]).sum().item()),
        "practical_correct": int((practical_pred == batch["labels"]).sum().item()),
        "oracle_correct": int((oracle_pred == batch["labels"]).sum().item()),
        "count": batch["labels"].numel(),
        "delta_abs": float(outputs["delta_mu"].abs().mean().item()),
        "residual_scale": float(outputs["residual_scale"].detach().item()),
        "snr_coord_weight": float(snr_coord_weight.mean().item()),
    }


def _run_epoch(model, config, args, optimizer, scaler, steps, *, train: bool, epoch: int):
    totals = {
        "loss": 0.0, "rank_loss": 0.0, "nll_loss": 0.0,
        "coord_loss": 0.0, "weighted_coord_loss": 0.0,
        "center_mse_practical": 0.0, "center_mse_corrected": 0.0,
        "corrected_correct": 0, "practical_correct": 0, "oracle_correct": 0,
        "count": 0, "delta_abs": 0.0, "residual_scale": 0.0, "snr_coord_weight": 0.0,
    }

    model.train(train)
    for _ in range(steps):
        batch = random_tmc_batch(config, args.batch_size, torch.device(args.device))
        if train:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=args.device_type, dtype=torch.float16, enabled=args.use_amp):
                metrics = _loss_and_metrics(model, batch, args, epoch)
            scaler.scale(metrics["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.inference_mode():
                with torch.autocast(device_type=args.device_type, dtype=torch.float16, enabled=args.use_amp):
                    metrics = _loss_and_metrics(model, batch, args, epoch)

        count = metrics["count"]
        totals["loss"] += float(metrics["loss"].detach().item()) * count
        totals["rank_loss"] += float(metrics["rank_loss"].detach().item()) * count
        totals["nll_loss"] += float(metrics["nll_loss"].detach().item()) * count
        totals["coord_loss"] += float(metrics["coord_loss"].detach().item()) * count
        totals["weighted_coord_loss"] += float(metrics["weighted_coord_loss"].detach().item()) * count
        totals["center_mse_practical"] += float(metrics["center_mse_practical"].detach().item()) * count
        totals["center_mse_corrected"] += float(metrics["center_mse_corrected"].detach().item()) * count
        totals["corrected_correct"] += metrics["corrected_correct"]
        totals["practical_correct"] += metrics["practical_correct"]
        totals["oracle_correct"] += metrics["oracle_correct"]
        totals["count"] += count
        totals["delta_abs"] += metrics["delta_abs"] * count
        totals["residual_scale"] += metrics["residual_scale"] * count
        totals["snr_coord_weight"] += metrics["snr_coord_weight"] * count

    n = max(totals["count"], 1)
    center_mse_practical = totals["center_mse_practical"] / n
    center_mse_corrected = totals["center_mse_corrected"] / n
    return {
        "loss": totals["loss"] / n,
        "rank_loss": totals["rank_loss"] / n,
        "nll_loss": totals["nll_loss"] / n,
        "coord_loss": totals["coord_loss"] / n,
        "weighted_coord_loss": totals["weighted_coord_loss"] / n,
        "center_mse_practical": center_mse_practical,
        "center_mse_corrected": center_mse_corrected,
        "center_gain_db": 10.0 * math.log10(max(center_mse_practical, 1e-12) / max(center_mse_corrected, 1e-12)),
        "corrected_ml_acc": totals["corrected_correct"] / n,
        "practical_baseline_acc": totals["practical_correct"] / n,
        "oracle_acc": totals["oracle_correct"] / n,
        "delta_abs": totals["delta_abs"] / n,
        "residual_scale": totals["residual_scale"] / n,
        "snr_coord_weight": totals["snr_coord_weight"] / n,
    }


def _resolve_step_schedule(config, args):
    if args.steps_per_epoch is not None and args.steps_per_epoch < 1:
        raise ValueError(f"steps_per_epoch must be >= 1, got {args.steps_per_epoch}")
    snr_points = len(config.snr_range)
    target_samples_per_epoch = args.samples_per_snr * snr_points
    if args.steps_per_epoch is None:
        steps_per_epoch = max(1, math.ceil(target_samples_per_epoch / args.batch_size))
        step_source = "samples_per_snr"
    else:
        steps_per_epoch = args.steps_per_epoch
        step_source = "manual_override"
    actual_samples_per_epoch = steps_per_epoch * args.batch_size
    effective_samples_per_snr = actual_samples_per_epoch / snr_points
    return {
        "snr_points": snr_points,
        "steps_per_epoch": steps_per_epoch,
        "step_source": step_source,
        "target_samples_per_epoch": target_samples_per_epoch,
        "actual_samples_per_epoch": actual_samples_per_epoch,
        "effective_samples_per_snr": effective_samples_per_snr,
    }


def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = torch.device(args.device)
    args.device_type = device.type
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    args.use_amp = device.type == "cuda" if args.amp is None else bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device=device.type, enabled=args.use_amp)

    config = build_system_config(
        paper_preset=args.paper_preset,
        n_t=args.n_t,
        n_ris=args.n_ris,
        s=args.s,
        signal_energy=args.signal_energy,
        csi_error_var=args.csi_error_var,
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
        snr_start=args.snr_start,
        snr_stop=args.snr_stop,
        snr_step=args.snr_step,
    )
    schedule = _resolve_step_schedule(config, args)

    model_args = {
        "token_dim": args.token_dim,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "dropout": args.dropout,
        "n_t": config.n_t,
        "n_ris": config.n_ris,
        "s": config.s,
        "residual_scale_init": 0.1,
    }
    model = TMCNet(**model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    warmup_epochs = min(max(args.warmup_epochs, 0), max(args.epochs - 1, 0))
    if warmup_epochs > 0:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(args.epochs - warmup_epochs, 1),
                    eta_min=1e-6,
                ),
            ],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=1e-6)

    print(
        f"Training TMC-Net (Improved): n_t={config.n_t}, n_ris={config.n_ris}, s={config.s}, "
        f"σ²_e={config.csi_error_var}, batch={args.batch_size}, steps/epoch={schedule['steps_per_epoch']}, "
        f"phase_bits={config.ris_phase_bits}, amp={config.ris_amplitude_bias:.2f}+{config.ris_amplitude_scale:.2f}cos(phi), "
        f"nll_w={args.nll_weight:.2f}, coord_w={args.coord_weight:.2f}, "
        f"snr_coord_start={args.snr_coord_start:.1f}, snr_coord_max_mult={args.snr_coord_max_mult:.1f}, "
        f"hard_neg_k={args.hard_neg_k}, amp={'on' if args.use_amp else 'off'}"
        + (f", preset={paper_preset_description(args.paper_preset)}" if args.paper_preset else ""),
        flush=True,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_metrics = _run_epoch(
            model, config, args, optimizer, scaler,
            schedule["steps_per_epoch"], train=True, epoch=epoch,
        )
        val_metrics = _run_epoch(
            model, config, args, optimizer, scaler,
            args.val_batches, train=False, epoch=epoch,
        )
        scheduler.step()
        elapsed = time.perf_counter() - t0
        history.append(
            {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "val": val_metrics,
                "epoch_seconds": elapsed,
            }
        )

        improved = val_metrics["loss"] < best_val_loss - 0.01
        if improved:
            best_val_acc = val_metrics["corrected_ml_acc"]
            best_val_loss = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4e} corrected={train_metrics['corrected_ml_acc']:.4f} "
            f"baseline={train_metrics['practical_baseline_acc']:.4f} oracle={train_metrics['oracle_acc']:.4f} "
            f"gain={train_metrics['center_gain_db']:.2f}dB | "
            f"val_loss={val_metrics['loss']:.4e} corrected={val_metrics['corrected_ml_acc']:.4f} "
            f"baseline={val_metrics['practical_baseline_acc']:.4f} oracle={val_metrics['oracle_acc']:.4f} "
            f"gain={val_metrics['center_gain_db']:.2f}dB {elapsed:.1f}s",
            flush=True,
        )

        if epochs_without_improvement >= args.patience:
            break

    checkpoint_path = ensure_dir(ROOT / args.output_dir) / "tmc_improved_best.pt"
    system_config_payload = {
        "n_t": config.n_t, "n_ris": config.n_ris, "s": config.s,
        "signal_energy": config.signal_energy, "paper_preset": args.paper_preset,
        "spectral_efficiency": config.spectral_efficiency,
        "snr_range": config.snr_range.tolist(),
        "csi_error_var": config.csi_error_var,
        "csi_error_model": config.csi_error_model,
        "csi_error_target": args.csi_error_target,
        "csi_error_snr_coupled": args.csi_error_snr_coupled,
        "csi_error_snr_ref_db": args.csi_error_snr_ref_db,
        "ris_phase_bits": config.ris_phase_bits,
        "ris_amplitude_bias": config.ris_amplitude_bias,
        "ris_amplitude_scale": config.ris_amplitude_scale,
        "ris_coupling_decay": config.ris_coupling_decay,
        "enable_phase_quantization": config.enable_phase_quantization,
        "enable_amplitude_coupling": config.enable_amplitude_coupling,
        "enable_mutual_coupling": config.enable_mutual_coupling,
    }
    save_torch_checkpoint(
        checkpoint_path,
        {
            "model_state_dict": best_state,
            "model_args": model_args,
            "model_class": "TMCNet",
            "system_config": system_config_payload,
            "best_val_corrected_ml_acc": best_val_acc,
            "best_val_loss": best_val_loss,
        },
    )

    report = {
        "checkpoint": str(checkpoint_path),
        "model_class": "TMCNet",
        "model_args": model_args,
        "system_config": system_config_payload,
        "best_val_corrected_ml_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "epochs_trained": len(history),
        "history": history,
        "amp_enabled": args.use_amp,
        "batch_size": args.batch_size,
        "steps_per_epoch": schedule["steps_per_epoch"],
        "nll_weight": args.nll_weight,
        "coord_weight": args.coord_weight,
        "snr_coord_start": args.snr_coord_start,
        "snr_coord_max_mult": args.snr_coord_max_mult,
        "hard_neg_k": args.hard_neg_k,
    }
    report_path = Path(args.report_path) if args.report_path else ROOT / "outputs" / "reports" / "tmc_improved_train.json"
    save_json(report_path, report)


if __name__ == "__main__":
    main()
