"""Train TMC-Net models with different Nt/s parameter combinations.

Configurations:
- Nt=4, s=2
- Nt=4, s=4
- Nt=8, s=4
- Nt=8, s=8
- Nt=16, s=8
- Nt=16, s=16
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
    CANDIDATE_STRATEGIES,
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


# Parameter combinations to sweep
PARAM_CONFIGS = [
    {"n_t": 4, "s": 2, "name": "Nt4_s2"},
    {"n_t": 4, "s": 4, "name": "Nt4_s4"},
    {"n_t": 8, "s": 4, "name": "Nt8_s4"},
    {"n_t": 8, "s": 8, "name": "Nt8_s8"},
    {"n_t": 16, "s": 8, "name": "Nt16_s8"},
    {"n_t": 16, "s": 16, "name": "Nt16_s16"},
]


def parse_args():
    parser = argparse.ArgumentParser(description="TMC-Net ablation: train with different Nt/s combinations")
    parser.add_argument("--samples-per-snr", type=int, default=10000)
    parser.add_argument("--steps-per-epoch", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val-batches", type=int, default=50)
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--rank-margin", type=float, default=0.25)
    parser.add_argument("--rank-weight", type=float, default=0.0)
    parser.add_argument("--rank-warmup-epochs", type=int, default=10)
    parser.add_argument("--decision-weight", type=float, default=0.0)
    parser.add_argument("--decision-temperature", type=float, default=2.0)
    parser.add_argument("--coord-weight", type=float, default=1.0)
    parser.add_argument("--csi-error-var", type=float, default=0.2)
    parser.add_argument(
        "--csi-mix",
        type=str,
        default=None,
        help=(
            "Comma-separated csi_error_var values to mix per-sample during training "
            "(e.g. '0.05,0.1,0.2,0.3,0.5'). Overrides --csi-error-var for batch sampling."
        ),
    )
    parser.add_argument("--csi-error-model", type=str, default="additive", choices=CSI_ERROR_MODELS)
    parser.add_argument("--csi-error-target", type=str, default="dual_link", choices=CSI_ERROR_TARGETS)
    parser.add_argument("--csi-error-snr-coupled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--csi-error-snr-ref-db", type=float, default=10.0)
    parser.add_argument("--csi-outlier-prob", type=float, default=0.0)
    parser.add_argument("--csi-outlier-scale", type=float, default=0.0)
    parser.add_argument("--ris-phase-bits", type=int, default=2)
    parser.add_argument("--ris-amplitude-bias", type=float, default=0.9)
    parser.add_argument("--ris-amplitude-scale", type=float, default=0.05)
    parser.add_argument("--ris-coupling-decay", type=float, default=0.05)
    parser.add_argument("--enable-phase-quantization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-amplitude-coupling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-mutual-coupling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--candidate-strategy", type=str, default="prefix", choices=CANDIDATE_STRATEGIES)
    parser.add_argument("--paper-preset", type=str, default="fig3-3b", choices=sorted(PAPER_SYSTEM_PRESETS))
    parser.add_argument("--n-ris", type=int, default=64)
    parser.add_argument("--signal-energy", type=float, default=1.0)
    parser.add_argument("--snr-start", type=int, default=0)
    parser.add_argument("--snr-stop", type=int, default=40)
    parser.add_argument("--snr-step", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--report-path", type=str, default=None)
    parser.add_argument("--csi-conditioned", action=argparse.BooleanOptionalAction, default=False,
                        help="Condition TMCNet on per-sample csi_error_var (recommended for --csi-mix).")
    parser.add_argument("--config-index", type=int, default=None,
                        help="Run only config at this index (0-5). If None, run all.")
    return parser.parse_args()


def _complex_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(torch.view_as_real(pred), torch.view_as_real(target))


def _ranking_loss(
    distances: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
    distance_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Margin-based ranking loss: correct candidate should be closer than wrong ones."""
    if distance_scale is not None:
        distances = distances / distance_scale.to(distances.dtype).clamp_min(1e-12)
    positive = distances.gather(1, labels[:, None])
    margin_terms = torch.relu(positive - distances + margin)
    label_mask = torch.zeros_like(margin_terms, dtype=torch.bool)
    label_mask.scatter_(1, labels[:, None], True)
    margin_terms = margin_terms.masked_fill(label_mask, 0.0)
    return (margin_terms.sum(dim=1) / max(distances.size(1) - 1, 1)).mean()


def _normalized_coord_loss(pred: torch.Tensor, target: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    residual = (pred - target) / scale.to(pred.real.dtype).clamp_min(1e-6)
    return F.smooth_l1_loss(
        torch.view_as_real(residual),
        torch.zeros_like(torch.view_as_real(residual)),
    )


def _rank_weight_for_epoch(args, epoch: int | None) -> float:
    if epoch is None or args.rank_warmup_epochs <= 0:
        return args.rank_weight
    return args.rank_weight * min(float(epoch) / float(args.rank_warmup_epochs), 1.0)


def _posterior_decision_kl(
    y: torch.Tensor,
    student_centers: torch.Tensor,
    teacher_centers: torch.Tensor,
    sigma_n: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    student_dist = candidate_distances_from_centers(y, student_centers)
    teacher_dist = candidate_distances_from_centers(y, teacher_centers)
    tau = (float(temperature) * sigma_n.square()).to(student_dist.dtype).clamp_min(1e-12)
    teacher_log_prob = F.log_softmax(-teacher_dist / tau, dim=1)
    student_log_prob = F.log_softmax(-student_dist / tau, dim=1)
    teacher_prob = teacher_log_prob.exp()
    return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")


def _loss_and_metrics(model, batch, args, epoch: int | None = None):
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
    practical_baseline_dist = candidate_distances_from_centers(batch["y"], batch["mu_practical"])

    rank_loss = _ranking_loss(
        corrected_dist,
        batch["labels"],
        args.rank_margin,
        distance_scale=batch["sigma_n"].square(),
    )
    coord_target = batch.get("mu_shrinkage_posterior", batch["mu_true"])
    coord_scale = coord_target.abs().square().mean(dim=1, keepdim=True).clamp_min(1e-12).sqrt()
    coord_loss = _normalized_coord_loss(outputs["mu_corrected"], coord_target, coord_scale)
    decision_loss = _posterior_decision_kl(
        batch["y"],
        outputs["mu_corrected"],
        coord_target,
        batch["sigma_n"],
        args.decision_temperature,
    )

    rank_weight = _rank_weight_for_epoch(args, epoch)
    loss = rank_weight * rank_loss + args.coord_weight * coord_loss + args.decision_weight * decision_loss

    center_mse_practical = _complex_mse(batch["mu_practical"], batch["mu_true"])
    center_mse_corrected = _complex_mse(outputs["mu_corrected"], batch["mu_true"])
    center_mse_to_coord_target = _complex_mse(outputs["mu_corrected"], coord_target)

    corrected_pred = corrected_dist.argmin(dim=1)
    practical_pred = practical_baseline_dist.argmin(dim=1)
    oracle_pred = candidate_distances_from_centers(batch["y"], batch["mu_true"]).argmin(dim=1)

    return {
        "loss": loss,
        "rank_loss": rank_loss,
        "coord_loss": coord_loss,
        "decision_loss": decision_loss,
        "rank_weight": rank_weight,
        "center_mse_practical": center_mse_practical,
        "center_mse_corrected": center_mse_corrected,
        "center_mse_to_coord_target": center_mse_to_coord_target,
        "corrected_correct": int((corrected_pred == batch["labels"]).sum().item()),
        "practical_correct": int((practical_pred == batch["labels"]).sum().item()),
        "oracle_correct": int((oracle_pred == batch["labels"]).sum().item()),
        "count": batch["labels"].numel(),
        "delta_abs": float(outputs["delta_mu"].abs().mean().item()),
    }


def _parse_csi_mix(csi_mix: str | None) -> list[float] | None:
    if csi_mix is None or not str(csi_mix).strip():
        return None
    values = [float(v) for v in str(csi_mix).split(",") if v.strip()]
    if not values:
        return None
    return values


def _sample_csi_error_var(
    csi_mix_values: list[float] | None,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    if csi_mix_values is None:
        return None
    pool = torch.as_tensor(csi_mix_values, device=device, dtype=torch.float32)
    idx = torch.randint(0, pool.numel(), (batch_size,), device=device)
    return pool[idx]


def _make_validation_batches(config, args, steps: int) -> list[dict[str, torch.Tensor]]:
    device = torch.device(args.device)
    csi_mix_values = _parse_csi_mix(getattr(args, "csi_mix", None))
    return [
        random_tmc_batch(
            config,
            args.batch_size,
            device,
            csi_error_var=_sample_csi_error_var(csi_mix_values, args.batch_size, device),
        )
        for _ in range(steps)
    ]


def _run_epoch(
    model,
    config,
    args,
    optimizer,
    scaler,
    steps,
    *,
    train: bool,
    epoch: int,
    fixed_batches: list[dict[str, torch.Tensor]] | None = None,
):
    totals = {
        "loss": 0.0, "rank_loss": 0.0, "coord_loss": 0.0, "decision_loss": 0.0,
        "center_mse_practical": 0.0, "center_mse_corrected": 0.0,
        "center_mse_to_coord_target": 0.0,
        "corrected_correct": 0, "practical_correct": 0, "oracle_correct": 0,
        "count": 0, "delta_abs": 0.0,
        "rank_weight": 0.0,
    }

    model.train(train)
    csi_mix_values = _parse_csi_mix(getattr(args, "csi_mix", None))
    train_device = torch.device(args.device)
    batch_iter = fixed_batches if fixed_batches is not None else (
        random_tmc_batch(
            config,
            args.batch_size,
            train_device,
            csi_error_var=_sample_csi_error_var(csi_mix_values, args.batch_size, train_device),
        )
        for _ in range(steps)
    )
    for batch in batch_iter:
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
        totals["coord_loss"] += float(metrics["coord_loss"].detach().item()) * count
        totals["decision_loss"] += float(metrics["decision_loss"].detach().item()) * count
        totals["rank_weight"] += float(metrics["rank_weight"]) * count
        totals["center_mse_practical"] += float(metrics["center_mse_practical"].detach().item()) * count
        totals["center_mse_corrected"] += float(metrics["center_mse_corrected"].detach().item()) * count
        totals["center_mse_to_coord_target"] += float(metrics["center_mse_to_coord_target"].detach().item()) * count
        totals["corrected_correct"] += metrics["corrected_correct"]
        totals["practical_correct"] += metrics["practical_correct"]
        totals["oracle_correct"] += metrics["oracle_correct"]
        totals["count"] += count
        totals["delta_abs"] += metrics["delta_abs"] * count

    n = max(totals["count"], 1)
    center_mse_practical = totals["center_mse_practical"] / n
    center_mse_corrected = totals["center_mse_corrected"] / n
    return {
        "loss": totals["loss"] / n,
        "rank_loss": totals["rank_loss"] / n,
        "coord_loss": totals["coord_loss"] / n,
        "decision_loss": totals["decision_loss"] / n,
        "rank_weight": totals["rank_weight"] / n,
        "center_mse_practical": center_mse_practical,
        "center_mse_corrected": center_mse_corrected,
        "center_mse_to_coord_target": totals["center_mse_to_coord_target"] / n,
        "center_gain_db": 10.0 * math.log10(max(center_mse_practical, 1e-12) / max(center_mse_corrected, 1e-12)),
        "corrected_ml_acc": totals["corrected_correct"] / n,
        "practical_baseline_acc": totals["practical_correct"] / n,
        "oracle_acc": totals["oracle_correct"] / n,
        "delta_abs": totals["delta_abs"] / n,
    }


def _resolve_step_schedule(config, args):
    snr_points = len(config.snr_range)
    target_samples_per_epoch = args.samples_per_snr * snr_points
    if args.steps_per_epoch is None:
        steps_per_epoch = max(1, math.ceil(target_samples_per_epoch / args.batch_size))
    else:
        steps_per_epoch = args.steps_per_epoch
    return {"snr_points": snr_points, "steps_per_epoch": steps_per_epoch}


def train_single_config(param_config, args, base_output_dir):
    """Train a single model configuration and return results."""
    n_t = param_config["n_t"]
    s = param_config["s"]
    config_name = param_config["name"]

    set_random_seed(args.seed)
    device = torch.device(args.device)
    device_type = device.type
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    use_amp = device.type == "cuda" if args.amp is None else bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)
    args.device_type = device_type
    args.use_amp = use_amp

    config = build_system_config(
        paper_preset=args.paper_preset,
        n_t=n_t,
        n_ris=args.n_ris,
        s=s,
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
        candidate_strategy=args.candidate_strategy,
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
        "candidate_strategy": config.candidate_strategy,
        "csi_conditioned": bool(args.csi_conditioned),
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
        f"\n{'='*60}\n"
        f"Training {config_name}: n_t={n_t}, s={s}, n_ris={args.n_ris}\n"
        f"{'='*60}",
        flush=True,
    )

    fixed_val_batches = _make_validation_batches(config, args, args.val_batches)

    best_coord_state = copy.deepcopy(model.state_dict())
    best_acc_state = copy.deepcopy(model.state_dict())
    best_val_acc = -float("inf")
    best_val_loss = float("inf")
    best_val_coord_loss = float("inf")
    best_coord_metrics = None
    best_acc_metrics = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_metrics = _run_epoch(model, config, args, optimizer, scaler, schedule["steps_per_epoch"],
                                   train=True, epoch=epoch)
        val_metrics = _run_epoch(
            model,
            config,
            args,
            optimizer,
            scaler,
            args.val_batches,
            train=False,
            epoch=epoch,
            fixed_batches=fixed_val_batches,
        )
        scheduler.step()
        elapsed = time.perf_counter() - t0
        history.append({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                         "train": train_metrics, "val": val_metrics, "epoch_seconds": elapsed})

        improved = math.isinf(best_val_loss) or val_metrics["loss"] < best_val_loss * 0.99
        if improved:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if val_metrics["coord_loss"] < best_val_coord_loss:
            best_val_coord_loss = val_metrics["coord_loss"]
            best_coord_state = copy.deepcopy(model.state_dict())
            best_coord_metrics = dict(val_metrics)

        if val_metrics["corrected_ml_acc"] > best_val_acc:
            best_val_acc = val_metrics["corrected_ml_acc"]
            best_acc_state = copy.deepcopy(model.state_dict())
            best_acc_metrics = dict(val_metrics)

        print(
            f"epoch {epoch:03d} | loss={train_metrics['loss']:.4e} acc={train_metrics['corrected_ml_acc']:.4f} "
            f"baseline={train_metrics['practical_baseline_acc']:.4f} oracle={train_metrics['oracle_acc']:.4f} "
            f"gain={train_metrics['center_gain_db']:.2f}dB | "
            f"val_loss={val_metrics['loss']:.4e} acc={val_metrics['corrected_ml_acc']:.4f} "
            f"oracle={val_metrics['oracle_acc']:.4f} {elapsed:.1f}s",
            flush=True,
        )

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    output_dir = ensure_dir(ROOT / base_output_dir / config_name)
    coord_checkpoint_path = output_dir / "tmc_best_coord_loss.pt"
    acc_checkpoint_path = output_dir / "tmc_best_acc.pt"
    checkpoint_payload = {
        "model_args": model_args,
        "model_class": "TMCNet",
        "system_config": {
            "n_t": config.n_t,
            "n_ris": config.n_ris,
            "s": config.s,
            "signal_energy": config.signal_energy,
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
            "candidate_strategy": config.candidate_strategy,
            "paper_preset": args.paper_preset,
        },
        "best_val_corrected_ml_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "best_val_coord_loss": best_val_coord_loss,
        "best_coord_metrics": best_coord_metrics,
        "best_acc_metrics": best_acc_metrics,
        "fixed_val_batches": args.val_batches,
        "fixed_val_samples": args.val_batches * args.batch_size,
    }
    save_torch_checkpoint(
        coord_checkpoint_path,
        {
            **checkpoint_payload,
            "model_state_dict": best_coord_state,
            "selection_metric": "coord_loss",
        },
    )
    save_torch_checkpoint(
        acc_checkpoint_path,
        {
            **checkpoint_payload,
            "model_state_dict": best_acc_state,
            "selection_metric": "corrected_ml_acc",
        },
    )

    report = {
        "config_name": config_name,
        "n_t": n_t,
        "s": s,
        "n_ris": args.n_ris,
        "checkpoint_best_coord_loss": str(coord_checkpoint_path),
        "checkpoint_best_acc": str(acc_checkpoint_path),
        "model_class": "TMCNet",
        "model_args": model_args,
        "best_val_corrected_ml_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "best_val_coord_loss": best_val_coord_loss,
        "best_coord_metrics": best_coord_metrics,
        "best_acc_metrics": best_acc_metrics,
        "fixed_val_batches": args.val_batches,
        "fixed_val_samples": args.val_batches * args.batch_size,
        "epochs_trained": len(history),
        "history": history,
        "rank_weight": args.rank_weight,
        "coord_weight": args.coord_weight,
        "decision_weight": args.decision_weight,
        "decision_temperature": args.decision_temperature,
        "decision_teacher": "mu_shrinkage_posterior",
        "coord_target": "mu_shrinkage_posterior",
        "residual_parameterization": "mu_base_scale",
        "candidate_features": "mu_ideal,mu_practical,mu_ideal_minus_mu_practical,active_antenna_ris",
        "csi_mix": _parse_csi_mix(getattr(args, "csi_mix", None)),
    }
    report_path = output_dir / "train_report.json"
    save_json(report_path, report)

    print(f"\nSaved coord-loss checkpoint: {coord_checkpoint_path}")
    print(f"Saved acc checkpoint: {acc_checkpoint_path}")
    print(f"Saved report: {report_path}")
    return report


def main():
    args = parse_args()

    configs_to_run = PARAM_CONFIGS
    if args.config_index is not None:
        if 0 <= args.config_index < len(PARAM_CONFIGS):
            configs_to_run = [PARAM_CONFIGS[args.config_index]]
        else:
            print(f"Error: config_index must be 0-{len(PARAM_CONFIGS)-1}")
            return

    all_reports = []
    for param_config in configs_to_run:
        report = train_single_config(param_config, args, args.output_dir)
        all_reports.append(report)

    if args.config_index is None:
        summary_path = ROOT / "outputs" / "reports" / "tmc_ablation_summary.json"
        save_json(summary_path, {"configs": all_reports})
        print(f"\n{'='*60}")
        print("Ablation Complete!")
        print(f"{'='*60}")
        for r in all_reports:
            print(f"  {r['config_name']}: n_t={r['n_t']}, s={r['s']} -> "
                  f"val_acc={r['best_val_corrected_ml_acc']:.4f}")


if __name__ == "__main__":
    main()
