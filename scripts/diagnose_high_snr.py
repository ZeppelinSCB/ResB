"""Diagnostic script to verify high-SNR plateau issues in TMC-Net.

This script validates the findings from update.md about:
1. Ranking loss becoming inactive at high SNR
2. Coordinate loss normalization killing gradients
3. Missing NLL loss with true noise variance
4. Lack of decision boundary-aware loss
"""

import argparse
import copy
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from resbdnn.config import build_system_config, paper_preset_description
from resbdnn.modeling import TMCNet
from resbdnn.simulation.torch_system import random_tmc_batch, candidate_distances_from_centers


def _complex_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(torch.view_as_real(pred), torch.view_as_real(target))


def _ranking_loss(distances: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    positive = distances.gather(1, labels[:, None])
    margin_terms = torch.relu(positive - distances + margin)
    label_mask = torch.zeros_like(margin_terms, dtype=torch.bool)
    label_mask.scatter_(1, labels[:, None], True)
    margin_terms = margin_terms.masked_fill(label_mask, 0.0)
    return (margin_terms.sum(dim=1) / max(distances.size(1) - 1, 1)).mean()


def _nll_loss_from_sigma(mu_corrected: torch.Tensor, y: torch.Tensor, labels: torch.Tensor, sigma_n: torch.Tensor) -> torch.Tensor:
    """NLL loss using true noise variance (as proposed in update.md)."""
    # dist_sq: (B, C)
    dist_sq = (y[:, None] - mu_corrected).abs().square()
    # Use true sigma_n^2 - at high SNR this will amplify small coordinate errors
    var = sigma_n.square().clamp_min(1e-6)  # (B, 1)
    log_likelihood = -dist_sq / (2.0 * var)
    return F.cross_entropy(log_likelihood, labels)


def diagnose_model(model, config, device, snr_points=[10, 15, 20, 25, 30, 35, 40], n_samples=2000):
    """Diagnose model behavior at different SNR levels."""
    results = {
        'snr': [],
        'ranking_loss': [],
        'ce_loss': [],
        'nll_loss': [],
        'coord_loss': [],
        'baseline_center_mse': [],
        'corrected_center_mse': [],
        'practical_acc': [],
        'corrected_acc': [],
        'oracle_acc': [],
        'delta_abs': [],
        'ideal_top2_gap': [],
        'corrected_top2_gap': [],
        'residual_scale': [],
        'ranking_loss_active_ratio': [],  # % of samples with non-zero ranking loss
    }

    batch_size = 256

    for snr_db in snr_points:
        print(f"\n=== SNR = {snr_db} dB ===")

        # Collect statistics over multiple batches
        totals = {
            'ranking_loss': 0.0, 'ce_loss': 0.0, 'nll_loss': 0.0,
            'coord_loss': 0.0, 'baseline_center_mse': 0.0, 'corrected_center_mse': 0.0,
            'practical_correct': 0, 'corrected_correct': 0, 'oracle_correct': 0,
            'delta_abs': 0.0, 'ideal_gap': 0.0, 'corrected_gap': 0.0,
            'residual_scale': 0.0, 'ranking_active': 0, 'count': 0,
        }

        n_batches = max(1, n_samples // batch_size)
        model.eval()

        with torch.inference_mode():
            for _ in range(n_batches):
                batch = random_tmc_batch(config, batch_size, device, snr_db=snr_db)

                outputs = model.forward_parts(
                    batch["h_hat"], batch["g_hat"], batch["sigma_n"],
                    batch["phi_config"], batch["mu_ideal"], batch["mu_practical"],
                )

                # Compute distances
                corrected_dist = candidate_distances_from_centers(batch["y"], outputs["mu_corrected"])
                practical_dist = candidate_distances_from_centers(batch["y"], batch["mu_practical"])

                # Ranking loss
                rank_loss = _ranking_loss(corrected_dist, batch["labels"], margin=0.25)
                # CE loss (current implementation)
                temperature = 1.0 / (1.0 + snr_db / 10.0)
                student_logits = (-corrected_dist / temperature).float()
                ce_loss = F.cross_entropy(student_logits, batch["labels"])
                # NLL loss with true sigma_n
                nll_loss = _nll_loss_from_sigma(outputs["mu_corrected"], batch["y"], batch["labels"], batch["sigma_n"])
                # Coordinate loss
                coord_loss = _complex_mse(outputs["mu_corrected"], batch["mu_true"])

                # Check if ranking loss is active (non-zero)
                positive = corrected_dist.gather(1, batch["labels"][:, None])
                wrong_dists = corrected_dist.clone()
                wrong_dists.scatter_(1, batch["labels"][:, None], float('inf'))
                hardest_wrong = wrong_dists.min(dim=1).values
                ranking_active = ((positive - hardest_wrong + 0.25) > 0).sum().item()

                # Predictions
                corrected_pred = corrected_dist.argmin(dim=1)
                practical_pred = practical_dist.argmin(dim=1)
                oracle_pred = candidate_distances_from_centers(batch["y"], batch["mu_true"]).argmin(dim=1)

                # Top-2 gap
                top2 = corrected_dist.topk(k=2, dim=1, largest=False).values
                corrected_gap = (top2[:, 1] - top2[:, 0]).mean().item()

                totals['ranking_loss'] += rank_loss.item() * batch_size
                totals['ce_loss'] += ce_loss.item() * batch_size
                totals['nll_loss'] += nll_loss.item() * batch_size
                totals['coord_loss'] += coord_loss.item() * batch_size
                totals['baseline_center_mse'] += _complex_mse(batch["mu_practical"], batch["mu_true"]).item() * batch_size
                totals['corrected_center_mse'] += _complex_mse(outputs["mu_corrected"], batch["mu_true"]).item() * batch_size
                totals['practical_correct'] += (practical_pred == batch["labels"]).sum().item()
                totals['corrected_correct'] += (corrected_pred == batch["labels"]).sum().item()
                totals['oracle_correct'] += (oracle_pred == batch["labels"]).sum().item()
                totals['delta_abs'] += outputs["delta_mu"].abs().mean().item() * batch_size
                totals['ideal_gap'] += 0.0  # Skip for efficiency
                totals['corrected_gap'] += corrected_gap * batch_size
                totals['residual_scale'] += outputs["residual_scale"].mean().item() * batch_size
                totals['ranking_active'] += ranking_active
                totals['count'] += batch_size

        # Compute averages
        n = totals['count']
        results['snr'].append(snr_db)
        results['ranking_loss'].append(totals['ranking_loss'] / n)
        results['ce_loss'].append(totals['ce_loss'] / n)
        results['nll_loss'].append(totals['nll_loss'] / n)
        results['coord_loss'].append(totals['coord_loss'] / n)
        results['baseline_center_mse'].append(totals['baseline_center_mse'] / n)
        results['corrected_center_mse'].append(totals['corrected_center_mse'] / n)
        results['practical_acc'].append(totals['practical_correct'] / n)
        results['corrected_acc'].append(totals['corrected_correct'] / n)
        results['oracle_acc'].append(totals['oracle_correct'] / n)
        results['delta_abs'].append(totals['delta_abs'] / n)
        results['corrected_top2_gap'].append(totals['corrected_gap'] / n)
        results['residual_scale'].append(totals['residual_scale'] / n)
        results['ranking_loss_active_ratio'].append(totals['ranking_active'] / n)

        print(f"  Accuracy: Practical={results['practical_acc'][-1]:.4f}, "
              f"Corrected={results['corrected_acc'][-1]:.4f}, Oracle={results['oracle_acc'][-1]:.4f}")
        print(f"  Losses: Ranking={results['ranking_loss'][-1]:.4f}, CE={results['ce_loss'][-1]:.4f}, "
              f"NLL={results['nll_loss'][-1]:.4f}, Coord={results['coord_loss'][-1]:.6f}")
        print(f"  Ranking active: {results['ranking_loss_active_ratio'][-1]*100:.1f}%")
        print(f"  Center MSE: Baseline={results['baseline_center_mse'][-1]:.6f}, "
              f"Corrected={results['corrected_center_mse'][-1]:.6f}")
        print(f"  Delta magnitude: {results['delta_abs'][-1]:.4f}, Residual scale: {results['residual_scale'][-1]:.4f}")

    return results


def print_diagnostic_summary(results):
    """Print diagnostic summary highlighting issues from update.md."""
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY - Checking update.md hypotheses")
    print("="*80)

    # Check 1: Ranking loss becoming inactive at high SNR
    print("\n[Issue 1] Ranking Loss at high SNR:")
    for i, snr in enumerate(results['snr']):
        if snr >= 20:
            active = results['ranking_loss_active_ratio'][i]
            print(f"  SNR={snr}dB: {active*100:.1f}% samples have active ranking loss "
                  f"(rank_loss={results['ranking_loss'][i]:.4f})")

    # Check 2: CE vs NLL loss comparison
    print("\n[Issue 2] CE Loss vs NLL Loss (true sigma_n):")
    for i, snr in enumerate(results['snr']):
        if snr >= 20:
            ce = results['ce_loss'][i]
            nll = results['nll_loss'][i]
            print(f"  SNR={snr}dB: CE={ce:.4f}, NLL={nll:.4f}, "
                  f"ratio={nll/max(ce, 1e-6):.2f}x")

    # Check 3: Accuracy gap between corrected and oracle
    print("\n[Issue 3] Accuracy Gap (corrected vs oracle):")
    for i, snr in enumerate(results['snr']):
        gap = results['oracle_acc'][i] - results['corrected_acc'][i]
        print(f"  SNR={snr}dB: Oracle={results['oracle_acc'][i]:.4f}, "
              f"Corrected={results['corrected_acc'][i]:.4f}, Gap={gap:.4f}")

    # Check 4: Center MSE improvement
    print("\n[Issue 4] Center MSE improvement:")
    for i, snr in enumerate(results['snr']):
        if results['baseline_center_mse'][i] > 0:
            improvement = 10 * math.log10(results['baseline_center_mse'][i] / max(results['corrected_center_mse'][i], 1e-12))
            print(f"  SNR={snr}dB: Baseline MSE={results['baseline_center_mse'][i]:.6f}, "
                  f"Corrected MSE={results['corrected_center_mse'][i]:.6f}, "
                  f"Improvement={improvement:.2f}dB")

    # Check 5: Delta magnitude vs residual scale
    print("\n[Issue 5] Delta magnitude and residual scale:")
    for i, snr in enumerate(results['snr']):
        print(f"  SNR={snr}dB: Delta_mag={results['delta_abs'][i]:.4f}, "
              f"Residual_scale={results['residual_scale'][i]:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose TMC-Net high-SNR plateau issues")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (if None, train a fresh model)")
    parser.add_argument("--paper-preset", type=str, default="fig3-3b")
    parser.add_argument("--snr-start", type=int, default=10)
    parser.add_argument("--snr-stop", type=int, default=40)
    parser.add_argument("--snr-step", type=int, default=5)
    parser.add_argument("--n-samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    snr_points = list(range(args.snr_start, args.snr_stop + 1, args.snr_step))

    config = build_system_config(
        paper_preset=args.paper_preset,
        snr_start=0, snr_stop=40, snr_step=2,
    )

    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model = TMCNet(**checkpoint['model_args']).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("No checkpoint provided, training a fresh model for diagnosis...")
        model = TMCNet(
            token_dim=256, n_layers=6, n_heads=8, dropout=0.1,
            n_t=config.n_t, n_ris=config.n_ris, s=config.s,
        ).to(device)

        # Quick training for diagnosis
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        batch_size = 512
        steps = 100

        print("Training for 100 steps to get a working model...")
        for step in range(steps):
            batch = random_tmc_batch(config, batch_size, device)
            outputs = model.forward_parts(
                batch["h_hat"], batch["g_hat"], batch["sigma_n"],
                batch["phi_config"], batch["mu_ideal"], batch["mu_practical"],
            )
            corrected_dist = candidate_distances_from_centers(batch["y"], outputs["mu_corrected"])
            rank_loss = _ranking_loss(corrected_dist, batch["labels"], margin=0.25)
            coord_loss = _complex_mse(outputs["mu_corrected"], batch["mu_true"])
            loss = rank_loss + 0.5 * coord_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 20 == 0:
                print(f"  Step {step}: loss={loss.item():.4f}")

    print(f"\nRunning diagnosis on {paper_preset_description(args.paper_preset)}")
    print(f"SNR range: {snr_points}")
    print(f"Device: {device}")

    results = diagnose_model(model, config, device, snr_points, args.n_samples)
    print_diagnostic_summary(results)

    # Save results
    output_path = ROOT / "outputs" / "diagnostic_results.json"
    import json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        # Convert numpy arrays and non-serializable types
        serializable = {k: [float(x) for x in v] if isinstance(v, list) else v for k, v in results.items()}
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
