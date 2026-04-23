"""Deep analysis of why accuracy plateau exists despite Center MSE improvement."""

import argparse
import sys
from pathlib import Path
import math

import torch
import torch.nn.functional as F
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from resbdnn.config import build_system_config
from resbdnn.modeling import TMCNet
from resbdnn.simulation.torch_system import random_tmc_batch, candidate_distances_from_centers


def _complex_mse(pred, target):
    return F.mse_loss(torch.view_as_real(pred), torch.view_as_real(target))


def analyze_error_sources(model, config, device, snr_points=[10, 20, 30, 40], n_samples=3000):
    """Analyze whether coordinate error or noise dominates at each SNR."""
    print("\n" + "="*80)
    print("ERROR SOURCE ANALYSIS")
    print("="*80)

    batch_size = 256

    for snr_db in snr_points:
        print(f"\n=== SNR = {snr_db} dB ===")

        totals = {
            'center_mse_practical': 0.0,
            'center_mse_corrected': 0.0,
            'sigma_n_sq_mean': 0.0,
            'sigma_n_mean': 0.0,
            'practical_correct': 0,
            'corrected_correct': 0,
            'oracle_correct': 0,
            'count': 0,
            'top2_gap_corrected': 0.0,
            'top2_gap_oracle': 0.0,
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
                oracle_dist = candidate_distances_from_centers(batch["y"], batch["mu_true"])

                # Top-2 gap (margin)
                top2_corrected = corrected_dist.topk(k=2, dim=1, largest=False).values
                top2_oracle = oracle_dist.topk(k=2, dim=1, largest=False).values

                # Predictions
                corrected_pred = corrected_dist.argmin(dim=1)
                practical_pred = practical_dist.argmin(dim=1)
                oracle_pred = oracle_dist.argmin(dim=1)

                totals['center_mse_practical'] += _complex_mse(batch["mu_practical"], batch["mu_true"]).item() * batch_size
                totals['center_mse_corrected'] += _complex_mse(outputs["mu_corrected"], batch["mu_true"]).item() * batch_size
                totals['sigma_n_sq_mean'] += batch["sigma_n"].square().mean().item() * batch_size
                totals['sigma_n_mean'] += batch["sigma_n"].mean().item() * batch_size
                totals['practical_correct'] += (practical_pred == batch["labels"]).sum().item()
                totals['corrected_correct'] += (corrected_pred == batch["labels"]).sum().item()
                totals['oracle_correct'] += (oracle_pred == batch["labels"]).sum().item()
                totals['count'] += batch_size
                totals['top2_gap_corrected'] += (top2_corrected[:, 1] - top2_corrected[:, 0]).mean().item() * batch_size
                totals['top2_gap_oracle'] += (top2_oracle[:, 1] - top2_oracle[:, 0]).mean().item() * batch_size

        n = totals['count']
        center_mse_practical = totals['center_mse_practical'] / n
        center_mse_corrected = totals['center_mse_corrected'] / n
        sigma_n_sq = totals['sigma_n_sq_mean'] / n
        sigma_n = totals['sigma_n_mean'] / n

        # Key ratio: coordinate error vs noise
        coord_to_noise_ratio = center_mse_corrected / max(sigma_n_sq, 1e-12)

        print(f"  Noise: σ_n = {sigma_n:.6f}, σ_n² = {sigma_n_sq:.8f}")
        print(f"  Center MSE: Practical={center_mse_practical:.4f}, Corrected={center_mse_corrected:.4f}")
        print(f"  Coord Error / Noise Ratio: {coord_to_noise_ratio:.2f}x")
        print(f"  Accuracy: Practical={totals['practical_correct']/n:.4f}, "
              f"Corrected={totals['corrected_correct']/n:.4f}, Oracle={totals['oracle_correct']/n:.4f}")
        print(f"  Top-2 Gap: Corrected={totals['top2_gap_corrected']/n:.4f}, Oracle={totals['top2_gap_oracle']/n:.4f}")

        # Diagnosis
        if coord_to_noise_ratio > 100:
            print(f"  ⚠️  DIAGNOSIS: Coordinate error dominates (ratio={coord_to_noise_ratio:.0f}x)")
            print(f"     Even if we eliminate noise entirely, detection would still fail!")
        elif coord_to_noise_ratio > 10:
            print(f"  ⚠️  DIAGNOSIS: Coordinate error is significant (ratio={coord_to_noise_ratio:.0f}x)")
        else:
            print(f"  ✅ DIAGNOSIS: Noise dominates at this SNR")


def main():
    parser = argparse.ArgumentParser(description="Deep analysis of accuracy plateau")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--paper-preset", type=str, default="fig3-3b")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = build_system_config(paper_preset=args.paper_preset)

    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model = TMCNet(**checkpoint['model_args']).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("No checkpoint, using random model (for testing)")
        model = TMCNet(token_dim=256, n_layers=6, n_heads=8, dropout=0.1,
                      n_t=config.n_t, n_ris=config.n_ris, s=config.s).to(device)

    analyze_error_sources(model, config, device, snr_points=[10, 20, 30, 40], n_samples=args.n_samples)


if __name__ == "__main__":
    main()
