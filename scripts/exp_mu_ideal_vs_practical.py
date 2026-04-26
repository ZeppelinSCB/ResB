#!/usr/bin/env python
"""Experiment: Compare mu_ideal vs mu_practical as model base features.

Hypothesis:
- mu_ideal has wrong magnitude (125+ vs ~16 for n_t=8)
- mu_practical is computable in real life and has correct magnitude
- Using mu_practical should fix the learning problem for larger n_t
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from resbdnn.simulation.torch_system import random_tmc_batch, candidate_distances_from_centers
from resbdnn.config import build_system_config


class SimpleTMCNet(nn.Module):
    """Simplified TMC-Net for comparison testing."""

    def __init__(self, n_t, n_ris, s, use_practical=True):
        super().__init__()
        self.use_practical = use_practical
        self.n_t = n_t
        self.n_ris = n_ris
        self.num_candidates = (n_t // 2) * s

        # Channel features
        self.chan_proj = nn.Sequential(
            nn.Linear(2 * n_t, 64),
            nn.ReLU(),
        )

        # mu features
        self.mu_proj = nn.Linear(4, 64)

        # Phase features
        self.phase_proj = nn.Sequential(
            nn.Linear(n_ris, 64),
            nn.ReLU(),
        )

        # Output head
        self.head = nn.Sequential(
            nn.Linear(64 + 64 + 64 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # delta_real, delta_imag
        )

    def forward(self, h_hat, g_hat, sigma_n, phi_config, mu_ideal, mu_practical=None):
        # Channel features: (batch, n_t, n_ris) -> (batch, n_ris, 2*n_t)
        h_concat = torch.cat([h_hat.real, h_hat.imag], dim=1).permute(0, 2, 1)
        chan_emb = self.chan_proj(h_concat)  # (batch, n_ris, 64)
        chan_emb = chan_emb.mean(dim=1)  # (batch, 64)

        # Choose mu base
        mu_base = mu_practical if (self.use_practical and mu_practical is not None) else mu_ideal

        # mu features: (batch, num_candidates, 4)
        mu_features = torch.stack([
            mu_base.real,
            mu_base.imag,
            mu_base.abs(),
            (torch.angle(mu_base) / np.pi),
        ], dim=-1)
        # Project and average over candidates
        mu_emb = self.mu_proj(mu_features)  # (batch, num_candidates, 64)
        mu_emb = mu_emb.mean(dim=1)  # (batch, 64)

        # Phase features: (batch, num_candidates, n_ris) -> (batch, num_candidates, 64) -> (batch, 64)
        phase_emb = self.phase_proj(phi_config)  # (batch, num_candidates, 64)
        phase_emb = phase_emb.mean(dim=1)  # (batch, 64)

        # SNR embedding
        snr_emb = torch.log1p(sigma_n.reshape(-1, 1))  # (batch, 1)

        # Combine features
        combined = torch.cat([chan_emb, mu_emb, phase_emb, snr_emb], dim=-1)
        delta = self.head(combined)  # (batch, 2)

        # Apply delta to mu_base (broadcast across candidates)
        mu_corrected = mu_base + torch.complex(delta[:, 0, None], delta[:, 1, None])

        return mu_corrected


def train_model(n_t, s, use_practical, steps=500, batch_size=256, lr=1e-3):
    """Train model and return final accuracy."""
    config = build_system_config(
        n_t=n_t, n_ris=64, s=s,
        csi_error_var=0.1,
        ris_phase_bits=4,
        enable_phase_quantization=True,
        enable_amplitude_coupling=True,
        enable_mutual_coupling=True
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SimpleTMCNet(n_t, 64, s, use_practical=use_practical).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for step in range(steps):
        batch = random_tmc_batch(config, batch_size=batch_size, device=device, snr_db=20.0)

        mu_corrected = model(
            batch['h_hat'], batch['g_hat'], batch['sigma_n'],
            batch['phi_config'], batch['mu_ideal'], batch['mu_practical']
        )

        # Ranking loss
        dist = candidate_distances_from_centers(batch['y'], mu_corrected)
        labels = batch['labels']
        pos_dist = dist.gather(1, labels[:, None]).squeeze(1)
        wrong_dist = dist.topk(2, dim=1, largest=False)[0][:, 1]
        rank_loss = F.relu(wrong_dist - pos_dist + 0.25).mean()

        # Coordinate loss
        coord_loss = F.mse_loss(torch.view_as_real(mu_corrected), torch.view_as_real(batch['mu_true']))

        loss = rank_loss + 0.1 * coord_loss
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Final evaluation
    with torch.no_grad():
        eval_batch = random_tmc_batch(config, batch_size=2048, device=device, snr_db=20.0)
        mu_corrected = model(
            eval_batch['h_hat'], eval_batch['g_hat'], eval_batch['sigma_n'],
            eval_batch['phi_config'], eval_batch['mu_ideal'], eval_batch['mu_practical']
        )

        # Model accuracy
        dist_corrected = candidate_distances_from_centers(eval_batch['y'], mu_corrected)
        model_acc = (dist_corrected.argmin(1) == eval_batch['labels']).float().mean().item()

        # Baseline accuracy (using mu_practical directly)
        dist_baseline = candidate_distances_from_centers(eval_batch['y'], eval_batch['mu_practical'])
        baseline_acc = (dist_baseline.argmin(1) == eval_batch['labels']).float().mean().item()

    return {
        'final_loss': np.mean(losses[-50:]),
        'model_acc': model_acc,
        'baseline_acc': baseline_acc,
        'improvement': model_acc - baseline_acc,
    }


def analyze_features(n_t, s):
    """Analyze the feature statistics."""
    config = build_system_config(
        n_t=n_t, n_ris=64, s=s,
        csi_error_var=0.1,
        ris_phase_bits=4,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = random_tmc_batch(config, batch_size=1000, device=device, snr_db=20.0)

    mu_ideal = batch['mu_ideal']
    mu_practical = batch['mu_practical']
    mu_true = batch['mu_true']

    # What correction is needed?
    delta_from_ideal = (mu_true - mu_ideal).abs().mean().item()
    delta_from_practical = (mu_true - mu_practical).abs().mean().item()

    return {
        'mu_ideal_mag': mu_ideal.abs().mean().item(),
        'mu_practical_mag': mu_practical.abs().mean().item(),
        'mu_true_mag': mu_true.abs().mean().item(),
        'delta_from_ideal': delta_from_ideal,
        'delta_from_practical': delta_from_practical,
    }


def main():
    print("=" * 70)
    print("Experiment: mu_ideal vs mu_practical as model base")
    print("=" * 70)
    print()

    # First, analyze the feature statistics
    print("=" * 70)
    print("PART 1: Feature Statistics Analysis")
    print("=" * 70)
    print()

    configs = [(4, 2), (4, 4), (8, 4), (8, 8), (16, 4), (16, 8)]

    print(f"{'n_t':>4} {'s':>4} {'ideal_mag':>12} {'prac_mag':>12} {'true_mag':>12} {'delta_ideal':>12} {'delta_prac':>12}")
    print("-" * 70)

    feature_stats = {}
    for n_t, s in configs:
        stats = analyze_features(n_t, s)
        feature_stats[(n_t, s)] = stats
        print(f"{n_t:4d} {s:4d} {stats['mu_ideal_mag']:12.2f} {stats['mu_practical_mag']:12.2f} "
              f"{stats['mu_true_mag']:12.2f} {stats['delta_from_ideal']:12.2f} {stats['delta_from_practical']:12.2f}")

    print()
    print("Key finding: mu_ideal magnitude grows with n_t, but mu_practical/mu_true stay similar")
    print()

    # Training comparison
    print("=" * 70)
    print("PART 2: Training Comparison")
    print("=" * 70)
    print()

    print("Training simple model with different mu bases...")
    print()

    results = []
    for n_t, s in [(8, 4), (8, 8), (16, 4), (16, 8)]:
        print(f"Testing n_t={n_t}, s={s}...", end=" ")

        # mu_ideal
        res_ideal = train_model(n_t, s, use_practical=False, steps=500)

        # mu_practical
        res_practical = train_model(n_t, s, use_practical=True, steps=500)

        results.append({
            'n_t': n_t, 's': s,
            'ideal': res_ideal,
            'practical': res_practical,
        })

        print(f"done")

    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()

    print(f"{'n_t':>4} {'s':>4} | {'Baseline':>10} {'mu_ideal':>10} {'mu_prac':>10} | {'Δ(ideal)':>10} {'Δ(prac)':>10}")
    print("-" * 70)

    for r in results:
        n_t, s = r['n_t'], r['s']
        base = r['ideal']['baseline_acc']
        acc_ideal = r['ideal']['model_acc']
        acc_prac = r['practical']['model_acc']

        delta_ideal = acc_ideal - base
        delta_prac = acc_prac - base

        print(f"{n_t:4d} {s:4d} | {base*100:10.1f}% {acc_ideal*100:10.1f}% {acc_prac*100:10.1f}% | "
              f"{delta_ideal*100:+10.1f}% {delta_prac*100:+10.1f}%")

    print()

    # Conclusion
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()

    total_improvement_ideal = sum(r['ideal']['improvement'] for r in results)
    total_improvement_prac = sum(r['practical']['improvement'] for r in results)

    print(f"Total improvement over baseline:")
    print(f"  Using mu_ideal:    {total_improvement_ideal*100:.1f}% (avg {total_improvement_ideal/len(results)*100:.1f}% per config)")
    print(f"  Using mu_practical: {total_improvement_prac*100:.1f}% (avg {total_improvement_prac/len(results)*100:.1f}% per config)")
    print()

    if total_improvement_prac > total_improvement_ideal:
        print("mu_practical shows better improvement overall!")
    else:
        print("mu_ideal shows better improvement (unexpected)")

    print()
    print("Real-life feasibility:")
    print("  mu_ideal: NOT computable - requires perfect phase compensation")
    print("  mu_practical: COMPUTABLE - uses actual RIS config from estimated CSI")


if __name__ == "__main__":
    main()