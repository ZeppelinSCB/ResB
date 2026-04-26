#!/usr/bin/env python3
"""Visualize training loss and validation accuracy from TMC ablation study."""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load data
with open("outputs/reports/tmc_ablation_summary.json", "r") as f:
    data = json.load(f)

configs = data["configs"]

# Set up figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("TMC Ablation Study: Training Analysis", fontsize=14, fontweight="bold")

# Color palette for configs
colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
markers = ["o", "s", "^", "D"]

# Plot 1: Training Loss
ax1 = axes[0, 0]
for i, config in enumerate(configs):
    epochs = [h["epoch"] for h in config["history"]]
    train_loss = [h["train"]["loss"] for h in config["history"]]
    ax1.plot(epochs, train_loss, color=colors[i], marker=markers[i],
             label=config["config_name"], markersize=4, linewidth=1.5)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Training Loss")
ax1.set_title("Training Loss vs Epoch")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Validation Accuracy
ax2 = axes[0, 1]
for i, config in enumerate(configs):
    epochs = [h["epoch"] for h in config["history"]]
    val_acc = [h["val"]["corrected_ml_acc"] for h in config["history"]]
    ax2.plot(epochs, val_acc, color=colors[i], marker=markers[i],
             label=config["config_name"], markersize=4, linewidth=1.5)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Validation Accuracy")
ax2.set_title("Validation Accuracy vs Epoch")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Validation Loss
ax3 = axes[1, 0]
for i, config in enumerate(configs):
    epochs = [h["epoch"] for h in config["history"]]
    val_loss = [h["val"]["loss"] for h in config["history"]]
    ax3.plot(epochs, val_loss, color=colors[i], marker=markers[i],
             label=config["config_name"], markersize=4, linewidth=1.5)
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Validation Loss")
ax3.set_title("Validation Loss vs Epoch")
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Best Validation Accuracy Comparison (Bar Chart)
ax4 = axes[1, 1]
config_names = [c["config_name"] for c in configs]
best_accs = [c["best_val_corrected_ml_acc"] for c in configs]
bar_colors = colors[:len(configs)]
bars = ax4.bar(config_names, best_accs, color=bar_colors, edgecolor="black", linewidth=1.2)
ax4.set_xlabel("Configuration")
ax4.set_ylabel("Best Validation Accuracy")
ax4.set_title("Best Validation Accuracy Comparison")
ax4.set_ylim([0, 1.0])
ax4.axhline(y=0.89, color="gray", linestyle="--", alpha=0.7, label="89% reference")
ax4.legend()
for bar, acc in zip(bars, best_accs):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"{acc:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("outputs/reports/tmc_ablation_visualization.png", dpi=150, bbox_inches="tight")
plt.show()

# Additional plot: Coordinate Loss and Rank Loss breakdown
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("Loss Component Analysis", fontsize=14, fontweight="bold")

# Rank Loss
ax_rank = axes2[0]
for i, config in enumerate(configs):
    epochs = [h["epoch"] for h in config["history"]]
    rank_loss = [h["train"]["rank_loss"] for h in config["history"]]
    ax_rank.plot(epochs, rank_loss, color=colors[i], marker=markers[i],
                 label=config["config_name"], markersize=4, linewidth=1.5)
ax_rank.set_xlabel("Epoch")
ax_rank.set_ylabel("Rank Loss (Train)")
ax_rank.set_title("Ranking Loss vs Epoch")
ax_rank.legend()
ax_rank.grid(True, alpha=0.3)

# Coordinate Loss
ax_coord = axes2[1]
for i, config in enumerate(configs):
    epochs = [h["epoch"] for h in config["history"]]
    coord_loss = [h["train"]["coord_loss"] for h in config["history"]]
    ax_coord.plot(epochs, coord_loss, color=colors[i], marker=markers[i],
                  label=config["config_name"], markersize=4, linewidth=1.5)
ax_coord.set_xlabel("Epoch")
ax_coord.set_ylabel("Coordinate Loss (Train)")
ax_coord.set_title("Coordinate Loss vs Epoch")
ax_coord.legend()
ax_coord.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/reports/tmc_ablation_loss_components.png", dpi=150, bbox_inches="tight")
plt.show()

# Summary table
print("\n" + "="*60)
print("TMC Ablation Study Summary")
print("="*60)
print(f"{'Config':<12} {'Best Val Acc':<15} {'Final Val Loss':<15} {'Epochs':<8}")
print("-"*60)
for config in configs:
    print(f"{config['config_name']:<12} {config['best_val_corrected_ml_acc']:.4f}          "
          f"{config['best_val_loss']:.4f}           {config['epochs_trained']}")
print("="*60)