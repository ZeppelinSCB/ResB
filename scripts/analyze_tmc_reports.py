#!/usr/bin/env python3
"""
Analyze TMC inference reports and generate comparison plots.

Usage:
    python scripts/analyze_tmc_reports.py report1.json report2.json ...
    python scripts/analyze_tmc_reports.py outputs/reports/*.json
"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any


def load_report(filepath: str) -> Tuple[str, Dict[str, Any]]:
    """Load a single report JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Handle ablation summary format (has 'configs' list)
    if 'configs' in data and isinstance(data['configs'], list) and len(data['configs']) > 0:
        # Flatten: return each config as a separate report
        return filepath, {'_is_container': True, '_configs': data['configs']}

    return filepath, data


def flatten_reports(reports: List[Tuple[str, Dict]]) -> List[Tuple[str, Dict]]:
    """Flatten container reports (ablation summaries) into individual reports."""
    flattened = []
    for filepath, data in reports:
        if data.get('_is_container'):
            for config in data['_configs']:
                flattened.append((filepath + '/' + config.get('config_name', 'unknown'), config))
        else:
            flattened.append((filepath, data))
    return flattened


def compute_ber_improvement(model_ber: List[float], baseline_ber: List[float]) -> List[float]:
    """Compute BER improvement (baseline - model) in percentage points."""
    return [(b - m) * 100 for b, m in zip(baseline_ber, model_ber)]


def compute_ber_ratio(model_ber: List[float], baseline_ber: List[float]) -> List[float]:
    """Compute BER ratio (model / baseline). Values < 1 mean model improves."""
    return [m / b if b > 0 else 0 for m, b in zip(model_ber, baseline_ber)]


def get_model_label(config_name: str) -> str:
    """Generate a readable label from config name."""
    # Extract key parameters from config name
    parts = config_name.split('_')
    return config_name


def plot_ber_vs_snr(reports: List[Tuple[str, Dict]], output_path: str = None):
    """
    Plot BER vs SNR for all models and their baselines.

    Each model uses a unique color, with solid lines for model and dashed for baseline.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color palette
    colors = plt.cm.tab10.colors + plt.cm.Set2.colors

    for idx, (filepath, data) in enumerate(reports):
        config_name = data.get('config_name', Path(filepath).stem)
        snr_db = data['snr_db']
        color = colors[idx % len(colors)]

        # Model performance (TMC corrected)
        if 'tmc_corrected' in data:
            model_ber = data['tmc_corrected']['ber']
            ax.semilogy(snr_db, model_ber, '-o', color=color, label=f'{config_name} (Model)',
                       linewidth=2, markersize=5)

        # Practical baseline
        if 'practical_baseline' in data:
            baseline_ber = data['practical_baseline']['ber']
            ax.semilogy(snr_db, baseline_ber, '--s', color=color, label=f'{config_name} (Baseline)',
                       linewidth=1.5, markersize=4, alpha=0.7)

        # True center oracle (optional reference)
        if 'true_center_oracle' in data:
            oracle_ber = data['true_center_oracle']['ber']
            ax.semilogy(snr_db, oracle_ber, ':', color=color, label=f'{config_name} (Oracle)',
                       linewidth=1, alpha=0.5)

    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=12)
    ax.set_title('BER vs SNR Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved BER plot to {output_path}")
    plt.show()


def plot_ber_improvement(reports: List[Tuple[str, Dict]], output_path: str = None):
    """
    Plot BER improvement (baseline - model) across SNR levels.

    Makes it easy to see which models provide the best improvement at each SNR.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Color palette
    colors = plt.cm.tab10.colors + plt.cm.Set2.colors

    # Subplot 1: Absolute improvement (percentage points)
    ax1 = axes[0]
    width = 0.8 / len(reports)
    x_positions = np.arange(21)  # Assuming 21 SNR points

    for idx, (filepath, data) in enumerate(reports):
        config_name = data.get('config_name', Path(filepath).stem)
        snr_db = data['snr_db']
        color = colors[idx % len(colors)]

        if 'tmc_corrected' in data and 'practical_baseline' in data:
            model_ber = data['tmc_corrected']['ber']
            baseline_ber = data['practical_baseline']['ber']
            improvement = compute_ber_improvement(model_ber, baseline_ber)

            # Shift positions for grouped bars
            offset = (idx - len(reports)/2 + 0.5) * width
            bars = ax1.bar(x_positions + offset, improvement, width, label=config_name,
                          color=color, alpha=0.8)

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('SNR Index (dB)', fontsize=12)
    ax1.set_ylabel('BER Improvement (pp)', fontsize=12)
    ax1.set_title('BER Improvement: Baseline - Model', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_positions[::3])
    ax1.set_xticklabels([f'{snr}dB' for snr in reports[0][1]['snr_db'][::3]], rotation=45)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # Subplot 2: Improvement line plot (cleaner for many models)
    ax2 = axes[1]
    for idx, (filepath, data) in enumerate(reports):
        config_name = data.get('config_name', Path(filepath).stem)
        snr_db = data['snr_db']
        color = colors[idx % len(colors)]

        if 'tmc_corrected' in data and 'practical_baseline' in data:
            model_ber = data['tmc_corrected']['ber']
            baseline_ber = data['practical_baseline']['ber']
            improvement = compute_ber_improvement(model_ber, baseline_ber)

            ax2.plot(snr_db, improvement, '-o', color=color, label=config_name,
                    linewidth=2, markersize=5)

    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.fill_between(ax2.get_xlim(), 0, ax2.get_ylim()[1] if ax2.get_ylim()[1] > 0 else 10,
                     alpha=0.1, color='green', label='Improvement Zone')
    ax2.set_xlabel('SNR (dB)', fontsize=12)
    ax2.set_ylabel('BER Improvement (percentage points)', fontsize=12)
    ax2.set_title('BER Improvement vs SNR', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved improvement plot to {output_path}")
    plt.show()


def plot_ber_ratio(reports: List[Tuple[str, Dict]], output_path: str = None):
    """
    Plot BER ratio (model/baseline) - values below 1 mean improvement.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab10.colors + plt.cm.Set2.colors

    for idx, (filepath, data) in enumerate(reports):
        config_name = data.get('config_name', Path(filepath).stem)
        snr_db = data['snr_db']
        color = colors[idx % len(colors)]

        if 'tmc_corrected' in data and 'practical_baseline' in data:
            model_ber = data['tmc_corrected']['ber']
            baseline_ber = data['practical_baseline']['ber']
            ratio = compute_ber_ratio(model_ber, baseline_ber)

            ax.plot(snr_db, ratio, '-o', color=color, label=config_name,
                   linewidth=2, markersize=6)

    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='No Improvement')
    ax.fill_between(ax.get_xlim(), 0, 1, alpha=0.1, color='green')
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('BER Ratio (Model / Baseline)', fontsize=12)
    ax.set_title('BER Ratio: Model vs Baseline\n(Values < 1 indicate improvement)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 2)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved ratio plot to {output_path}")
    plt.show()


def plot_calibration_quality(reports: List[Tuple[str, Dict]], output_path: str = None):
    """
    Plot calibration quality (center MSE) comparison.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.tab10.colors + plt.cm.Set2.colors

    # Left: Center MSE comparison
    ax1 = axes[0]
    for idx, (filepath, data) in enumerate(reports):
        config_name = data.get('config_name', Path(filepath).stem)
        snr_db = data['snr_db']
        color = colors[idx % len(colors)]

        if 'calibration' in data:
            mse_practical = data['calibration']['center_mse_practical']
            mse_corrected = data['calibration']['center_mse_corrected']

            ax1.semilogy(snr_db, mse_practical, '--', color=color, label=f'{config_name} (Baseline)',
                        linewidth=1.5, alpha=0.7)
            ax1.semilogy(snr_db, mse_corrected, '-', color=color, label=f'{config_name} (Model)',
                        linewidth=2)

    ax1.set_xlabel('SNR (dB)', fontsize=12)
    ax1.set_ylabel('Center MSE (log scale)', fontsize=12)
    ax1.set_title('Calibration Quality: Center MSE', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3, which='both')

    # Right: Gain in dB
    ax2 = axes[1]
    for idx, (filepath, data) in enumerate(reports):
        config_name = data.get('config_name', Path(filepath).stem)
        snr_db = data['snr_db']
        color = colors[idx % len(colors)]

        if 'calibration' in data:
            gain_db = data['calibration']['center_gain_vs_practical_db']
            ax2.plot(snr_db, gain_db, '-o', color=color, label=config_name,
                    linewidth=2, markersize=5)

    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('SNR (dB)', fontsize=12)
    ax2.set_ylabel('Gain vs Baseline (dB)', fontsize=12)
    ax2.set_title('Calibration Improvement (dB)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved calibration plot to {output_path}")
    plt.show()


def plot_symbol_accuracy(reports: List[Tuple[str, Dict]], output_path: str = None):
    """
    Plot symbol accuracy comparison.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab10.colors + plt.cm.Set2.colors

    for idx, (filepath, data) in enumerate(reports):
        config_name = data.get('config_name', Path(filepath).stem)
        snr_db = data['snr_db']
        color = colors[idx % len(colors)]

        if 'tmc_corrected' in data:
            model_acc = data['tmc_corrected']['symbol_acc']
            ax.plot(snr_db, [a * 100 for a in model_acc], '-o', color=color,
                   label=f'{config_name} (Model)', linewidth=2, markersize=5)

        if 'practical_baseline' in data:
            baseline_acc = data['practical_baseline']['symbol_acc']
            ax.plot(snr_db, [a * 100 for a in baseline_acc], '--s', color=color,
                   label=f'{config_name} (Baseline)', linewidth=1.5, markersize=4, alpha=0.7)

    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Symbol Accuracy (%)', fontsize=12)
    ax.set_title('Symbol Accuracy vs SNR', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved accuracy plot to {output_path}")
    plt.show()


def print_summary(reports: List[Tuple[str, Dict]]):
    """Print a text summary of all reports."""
    print("\n" + "="*80)
    print("TMC INFERENCE RESULTS SUMMARY")
    print("="*80)

    for filepath, data in reports:
        config_name = data.get('config_name', Path(filepath).stem)
        print(f"\n{'='*60}")
        print(f"Config: {config_name}")
        print(f"{'='*60}")

        snr_db = data['snr_db']
        print(f"SNR Range: {snr_db[0]} - {snr_db[-1]} dB ({len(snr_db)} points)")

        if 'tmc_corrected' in data and 'practical_baseline' in data:
            model_ber = np.array(data['tmc_corrected']['ber'])
            baseline_ber = np.array(data['practical_baseline']['ber'])

            # Find where model beats baseline
            improvement = baseline_ber - model_ber
            beats_baseline = np.sum(improvement > 0)

            print(f"\nBER Performance:")
            print(f"  Best Model BER:  {model_ber.min():.4f} (at {snr_db[np.argmin(model_ber)]} dB)")
            print(f"  Best Baseline BER: {baseline_ber.min():.4f} (at {snr_db[np.argmin(baseline_ber)]} dB)")
            print(f"  SNR points where Model beats Baseline: {beats_baseline}/{len(snr_db)}")

            # Average improvement
            avg_improvement = np.mean(improvement) * 100
            print(f"  Average BER Improvement: {avg_improvement:.2f} percentage points")

        if 'calibration' in data:
            mse_corrected = np.array(data['calibration']['center_mse_corrected'])
            mse_practical = np.array(data['calibration']['center_mse_practical'])
            print(f"\nCalibration Quality:")
            print(f"  Final (high SNR) MSE - Model: {mse_corrected[-1]:.4f}, Baseline: {mse_practical[-1]:.4f}")
            print(f"  MSE Improvement: {(1 - mse_corrected[-1]/mse_practical[-1])*100:.1f}%")

        # Check if oracle is available
        if 'true_center_oracle' in data:
            oracle_ber = np.array(data['true_center_oracle']['ber'])
            model_ber = np.array(data['tmc_corrected']['ber'])
            gap_to_oracle = np.mean((model_ber - oracle_ber) * 100)
            print(f"\nOracle Gap: {gap_to_oracle:.2f} pp (higher is better)")


def main():
    parser = argparse.ArgumentParser(description='Analyze TMC inference reports')
    parser.add_argument('reports', nargs='+', help='Path to report JSON files')
    parser.add_argument('--output-dir', '-o', default=None, help='Output directory for plots')
    parser.add_argument('--skip-ber', action='store_true', help='Skip BER plot')
    parser.add_argument('--skip-improvement', action='store_true', help='Skip improvement plot')
    parser.add_argument('--skip-ratio', action='store_true', help='Skip ratio plot')
    parser.add_argument('--skip-calibration', action='store_true', help='Skip calibration plot')
    parser.add_argument('--skip-accuracy', action='store_true', help='Skip accuracy plot')
    parser.add_argument('--skip-summary', action='store_true', help='Skip text summary')
    parser.add_argument('--all', action='store_true', help='Generate all plots')

    args = parser.parse_args()

    # Load all reports
    reports = []
    for filepath in args.reports:
        try:
            path, data = load_report(filepath)
            reports.append((path, data))
            print(f"Loaded: {filepath}")
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    if not reports:
        print("No valid reports loaded.")
        return

    # Flatten container formats (ablation summaries)
    reports = flatten_reports(reports)

    # Determine output directory
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    def get_output_path(name):
        if output_dir:
            return str(output_dir / f"{name}.png")
        return None

    # Generate plots
    if args.all or not any([args.skip_ber, args.skip_improvement, args.skip_ratio,
                            args.skip_calibration, args.skip_accuracy]):
        # Default: generate all plots if none specified
        should_skip = {
            'ber': args.skip_ber,
            'improvement': args.skip_improvement,
            'ratio': args.skip_ratio,
            'calibration': args.skip_calibration,
            'accuracy': args.skip_accuracy
        }
    else:
        should_skip = {
            'ber': args.skip_ber,
            'improvement': args.skip_improvement,
            'ratio': args.skip_ratio,
            'calibration': args.skip_calibration,
            'accuracy': args.skip_accuracy
        }

    # BER vs SNR
    if not should_skip['ber']:
        print("\nGenerating BER vs SNR plot...")
        plot_ber_vs_snr(reports, get_output_path('ber_vs_snr'))

    # BER Improvement
    if not should_skip['improvement']:
        print("\nGenerating BER improvement plot...")
        plot_ber_improvement(reports, get_output_path('ber_improvement'))

    # BER Ratio
    if not should_skip['ratio']:
        print("\nGenerating BER ratio plot...")
        plot_ber_ratio(reports, get_output_path('ber_ratio'))

    # Calibration Quality
    if not should_skip['calibration']:
        print("\nGenerating calibration quality plot...")
        plot_calibration_quality(reports, get_output_path('calibration_quality'))

    # Symbol Accuracy
    if not should_skip['accuracy']:
        print("\nGenerating symbol accuracy plot...")
        plot_symbol_accuracy(reports, get_output_path('symbol_accuracy'))

    # Text summary
    if not args.skip_summary:
        print_summary(reports)

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
