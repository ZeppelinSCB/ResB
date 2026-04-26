"""Build a BER panel from a matched-baseline CSI sweep report."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "outputs" / "reports"
FIG_DIR = ROOT / "outputs" / "figures"


def parse_args():
    parser = argparse.ArgumentParser(description="Plot matched-baseline BER panel from eval_sweep_tmc.py output.")
    parser.add_argument(
        "--report-path",
        type=str,
        default=str(REPORT_DIR / "matched_baseline_Nt8_s4_phase_only.json"),
    )
    parser.add_argument(
        "--figure-path",
        type=str,
        default=str(FIG_DIR / "matched_baseline_Nt8_s4_phase_only.png"),
    )
    return parser.parse_args()


def load_report(path: Path):
    with open(path) as f:
        return json.load(f)


def main():
    args = parse_args()
    report = load_report(Path(args.report_path))
    rows = sorted(report["sweep"], key=lambda row: row["csi_error_var"])
    if not rows:
        raise ValueError(f"No sweep rows found in {args.report_path}")

    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rows)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, row in zip(axes, rows):
        r = row["results"]
        snr = report["snr_db"]
        feas = r["feasibility"]
        rw = feas.get("R_tmc_weighted")
        csi = row["csi_error_var"]

        ax.semilogy(snr, np.maximum(r["practical_baseline"]["ber"], 1e-7),
                    "b^-.", label="BER_practical", linewidth=1.5)
        ax.semilogy(snr, np.maximum(r["tmc_corrected"]["ber"], 1e-7),
                    "ro-", label="BER_TMC (gen)", linewidth=1.8)
        ax.semilogy(snr, np.maximum(r["shrinkage_posterior"]["ber"], 1e-7),
                    "gs--", label="BER_posterior", linewidth=1.5)
        ax.semilogy(snr, np.maximum(r["true_center_oracle"]["ber"], 1e-7),
                    "m:", label="BER_true_oracle", linewidth=1.5)

        title = rf"$\sigma_e^2={csi}$"
        if rw is not None:
            title += rf"   $R_{{tmc}}^w={rw:.3f}$"
        ax.set_title(title)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.set_ylim(5e-3, 0.6)

    for ax in axes[len(rows):]:
        ax.axis("off")

    for ax in axes[-ncols:]:
        ax.set_xlabel("SNR (dB)")
    for ax in axes[::ncols]:
        ax.set_ylabel("BER")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        "TMC-Net matched baseline",
        y=1.06, fontsize=12,
    )
    fig.tight_layout()

    out = Path(args.figure_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
