# TMC-Net for RIS-TFSSK

Minimal PyTorch implementation of the maintained **Transformer-based Mismatch Calibration (TMC-Net)** workflow for RIS-TFSSK.

## Current workflow

The maintained path now follows `design.md` directly:

1. The simulator injects **non-ideal RIS mismatch** through discrete phase quantization, phase-dependent amplitude loss, and RIS mutual coupling.
2. The maintained model is **TMC-Net**, an SNR-conditioned RIS-token transformer that predicts a complex CSI-residual correction `delta_mu` for every candidate template on top of the practical non-ideal baseline.
3. Detection stays geometric: compare **TMC-corrected templates**, **ideal-template ML**, the **practical baseline** (`practical_oracle` remains as a compatibility report key), and the **configuration-locked true-center oracle**.
4. The paper-text imperfect-CSI ML reproduction remains separate in `scripts/reproduce_paper_text.py`.

## Setup

```bash
conda activate resb
pip install -r requirements.txt
```

## Main scripts

```text
scripts/
  train_tmc.py          # train the maintained TMC-Net calibration model
  infer_tmc.py          # evaluate TMC-corrected templates vs ideal ML / oracle
  eval_sweep_tmc.py     # sweep one checkpoint across multiple CSI error levels
  run_tmc_ablations.sh  # run the main TMC capacity ablations
  diagnose_tmc.py       # confusion/alignment diagnostics for a trained checkpoint
  run_tmc_diagnostics.sh # loss and hardware diagnostic experiment bundle
  reproduce_paper_text.py
```

## Train TMC-Net

```bash
python scripts/train_tmc.py \
    --paper-preset fig3-3b \
    --csi-error-var 0.1 \
    --samples-per-snr 20000 \
    --epochs 80 \
    --patience 15 \
    --batch-size 512 \
    --learning-rate 2e-4 \
    --token-dim 256 \
    --n-layers 6 \
    --n-heads 8
```

This writes the maintained checkpoint to `outputs/checkpoints/tmc_best.pt`.

The default non-ideal hardware matches the design note:

- `--ris-phase-bits 2`
- `--ris-amplitude-bias 0.8`
- `--ris-amplitude-scale 0.2`
- `--ris-coupling-decay 0.15`

The training loss is:

```text
ranking_loss(y, mu_corrected) + distill_weight * mse(delta_mu, mu_true - mu_practical)
```

## Evaluate the maintained model

```bash
python scripts/infer_tmc.py \
    --checkpoint outputs/checkpoints/tmc_best.pt \
    --csi-error-var 0.1 \
    --num-bits 100000

python scripts/eval_sweep_tmc.py \
    --checkpoint outputs/checkpoints/tmc_best.pt \
    --csi-error-vars 0.0 0.05 0.1 0.15 0.2 \
    --num-bits 100000

bash scripts/run_tmc_ablations.sh
```

Reports include:

- `tmc_corrected`: BER after template calibration
- `ideal_ml`: BER from the mismatched ideal-template detector
- `practical_baseline`: BER from the strongest model-based detector that uses the same noisy CSI as the receiver plus the true non-ideal hardware model
- `practical_oracle`: compatibility alias for `practical_baseline`
- `true_center_oracle`: BER from the configuration-locked detector that knows the real non-ideal centers
- `calibration`: center-MSE and calibration-gain diagnostics

## Diagnose the oracle gap

```bash
python scripts/diagnose_tmc.py \
    --checkpoint outputs/checkpoints/tmc_best.pt \
    --num-bits 40000

bash scripts/run_tmc_diagnostics.sh
```

`diagnose_tmc.py` writes a JSON report plus confusion and vector-alignment figures. The diagnostic bundle compares coordinate-only training, probability-distillation training, and a milder hardware setting; optional hardware-factor ablations can be enabled with `RUN_HARDWARE_ABLATIONS=1`.

## Reproduce the paper-text ML path

```bash
python scripts/reproduce_paper_text.py \
    --paper-preset fig3-3b \
    --csi-error-var 0.5
```

This path is intentionally separate. It keeps the original paper-text assumption: estimated CSI is used for phase design, while receiver-side detection still uses the paper's perfect-CSI ML rule.

## Quick smoke check

```bash
python scripts/train_tmc.py \
    --paper-preset fig3-3b \
    --csi-error-var 0.1 \
    --samples-per-snr 32 \
    --epochs 1 \
    --val-batches 1 \
    --batch-size 16 \
    --device cpu \
    --output-dir outputs/checkpoints-smoke \
    --report-path outputs/reports/tmc_train_smoke.json

python scripts/infer_tmc.py \
    --checkpoint outputs/checkpoints-smoke/tmc_best.pt \
    --num-bits 96 \
    --batch-size 16 \
    --device cpu \
    --report-path outputs/reports/tmc_infer_smoke.json \
    --figure-path outputs/figures/tmc_smoke.png
```
