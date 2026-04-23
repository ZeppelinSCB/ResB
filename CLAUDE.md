# ResB Project Documentation

## Project Overview

ResB is a deep learning project for **TMC-Net (Template Mismatch Correction Network)** in RIS-assisted communication systems. The goal is to correct non-ideal hardware effects on RIS template generation.

## Key Files

### Model
- `src/resbdnn/modeling/backbones.py` - TMCNet architecture

### Training
- `scripts/train_tmc_simple.py` - **Simplified training with Rank + Coord loss** (recommended)
- `scripts/train_tmc.py.bak` - Original training script (deprecated)
- `scripts/train_tmc_improved.py.bak` - Improved training script (deprecated)

### Diagnostics
- `scripts/diagnose_high_snr.py` - High-SNR plateau diagnosis
- `scripts/analyze_error_sources.py` - Error source analysis

## Training

```bash
# Basic training
python scripts/train_tmc_simple.py --epochs 80 --seed 42

# With reduced hardware non-ideality (方案B)
python scripts/train_tmc_simple.py \
  --ris-phase-bits 8 \
  --ris-amplitude-bias 1.0 \
  --ris-amplitude-scale 0.0 \
  --ris-coupling-decay 0.0 \
  --enable-phase-quantization False \
  --enable-amplitude-coupling False \
  --enable-mutual-coupling False \
  --csi-error-var 0.0
```

## Loss Function Design

### Recommended: Rank + Coord

The simplified training uses only two effective losses:

1. **Ranking Loss**: Ensures correct candidate is closer than wrong candidates
   ```python
   rank_loss = margin_based_ranking(corrected_dist, labels, margin=0.25)
   ```

2. **Coordinate Loss**: Direct MSE to true centers
   ```python
   coord_loss = MSE(mu_corrected, mu_true)
   ```

### Removed Losses (Ineffective/Harmful)

| Loss | Status | Reason |
|------|--------|--------|
| CE Loss | Removed | Minimal benefit (+0.3-1.5%) |
| KL Divergence | **Removed** | **Harmful** (-3-5%): locks model to sub-optimal baseline |
| NLL Loss | Removed | Ineffective with hardware non-ideality |
| SNR-adaptive weighting | Removed | No significant improvement |

## Key Findings

### Root Cause of High-SNR Plateau

**Hardware non-ideality dominates over noise at high SNR**:

| Metric | Value |
|--------|-------|
| `mu_practical` vs `mu_true` RMSE | ~12 |
| Required RMSE for 99% accuracy | < 1-2 |
| Coordinate error / Noise ratio @ 30dB | ~185x |

The model cannot achieve Oracle performance because:
1. It only observes `h_hat, g_hat` (noisy channel estimates)
2. `mu_true` requires true channel `h_true, g_true`
3. This is an **information bottleneck**

### Loss Function Ablation

| Loss Combination | Current Config | Ideal Hardware |
|-----------------|----------------|----------------|
| Rank Only | 34.9% | 81.8% |
| Coord Only | 36.8% | 82.4% |
| **Rank + Coord** | **39.2%** | **83.3%** |
| Rank + CE | 37.6% | - |
| Rank + Coord + CE | 37.6% | - |
| Rank + Coord + KL | 33.2% | - |
| All (original) | 34.2% | - |

**Conclusion**: Rank + Coord is optimal. KL Divergence is harmful.

## Architecture Information Bottleneck

Despite the information bottleneck, architecture improvements may still help:
- Better utilization of available input features
- More efficient feature extraction
- Better cross-candidate reasoning

## Hardware Configuration Reference

| Parameter | Current | 方案A (Mild) | 方案B (Ideal) |
|-----------|---------|--------------|---------------|
| `ris_phase_bits` | 4 | 5 | 8 |
| `ris_amplitude_bias` | 0.9 | 0.95 | 1.0 |
| `ris_amplitude_scale` | 0.05 | 0.02 | 0.0 |
| `ris_coupling_decay` | 0.05 | 0.02 | 0.0 |
| `csi_error_var` | 0.1 | 0.05 | 0.0 |

## Memory

Key findings are documented in:
- `memory/tmc_snr_plateau_analysis.md` - Root cause analysis
