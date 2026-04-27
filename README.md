# TMC-Net for RIS-TFSSK

PyTorch implementation of **Transformer-based Mismatch Calibration (TMC-Net)** for RIS-assisted communication systems with non-ideal hardware.

## System Overview

### Simulation Setup

RIS-empowered transmission with TFS (Transmit Field Switching):
- **n_t**: Number of transmit antennas (power of 2)
- **n_ris**: Number of RIS elements
- **s**: Number of phase combinations per active RIS configuration
- **Candidates**: `(n_t/2) × s` template candidates per transmission

### Non-Ideal Hardware Model

RIS hardware non-ideality is injected through three effects:

| Effect | Parameter | Default | Description |
|--------|-----------|---------|-------------|
| Phase Quantization | `ris_phase_bits` | 2 bits | Discrete phase levels (4 levels) |
| Amplitude Loss | `ris_amplitude_bias` | 0.9 | Base amplitude scaling |
| | `ris_amplitude_scale` | 0.05 | Phase-dependent amplitude variation |
| Mutual Coupling | `ris_coupling_decay` | 0.05 | Inter-element coupling decay |

### CSI Error Model

Channel state information (CSI) is corrupted by estimation errors:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `csi_error_var` | 0.5 | Variance of CSI estimation error |
| `csi_error_model` | additive | Error injection model |
| `csi_error_target` | dual_link | Applies to h and g jointly |
| `csi_error_snr_coupled` | True | Error variance scales with SNR |

## Model Architecture

### TMCNet Backbone

**Location**: [backbones.py](src/resbdnn/modeling/backbones.py)

TMCNet is a transformer-based model that predicts a complex CSI-residual correction `delta_mu` for each template candidate.

**Architecture Components**:

```
Input Processing
├── Channel features (h_hat, g_hat) → token_dim projection
├── SNR conditioning → SiLU embedding
└── Phase configuration → projection

Transformer Backbone
└── 6 × ConditionedSelfAttentionBlock
    ├── LayerNorm + FiLM conditioning on SNR
    ├── Multi-head self-attention
    └── Feed-forward network

Candidate Processing
├── Candidate query embeddings
├── Candidate attribute projection (n_a, s_idx)
├── Active RIS feature projection (11-dim)
├── Mismatch feature projection (mu_ideal - mu_practical)
└── Global representation (mean pooling)

Output Head
└── Fusion projection → residual head → delta_mu (complex)
```

**Key Parameters**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `token_dim` | 256 | Token embedding dimension |
| `n_layers` | 6 | Number of transformer layers |
| `n_heads` | 8 | Attention heads |
| `dropout` | 0.1 | Dropout rate |

## Training

### Training Script

```bash
python scripts/train_tmc.py \
    --paper-preset fig3-3b \
    --csi-error-var 0.5 \
    --samples-per-snr 10000 \
    --epochs 80 \
    --patience 10 \
    --batch-size 512 \
    --learning-rate 2e-4 \
    --token-dim 256 \
    --n-layers 6 \
    --n-heads 8
```

### Loss Function

The training uses a combination of two effective losses:

1. **Ranking Loss** (weight: 1.0, margin: 0.25)
   - Ensures the correct candidate has smaller distance than incorrect candidates

2. **Coordinate Loss** (weight: 1.0)
   - MSE between corrected centers and shrinkage posterior targets
   - Anchor target: `mu_shrinkage_posterior` (achievable from noisy CSI estimates)

```python
loss = rank_weight * ranking_loss(corrected_dist, labels, margin=0.25) \
     + coord_weight * MSE(mu_corrected, mu_shrinkage_posterior)
```

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `samples_per_snr` | 10000 | Training samples per SNR point |
| `steps_per_epoch` | 64 | Gradient steps per epoch |
| `val_batches` | 10 | Validation batches |
| `warmup_epochs` | 3 | Learning rate warmup |
| `grad_clip` | 1.0 | Gradient clipping norm |
| `weight_decay` | 1e-4 | AdamW weight decay |
| `amp` | auto | Automatic mixed precision (CUDA) |

### Checkpoint Output

Trained checkpoints are saved to:
```
outputs/checkpoints/tmc_simple_best.pt
```

## Inference

### Evaluation Script

```bash
python scripts/infer_tmc.py \
    --checkpoint outputs/checkpoints/tmc_simple_best.pt \
    --num-bits 50000 \
    --snr-start 0 \
    --snr-stop 40 \
    --snr-step 2
```

### Evaluation Metrics

The inference script reports BER for multiple detection strategies:

| Metric | Description |
|--------|-------------|
| `tmc_corrected` | BER after TMC-Net template calibration |
| `practical_baseline` | BER from model-based detector with noisy CSI |
| `shrinkage_posterior` | BER from shrinkage posterior oracle |
| `true_center_oracle` | BER from configuration-locked true-center oracle |
| `practical_oracle` | Compatibility alias for `practical_baseline` |

### Calibration Diagnostics

| Metric | Description |
|--------|-------------|
| `center_mse_practical` | MSE between practical and true centers |
| `center_mse_corrected` | MSE between corrected and true centers |
| `center_gain_vs_practical_db` | Improvement in center estimation (dB) |
| `delta_abs` | Average magnitude of predicted correction |

### Feasibility Metrics

| Metric | Description |
|--------|-------------|
| `G_learnable` | Gap closable from observations |
| `G_hidden` | Gap not observable from h_hat, g_hat |
| `R_tmc` | Ratio of learnable gap closed by TMC |
| `R_observable` | Fraction of observable gap |

## System Presets

| Preset | n_t | n_ris | s | Spectral Efficiency |
|--------|-----|-------|---|---------------------|
| `fig3-2b` | 4 | 64 | 2 | 2 bits/s/Hz |
| `fig3-3b` | 4 | 64 | 4 | 3 bits/s/Hz |
| `fig4-4b` | 8 | 64 | 4 | 4 bits/s/Hz |
| `fig4-5b` | 8 | 64 | 8 | 5 bits/s/Hz |

## Quick Start

### Full Training and Evaluation

```bash
# Train
python scripts/train_tmc.py \
    --paper-preset fig3-3b \
    --csi-error-var 0.5 \
    --epochs 80

# Evaluate
python scripts/infer_tmc.py \
    --checkpoint outputs/checkpoints/tmc_simple_best.pt \
    --num-bits 50000
```

### Smoke Test (CPU, Fast)

```bash
python scripts/train_tmc.py \
    --paper-preset fig3-3b \
    --csi-error-var 0.1 \
    --samples-per-snr 32 \
    --epochs 1 \
    --val-batches 1 \
    --batch-size 16 \
    --device cpu \
    --output-dir outputs/checkpoints-smoke

python scripts/infer_tmc.py \
    --checkpoint outputs/checkpoints-smoke/tmc_simple_best.pt \
    --num-bits 96 \
    --batch-size 16 \
    --device cpu
```

## Project Structure

```text
scripts/
  train_tmc.py      # Maintained training script
  infer_tmc.py      # Maintained evaluation script

src/resbdnn/
  modeling/
    backbones.py    # TMCNet architecture
  simulation/
    torch_system.py # System simulation and batch generation
  config.py         # System configuration

outputs/
  checkpoints/     # Model checkpoints
  reports/         # Training/evaluation reports
  figures/          # Generated plots
```
