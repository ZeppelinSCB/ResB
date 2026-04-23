#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT"

if [[ "${CONDA_DEFAULT_ENV-}" != "resb" ]]; then
  if [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
  fi
  conda activate resb
fi

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/outputs/.matplotlib}"

PAPER_PRESET="${PAPER_PRESET:-fig3-3b}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"

CSI_ERROR_VAR="${CSI_ERROR_VAR:-0.1}"
CSI_ERROR_MODEL="${CSI_ERROR_MODEL:-normalized}"
CSI_ERROR_TARGET="${CSI_ERROR_TARGET:-dual_link}"

RIS_PHASE_BITS="${RIS_PHASE_BITS:-2}"
RIS_AMPLITUDE_BIAS="${RIS_AMPLITUDE_BIAS:-0.8}"
RIS_AMPLITUDE_SCALE="${RIS_AMPLITUDE_SCALE:-0.2}"
RIS_COUPLING_DECAY="${RIS_COUPLING_DECAY:-0.15}"

SAMPLES_PER_SNR="${SAMPLES_PER_SNR:-20000}"
TRAIN_STEPS_PER_EPOCH="${TRAIN_STEPS_PER_EPOCH:-}"
EPOCHS="${EPOCHS:-80}"
PATIENCE="${PATIENCE:-15}"
VAL_BATCHES="${VAL_BATCHES:-10}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4000}"
NUM_BITS="${NUM_BITS:-100000}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"

BASE_TOKEN_DIM="${BASE_TOKEN_DIM:-256}"
BASE_LAYERS="${BASE_LAYERS:-4}"
BASE_HEADS="${BASE_HEADS:-8}"
LARGE_TOKEN_DIM="${LARGE_TOKEN_DIM:-384}"
LARGE_LAYERS="${LARGE_LAYERS:-6}"
LARGE_HEADS="${LARGE_HEADS:-8}"

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-outputs/experiments/tmc-ablations}"
CHECKPOINT_ROOT="$EXPERIMENT_ROOT/checkpoints"
REPORT_ROOT="$EXPERIMENT_ROOT/reports"
FIGURE_ROOT="$EXPERIMENT_ROOT/figures"

mkdir -p "$CHECKPOINT_ROOT" "$REPORT_ROOT" "$FIGURE_ROOT"

COMMON_ARGS=(
  --paper-preset "$PAPER_PRESET"
  --csi-error-var "$CSI_ERROR_VAR"
  --csi-error-model "$CSI_ERROR_MODEL"
  --csi-error-target "$CSI_ERROR_TARGET"
  --ris-phase-bits "$RIS_PHASE_BITS"
  --ris-amplitude-bias "$RIS_AMPLITUDE_BIAS"
  --ris-amplitude-scale "$RIS_AMPLITUDE_SCALE"
  --ris-coupling-decay "$RIS_COUPLING_DECAY"
  --samples-per-snr "$SAMPLES_PER_SNR"
  --epochs "$EPOCHS"
  --patience "$PATIENCE"
  --val-batches "$VAL_BATCHES"
  --batch-size "$TRAIN_BATCH_SIZE"
  --device "$DEVICE"
  --seed "$SEED"
  --learning-rate "$LEARNING_RATE"
)

if [[ -n "$TRAIN_STEPS_PER_EPOCH" ]]; then
  COMMON_ARGS+=(--steps-per-epoch "$TRAIN_STEPS_PER_EPOCH")
fi

SWEEP_VARS_STRING="${EVAL_SWEEP_VARS:-0.0 0.05 0.1 0.15 0.2}"
read -r -a EVAL_SWEEP_VARS <<< "$SWEEP_VARS_STRING"

run_ablation() {
  local name="$1"
  shift

  echo
  echo "== $name =="
  python scripts/train_tmc.py \
    "${COMMON_ARGS[@]}" \
    --output-dir "$CHECKPOINT_ROOT/$name" \
    --report-path "$REPORT_ROOT/${name}_train.json" \
    "$@"

  python scripts/infer_tmc.py \
    --checkpoint "$CHECKPOINT_ROOT/$name/tmc_best.pt" \
    --csi-error-var "$CSI_ERROR_VAR" \
    --csi-error-model "$CSI_ERROR_MODEL" \
    --csi-error-target "$CSI_ERROR_TARGET" \
    --ris-phase-bits "$RIS_PHASE_BITS" \
    --ris-amplitude-bias "$RIS_AMPLITUDE_BIAS" \
    --ris-amplitude-scale "$RIS_AMPLITUDE_SCALE" \
    --ris-coupling-decay "$RIS_COUPLING_DECAY" \
    --num-bits "$NUM_BITS" \
    --batch-size "$EVAL_BATCH_SIZE" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --report-path "$REPORT_ROOT/${name}_infer.json" \
    --figure-path "$FIGURE_ROOT/${name}.png"

  python scripts/eval_sweep_tmc.py \
    --checkpoint "$CHECKPOINT_ROOT/$name/tmc_best.pt" \
    --csi-error-vars "${EVAL_SWEEP_VARS[@]}" \
    --csi-error-model "$CSI_ERROR_MODEL" \
    --csi-error-target "$CSI_ERROR_TARGET" \
    --ris-phase-bits "$RIS_PHASE_BITS" \
    --ris-amplitude-bias "$RIS_AMPLITUDE_BIAS" \
    --ris-amplitude-scale "$RIS_AMPLITUDE_SCALE" \
    --ris-coupling-decay "$RIS_COUPLING_DECAY" \
    --num-bits "$NUM_BITS" \
    --batch-size "$EVAL_BATCH_SIZE" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --report-path "$REPORT_ROOT/${name}_sweep.json"
}

run_ablation tmc_base \
  --token-dim "$BASE_TOKEN_DIM" \
  --n-layers "$BASE_LAYERS" \
  --n-heads "$BASE_HEADS"

run_ablation tmc_large \
  --token-dim "$LARGE_TOKEN_DIM" \
  --n-layers "$LARGE_LAYERS" \
  --n-heads "$LARGE_HEADS"

export ABLATION_REPORT_ROOT="$REPORT_ROOT"
python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ABLATION_REPORT_ROOT"])
rows = []
for infer_path in sorted(root.glob("*_infer.json")):
    name = infer_path.stem.removesuffix("_infer")
    train_path = root / f"{name}_train.json"
    infer = json.loads(infer_path.read_text())
    train = json.loads(train_path.read_text())
    rows.append(
        {
            "name": name,
            "best_val": train.get("best_val_corrected_ml_acc"),
            "tmc_ber": sum(infer["tmc_corrected"]["ber"]) / len(infer["tmc_corrected"]["ber"]),
            "ideal_ber": sum(infer["ideal_ml"]["ber"]) / len(infer["ideal_ml"]["ber"]),
            "oracle_ber": sum(infer["true_center_oracle"]["ber"]) / len(infer["true_center_oracle"]["ber"]),
        }
    )

print("\nSummary:")
for row in rows:
    print(
        f"{row['name']:>20} | "
        f"best_val={row['best_val']:.4f} "
        f"tmc_ber={row['tmc_ber']:.5f} "
        f"ideal_ber={row['ideal_ber']:.5f} "
        f"oracle_ber={row['oracle_ber']:.5f}"
    )
PY
