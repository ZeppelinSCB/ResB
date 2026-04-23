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

AGGR_PHASE_BITS="${AGGR_PHASE_BITS:-2}"
AGGR_AMP_BIAS="${AGGR_AMP_BIAS:-0.8}"
AGGR_AMP_SCALE="${AGGR_AMP_SCALE:-0.2}"
AGGR_COUPLING_DECAY="${AGGR_COUPLING_DECAY:-0.15}"

MILD_PHASE_BITS="${MILD_PHASE_BITS:-3}"
MILD_AMP_BIAS="${MILD_AMP_BIAS:-0.9}"
MILD_AMP_SCALE="${MILD_AMP_SCALE:-0.1}"
MILD_COUPLING_DECAY="${MILD_COUPLING_DECAY:-0.05}"

SAMPLES_PER_SNR="${SAMPLES_PER_SNR:-20000}"
TRAIN_STEPS_PER_EPOCH="${TRAIN_STEPS_PER_EPOCH:-}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-8}"
VAL_BATCHES="${VAL_BATCHES:-10}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4000}"
NUM_BITS="${NUM_BITS:-100000}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"

TOKEN_DIM="${TOKEN_DIM:-256}"
LAYERS="${LAYERS:-6}"
HEADS="${HEADS:-8}"

BASELINE_RANK_WEIGHT="${BASELINE_RANK_WEIGHT:-1.0}"
BASELINE_CE_WEIGHT="${BASELINE_CE_WEIGHT:-0.0}"
BASELINE_PROB_WEIGHT="${BASELINE_PROB_WEIGHT:-0.0}"
BASELINE_COORD_WEIGHT="${BASELINE_COORD_WEIGHT:-0.25}"
BASELINE_TEMP="${BASELINE_TEMP:-1.0}"
BASELINE_LOSS_WARMUP="${BASELINE_LOSS_WARMUP:-0}"

PROB_RANK_WEIGHT="${PROB_RANK_WEIGHT:-0.0}"
PROB_CE_WEIGHT="${PROB_CE_WEIGHT:-1.0}"
PROB_PROB_WEIGHT="${PROB_PROB_WEIGHT:-0.5}"
PROB_COORD_WEIGHT="${PROB_COORD_WEIGHT:-0.1}"
PROB_TEMP="${PROB_TEMP:-1.0}"
PROB_LOSS_WARMUP="${PROB_LOSS_WARMUP:-3}"

RUN_HARDWARE_ABLATIONS="${RUN_HARDWARE_ABLATIONS:-0}"

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-outputs/experiments/tmc-diagnostics}"
CHECKPOINT_ROOT="$EXPERIMENT_ROOT/checkpoints"
REPORT_ROOT="$EXPERIMENT_ROOT/reports"
FIGURE_ROOT="$EXPERIMENT_ROOT/figures"

mkdir -p "$CHECKPOINT_ROOT" "$REPORT_ROOT" "$FIGURE_ROOT"

COMMON_ARGS=(
  --paper-preset "$PAPER_PRESET"
  --csi-error-var "$CSI_ERROR_VAR"
  --csi-error-model "$CSI_ERROR_MODEL"
  --csi-error-target "$CSI_ERROR_TARGET"
  --samples-per-snr "$SAMPLES_PER_SNR"
  --epochs "$EPOCHS"
  --patience "$PATIENCE"
  --val-batches "$VAL_BATCHES"
  --batch-size "$TRAIN_BATCH_SIZE"
  --device "$DEVICE"
  --seed "$SEED"
  --learning-rate "$LEARNING_RATE"
  --token-dim "$TOKEN_DIM"
  --n-layers "$LAYERS"
  --n-heads "$HEADS"
)

if [[ -n "$TRAIN_STEPS_PER_EPOCH" ]]; then
  COMMON_ARGS+=(--steps-per-epoch "$TRAIN_STEPS_PER_EPOCH")
fi

run_case() {
  local name="$1"
  shift
  local -a train_extra=()
  local -a eval_extra=()

  while (($#)); do
    case "$1" in
      --enable-phase-quantization|--no-enable-phase-quantization|--enable-amplitude-coupling|--no-enable-amplitude-coupling|--enable-mutual-coupling|--no-enable-mutual-coupling)
        train_extra+=("$1")
        eval_extra+=("$1")
        shift
        ;;
      --rank-weight|--ce-weight|--distill-prob-weight|--distill-coord-weight|--temperature|--loss-warmup-epochs)
        train_extra+=("$1" "$2")
        shift 2
        ;;
      *)
        train_extra+=("$1" "$2")
        eval_extra+=("$1" "$2")
        shift 2
        ;;
    esac
  done

  echo
  echo "== $name =="
  python scripts/train_tmc.py \
    "${COMMON_ARGS[@]}" \
    --output-dir "$CHECKPOINT_ROOT/$name" \
    --report-path "$REPORT_ROOT/${name}_train.json" \
    "${train_extra[@]}"

  python scripts/infer_tmc.py \
    --checkpoint "$CHECKPOINT_ROOT/$name/tmc_best.pt" \
    --num-bits "$NUM_BITS" \
    --batch-size "$EVAL_BATCH_SIZE" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --report-path "$REPORT_ROOT/${name}_infer.json" \
    --figure-path "$FIGURE_ROOT/${name}_ber.png" \
    "${eval_extra[@]}"

  python scripts/diagnose_tmc.py \
    --checkpoint "$CHECKPOINT_ROOT/$name/tmc_best.pt" \
    --num-bits "$NUM_BITS" \
    --batch-size "$EVAL_BATCH_SIZE" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --report-path "$REPORT_ROOT/${name}_diagnostics.json" \
    --figure-dir "$FIGURE_ROOT/$name" \
    "${eval_extra[@]}"
}


run_case aggressive_prob_distill \
  --ris-phase-bits "$AGGR_PHASE_BITS" \
  --ris-amplitude-bias "$AGGR_AMP_BIAS" \
  --ris-amplitude-scale "$AGGR_AMP_SCALE" \
  --ris-coupling-decay "$AGGR_COUPLING_DECAY" \
  --rank-weight "$PROB_RANK_WEIGHT" \
  --ce-weight "$PROB_CE_WEIGHT" \
  --distill-prob-weight "$PROB_PROB_WEIGHT" \
  --distill-coord-weight "$PROB_COORD_WEIGHT" \
  --temperature "$PROB_TEMP" \
  --loss-warmup-epochs "$PROB_LOSS_WARMUP"

run_case mild_prob_distill \
  --ris-phase-bits "$MILD_PHASE_BITS" \
  --ris-amplitude-bias "$MILD_AMP_BIAS" \
  --ris-amplitude-scale "$MILD_AMP_SCALE" \
  --ris-coupling-decay "$MILD_COUPLING_DECAY" \
  --rank-weight "$PROB_RANK_WEIGHT" \
  --ce-weight "$PROB_CE_WEIGHT" \
  --distill-prob-weight "$PROB_PROB_WEIGHT" \
  --distill-coord-weight "$PROB_COORD_WEIGHT" \
  --temperature "$PROB_TEMP" \
  --loss-warmup-epochs "$PROB_LOSS_WARMUP"

if [[ "$RUN_HARDWARE_ABLATIONS" == "1" ]]; then
  run_case csi_only_prob \
    --ris-phase-bits "$AGGR_PHASE_BITS" \
    --ris-amplitude-bias "$AGGR_AMP_BIAS" \
    --ris-amplitude-scale "$AGGR_AMP_SCALE" \
    --ris-coupling-decay "$AGGR_COUPLING_DECAY" \
    --no-enable-phase-quantization \
    --no-enable-amplitude-coupling \
    --no-enable-mutual-coupling \
    --rank-weight "$PROB_RANK_WEIGHT" \
    --ce-weight "$PROB_CE_WEIGHT" \
    --distill-prob-weight "$PROB_PROB_WEIGHT" \
    --distill-coord-weight "$PROB_COORD_WEIGHT" \
    --temperature "$PROB_TEMP" \
    --loss-warmup-epochs "$PROB_LOSS_WARMUP"

  run_case quant_only_prob \
    --ris-phase-bits "$AGGR_PHASE_BITS" \
    --ris-amplitude-bias "$AGGR_AMP_BIAS" \
    --ris-amplitude-scale "$AGGR_AMP_SCALE" \
    --ris-coupling-decay "$AGGR_COUPLING_DECAY" \
    --enable-phase-quantization \
    --no-enable-amplitude-coupling \
    --no-enable-mutual-coupling \
    --rank-weight "$PROB_RANK_WEIGHT" \
    --ce-weight "$PROB_CE_WEIGHT" \
    --distill-prob-weight "$PROB_PROB_WEIGHT" \
    --distill-coord-weight "$PROB_COORD_WEIGHT" \
    --temperature "$PROB_TEMP" \
    --loss-warmup-epochs "$PROB_LOSS_WARMUP"

  run_case quant_amp_prob \
    --ris-phase-bits "$AGGR_PHASE_BITS" \
    --ris-amplitude-bias "$AGGR_AMP_BIAS" \
    --ris-amplitude-scale "$AGGR_AMP_SCALE" \
    --ris-coupling-decay "$AGGR_COUPLING_DECAY" \
    --enable-phase-quantization \
    --enable-amplitude-coupling \
    --no-enable-mutual-coupling \
    --rank-weight "$PROB_RANK_WEIGHT" \
    --ce-weight "$PROB_CE_WEIGHT" \
    --distill-prob-weight "$PROB_PROB_WEIGHT" \
    --distill-coord-weight "$PROB_COORD_WEIGHT" \
    --temperature "$PROB_TEMP" \
    --loss-warmup-epochs "$PROB_LOSS_WARMUP"
fi

export DIAG_REPORT_ROOT="$REPORT_ROOT"
python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["DIAG_REPORT_ROOT"])
rows = []
for infer_path in sorted(root.glob("*_infer.json")):
    name = infer_path.stem.removesuffix("_infer")
    diag_path = root / f"{name}_diagnostics.json"
    train_path = root / f"{name}_train.json"
    infer = json.loads(infer_path.read_text())
    diag = json.loads(diag_path.read_text())
    train = json.loads(train_path.read_text())
    rows.append(
        {
            "name": name,
            "best_val": train.get("best_val_corrected_ml_acc"),
            "tmc_acc": sum(infer["tmc_corrected"]["symbol_acc"]) / len(infer["tmc_corrected"]["symbol_acc"]),
            "ideal_acc": sum(infer["ideal_ml"]["symbol_acc"]) / len(infer["ideal_ml"]["symbol_acc"]),
            "practical_acc": sum(infer["practical_oracle"]["symbol_acc"]) / len(infer["practical_oracle"]["symbol_acc"]),
            "genie_acc": sum(infer["true_center_oracle"]["symbol_acc"]) / len(infer["true_center_oracle"]["symbol_acc"]),
            "largest_gap_snr": diag["largest_practical_oracle_gap_snr"],
            "largest_gap": diag["largest_practical_oracle_gap"],
            "delta_cos": diag["alignment"]["delta_direction_cosine"],
        }
    )

print("\nSummary:")
for row in rows:
    print(
        f"{row['name']:>24} | "
        f"best_val={row['best_val']:.4f} "
        f"tmc_acc={row['tmc_acc']:.4f} "
        f"ideal_acc={row['ideal_acc']:.4f} "
        f"practical_acc={row['practical_acc']:.4f} "
        f"genie_acc={row['genie_acc']:.4f} "
        f"gap@{row['largest_gap_snr']:>2}dB={row['largest_gap']:.4f} "
        f"delta_cos={row['delta_cos']:.4f}"
    )
PY
