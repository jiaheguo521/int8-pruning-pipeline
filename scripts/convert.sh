#!/usr/bin/env bash
# convert.sh — Phase 2 driver: PyTorch (.pt) -> TFLite int8, every .pt under
# PT_PRUNED_DIR. Calibration is built once per family and cached under the output dir.
#
# Usage:
#   ./scripts/convert.sh
#   GLOB='mobilenetv2_*pct_*.pt' EXTRA_ARGS='--skip-existing' ./scripts/convert.sh
#   . ./onnx2tf-env/bin/activate && EXPORT_PATH=onnx2tf ./scripts/convert.sh
# Env vars, the three export paths and what each is for: docs/SETUP.md section 4.

set -euo pipefail

# Shared paths + require_venv live in scripts/config.sh.
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
require_venv

GLOB="${GLOB:-*.pt}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
EXPORT_PATH="${EXPORT_PATH:-litert}"
QUANT_TYPE="${QUANT_TYPE:-per-channel}"

case "$EXPORT_PATH" in
    litert)  OUT_DIR="$TFLITE_DIR" ;;
    onnx2tf) OUT_DIR="$TFLITE_ONNX2TF_DIR"
             EXTRA_ARGS="$EXTRA_ARGS --quant-type $QUANT_TYPE" ;;
    onnx)    OUT_DIR="$ONNX_FP32_DIR" ;;
    *) echo "ERROR: EXPORT_PATH must be litert, onnx2tf or onnx (got: $EXPORT_PATH)" >&2
       exit 1 ;;
esac

if [[ ! -d "$PT_PRUNED_DIR" ]]; then
    echo "ERROR: PT_PRUNED_DIR not found: $PT_PRUNED_DIR" >&2
    exit 1
fi
# Only the quantizing paths read a dataset; the fp32 ONNX path needs neither calibration nor data.
if [[ "$EXPORT_PATH" != "onnx" && ! -d "$DATASET_DIR" ]]; then
    echo "ERROR: DATASET_DIR not found: $DATASET_DIR" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/convert_${TS}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[$(date)] convert.sh starting"
echo "  PROJECT_DIR=$PROJECT_DIR"
if [[ "$EXPORT_PATH" == "onnx" ]]; then
    echo "  DATASET_DIR=(unused: fp32 export does not calibrate)"
else
    echo "  DATASET_DIR=$DATASET_DIR"
fi
echo "  PT_PRUNED_DIR=$PT_PRUNED_DIR"
echo "  EXPORT_PATH=$EXPORT_PATH  OUT_DIR=$OUT_DIR"
echo "  NUM_CALIB=$NUM_CALIB SEED=$SEED GLOB='$GLOB'"
echo "  EXTRA_ARGS='$EXTRA_ARGS'"
echo "  LOG=$LOG_FILE"
echo

# shellcheck disable=SC2086
python "$PROJECT_DIR/src/int8_pruning/convert/tflite.py" \
    --input-dir "$PT_PRUNED_DIR" \
    --output-dir "$OUT_DIR" \
    --export-path "$EXPORT_PATH" \
    --dataset-dir "$DATASET_DIR" \
    --project-dir "$PROJECT_DIR" \
    --num-calib "$NUM_CALIB" \
    --seed "$SEED" \
    --glob "$GLOB" \
    $EXTRA_ARGS

echo
echo "[$(date)] convert.sh done"
