# config.sh — shared paths and helpers, sourced by every driver; every var keeps its ":-" guard.

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJECT_DIR
export DATASET_DIR="${DATASET_DIR:-/datasets}"

# Canonical output dirs, each named by what it holds: convert.sh writes TFLITE_DIR, compile_edgetpu.sh reads it.
export MODELS_DIR="${MODELS_DIR:-$PROJECT_DIR/outputs/models}"
export PT_PRUNED_DIR="${PT_PRUNED_DIR:-$PROJECT_DIR/outputs/pytorch_pruned}"
# The default export path (convert.sh EXPORT_PATH=litert) writes here.
export TFLITE_DIR="${TFLITE_DIR:-$PROJECT_DIR/outputs/tflite_int8_litert}"
# The onnx2tf path writes here, separately: the same .pt gives a different file on each path.
export TFLITE_ONNX2TF_DIR="${TFLITE_ONNX2TF_DIR:-$PROJECT_DIR/outputs/tflite_int8_onnx2tf}"
# The onnx path: fp32 ONNX graphs, for the toolchains that do not read TFLite. Unused downstream.
export ONNX_FP32_DIR="${ONNX_FP32_DIR:-$PROJECT_DIR/outputs/onnx_fp32}"
# READ-ONLY archive of the pre-2026-08-22 onnx2tf build; do not convert into it. Some ablation
# rungs exist only here, so deliver.sh resolves against both and records where each file came from.
export TFLITE_LEGACY_DIR="${TFLITE_LEGACY_DIR:-$PROJECT_DIR/outputs/tflite_int8}"
export EDGETPU_DIR="${EDGETPU_DIR:-$PROJECT_DIR/outputs/edgetpu}"
export LOG_DIR="${LOG_DIR:-$PROJECT_DIR/outputs/pruning_logs}"

# src-layout: `import int8_pruning` works for the drivers without an editable install.
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# Shared tuning knobs.
export SEED="${SEED:-42}"
export NUM_CALIB="${NUM_CALIB:-100}"

# Active-venv gate. Drivers that only invoke external binaries (compile_edgetpu.sh) skip it.
require_venv() {
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        echo "[ERROR] No active virtualenv (\$VIRTUAL_ENV is empty)." >&2
        echo "        Activate the project venv before running:" >&2
        echo "          source ./pruning-env/bin/activate" >&2
        exit 1
    fi
}
