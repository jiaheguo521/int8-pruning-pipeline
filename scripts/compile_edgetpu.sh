#!/usr/bin/env bash
# compile_edgetpu.sh — Phase 3 driver: TFLite int8 -> Coral Edge TPU, one run dir per
# invocation. PAIR mode co-compiles two models so they share the on-chip cache.
#
# Usage:
#   ./scripts/compile_edgetpu.sh
#   GLOB='mobilenetv2_*_int8.tflite' SHOW_OPS=1 ./scripts/compile_edgetpu.sh
#   PAIR="<stemA> <stemB>" ./scripts/compile_edgetpu.sh
# Env vars and the pre-flight apt install: docs/SETUP.md section 5.

set -euo pipefail

# Shared paths live in scripts/config.sh. No venv needed: edgetpu_compiler is a standalone binary.
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
GLOB="${GLOB:-*_int8.tflite}"
PAIR="${PAIR:-}"
SHOW_OPS="${SHOW_OPS:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
EDGETPU_COMPILER="${EDGETPU_COMPILER:-edgetpu_compiler}"

if ! command -v "$EDGETPU_COMPILER" >/dev/null 2>&1; then
    echo "ERROR: edgetpu_compiler not found (set EDGETPU_COMPILER or apt install edgetpu-compiler)" >&2
    exit 1
fi
if [[ ! -d "$TFLITE_DIR" ]]; then
    echo "ERROR: TFLITE_DIR not found: $TFLITE_DIR" >&2
    exit 1
fi

mkdir -p "$EDGETPU_DIR"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/compile_edgetpu_${TS}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

resolve_pair_input() {
    # Resolve a PAIR stem to its absolute *_int8.tflite path, bare or already suffixed.
    local stem="$1"
    local path
    if [[ "$stem" == *"_int8.tflite" ]]; then
        path="$TFLITE_DIR/$stem"
    else
        path="$TFLITE_DIR/${stem}_int8.tflite"
    fi
    if [[ ! -f "$path" ]]; then
        echo "ERROR: PAIR input not found: $path" >&2
        exit 1
    fi
    printf '%s\n' "$path"
}

copy_sidecar() {
    # Copy <name>_int8.size.json beside the artifact: i/o dtype/shape and calib metadata.
    local src="$1" dst_dir="$2"
    local sidecar="${src%.tflite}.size.json"
    if [[ -f "$sidecar" ]]; then
        cp -- "$sidecar" "$dst_dir/"
    fi
}

echo "[$(date)] compile_edgetpu.sh starting"
echo "  EDGETPU_COMPILER=$EDGETPU_COMPILER ($($EDGETPU_COMPILER --version 2>&1 | head -1))"
echo "  PROJECT_DIR=$PROJECT_DIR"
echo "  TFLITE_DIR=$TFLITE_DIR"
echo "  EDGETPU_DIR=$EDGETPU_DIR"
echo "  SHOW_OPS=$SHOW_OPS"
echo "  EXTRA_ARGS='$EXTRA_ARGS'"
echo "  LOG=$LOG_FILE"

if [[ -n "$PAIR" ]]; then
    # ---------- pair / co-compile mode ----------
    # shellcheck disable=SC2206
    PAIR_STEMS=($PAIR)
    if [[ "${#PAIR_STEMS[@]}" -ne 2 ]]; then
        echo "ERROR: PAIR must contain exactly 2 space-separated stems; got ${#PAIR_STEMS[@]}" >&2
        exit 1
    fi
    STEM_A="${PAIR_STEMS[0]%_int8.tflite}"
    STEM_B="${PAIR_STEMS[1]%_int8.tflite}"
    SRC_A="$(resolve_pair_input "${PAIR_STEMS[0]}")"
    SRC_B="$(resolve_pair_input "${PAIR_STEMS[1]}")"
    RUN_DIR="$EDGETPU_DIR/pair_${STEM_A}__${STEM_B}_${TS}"
    mkdir -p "$RUN_DIR"
    echo "  MODE=pair"
    echo "  RUN_DIR=$RUN_DIR"
    echo "  PAIR=($SRC_A) + ($SRC_B)"
    echo

    if [[ "$SHOW_OPS" == "1" ]]; then
        # --show_operations does not compile, so run it separately per input to capture the op map as text.
        # shellcheck disable=SC2086
        "$EDGETPU_COMPILER" --show_operations $EXTRA_ARGS "$SRC_A" \
            > "$RUN_DIR/${STEM_A}.show_ops.txt"
        # shellcheck disable=SC2086
        "$EDGETPU_COMPILER" --show_operations $EXTRA_ARGS "$SRC_B" \
            > "$RUN_DIR/${STEM_B}.show_ops.txt"
    fi

    # shellcheck disable=SC2086
    "$EDGETPU_COMPILER" -o "$RUN_DIR" $EXTRA_ARGS "$SRC_A" "$SRC_B"

    copy_sidecar "$SRC_A" "$RUN_DIR"
    copy_sidecar "$SRC_B" "$RUN_DIR"
else
    # ---------- single-model batch mode ----------
    RUN_DIR="$EDGETPU_DIR/single_${TS}"
    mkdir -p "$RUN_DIR"
    echo "  MODE=single  GLOB='$GLOB'"
    echo "  RUN_DIR=$RUN_DIR"
    echo

    shopt -s nullglob
    INPUTS=("$TFLITE_DIR"/$GLOB)
    shopt -u nullglob
    if [[ "${#INPUTS[@]}" -eq 0 ]]; then
        echo "ERROR: no .tflite matched '$GLOB' under $TFLITE_DIR" >&2
        exit 1
    fi

    for src in "${INPUTS[@]}"; do
        name="$(basename "$src")"
        stem="${name%_int8.tflite}"
        echo "--- compiling $name ---"
        if [[ "$SHOW_OPS" == "1" ]]; then
            # shellcheck disable=SC2086
            "$EDGETPU_COMPILER" --show_operations $EXTRA_ARGS "$src" \
                > "$RUN_DIR/${stem}.show_ops.txt"
        fi
        # shellcheck disable=SC2086
        "$EDGETPU_COMPILER" -o "$RUN_DIR" $EXTRA_ARGS "$src"
        copy_sidecar "$src" "$RUN_DIR"
    done
fi

echo
echo "[$(date)] compile_edgetpu.sh done"
echo "Artifacts in: $RUN_DIR"
ls -1 "$RUN_DIR"
