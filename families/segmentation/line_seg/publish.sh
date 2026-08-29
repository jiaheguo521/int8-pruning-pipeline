#!/usr/bin/env bash
# publish.sh — Phase 4 driver for the line-seg family: assemble the upstream-shaped model
# dirs and score them with the UPSTREAM eval protocol. The output is a [1,1,H/2,W/2]
# mask-logit map, so this repo's zero-shot and mAP paths do not apply.
#
# Usage:
#   LANE_SEG_UPSTREAM=... GLOB=... EVAL_DIR=... OUT_JSON=... \
#     ./families/segmentation/line_seg/publish.sh
#
# The shape <upstream>/training/eval_int8_lane_seg.py wants:
#   <stem>/meta.json                        geometry + normalization, copied unchanged
#   <stem>/tf/*_full_integer_quant.tflite   the int8 graph (scored on CPU)
#   <stem>/model_edgetpu.tflite             the compiled graph (--edgetpu)
# Accuracy is scored on the int8 graph on CPU. Upstream's --edgetpu leg is unreachable
# from either runtime available here (DELIVERABLE.md says why), so on-device numbers come
# from bench.sh instead.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../../../scripts/config.sh"
require_venv

LANE_SEG_UPSTREAM="${LANE_SEG_UPSTREAM:-$PROJECT_DIR/../MicroROS-Pi5_Coral_TPU/training}"
GLOB="${GLOB:-line_seg_*_int8.tflite}"
EVAL_DIR="${EVAL_DIR:-$PROJECT_DIR/outputs/line_seg_eval}"
OUT_JSON="${OUT_JSON:-$EVAL_DIR/results.json}"

UPSTREAM="$(cd "$LANE_SEG_UPSTREAM" && pwd)"
[[ -f "$UPSTREAM/eval_int8_lane_seg.py" ]] || {
    echo "ERROR: $UPSTREAM/eval_int8_lane_seg.py not found" >&2; exit 1; }

mkdir -p "$EVAL_DIR"

shopt -s nullglob
INPUTS=("$TFLITE_DIR"/$GLOB)
shopt -u nullglob
[[ "${#INPUTS[@]}" -gt 0 ]] || { echo "ERROR: no tflite matched '$GLOB' in $TFLITE_DIR" >&2; exit 1; }

echo "[$(date)] families/segmentation/line_seg/publish.sh"
echo "  UPSTREAM      = $UPSTREAM"
echo "  TFLITE_DIR    = $TFLITE_DIR   GLOB='$GLOB' (${#INPUTS[@]} models)"
echo "  EDGETPU_DIR   = $EDGETPU_DIR"
echo "  EVAL_DIR      = $EVAL_DIR"
echo

rows=()
for src in "${INPUTS[@]}"; do
    name="$(basename "$src")"
    stem="${name%_int8.tflite}"
    variant="${stem%%_pruned*}"
    dir="$EVAL_DIR/$stem"

    meta="$DATASET_DIR/line_seg/$variant/meta.json"
    [[ -f "$meta" ]] || { echo "ERROR: $meta missing (run families/segmentation/line_seg/setup.sh)" >&2; exit 1; }

    # Newest compiled artifact wins: one timestamped run dir per invocation, so never score a stale copy.
    edge=""
    while IFS= read -r cand; do edge="$cand"; done < <(
        find "$EDGETPU_DIR" -name "${stem}_int8_edgetpu.tflite" -printf '%T@ %p\n' 2>/dev/null \
            | sort -n | cut -d' ' -f2-)

    mkdir -p "$dir/tf"
    cp -- "$meta" "$dir/meta.json"
    cp -- "$src" "$dir/tf/${stem}_full_integer_quant.tflite"
    # `if`, not `&&`: a bare `[[ ]] && cmd` that tests false returns 1 and aborts the run under `set -e`.
    if [[ -n "$edge" ]]; then cp -- "$edge" "$dir/model_edgetpu.tflite"; fi

    echo "--- $stem (variant=$variant) ---"
    cpu_json="$(python "$UPSTREAM/eval_int8_lane_seg.py" "$dir" 2>/dev/null | tail -1 || true)"
    [[ -n "$cpu_json" ]] || { echo "ERROR: int8-cpu eval produced no output for $stem" >&2; exit 1; }
    echo "  int8-cpu : $cpu_json"

    # Four keys, not three. `edgetpu` is null here because on-device accuracy needs the Coral plus the
    # upstream scorer in one process, which this repo cannot do -- but report_lane_seg.py reads the key
    # and the shipped results.json has it, so omitting it made a regenerated file silently lose one.
    rows+=("$stem" "$variant" "${cpu_json:-null}")
done

mkdir -p "$(dirname "$OUT_JSON")"
# Serialise with json, not shell interpolation: a variant name holding a quote emitted invalid JSON.
python - "$OUT_JSON" "${rows[@]}" <<'PYJSON'
import json, sys
out, flat = sys.argv[1], sys.argv[2:]
rows = []
for stem, variant, cpu in zip(flat[0::3], flat[1::3], flat[2::3]):
    rows.append({"stem": stem, "variant": variant,
                 "int8_cpu": json.loads(cpu) if cpu != "null" else None,
                 "edgetpu": None})
with open(out, "w") as fh:
    json.dump(rows, fh, indent=2)
    fh.write("\n")
PYJSON
echo
echo "[$(date)] wrote $OUT_JSON"
