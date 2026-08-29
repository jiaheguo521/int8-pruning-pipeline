#!/usr/bin/env bash
# bench.sh — single-inference latency of the line-seg models on the PHYSICAL Coral USB
# Edge TPU, as JSON for report.py. Measurement: src/int8_pruning/backends/edgetpu/bench.py.
#
# Usage:
#   DIRS="d1 d2" RUNTIME=docker|venv OUT_JSON=... ./families/segmentation/line_seg/bench.sh
#
# RUNTIME=venv is BROKEN in this checkout (ai_edge_litert 2.1.6 fails at invoke, pycoral is
# in neither venv), so the edgetpu_ros docker image is a single point of failure for every
# on-device number published here. If the delegate fails to load, unplug and replug the
# Coral: it re-enumerates 1a6e:089a -> 18d1:9302 after a first successful open, and a run
# that died holding it blocks later opens. The dense latency-vs-bytes sweep that once ran
# here found no cliff -- docs/PRUNING_HAZARDS.md section 2, rows in results/line_seg/.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../../../scripts/config.sh"

RUNTIME="${RUNTIME:-docker}"   # venv path is broken here -- see the header
DOCKER_IMAGE="${DOCKER_IMAGE:-edgetpu_ros:latest}"
OUT_JSON="${OUT_JSON:-$PROJECT_DIR/outputs/line_seg_eval/latency.json}"
DIRS="${DIRS:-$EDGETPU_DIR}"

# Newest compiled artifact per stem: re-compiles write a fresh run dir, and a stale one would be benched.
declare -A latest
while IFS= read -r line; do
    p="${line#* }"
    latest["$(basename "$p")"]="$p"
done < <(for d in $DIRS; do
             [[ -d "$d" ]] && find "$d" -name 'line_seg_*_int8_edgetpu.tflite' -printf '%T@ %p\n'
         done | sort -n)

MODELS=("${latest[@]}")
[[ "${#MODELS[@]}" -gt 0 ]] || { echo "ERROR: no line_seg_*_int8_edgetpu.tflite under: $DIRS" >&2; exit 1; }

echo "[$(date)] families/segmentation/line_seg/bench.sh  runtime=$RUNTIME  ${#MODELS[@]} models"

if [[ "$RUNTIME" == "docker" ]]; then
    mounts=()
    for d in $DIRS; do [[ -d "$d" ]] && mounts+=(-v "$d":"$d":ro); done
    RAW="$(docker run --rm --privileged -v /dev/bus/usb:/dev/bus/usb \
        -v "$PROJECT_DIR/src/int8_pruning/backends/edgetpu":/src:ro "${mounts[@]}" "$DOCKER_IMAGE" \
        python3 /src/bench.py "${MODELS[@]}")"
else
    require_venv
    if [[ ! -d "$PROJECT_DIR/sources/_compat" ]]; then
        echo "ERROR: RUNTIME=venv needs \$PROJECT_DIR/sources/_compat, which is not in this" >&2
        echo "       checkout. Neither venv can drive the Coral either (see the header)." >&2
        echo "       Use RUNTIME=docker." >&2
        exit 1
    fi
    RAW="$(PYTHONPATH="$PROJECT_DIR/sources/_compat${PYTHONPATH:+:$PYTHONPATH}" \
        python "$PROJECT_DIR/src/int8_pruning/backends/edgetpu/bench.py" "${MODELS[@]}")"
fi
echo "$RAW"

if grep -q "ERROR:" <<<"$RAW"; then
    echo
    echo "[ERROR] at least one model failed to run on the Coral." >&2
    echo "        If it says 'Failed to load delegate': unplug and replug the Coral, then re-run." >&2
    exit 1
fi

mkdir -p "$(dirname "$OUT_JSON")"
awk 'BEGIN{print "["; sep=""}
     /_int8_edgetpu\.tflite/ {
        stem=$1; sub(/_int8_edgetpu\.tflite$/, "", stem);
        printf "%s  {\"stem\": \"%s\", \"mean\": %s, \"median\": %s, \"std\": %s, \"min\": %s}",
               sep, stem, $2, $3, $4, $5; sep=",\n" }
     END{print "\n]"}' <<<"$RAW" > "$OUT_JSON"

echo
echo "[$(date)] wrote $OUT_JSON"
