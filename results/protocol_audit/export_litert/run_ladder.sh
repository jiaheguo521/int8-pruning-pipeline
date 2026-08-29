#!/usr/bin/env bash
# Full-mapping export of every detection rung: .pt -> float tflite (litert-torch) -> int8
# (ai-edge-quantizer) -> buffers inlined -> edgetpu_compiler. Target: 0 operations on CPU,
# and any rung that misses it is reported rather than hidden. QSV_MINMAX=1 swaps
# ai-edge-quantizer's smoothed activation ranges for the true min/max: +0.020 mAP on
# efficientdet_lite1 (0.2932 -> 0.3134).
set -uo pipefail

# L: this dir. R: repo root ($ETPU_REPO). A: the litert-torch venv, NOT part of this repo -- see README.md.
L="$(cd "$(dirname "$0")" && pwd)"
R="${ETPU_REPO:-$(cd "$L/../../.." && pwd)}"
A="${AIEDGE_ENV:?set AIEDGE_ENV to a venv with litert-torch 0.9.3 + ai-edge-quantizer 0.8.0}"
FLOAT_DIR="${FLOAT_DIR:-$R/outputs/tflite_float_litert}"
INT8_DIR="$R/outputs/tflite_int8_litert"
ETPU_DIR="$R/outputs/edgetpu/litert_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$FLOAT_DIR" "$INT8_DIR" "$ETPU_DIR"
SUMMARY="$ETPU_DIR/mapping_summary.tsv"
printf 'stem\tstatus\ttpu_ops\tcpu_ops\tonchip\toffchip\tint8_MiB\tedgetpu_MiB\n' > "$SUMMARY"

for pt in "$R"/outputs/pytorch_pruned/efficientdet_*coco-train2017*.pt; do
    stem=$(basename "$pt" .pt)
    # --pool-ceil is NOT safe on lite2: ceil_mode=True equals timm's SAME-pool only when the
    # pool input is even. lite1 @384 pools 48->24->12->6->3, all even, so it is bit-exact;
    # lite2 @448 pools 56->28->14->7->4, and at input 7 SAME gives 4x4 while ceil gives 3x3,
    # a different network. The 5 lite2 rungs built before 2026-08-22 carry that defect.
    case "$stem" in
        *lite2*) calib="$R/outputs/tflite_int8/calib_cache/efficientdet_lite2_coco_n100_seed42.npy"
                 POOL_FLAG="" ;;
        *)       calib="$R/outputs/tflite_int8/calib_cache/efficientdet_lite1_coco_n100_seed42.npy"
                 POOL_FLAG="--pool-ceil" ;;
    esac
    f32="$FLOAT_DIR/${stem}_f32.tflite"
    int8_raw="$FLOAT_DIR/${stem}_int8_raw.tflite"
    int8="$INT8_DIR/${stem}_int8.tflite"
    echo "=== [$(date +%H:%M:%S)] $stem"

    if [[ ! -f "$int8" || "${FORCE:-0}" == "1" ]]; then
        "$A/bin/python" "$L/aiedge_float.py" --patch-fpn $POOL_FLAG --nhwc-out \
            "$pt" "$f32" 2>&1 | grep -E '^\[float\]' || { echo "  FLOAT FAILED"; \
            printf '%s\tfloat_failed\t-\t-\t-\t-\t-\t-\n' "$stem" >> "$SUMMARY"; continue; }
        CALIB_NPY="$calib" CALIB_N=100 QSV_MINMAX=1 "$A/bin/python" "$L/aeq_quantize.py" \
            "$f32" "$int8_raw" 2>&1 | grep -E '^\[int8\]' || { echo "  QUANT FAILED"; \
            printf '%s\tquant_failed\t-\t-\t-\t-\t-\t-\n' "$stem" >> "$SUMMARY"; continue; }
        PYTHONPATH="$R/src" "$R/pruning-env/bin/python" -m etpu.convert.flatbuffer \
            inline "$int8_raw" "$int8" 2>&1 | grep -E '^\[inline\]' || { echo "  INLINE FAILED"; \
            printf '%s\tinline_failed\t-\t-\t-\t-\t-\t-\n' "$stem" >> "$SUMMARY"; continue; }
        rm -f "$int8_raw"     # keep $f32: re-quantizing is 3 min, re-exporting is 5
    else
        echo "  [skip] $(basename "$int8") exists"
    fi

    log=$(edgetpu_compiler -o "$ETPU_DIR" -s "$int8" 2>&1)
    echo "$log" | tail -25 > "$ETPU_DIR/${stem}_compile.log"
    tpu=$(echo "$log" | grep -oP 'run on Edge TPU: \K[0-9]+' | tail -1)
    cpu=$(echo "$log" | grep -oP 'run on CPU: \K[0-9]+' | tail -1)
    on=$( echo "$log" | grep -oP 'On-chip memory used for caching model parameters: \K.*'  | tail -1)
    off=$(echo "$log" | grep -oP 'Off-chip memory used for streaming uncached model parameters: \K.*' | tail -1)
    # "0 on CPU" prints no CPU line at all -- the compiler omits the warning block.
    if [[ -z "${tpu:-}" ]]; then
        tpu=$(echo "$log" | grep -oP 'Total number of operations: \K[0-9]+' | tail -1); cpu=0
    fi
    etpu="$ETPU_DIR/${stem}_int8_edgetpu.tflite"
    if [[ -f "$etpu" ]]; then
        printf '%s\tok\t%s\t%s\t%s\t%s\t%.2f\t%.2f\n' "$stem" "${tpu:-?}" "${cpu:-?}" \
            "${on:-?}" "${off:-?}" \
            "$(stat -c%s "$int8" | awk '{print $1/1048576}')" \
            "$(stat -c%s "$etpu" | awk '{print $1/1048576}')" >> "$SUMMARY"
        echo "  -> TPU ${tpu:-?} / CPU ${cpu:-?}   on-chip ${on:-?}  off-chip ${off:-?}"
    else
        printf '%s\tcompile_failed\t-\t-\t-\t-\t-\t-\n' "$stem" >> "$SUMMARY"
        echo "  COMPILE FAILED"
    fi
done

echo; echo "=== summary: $SUMMARY"; column -t -s $'\t' "$SUMMARY"
