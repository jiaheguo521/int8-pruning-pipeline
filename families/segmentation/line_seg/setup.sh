#!/usr/bin/env bash
# setup.sh — one-time wiring of the upstream line-seg deliverables. Nothing is copied,
# only symlinked, so there is exactly one copy of every artifact on disk:
#   MODELS_DIR/line_seg_<v>.pt          -> <upstream>/outputs/line_seg_<v>/line_seg_<v>.pt
#   DATASET_DIR/line_seg/line_seg_<v>/  -> <upstream>/outputs/line_seg_<v>/
# The second is int8_pruning.convert.tflite's `dataset_subdir`: it supplies the delivered
# cal.npy (200 raw-RGB frames, 140 track + 60 floor, ROI-cropped) and meta.json.
# lane_seg_models.py must stay byte-identical to the upstream file -- it is the module the
# delivered .pt files unpickle against, and prune.py re-checks the sha256 every run.
#
# Usage:
#   LANE_SEG_UPSTREAM=... ./families/segmentation/line_seg/setup.sh

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../../../scripts/config.sh"

LANE_SEG_UPSTREAM="${LANE_SEG_UPSTREAM:-$PROJECT_DIR/../MicroROS-Pi5_Coral_TPU/training}"
VARIANTS=(line_seg_base128 line_seg_w96 line_seg_link_r34)

if [[ ! -d "$LANE_SEG_UPSTREAM" ]]; then
    echo "[ERROR] LANE_SEG_UPSTREAM not found: $LANE_SEG_UPSTREAM" >&2
    echo "        Point it at the MicroROS-Pi5_Coral_TPU repo's training/ directory." >&2
    exit 1
fi
UPSTREAM="$(cd "$LANE_SEG_UPSTREAM" && pwd)"

echo "  LANE_SEG_UPSTREAM = $UPSTREAM"
echo "  MODELS_DIR        = $MODELS_DIR"
echo "  DATASET_DIR       = $DATASET_DIR"
echo

# lane_seg_models.py — the unpickling module (handoff §B: "copy this one file").
src_mod="$UPSTREAM/lane_seg_models.py"
dst_mod="$PROJECT_DIR/families/segmentation/line_seg/lane_seg_models.py"
[[ -f "$src_mod" ]] || { echo "[ERROR] missing $src_mod" >&2; exit 1; }
cp -- "$src_mod" "$dst_mod"
echo "  [module] families/segmentation/line_seg/lane_seg_models.py <- upstream ($(sha256sum "$dst_mod" | cut -c1-12))"

mkdir -p "$MODELS_DIR" "$DATASET_DIR/line_seg"
for v in "${VARIANTS[@]}"; do
    src_dir="$UPSTREAM/outputs/$v"
    if [[ ! -f "$src_dir/$v.pt" ]] || [[ ! -f "$src_dir/cal.npy" ]]; then
        echo "[ERROR] incomplete upstream delivery for $v: need $src_dir/{$v.pt,cal.npy}" >&2
        exit 1
    fi
    ln -sfn "$src_dir/$v.pt" "$MODELS_DIR/$v.pt"
    ln -sfn "$src_dir"       "$DATASET_DIR/line_seg/$v"
    echo "  [link] $v  baseline + cal.npy/meta.json"
done

echo
echo "  ✓ ready. Next:"
echo "     CLS_DATASET=line_seg CLS_MODEL=line_seg_w96 CHECKPOINTS='10 30' ./scripts/pruning.sh"
