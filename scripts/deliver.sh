#!/usr/bin/env bash
# deliver.sh — assemble deliverables/<dir>/ for one family, driven by the `deliver`
# section of families/<task>/<name>/family.yaml. The layout is the contract the
# downstream repo consumes:
#   deliverables/<dir>/{checkpoints_pytorch,models_int8_tflite,edgetpu,reference}/
# reference/<module>.py ships beside the checkpoints because the .pt files are full-module
# pickles: torch.load(weights_only=False) needs that module importable under its exact name
# (the pickles name `lane_seg_models` and `youreid_model`). The Re-ID package predates this
# script and was assembled by hand, which is why its reference JSON key names do not match
# what its own producer emits.
#
# Usage:
#   ./scripts/deliver.sh <family>            # line_seg | reid | effdet[:lite2]
#   DELIVER_DIR=... ./scripts/deliver.sh <family>

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

# Compiler logs are captured stdout and carry absolute paths. Stripping "$PROJECT_DIR/" alone
# misses the log's own "  PROJECT_DIR=<abs>" header and logs written before the repo moved, so:
# strip the current root, rewrite the header, strip any prefix ahead of outputs/.
strip_local_paths() {
    local src="$1" dst="$2"
    sed -e "s|$PROJECT_DIR/||g" \
        -e "s|^\( *\)PROJECT_DIR=/.*|\1PROJECT_DIR=.|" \
        -e "s#/[^ ]*/outputs/#outputs/#g" "$src" > "$dst"
    if grep -qE "(/media/|/home/|/Users/)" "$dst"; then
        echo "[ERROR] absolute path survived redaction in $dst" >&2
        grep -nE "(/media/|/home/|/Users/)" "$dst" | head -3 >&2
        exit 1
    fi
}

FAMILY="${1:-}"
if [[ -z "$FAMILY" ]]; then
    echo "usage: $0 <family>" >&2
    echo "  families with a deliver section:" >&2
    python -m int8_pruning.manifest deliver 2>&1 | sed 's/^/    /' >&2
    exit 2
fi

# Capture first: `eval "$(...)" || exit 1` tests eval, not the substitution, so a manifest error surfaced as `D_DIR: unbound variable`.
D_ASSIGNMENTS="$(python -m int8_pruning.manifest deliver "$FAMILY")" || exit 1
eval "$D_ASSIGNMENTS"

DELIVER_DIR="${DELIVER_DIR:-$PROJECT_DIR/deliverables/$D_DIR}"
EVAL_DIR="${EVAL_DIR:-$PROJECT_DIR/$D_EVAL_DIR}"

mkdir -p "$DELIVER_DIR"/{checkpoints_pytorch,models_int8_tflite,reference}
if [[ "$D_SHIP_EDGETPU" == "1" ]]; then mkdir -p "$DELIVER_DIR/edgetpu"; fi

n=0
legacy=0
edge_n=0
export_paths=()
for pt in "$PT_PRUNED_DIR"/$D_GLOB.pt; do
    [[ -e "$pt" ]] || continue
    stem="$(basename "$pt" .pt)"
    # -L: 0% rows may be symlinks into an upstream delivery, so ship the content. The reid 0% baseline
    # is a symlink INTO this package, which holds the only copy, and cp would abort with "are the same
    # file" -- so skip when the link already resolves to the destination.
    dest_pt="$DELIVER_DIR/checkpoints_pytorch/$(basename "$pt")"
    if [[ "$(readlink -f "$pt")" != "$(readlink -f "$dest_pt" 2>/dev/null)" ]]; then
        cp -L -- "$pt" "$dest_pt"
    fi

    # Two export generations coexist under outputs/: the current litert-torch path (TFLITE_DIR)
    # and the retired onnx2tf one (TFLITE_LEGACY_DIR). Ablation rungs never re-converted exist
    # only in the latter, so resolve instead of assuming. Which one each file came from is
    # recorded in reference/export_paths.json: byte counts and packaging_frac are NOT
    # comparable across the two (docs/PRUNING_HAZARDS.md section 4).
    int8="$TFLITE_DIR/${stem}_int8.tflite"
    epath="litert-torch+ai-edge-quantizer"
    if [[ ! -f "$int8" ]]; then
        int8="$TFLITE_LEGACY_DIR/${stem}_int8.tflite"
        epath="onnx2tf"
        if [[ ! -f "$int8" ]]; then
            echo "[ERROR] no int8 tflite for $stem in either" >&2
            echo "        $TFLITE_DIR" >&2
            echo "        $TFLITE_LEGACY_DIR" >&2
            exit 1
        fi
        legacy=$((legacy + 1))
    fi
    cp -- "$int8" "$DELIVER_DIR/models_int8_tflite/"
    export_paths+=("$stem|$epath")

    if [[ "$D_SHIP_EDGETPU" == "1" ]]; then
        # Newest compiled artifact wins: compile_edgetpu.sh writes a fresh run dir per invocation,
        # so a re-compile must not be shipped from a stale copy. Multi-model runs are skipped:
        # cocompile_*/ and pair_*/ hold artifacts compiled two-at-a-time to share one on-chip
        # allocation, under the SAME filename as the standalone build. For efficientdet_lite1 the
        # co-compile run (2026-08-22 06:54) is newer than its standalone one (08-21 23:34), so
        # "newest wins" alone would ship 12 of 19 rungs as co-compiled halves.
        edge=""
        while IFS= read -r cand; do edge="$cand"; done < <(
            find "$EDGETPU_DIR" -name "${stem}_int8_edgetpu.tflite" \
                -not -path '*/cocompile_*' -not -path '*/pair_*' \
                -printf '%T@ %p\n' 2>/dev/null \
                | sort -n | cut -d' ' -f2-)
        if [[ -n "$edge" ]]; then
            cp -- "$edge" "$DELIVER_DIR/edgetpu/"
            edge_n=$((edge_n + 1))
            # The compiler writes absolute paths into every log; strip_local_paths redacts them. `_compile.log`
            # is the `-s` stdout and carries the on-chip/off-chip split, `_int8_edgetpu.log` only the op table.
            for l in "${edge%.tflite}.log" "${edge%_int8_edgetpu.tflite}_compile.log"; do
                [[ -f "$l" ]] || continue
                strip_local_paths "$l" "$DELIVER_DIR/edgetpu/$(basename "$l")"
            done
        fi
    fi
    n=$((n + 1))
done

# Per-file export-path record: without it the package mixes two export generations undetectably.
{
    echo "{"
    echo "  \"what\": \"which export path each models_int8_tflite/ file was built with. onnx2tf was retired 2026-08-22; byte counts are not comparable across the two paths.\","
    echo "  \"models\": {"
    last=$(( ${#export_paths[@]} - 1 ))
    for i in "${!export_paths[@]}"; do
        s="${export_paths[$i]%%|*}"; p="${export_paths[$i]##*|}"
        printf '    "%s": "%s"%s\n' "$s" "$p" "$([[ $i -lt $last ]] && echo ,)"
    done
    echo "  }"
    echo "}"
} > "$DELIVER_DIR/reference/export_paths.json"

if [[ -n "$D_REF_MODULE" ]]; then
    cp -- "$PROJECT_DIR/$D_REF_MODULE" "$DELIVER_DIR/reference/"
fi

# The load contract has to travel with the files. line_seg and reid ship reference/<module>.py;
# effdet's pickles name pip packages (effdet, timm, omegaconf), so it writes the note instead.
if [[ -n "$D_CKPT_NOTE" ]]; then
    { echo "# Read this before loading anything in this directory"; echo
      echo "$D_CKPT_NOTE" | fold -s -w 88
    } > "$DELIVER_DIR/checkpoints_pytorch/READ_THIS_FIRST.md"
fi
for f in $D_REF_JSON; do
    if [[ -f "$EVAL_DIR/$f" ]]; then cp -- "$EVAL_DIR/$f" "$DELIVER_DIR/reference/"; fi
done

# The co-compile warning travels WITH the artifacts: in a model card it is one click from a file that looks drop-in and is not.
if [[ "$D_SHIP_EDGETPU" == "1" && "$edge_n" -gt 0 && -n "$D_EDGETPU_NOTE" ]]; then
    mkdir -p "$DELIVER_DIR/edgetpu"
    { echo "# Read this before using anything in this directory"; echo
      echo "$D_EDGETPU_NOTE" | fold -s -w 88
    } > "$DELIVER_DIR/edgetpu/READ_THIS_FIRST.md"
fi

if [[ "$D_SHIP_EDGETPU" == "1" && "$edge_n" -gt 0 && -n "$D_STDOUT_GREP" ]]; then
    # The full `-s` stdout (memory + subgraph lines) exists only in the run-level log.
    mkdir -p "$DELIVER_DIR/edgetpu/compiler_stdout"
    for f in "$LOG_DIR"/compile_edgetpu_*.log; do
        if grep -ql "$D_STDOUT_GREP" "$f" 2>/dev/null; then
            strip_local_paths "$f" \
                "$DELIVER_DIR/edgetpu/compiler_stdout/$(basename "$f")"
        fi
    done
fi

# __pycache__ next to a shipped reference module was going out in the package.
find "$DELIVER_DIR" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

echo "  ✓ $FAMILY: $n models -> $DELIVER_DIR"
if [[ "$legacy" -gt 0 ]]; then
    echo "    ($legacy of $n int8 files came from the retired onnx2tf build — they were"
    echo "     never re-converted; see reference/export_paths.json)"
fi
if [[ "$D_SHIP_EDGETPU" != "1" ]]; then
    echo "    (no edgetpu/ by design — this family's compiled artifacts are only"
    echo "     valid as part of a co-compile with the downstream detector)"
elif [[ "$edge_n" -eq 0 ]]; then
    # Edge TPU compilation is optional, so reaching here is not an error -- but it used to be SILENT:
    # nothing was copied, READ_THIS_FIRST.md still described files that were not there, and the summary
    # printed green. The absence surfaced one script later, as a manifest_deliverables.sh hash mismatch.
    echo "    [SKIPPED] edgetpu/: this family ships compiled artifacts, but none were"
    echo "     found under $EDGETPU_DIR — packaging int8 only."
    echo "     To add them: ./scripts/compile_edgetpu.sh"
    rmdir "$DELIVER_DIR/edgetpu" 2>/dev/null || true
elif [[ "$edge_n" -lt "$n" ]]; then
    echo "    (edgetpu/: $edge_n of $n models compiled — the rest have no artifact"
    echo "     under $EDGETPU_DIR)"
fi
find "$DELIVER_DIR" -type f | sed "s|$DELIVER_DIR/|    |" | sort
