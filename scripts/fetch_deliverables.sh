#!/usr/bin/env bash
# fetch_deliverables.sh — download the published model artifacts from HuggingFace,
# verified against results/deliverables.sha256. Two tiers:
#   deploy (default)  models_int8_tflite/ + edgetpu/ + reference/  162 files, 354 MB
#   full              everything, adds checkpoints_pytorch/        202 files, 763 MB
#
# Usage:
#   ./scripts/fetch_deliverables.sh [--tier full] [--list] [--verify] [--force] [<family>]
# Tiers, the HF token and the checkpoints' pip pins: docs/SETUP.md section 6.
set -o pipefail

HF_REPO="${HF_REPO:-jiaheguo521/int8-pruning-pipeline-models}"
BASE_URL="https://huggingface.co/${HF_REPO}/resolve/main"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MANIFEST="$PROJECT_DIR/results/deliverables.sha256"
DEST="${DELIVERABLES_DIR:-$PROJECT_DIR/deliverables}"

[[ -f "$MANIFEST" ]] || { echo "[ERROR] missing manifest: $MANIFEST" >&2; exit 1; }

FORCE=0
VERIFY_ONLY=0
TIER=deploy
FILTERS=()

# A path belongs to the deploy tier unless it is a PyTorch checkpoint.
in_tier() {  # <relative path>
    [[ "$TIER" == "full" ]] && return 0
    [[ "$1" == */checkpoints_pytorch/* ]] && return 1
    return 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --list)
            awk '{split($2,a,"/"); print a[1]}' "$MANIFEST" | sort -u | while read -r fam; do
                n=$(grep -c "^[0-9a-f]*  $fam/" "$MANIFEST")
                d=$(grep "^[0-9a-f]*  $fam/" "$MANIFEST" | grep -vc "/checkpoints_pytorch/")
                echo "  $fam  ($n files; $d in the deploy tier)"
            done
            exit 0 ;;
        --tier)
            TIER="$2"
            [[ "$TIER" == deploy || "$TIER" == full ]] || {
                echo "[ERROR] --tier must be deploy or full (got: $TIER)" >&2; exit 2; }
            shift 2 ;;
        --force)  FORCE=1; shift ;;
        --verify) VERIFY_ONLY=1; shift ;;
        -h|--help) sed -n '3,42p' "$0"; exit 0 ;;
        -*) echo "[ERROR] unknown flag: $1" >&2; exit 2 ;;
        *)  FILTERS+=("$1"); shift ;;
    esac
done

# While the repo is private every request needs a token: $HF_TOKEN, else the file `hf auth login` writes.
TOKEN="${HF_TOKEN:-}"
if [[ -z "$TOKEN" && -r "${HF_HOME:-$HOME/.cache/huggingface}/token" ]]; then
    TOKEN="$(< "${HF_HOME:-$HOME/.cache/huggingface}/token")"
fi

fetch() {  # <url> <out>
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL ${TOKEN:+-H "Authorization: Bearer $TOKEN"} "$1" -o "$2"
    else
        wget -q ${TOKEN:+--header="Authorization: Bearer $TOKEN"} "$1" -O "$2"
    fi
}

ok=0; got=0; failed=0; skipped=0
while read -r want rel; do
    [[ -n "$rel" ]] || continue
    in_tier "$rel" || { skipped=$((skipped+1)); continue; }
    if [[ ${#FILTERS[@]} -gt 0 ]]; then
        match=0
        for f in "${FILTERS[@]}"; do [[ "$rel" == "$f"/* ]] && match=1; done
        [[ $match -eq 1 ]] || { skipped=$((skipped+1)); continue; }
    fi

    out="$DEST/$rel"
    if [[ -f "$out" && $FORCE -eq 0 ]]; then
        have="$(sha256sum "$out" | cut -d' ' -f1)"
        if [[ "$have" == "$want" ]]; then ok=$((ok+1)); continue; fi
        echo "  [stale] $rel"
    fi
    if [[ $VERIFY_ONLY -eq 1 ]]; then
        echo "  [MISSING] $rel"; failed=$((failed+1)); continue
    fi

    mkdir -p "$(dirname "$out")"
    echo "  [get] $rel"
    if ! fetch "$BASE_URL/$rel" "$out"; then
        echo "  [FAIL] download: $rel" >&2; rm -f "$out"; failed=$((failed+1)); continue
    fi
    have="$(sha256sum "$out" | cut -d' ' -f1)"
    if [[ "$have" != "$want" ]]; then
        echo "  [FAIL] sha256 mismatch: $rel" >&2
        echo "         want $want" >&2
        echo "         got  $have" >&2
        rm -f "$out"; failed=$((failed+1)); continue
    fi
    got=$((got+1))
done < "$MANIFEST"

echo
if [[ $VERIFY_ONLY -eq 1 ]]; then
    echo "verify [$TIER]: $ok present and valid, $failed missing/corrupt$([[ $skipped -gt 0 ]] && echo ", $skipped not in tier/filter")"
else
    echo "done [$TIER]: $got downloaded, $ok already valid, $failed failed$([[ $skipped -gt 0 ]] && echo ", $skipped not in tier/filter")"
    if [[ $failed -gt 0 && -z "$TOKEN" ]]; then
        echo "hint: no token found. If $HF_REPO is still private, run \`hf auth login\`" >&2
        echo "      or set HF_TOKEN=... — without one every request returns HTTP 401." >&2
    fi
fi
[[ $failed -eq 0 ]]
