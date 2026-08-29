#!/usr/bin/env bash
# manifest_deliverables.sh — regenerate results/deliverables.sha256, which is what
# fetch_deliverables.sh verifies every download against. Generated, not hand-kept: a
# hand-kept one drifts the moment deliverables/ is rebuilt. Paths in it are relative to
# deliverables/, which is what fetch_deliverables.sh joins them onto.
#
# Usage:
#   ./scripts/manifest_deliverables.sh [--check]   # --check verifies, changes nothing

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

DEST="${DELIVERABLES_DIR:-$PROJECT_DIR/deliverables}"
MANIFEST="$PROJECT_DIR/results/deliverables.sha256"
CHECK=false
[[ "${1:-}" == "--check" ]] && CHECK=true

[[ -d "$DEST" ]] || { echo "[ERROR] no deliverables tree at $DEST" >&2; exit 1; }

# deliverables/ holds more than what gets published, so the manifest is a subset of the
# tree: line_seg is pruned against a self-collected dataset, and its Re-ID and co-compile
# siblings are downstream handoffs. Both EfficientDet rungs qualify on two grounds, a public
# baseline on a public dataset and a single export path (19/19 litert-torch +
# ai-edge-quantizer, no onnx2tf leftovers). Build lite2 with `./scripts/deliver.sh effdet:lite2`.
PUBLISHED=(effdet_lite1_pruning effdet_lite2_pruning)

# deliverables/ is gitignored and absent from a fresh clone. Without this guard the script would
# write an empty manifest over the real one and exit 0.
missing=()
for pkg in "${PUBLISHED[@]}"; do
    [[ -d "$DEST/$pkg" ]] || missing+=("$pkg")
done
if [[ ${#missing[@]} -gt 0 ]]; then
    echo "[ERROR] published package(s) not in $DEST: ${missing[*]}" >&2
    echo "        deliverables/ is generated: run ./scripts/deliver.sh <family> first." >&2
    exit 1
fi

new="$(cd "$DEST" && find "${PUBLISHED[@]}" -type f | LC_ALL=C sort | xargs -d '\n' sha256sum)"
count="$(echo "$new" | wc -l)"

# The manifest only ties deliverables/ to itself. A package also carries copies of committed
# results files (deliver.reference_json), and those can go stale against results/ with the
# manifest and check.sh all green, because the manifest is regenerated FROM the stale copies.
# Only targets whose eval_dir is under results/ can be checked: line_seg and reid source
# their reference JSONs from the gitignored outputs/, so they are reported as skipped.
check_reference_copies() {
    PROJECT_DIR="$PROJECT_DIR" DEST="$DEST" python3 - <<'PYEOF'
import os, sys, glob, filecmp
try:
    import yaml
except ImportError:
    print("  [SKIP] reference/ freshness: needs pyyaml (pip install -e .)")
    sys.exit(0)

repo, dest = os.environ["PROJECT_DIR"], os.environ["DEST"]
bad, checked, skipped = [], 0, []


def targets(dl):
    """(package dir, eval_dir, reference_json) for the family and each variant."""
    yield dl.get("dir"), dl.get("eval_dir"), dl.get("reference_json") or []
    for _, v in (dl.get("variants") or {}).items():
        yield (v.get("dir") or dl.get("dir"),
               v.get("eval_dir") or dl.get("eval_dir"),
               v.get("reference_json") or dl.get("reference_json") or [])


for fam in sorted(glob.glob(os.path.join(repo, "families/*/*/family.yaml"))):
    dl = (yaml.safe_load(open(fam)) or {}).get("deliver")
    if not dl:
        continue
    for pkg, eval_dir, refs in targets(dl):
        if not pkg or not os.path.isdir(os.path.join(dest, pkg)):
            continue
        if not (eval_dir or "").startswith("results/"):
            skipped.append(f"{pkg} (eval_dir {eval_dir} is not under results/)")
            continue
        for name in refs:
            shipped = os.path.join(dest, pkg, "reference", name)
            source = os.path.join(repo, eval_dir, name)
            if not os.path.exists(shipped):
                bad.append(f"{pkg}/reference/{name} is missing from the package")
            elif not os.path.exists(source):
                bad.append(f"{eval_dir}/{name} does not exist, but {pkg} ships a copy")
            elif not filecmp.cmp(shipped, source, shallow=False):
                bad.append(f"{pkg}/reference/{name} differs from {eval_dir}/{name}")
            else:
                checked += 1

for s in skipped:
    print(f"  [skip] reference/ freshness: {s}")
if bad:
    print("[FAIL] a package ships a reference copy that no longer matches results/:",
          file=sys.stderr)
    for b in bad:
        print(f"    {b}", file=sys.stderr)
    print("    fix: ./scripts/deliver.sh <family>[:<variant>], then re-run this "
          "script without --check to refresh the manifest.", file=sys.stderr)
    sys.exit(1)
print(f"reference/ copies match results/ ({checked} files)")
PYEOF
}

if $CHECK; then
    if diff -q <(echo "$new") "$MANIFEST" >/dev/null 2>&1; then
        echo "manifest matches deliverables/ ($count files)"
    else
        echo "[FAIL] $MANIFEST is out of date with $DEST:" >&2
        diff <(cat "$MANIFEST") <(echo "$new") | head -20 >&2
        exit 1
    fi
    check_reference_copies
else
    echo "$new" > "$MANIFEST"
    echo "wrote $MANIFEST ($count files)"
fi
