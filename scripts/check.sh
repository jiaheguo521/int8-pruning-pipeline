#!/usr/bin/env bash
# check.sh — re-derive the numbers this repository publishes, enforcing results/README.md's
# rule: a number in a document is reachable in one click and re-derivable in one command.
#
# Usage:
#   ./scripts/check.sh                 # every tier this environment supports
#   ./scripts/check.sh --clean-clone   # the CI tier: stdlib only, no venv, no outputs/
# A skip is never a pass: the other tiers skip loudly when a prerequisite is missing.
# Several checks regenerate a tracked .json and must write it back byte-identically;
# the final gate asserts that.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

CLEAN_CLONE=false
[[ "${1:-}" == "--clean-clone" ]] && CLEAN_CLONE=true

PY="${PYTHON:-python3}"
pass=0; fail=0; skip=0

have() { "$PY" -c "import $1" >/dev/null 2>&1; }

# Content of every tracked file the checks could rewrite, as of right now.
snapshot() { git ls-files -z -- results/ docs/ 2>/dev/null | xargs -0 -r sha1sum 2>/dev/null | sort; }
SNAPSHOT_BEFORE=""
git rev-parse --is-inside-work-tree >/dev/null 2>&1 && SNAPSHOT_BEFORE="$(snapshot)"

run() {  # run <label> <cmd...>
    local label="$1"; shift
    printf '\n\033[1m>>> %s\033[0m\n    $ %s\n' "$label" "$*"
    if "$@"; then
        echo "    [PASS] $label"; pass=$((pass+1))
    else
        echo "    [FAIL] $label"; fail=$((fail+1))
    fi
}

skip_it() {
    printf '\n>>> %s\n    [SKIP] %s\n' "$1" "$2"; skip=$((skip+1))
}

# ---- tier 1: no third-party package, no outputs/ tree -----------------------
# int8_pruning.prune.ladder declares three contracts it cannot enforce from inside: get one wrong
# and the run finishes and reports numbers. These are that enforcement.
run "unit tests" \
    "$PY" -m unittest discover -s tests
run "latency law reproduces PRUNING_HAZARDS section 2" \
    "$PY" results/latency_law/fit_latency_law.py --check
# Recounts every measured Coral latency under results/; fails if a README stops printing its total.
run "Edge TPU census matches the READMEs" \
    "$PY" results/protocol_audit/edgetpu_census.py
# Which pipeline branches produced each results file. Must run before the generated-tables check.
run "pipeline provenance per results file" \
    "$PY" results/protocol_audit/pipeline_provenance.py
run "figure inputs are complete" \
    "$PY" results/figures/make_figures.py --check
# Projections of results/*.json, rendered and compared against what the documents ship.
run "generated tables match their JSON sources" \
    "$PY" results/tables.py --check
# Doc structure, not doc prose: links, images, mermaid, en/zh parity, numbers. The prose half needs a reader.
run "README structure" ./scripts/check_docs.sh

if $CLEAN_CLONE; then
    printf '\n--clean-clone: stopping here.\n'
else
    # ---- tier 1b: needs pyyaml, nothing else --------------------------------
    # Re-derives results/capabilities.json from family.yaml; the gate below fails if a byte moved.
    if have yaml; then
        run "capabilities JSON matches families/*/family.yaml" \
            "$PY" results/capabilities.py
    else
        skip_it "capabilities JSON" "needs pyyaml (pip install -e .)"
    fi

    # ---- tier 2: needs the pipeline's outputs/ tree -------------------------
    # criterion_census.py has no guard against an empty outputs/ and would write a degraded census.
    if [ -d outputs ] && [ -n "$(find outputs -name '*.pt' -print -quit 2>/dev/null)" ]; then
        run "criterion census" "$PY" results/protocol_audit/criterion_census.py
    else
        skip_it "criterion census" "needs a populated outputs/ tree (gitignored; run the pipeline first)"
    fi

    # ---- tier 2b: needs the generated deliverables/ tree --------------------
    # The manifest still describes the packages on disk, and their copies of committed results files
    # have not gone stale. Why the second needs its own check is in manifest_deliverables.sh.
    if [ -d deliverables ]; then
        run "deliverables manifest and reference copies" \
            ./scripts/manifest_deliverables.sh --check
    else
        skip_it "deliverables manifest" "needs a built deliverables/ tree (gitignored; run ./scripts/deliver.sh <family>)"
    fi

    # ---- tier 3: needs torch ------------------------------------------------
    if have torch; then
        run "channel census (lite1)" "$PY" results/protocol_audit/channel_census.py --check
        run "channel census (lite2)" \
            "$PY" results/protocol_audit/channel_census.py --model lite2 --check
        run "allocation stats" "$PY" results/protocol_audit/allocation_stats.py
    else
        skip_it "channel census (both ladders) / allocation stats" "needs torch (pip install -e .)"
    fi

    # Redrawing the figures is a build step: matplotlib renders different bytes on a different version.
fi

# ---- gate: a check that rewrote a tracked file must have written it back ----
# A byte change in a regenerated .json means a published number moved. The snapshot is taken before
# the run, so unrelated work in progress is not reported.
if [ -n "${SNAPSHOT_BEFORE:-}" ]; then
    changed="$(comm -13 <(echo "$SNAPSHOT_BEFORE") <(snapshot) | awk '{print $2}')"
    if [ -n "$changed" ]; then
        printf '\n\033[1m[FAIL]\033[0m these tracked files changed WHILE re-deriving them:\n'
        echo "$changed" | sed 's/^/    /'
        echo "    A regenerated file that no longer matches the committed one means a"
        echo "    published number moved. Read the diff before committing it."
        fail=$((fail+1))
    fi
fi

printf '\n%s\n' "$(printf '=%.0s' {1..64})"
printf 'passed %d   failed %d   skipped %d\n' "$pass" "$fail" "$skip"
[ "$fail" -eq 0 ] || exit 1
