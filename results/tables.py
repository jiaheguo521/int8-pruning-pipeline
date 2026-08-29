#!/usr/bin/env python3
"""Render the tables that are projections of committed JSON, in place.

Four blocks across three documents. In docs/PRUNING_HAZARDS.md: the criterion
census and the two EfficientDet recovery ladders. In both READMEs: the capabilities
table, which is what `python -m int8_pruning.manifest capabilities` prints -- the
lines come from results/capabilities.json, written by results/capabilities.py,
which is the half allowed to read YAML.

The three in PRUNING_HAZARDS were transcribed
by hand, and the transcription has drifted before -- the document's own
corrections table records `magnitude_l2` being published as 89 rungs, then 94,
before the census script settled it at 101. This script is what stops that: the
numbers live in the JSON, the prose lives in the document, and `--check` fails
if the two disagree.

    python3 results/tables.py --check   # assert every document matches (CI)
    python3 results/tables.py --write   # rewrite the tables in place
    python3 results/tables.py --print   # emit to stdout, change nothing

Deliberately stdlib-only, for the same reason as fit_latency_law.py: a check
that needs an environment built first is not much of a check. It runs in
check.sh's first tier, the one a fresh clone can pass.

WHAT IS NOT GENERATED, and why. Most tables in that document are not projections
of anything -- section 1's worst-hit layers, section 4's export-path matrix, the
Lite1/Lite2 status grid, Provenance and the corrections table are judgements, and
a script cannot hold them. Two more are numeric but still editorial: the
five-model latency table in README.md and section 2's three-model table pick rows
that straddle the 8 MiB cache, and encoding "which rows make the point" in code
is worse than leaving them written down. Those stay covered by
fit_latency_law.py's `quoted_in` cross-check instead.

The editorial parts of the PRUNING_HAZARDS tables below -- row order, the status column, the
bold emphasis, the footnote markers -- are held HERE rather than in the document,
so that a number and the sentence next to it cannot drift apart. Everything with
a digit in it comes from JSON.
"""
import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HAZARDS = REPO / "docs" / "PRUNING_HAZARDS.md"
READMES = [REPO / "README.md", REPO / "README.zh-CN.md"]
CENSUS = REPO / "results" / "criterion_census.json"
CAPS = REPO / "results" / "capabilities.json"
GRID = REPO / "results" / "detection" / "full_scope_grid.json"
PROVENANCE = REPO / "results" / "pipeline_provenance.json"
RESULTS_README = REPO / "results" / "README.md"
LADDER = {
    "lite1": REPO / "results" / "detection" / "full_mapping_ladder_lite1.json",
    "lite2": REPO / "results" / "detection" / "full_mapping_ladder_lite2.json",
}

FENCE = re.compile(r"(<!-- generated: (\w+) -->\n)(.*?)(\n<!-- /generated -->)",
                   re.DOTALL)


def _row(cells):
    return "| " + " | ".join(cells) + " |"


def _sep(aligns):
    """The separator line, in the document's own style: no padding spaces."""
    return "|" + "|".join(aligns) + "|"


# 1. Criterion census  <- results/criterion_census.json
# Row order and the status column are editorial: the two criteria with evidence first,
# then `random`, whose zero means something different from the other two -- it was
# measured, by scripts that were never committed.
CENSUS_ORDER = ["magnitude_l2", "lamp", "random", "magnitude_l1", "fpgm"]
CENSUS_STATUS = {
    "magnitude_l2": "every family except `imagenet_backbones` and `relu_clip`",
    "lamp": "detection and Re-ID only",
    "random": ("measured **only** in ad-hoc control scripts that were never "
               "committed; the numbers are on record but are **not "
               "reproducible from this repo**"),
    "magnitude_l1": "implemented, never executed",
    "fpgm": "implemented, never executed",
}


def census_table():
    rows = {r["criterion"]: r for r in json.loads(CENSUS.read_text())["table"]}
    out = [
        _row(["criterion", "rungs", "local / cluster", "sweep `.pt`",
              "fine-tuned `.pt`", "families", "status"]),
        _sep(["---", "---:", "---", "---:", "---:", "---", "---"]),
    ]
    for name in CENSUS_ORDER:
        r = rows[name]
        n = r["rungs"]
        # The families cell used to read "five of six", a count that went stale when relu_clip landed. Listing cannot.
        fams = ", ".join(f"`{f}`" for f in r["families"]) or "—"
        out.append(_row([
            f"`{name}`",
            f"**{n}**" if n else "0",
            f'{r["rungs_local"]} / {r["rungs_cluster"]}' if n else "—",
            str(r["sweep_pt"]), str(r["finetuned_pt"]),
            fams, CENSUS_STATUS[name],
        ]))
    return "\n".join(out)


# 2-3. The two EfficientDet recovery ladders  <- full_mapping_ladder_lite{1,2}
# Cells come from the full-mapping ladders, which is what section 3 cites as the evidence
# for them. The DENOMINATOR is full_scope_grid.json's `baselines` -- 0.3220 and 0.3589,
# each rounded from the ladder's own dense row (0.32193612..., 0.35883937...) -- because
# that is what the published percentages were computed against. The unrounded dense mAP
# would move about half these cells by 0.1pp. Change it only on purpose.
LADDERS = {
    # first_col: which column the reduction axis prints. Lite1 was published on the TARGET and Lite2 on
    # the REALIZED reduction; both bottom rows are saturated and carry a footnote marker.
    "lite1": {"baseline": "efficientdet_lite1", "header": "reduction",
              "axis": "target", "marker": "†",
              # The ladder's endpoint under the protocol's own criterion, the number section 3 argues from.
              "bold_cells": {(90, "magnitude_l2", "of_base")}},
    "lite2": {"baseline": "efficientdet_lite2", "header": "realized reduction",
              "axis": "realized", "marker": "‡", "bold_cells": set()},
}
# 60% is the paper's headline reduction; its whole row is bold in both tables.
HEADLINE_PCT = 60


def ladder_table(tag):
    spec = LADDERS[tag]
    base = json.loads(GRID.read_text())["baselines"][spec["baseline"]]
    rows = json.loads(LADDER[tag].read_text())["rows"]
    by = {(r["importance"], r["target_pct"]): r
          for r in rows if r["importance"] != "dense"}
    targets = sorted({t for _, t in by})

    out = [
        _row([spec["header"], "`lamp`", "of baseline", "`magnitude_l2`",
              "of baseline"]),
        _sep(["---:", "---:", "---:", "---:", "---:"]),
    ]
    for t in targets:
        last = t == targets[-1]
        head = spec["header"]  # unused, kept readable below
        if spec["axis"] == "target":
            axis = f"{t}%"
        else:
            axis = f'{by[("lamp", t)]["realized_full_pct"]:.1f}%'
        cells = [axis + (spec["marker"] if last else "")]
        for imp in ("lamp", "magnitude_l2"):
            r = by[(imp, t)]
            cells.append(f'{r["fp32_mAP"]:.4f}')
            cells.append(f'{100 * r["fp32_mAP"] / base:.1f}%')
        if t == HEADLINE_PCT:
            cells = [f"**{c}**" for c in cells]
        else:
            for (bt, bimp, _kind) in spec["bold_cells"]:
                if bt == t:
                    i = 2 if bimp == "lamp" else 4
                    cells[i] = f"**{cells[i]}**"
        out.append(_row(cells))
    del head
    return "\n".join(out)


def capabilities_table():
    """The `manifest capabilities` table, quoted verbatim in both READMEs.

    Not built here: emitted by results/capabilities.py, which is the half that
    can read YAML. This end only has to stay stdlib, and copying the stored lines
    is what makes "the README shows what the command prints" a checkable claim
    rather than a hopeful one.
    """
    return "```\n" + "\n".join(json.loads(CAPS.read_text())["rendered"]) + "\n```"


def pipeline_table():
    """Which branches of the pipeline produced each results file.

    Built from results/pipeline_provenance.json, which reads the files themselves.
    A cell reading "n/a" means the stage was never reached (a pruning-stage file has
    no export path); an em-rule means the file reached it and does not say which.
    """
    doc = json.loads(PROVENANCE.read_text())
    out = [_row(["file", "scope", "criteria", "recovery", "mode", "export path"]),
           _sep(["---", "---", "---", "---", "---", "---"])]
    for r in doc["rows"]:
        rel = r["file"][len("results/"):]
        cells = [f"[`{rel}`]({rel})", r["scope"],
                 ", ".join(f"`{c}`" for c in r["criteria"]) if r["criteria"] else "—",
                 r["recovery"] or "—",
                 r["prune_mode"] or "—",
                 ", ".join(r["export_paths"]) if r["export_paths"] else "—"]
        cells[2] = cells[2].replace("`n/a`", "n/a")
        out.append(_row(cells))
    return "\n".join(out)


TABLES = {
    "census": census_table,
    "pipeline": pipeline_table,
    "ladder_lite1": lambda: ladder_table("lite1"),
    "ladder_lite2": lambda: ladder_table("lite2"),
    "capabilities": capabilities_table,
}

# Which document owns which fences. `capabilities` is a command's output, so both languages show it.
DOCS = {
    HAZARDS: {"census", "ladder_lite1", "ladder_lite2"},
    READMES[0]: {"capabilities"},
    READMES[1]: {"capabilities"},
    RESULTS_README: {"pipeline"},
}


def render(text, doc):
    """Replace every fenced block in `text`. Returns (new_text, seen_names)."""
    seen = []

    def sub(m):
        name = m.group(2)
        if name not in TABLES:
            raise SystemExit(f"{doc}: unknown generated table {name!r} "
                             f"(known: {' '.join(sorted(TABLES))})")
        seen.append(name)
        return m.group(1) + TABLES[name]() + m.group(4)

    return FENCE.sub(sub, text), seen


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true",
                   help="fail if the document does not match the JSON")
    g.add_argument("--write", action="store_true",
                   help="rewrite the generated tables in place")
    g.add_argument("--print", dest="show", action="store_true",
                   help="print every table to stdout, change nothing")
    args = p.parse_args(argv)

    if args.show:
        for name in TABLES:
            print(f"<!-- generated: {name} -->")
            print(TABLES[name]())
            print("<!-- /generated -->\n")
        return 0

    failed, total = False, 0
    for doc, expected in DOCS.items():
        old = doc.read_text()
        new, seen = render(old, doc)
        missing = sorted(expected - set(seen))
        if missing:
            print(f"[FAIL] {doc.relative_to(REPO)} has no fence for: "
                  f"{' '.join(missing)}")
            failed = True
            continue
        total += len(seen)

        if args.write:
            if new != old:
                doc.write_text(new)
                print(f"[write] {doc.relative_to(REPO)}: {len(seen)} rendered")
            else:
                print(f"[write] {doc.relative_to(REPO)}: already up to date")
            continue

        if new != old:
            import difflib
            print(f"[FAIL] {doc.relative_to(REPO)} disagrees with results/*.json.\n"
                  f"       Re-render with: python3 results/tables.py --write\n")
            sys.stdout.writelines(difflib.unified_diff(
                old.splitlines(keepends=True), new.splitlines(keepends=True),
                fromfile="committed", tofile="from JSON", n=1))
            failed = True

    if failed:
        return 1
    if not args.write:
        print(f"[ok] {total} generated tables match their JSON sources")
    return 0


if __name__ == "__main__":
    sys.exit(main())
