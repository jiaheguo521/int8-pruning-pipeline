#!/usr/bin/env python3
"""Commit what each family declares it implements, as JSON.

`families/*/family.yaml` is the source of truth, and it stays YAML: it is edited
by hand and carries multi-paragraph notes (`edgetpu_note`, `checkpoint_note`) that
JSON would turn into one unreadable escaped line. But reading YAML needs pyyaml,
and the check that guards the README block has to pass on a fresh clone with no
third-party package. So this script is the bridge, exactly as criterion_census.py
is for the census: the dependency-carrying step runs here and writes a JSON, and
the stdlib-only step (results/tables.py) renders the document from that JSON.

`rendered` holds the table LINES rather than only the fields, because the claim
the README makes is "this is what the command prints". Storing the text is what
makes that claim checkable instead of merely likely.

    python3 results/capabilities.py           # rewrite results/capabilities.json
    python3 results/capabilities.py --check   # print it, write nothing

Needs pyyaml (a core dependency of the package, so any real install has it) and
nothing else -- no torch, no outputs/ tree.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from int8_pruning.manifest import capability_rows, capability_table  # noqa: E402

OUT = REPO / "results" / "capabilities.json"


def main(check):
    rows = capability_rows()
    doc = {
        "produced_by": "results/capabilities.py",
        "what": "What each families/*/family.yaml declares its worker implements: which "
                "recovery arm, which ladder modes, how many int8 convert configs, whether "
                "scripts/deliver.sh can package it, whether that package ships Edge TPU "
                "artifacts.",
        "note": "Declared, NOT measured. A family listing `iterative` means its worker "
                "implements it, not that any committed ladder was produced with it -- see "
                "criterion_census.json for what has actually been run.",
        "rendered_by": "python -m int8_pruning.manifest capabilities",
        "rows": rows,
        "rendered": capability_table(rows),
    }
    text = json.dumps(doc, indent=1) + "\n"
    if check:
        sys.stdout.write(text)
    else:
        OUT.write_text(text)
        print(f"wrote {OUT.relative_to(REPO)} ({len(rows)} families)")
    return 0


if __name__ == "__main__":
    sys.exit(main("--check" in sys.argv))
