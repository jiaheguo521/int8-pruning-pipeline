#!/usr/bin/env bash
# check_docs.sh — the mechanical half of a README review: only what a script can decide
# without an opinion. The judgement half (reader voice, jargon, over-argued paragraphs)
# needs a reader. Standard library only, so it runs in check.sh's clean-clone tier.
#
# Usage:
#   ./scripts/check_docs.sh [--strict]   # --strict makes warnings fail too

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

STRICT=0
[[ "${1:-}" == "--strict" ]] && STRICT=1

STRICT=$STRICT python3 - <<'PY'
import json, os, re, sys

EN, ZH = "README.md", "README.zh-CN.md"
STRICT = os.environ.get("STRICT") == "1"
hard, soft = [], []


def read(p):
    return open(p, encoding="utf-8").read() if os.path.exists(p) else None


docs = {p: read(p) for p in (EN, ZH)}
for p, s in docs.items():
    if s is None:
        hard.append(f"{p} is missing")
docs = {p: s for p, s in docs.items() if s}
if not docs:
    print("[FAIL] no README to check", file=sys.stderr)
    sys.exit(1)


# ---- 1. relative links resolve ---------------------------------------------
for p, s in docs.items():
    for m in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", s):
        target = m.group(1)
        if target.startswith(("http", "#", "mailto")):
            continue
        path = target.split("#")[0]
        if path and not os.path.exists(path):
            hard.append(f"{p}: broken link -> {target}")


# ---- 1b. every link survives the public mirror -----------------------------
# Links above are checked against the LOCAL tree, but readers see the public mirror, a whitelist
# subset: a link to a synced-out path resolves here and 404s there.
SYNC = "sync_to_public.sh"
if os.path.exists(SYNC):
    src = open(SYNC, encoding="utf-8").read()
    block = re.search(r"^MAPPINGS=\((.*?)^\)", src, re.S | re.M)
    excl = re.search(r"^RSYNC_EXCLUDES=\((.*?)\)", src, re.S | re.M)
    if block:
        mapped = [m.group(1) for m in
                  re.finditer(r'^\s*"([^":]+)::', block.group(1), re.M)]
        drops = re.findall(r"--exclude=(\S+?)[\s)]", excl.group(1)) if excl else []
        drops = [d.rstrip("/") for d in drops if not d.startswith("*") and "__" not in d]

        def public(path):
            if not any(path == m.rstrip("/") or path.startswith(m)
                       for m in mapped):
                return "not in MAPPINGS"
            for d in drops:
                if f"/{d}/" in f"/{path}/":
                    return f"excluded by --exclude={d}"
            return None

        for p, s in docs.items():
            for m in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", s):
                t = m.group(1)
                if t.startswith(("http", "#", "mailto")):
                    continue
                path = t.split("#")[0].rstrip("/")
                if not path:
                    continue
                why = public(path)
                if why:
                    hard.append(f"{p}: link dies in the public mirror ({why}) -> {t}")


# ---- 2. images exist, and one appears above the fold -----------------------
# A repository a recruiter may open has about one screen; the strongest thing on it is a figure.
FOLD = 40
for p, s in docs.items():
    imgs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", s)
    for src in imgs:
        if not src.startswith("http") and not os.path.exists(src):
            hard.append(f"{p}: image not on disk -> {src}")
    head = "\n".join(s.splitlines()[:FOLD])
    if not re.search(r"!\[[^\]]*\]\(", head) and "```mermaid" not in head:
        soft.append(f"{p}: no figure or diagram in the first {FOLD} lines")
    for src in imgs:
        # Badges carry their meaning in the rendered pill, not the alt text.
        if "shields.io" in src or "badge.svg" in src:
            continue
        alt = next(m.group(1) for m in re.finditer(r"!\[([^\]]*)\]\(" + re.escape(src), s))
        if len(alt.strip()) < 12:
            soft.append(f"{p}: figure has a thin alt text -> {src!r} alt={alt!r}")


# ---- 3. mermaid is parseable and small enough to read ----------------------
RESERVED = {"end", "graph", "subgraph", "class", "style", "click", "o", "x"}
MAX_NODES = 8
for p, s in docs.items():
    for i, block in enumerate(re.findall(r"```mermaid\n(.*?)```", s, re.S), 1):
        ids = set(re.findall(r"^\s*([A-Za-z][A-Za-z0-9_]*)\[", block, re.M))
        used = set(re.findall(r"([A-Za-z][A-Za-z0-9_]*)\s*-\.?->", block)) | set(
            re.findall(r"-\.?->(?:\|[^|]*\|)?\s*([A-Za-z][A-Za-z0-9_]*)", block))
        tag = f"{p}: mermaid #{i}"
        if block.count('"') % 2:
            hard.append(f"{tag}: unbalanced quotes")
        for n in sorted(used - ids):
            hard.append(f"{tag}: edge references undeclared node {n}")
        for n in sorted(ids - used):
            soft.append(f"{tag}: node {n} is declared but never connected")
        for n in sorted(i for i in ids if i.lower() in RESERVED):
            hard.append(f"{tag}: {n} is a mermaid reserved word")
        styled = set()
        for c in re.findall(r"^\s*class\s+([A-Za-z0-9_,]+)\s", block, re.M):
            styled |= set(c.split(","))
        for n in sorted(styled - ids):
            hard.append(f"{tag}: class applied to undeclared node {n}")
        if len(ids) > MAX_NODES:
            soft.append(f"{tag}: {len(ids)} nodes — over {MAX_NODES}, likely unreadable")


# ---- 4. the two languages stay structurally parallel -----------------------
if EN in docs and ZH in docs:
    def shape(s):
        return {
            "h2": len(re.findall(r"^##[^#]", s, re.M)),
            "h3": len(re.findall(r"^###[^#]", s, re.M)),
            "images": len(re.findall(r"!\[[^\]]*\]\(", s)),
            "mermaid": len(re.findall(r"```mermaid", s)),
            "tables": len(re.findall(r"^\|", s, re.M)),
        }
    a, b = shape(docs[EN]), shape(docs[ZH])
    for k in a:
        if a[k] != b[k]:
            (hard if k in ("h2", "images", "mermaid") else soft).append(
                f"en/zh differ in {k}: {a[k]} vs {b[k]}")


# ---- 5. the zh document keeps one punctuation convention ------------------
if ZH in docs:
    for n, line in enumerate(docs[ZH].splitlines(), 1):
        if line.lstrip().startswith(("|", ">", "    ", "```")):
            continue
        if re.search(r"[一-龥][,;:!?]|[,;:!?][一-龥]", line):
            soft.append(f"{ZH}:{n}: half-width punctuation inside Chinese text")


# ---- 6. every number in the prose should exist under results/ -------------
# Substring matching, deliberately: prose rounds (2.47 against a stored 2.4657), so an
# exact token would flag every rounded figure. The cost is that a wrong number can hide
# inside a longer right one (99.87 inside 99.8733), so this warns, it does not prove.
corpus = []
for root, _, files in os.walk("results"):
    if "_deleted" in root:
        continue
    for f in files:
        if f.endswith((".json", ".md", ".py", ".sha256")):
            try:
                corpus.append(open(os.path.join(root, f), encoding="utf-8",
                                   errors="ignore").read())
            except OSError:
                pass
corpus = "\n".join(corpus)

SKIP = re.compile(r"^(19|20)\d\d$")          # years
untraceable = set()
for p, s in docs.items():
    prose = re.sub(r"```.*?```", "", s, flags=re.S)      # not code blocks
    prose = re.sub(r"\[[^\]]*\]\([^)]*\)", "", prose)    # not link targets
    for tok in re.findall(r"\d+\.\d{2,}", prose):        # 2+ decimals only
        if SKIP.match(tok):
            continue
        if tok not in corpus and tok.rstrip("0").rstrip(".") not in corpus:
            untraceable.add(f"{p}: {tok}")
for t in sorted(untraceable):
    soft.append(f"not literally under results/ (derived, or transcribed by hand?) -> {t}")


# ---- 7. the hand-carried scale integers still match their sources ---------
# Check 6 matches `\d+\.\d{2,}` only, so every integer in the prose is unchecked. Each
# entry below recomputes a count from its source file and requires the document to still
# print it, in the phrase it prints it in. 89 models / 131 artifacts are guarded the same
# way by results/protocol_audit/edgetpu_census.py, which owns the count it asserts.
HAZ = "docs/PRUNING_HAZARDS.md"


def _load(path):
    try:
        return json.loads(read(path) or "")
    except Exception:
        return None


census = _load("results/criterion_census.json")
audit = _load("results/tflite_size_audit.json")
caps = _load("results/capabilities.json")

scale = []
if census:
    scale.append(("rungs, from criterion_census.rungs_total", census["rungs_total"],
                  {EN: r"\*\*{n}\*\* of them here", ZH: r"这里有 \*\*{n}\*\* 档"}))
    modes = census.get("prune_modes") or {}
    if "rungs_iterative" in modes:
        scale.append(("iterative rungs, from criterion_census.prune_modes",
                      modes["rungs_iterative"],
                      {EN: r"\*\*{n} of the %d rungs\*\*" % census["rungs_total"],
                       ZH: r"\*\*%d 档里 {n} 档\*\*" % census["rungs_total"]}))
if caps:
    scale.append(("families, from capabilities.rows", len(caps["rows"]),
                  {EN: r"across \*\*{n}\*\* families", ZH: r"横跨 \*\*{n}\*\* 个家族"}))
if audit:
    rows = audit["models"]
    scale.append(("int8 graphs, from tflite_size_audit.models", len(rows),
                  {EN: r"\*\*{n}\*\* int8 graphs", ZH: r"\*\*{n}\*\* 张 int8 图",
                   HAZ: r"across the {n} audited"}))
    per = {}
    for r in rows:
        per[r.get("export_path")] = per.get(r.get("export_path"), 0) + 1
    if len(set(per.values())) == 1:
        scale.append(("audit rows per export path, from tflite_size_audit.models",
                      next(iter(per.values())), {HAZ: r"{n} rows on each path"}))

# Stated in six places, two of them code comments. The MB figure beside it is NOT guarded.
manifest = read("results/deliverables.sha256")
if manifest:
    scale.append(("published files, from deliverables.sha256 line count",
                  len([l for l in manifest.splitlines() if l.strip()]),
                  {EN: r"{n} files and \d+ MB together",
                   ZH: r"合计 {n} 件、\d+ MB",
                   "results/README.md": r"the {n} hashes in",
                   "scripts/fetch_deliverables.sh": r"{n} files, \d+ MB",
                   "src/int8_pruning/prune/ladder.py": r"all {n} hashes",
                   "tests/test_ladder.py": r"all {n} hashes"}))

for label, value, wanted in scale:
    for path, pattern in wanted.items():
        text = docs.get(path) if path in docs else read(path)
        if text is None:
            continue
        if not re.search(pattern.format(n=value), text):
            hard.append(f"{path}: stale or missing count -- {label} is now {value}, "
                        f"and no {pattern.format(n=value)!r} in the document")


# ---- 8. every "docs/SETUP.md section N" pointer resolves --------------------
# The driver headers stopped being self-contained when their env-var tables moved into
# SETUP; each one now ends in a pointer, and a renumbered section breaks all seven at once
# with nothing else noticing. Section headings are `## N. ...`.
setup = read("docs/SETUP.md") or ""
sections = set(re.findall(r"^## (\d+)\.", setup, re.M))
for root in ("scripts", "families", "results"):
    for dirpath, _, names in os.walk(root):
        for n in sorted(names):
            if not n.endswith(".sh"):
                continue
            src = read(os.path.join(dirpath, n)) or ""
            for want in re.findall(r"docs/SETUP\.md section (\d+)", src):
                if want not in sections:
                    hard.append(f"{os.path.join(dirpath, n)}: points at docs/SETUP.md "
                                f"section {want}, which does not exist")


# ---- 9. every driver in scripts/ is documented somewhere -------------------
# After check 8's move, a driver's own header no longer explains it, so a script named in
# no document has no usage anywhere. SETUP.md carries the pipeline; check.sh and deliver.sh
# are in README.md instead, so either counts. The zh half is a warning, not a failure:
# a missing translation is a parity defect, not an undocumented script.
docs_en = " ".join(filter(None, (read("docs/SETUP.md"), read(EN))))
docs_zh = " ".join(filter(None, (read("docs/SETUP.zh-CN.md"), read(ZH))))
for n in sorted(os.listdir("scripts")) if os.path.isdir("scripts") else []:
    if not n.endswith(".sh"):
        continue
    if n not in docs_en:
        hard.append(f"scripts/{n} is named in no document: add it to docs/SETUP.md "
                    f"(pipeline) or {EN} (tooling)")
    elif n not in docs_zh:
        soft.append(f"scripts/{n} is documented in English only")


# ---- report ---------------------------------------------------------------
for m in hard:
    print(f"  [FAIL] {m}")
for m in soft:
    print(f"  [warn] {m}")
print(f"\n{len(hard)} failures, {len(soft)} warnings"
      f"{' (strict: warnings fail)' if STRICT else ''}")
sys.exit(1 if hard or (STRICT and soft) else 0)
PY
