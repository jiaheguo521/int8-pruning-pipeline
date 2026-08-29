"""Single source of truth for per-family facts: `families/<name>/family.yaml`.

Family knowledge used to live in five hand-edited registries that nothing kept
in sync — `MODEL_REGISTRY` here in the converter, `FAMILY_NORMS` in the data
layer, and three `case` blocks in `scripts/pruning.sh`. They had already
drifted: the converter was missing `clip_rn50` entirely while the README
documented `--model-family clip_rn50`, so the published CLIP numbers were not
reproducible from a clean clone.

Adding a family is now: create `families/<name>/family.yaml`, drop in a
`prune.py` / `eval.py`, done.

Calibration loaders stay Python (they read datasets); the YAML names one and
this module resolves it. That keeps the manifest declarative without inventing
a config language.
"""

import shlex
from pathlib import Path

import yaml

FAMILIES_DIR = Path(__file__).resolve().parents[2] / "families"


def family_yaml(name):
    """Repo-relative path of one family's family.yaml.

    Built by searching, not by formatting `families/<name>/family.yaml`: the
    directories sit under a task bucket now, and an error message that names a
    path the reader cannot open is worse than no path at all.
    """
    hits = sorted(FAMILIES_DIR.glob(f"*/{name}/family.yaml"))
    if not hits:
        return f"families/*/{name}/family.yaml"
    return str(hits[0].relative_to(FAMILIES_DIR.parent))


def _load_docs():
    """Every families/<task>/<name>/family.yaml, keyed by family directory name.

    The task bucket is a filing convention, not an identifier: nothing keys off it,
    and a family keeps its name if it is refiled. That is why the glob reaches one
    level deeper rather than the key gaining a prefix.
    """
    docs = {}
    for p in sorted(FAMILIES_DIR.glob("*/*/family.yaml")):
        with open(p) as fh:
            doc = yaml.safe_load(fh) or {}
        docs[p.parent.name] = doc
    return docs


def load(family_name):
    """One family's manifest. Raises KeyError with the known names if absent."""
    docs = _load_docs()
    if family_name not in docs:
        raise KeyError(
            f"no families/{family_name}/family.yaml (known: {sorted(docs)})")
    return docs[family_name]


def convert_entries():
    """Flat list of (family_dir, entry-dict) for every declared convert target."""
    out = []
    for fam, doc in _load_docs().items():
        for entry in doc.get("convert", []):
            out.append((fam, entry))
    return out


def build_model_registry(model_family_cls, calib_loaders):
    """Materialise the converter's registry from the manifests.

    `calib_loaders` maps the YAML's `calib_loader` string to the function.
    Passed in rather than imported so this module stays free of torch/PIL.
    """
    registry = []
    for fam, e in convert_entries():
        loader_name = e["calib_loader"]
        if loader_name not in calib_loaders:
            raise KeyError(
                f"{family_yaml(fam)} names calib_loader={loader_name!r}, "
                f"which is not defined (known: {sorted(calib_loaders)})")
        kwargs = {k: v for k, v in e.items() if k != "calib_loader"}
        registry.append(model_family_cls(calib_loader=calib_loaders[loader_name],
                                         **kwargs))
    return registry


def family_norms():
    """{convert-entry name: (mean, std)} for every declared entry.

    Replaces the hand-maintained `FAMILY_NORMS` table, which held only the two
    families whose normalization differs from ImageNet and had to be kept in
    sync with the converter by hand.
    """
    return {e["name"]: (tuple(e["mean"]), tuple(e["std"]))
            for _, e in convert_entries()}


def prune_target(model):
    """The `prune` section owning `model`, plus its family name.

    Replaces the CLS_MODEL / DET_MODEL allowlists in scripts/pruning.sh, which
    listed every family by hand in three separate `case` blocks.
    """
    for fam, doc in _load_docs().items():
        sec = doc.get("prune") or {}
        if model in sec.get("models", []):
            return fam, sec
    return None, None


def capability_rows(docs=None):
    """What each family.yaml declares its worker implements, one dict per family.

    Structured rather than printed, because results/capabilities.py commits it as
    JSON and results/tables.py renders the README block from that -- the same
    two-stage shape criterion_census.py uses, and for the same reason: the cheap
    check has to run on a clone with no third-party package, and yaml is one.
    """
    docs = _load_docs() if docs is None else docs
    rows = []
    for fam, doc in sorted(docs.items()):
        pr = doc.get("prune") or {}
        dl = doc.get("deliver") or {}
        rows.append({
            "family": fam,
            "recovery": pr.get("recovery", "?"),
            "prune_modes": pr.get("prune_modes") or ["independent"],
            "convert_configs": len(doc.get("convert") or []),
            "deliver": bool(dl),
            "ship_edgetpu": bool(dl.get("ship_edgetpu")),
        })
    return rows


def capability_table(rows):
    """`capabilities` output, as lines. Both READMEs quote this VERBATIM, so a
    column width changed here moves text in two documents; results/tables.py
    --check is what notices."""
    lines = [f"{'family':<19} {'recovery':<9} {'prune modes':<24} "
             f"{'int8':<6} {'deliver':<8} edgetpu",
             "-" * 82]
    for r in rows:
        n = r["convert_configs"]
        lines.append(f"{r['family']:<19} {r['recovery']:<9} "
                     f"{' '.join(r['prune_modes']):<24} "
                     f"{(str(n) + ' cfg') if n else '--':<6} "
                     f"{('yes' if r['deliver'] else '--'):<8} "
                     f"{'yes' if r['ship_edgetpu'] else '--'}")
    return lines


def _cli(argv=None):
    """Tiny query interface for the shell drivers.

    `pruning.sh` shells out to this rather than carrying its own copy of the
    family list:

        python -m int8_pruning.manifest models              # every prunable model
        python -m int8_pruning.manifest worker <model>      # worker script path
        python -m int8_pruning.manifest check <model> <ds>  # exit 0 if the pair is valid
        python -m int8_pruning.manifest capabilities        # which legs each family implements
        python -m int8_pruning.manifest capabilities <fam>  # ... for one family
        python -m int8_pruning.manifest prune-mode <model> <mode>   # exit 0 if implemented
    """
    import sys
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(_cli.__doc__); return 2
    cmd = argv[0]

    if cmd == "capabilities":
        # Which legs of the pipeline a family implements. Half is derived from the manifest's own shape
        # (a `convert:` block IS the int8 leg, a `deliver:` block IS the packaging leg) and half is
        # declared under `prune:`, because it cannot be read off the YAML. See families/README.md.
        docs = _load_docs()
        wanted = argv[1] if len(argv) > 1 else None
        if wanted is not None and wanted not in docs:
            print(f"[ERROR] no family.yaml for {wanted!r} "
                  f"(known: {' '.join(sorted(docs))})", file=sys.stderr)
            return 1
        rows = capability_rows(docs if wanted is None
                               else {wanted: docs[wanted]})
        print("\n".join(capability_table(rows)))
        print()
        print("recovery     labelled = fine-tune on the task's own labels; "
              "distill = 1-cos to a frozen")
        print("             copy of the UNPRUNED model (no labels needed)")
        print("prune modes  independent = every ratio pruned from the dense baseline on its "
              "own;")
        print("             iterative = one warm-started trajectory (PRUNE_MODE=iterative)")
        print("int8         number of `convert:` configs -- '--' means this family has no "
              "int8 leg")
        print("deliver      has a `deliver:` section, so scripts/deliver.sh can package it")
        print("edgetpu      that package ships compiled Edge TPU artifacts")
        return 0

    if cmd == "prune-mode" and len(argv) >= 3:
        # Gate for scripts/pruning.sh: refuse an unimplemented mode instead of dropping the flag.
        model, mode = argv[1], argv[2]
        fam, sec = prune_target(model)
        if sec is None:
            print(f"[ERROR] unknown model {model!r}", file=sys.stderr)
            return 1
        modes = sec.get("prune_modes") or ["independent"]
        if mode not in modes:
            print(f"[ERROR] family {fam} does not implement prune_mode {mode!r} "
                  f"(implements: {' '.join(modes)}).", file=sys.stderr)
            print(f"        Implement it in {sec['worker']} first -- see the "
                  f"run_iterative in families/classification/clip_rn50/prune.py for the shape, and "
                  f"int8_pruning.prune.ladder for the contracts -- then add {mode!r} to "
                  f"prune_modes in {family_yaml(fam)}.", file=sys.stderr)
            return 1
        return 0

    if cmd == "models":
        for fam, doc in sorted(_load_docs().items()):
            for m in (doc.get("prune") or {}).get("models", []):
                print(m)
        return 0

    if cmd == "deliver" and len(argv) == 1:
        for f, d in sorted(_load_docs().items()):
            if d.get("deliver"):
                print(f)
        return 0

    if cmd == "deliver" and len(argv) >= 2:
        # Shell-eval'able assignments for scripts/deliver.sh. `<family>:<variant>` selects a
        # `deliver.variants.<variant>` override merged over the section defaults: effdet holds two
        # shippable rungs (Lite1 384 / Lite2 448) differing only in dir, stem_glob and a note or two.
        fam, _, variant = argv[1].partition(":")
        doc = _load_docs().get(fam)
        sec = (doc or {}).get("deliver")
        if sec is None:
            have = [f for f, d in _load_docs().items() if d.get("deliver")]
            print(f"[ERROR] {family_yaml(fam)} has no `deliver` section "
                  f"(families that do: {' '.join(sorted(have))})", file=sys.stderr)
            return 1
        if variant:
            vs = sec.get("variants") or {}
            if variant not in vs:
                print(f"[ERROR] {family_yaml(fam)} deliver has no variant "
                      f"'{variant}' (has: {' '.join(sorted(vs)) or 'none'})", file=sys.stderr)
                return 1
            sec = {**sec, **vs[variant]}
        sec = {k: v for k, v in sec.items() if k != "variants"}
        print(f'D_DIR="{sec["dir"]}"')
        print(f'D_GLOB="{sec["stem_glob"]}"')
        print(f'D_REF_MODULE="{sec.get("reference_module", "")}"')
        print(f'D_EVAL_DIR="{sec.get("eval_dir", "")}"')
        print(f'D_REF_JSON="{" ".join(sec.get("reference_json", []))}"')
        print(f'D_SHIP_EDGETPU={1 if sec.get("ship_edgetpu") else 0}')
        print(f'D_STDOUT_GREP="{sec.get("compiler_stdout_grep", "")}"')
        # Prose, so shell-quote it: the naive "{...}" interpolation above breaks on the first apostrophe.
        print(f'D_EDGETPU_NOTE={shlex.quote(sec.get("edgetpu_note", ""))}')
        print(f'D_CKPT_NOTE={shlex.quote(sec.get("checkpoint_note", ""))}')
        return 0

    if cmd == "det" and len(argv) >= 2:
        # Shell-eval'able assignments for the detection dispatch.
        fam, sec = prune_target(argv[1])
        if sec is None or sec.get("part") != "B":
            det = [m for _, d in _load_docs().items()
                   for m in ((d.get("prune") or {}).get("models", [])
                             if (d.get("prune") or {}).get("part") == "B" else [])]
            print(f"[ERROR] DET_MODEL must be one of: {' '.join(sorted(det))} "
                  f"(got: {argv[1]})", file=sys.stderr)
            return 1
        # A family can hold several rungs of one architecture (effdet Lite1 384 / Lite2 448);
        # `model_params` overrides the section defaults per model so they need not have a dir each.
        over = (sec.get("model_params") or {}).get(argv[1], {})
        baseline = over.get("baseline", sec["baseline"])
        print(f'DET_WORKER="{sec["worker"]}"')
        print(f'DET_IMAGE_SIZE={over.get("image_size", sec["image_size"])}')
        print(f'DET_NUM_CLASSES={over.get("num_classes", sec["num_classes"])}')
        print(f'DET_BASELINE_FILE="{baseline}"')
        print(f'DET_BASELINE_NAME="{baseline}"')
        return 0

    if cmd in ("worker", "check") and len(argv) >= 2:
        model = argv[1]
        fam, sec = prune_target(model)
        if sec is None:
            known = [m for _, d in _load_docs().items()
                     for m in (d.get("prune") or {}).get("models", [])]
            print(f"[ERROR] unknown model {model!r}. Known: {' '.join(sorted(known))}",
                  file=sys.stderr)
            return 1
        if cmd == "worker":
            print(sec["worker"]); return 0

        dataset = argv[2] if len(argv) > 2 else None
        allowed = sec.get("datasets", [])
        if dataset not in allowed:
            print(f"[ERROR] {model} is {'/'.join(allowed)}-only"
                  f"{'; ' + sec['note'] if sec.get('note') else ''} "
                  f"(got dataset={dataset})", file=sys.stderr)
            return 1
        only = (sec.get("dataset_restrictions") or {}).get(dataset)
        if only and model not in only:
            print(f"[ERROR] dataset={dataset} only has a {'/'.join(only)} baseline "
                  f"(got model={model})", file=sys.stderr)
            return 1
        return 0

    print(f"[ERROR] bad usage: {' '.join(argv)}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
