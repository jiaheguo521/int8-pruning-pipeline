# Deleted artifacts

Tombstones for pipeline products that were removed **deliberately**, not lost.
Nothing in here is a model — these are records *about* models.

Each entry records what was deleted, why, the exact byte size of every file, and
the command that regenerates it. The point is that a gap in `outputs/` should be
explainable years later, and that a stray file found somewhere else can be
identified as a retired artifact rather than a missing original.

- **`2026-08-20_superseded_ladders.json`** — 603.7 MB of `line_seg` artifacts
  dropped when that family switched from a per-layer **channel** ratio to a
  **parameter** ratio (commit `161817c`). Unpruned baselines were kept: 0% has no
  scope and anchors every curve. Regenerate with
  `families/segmentation/line_seg/prune.py --ratio_semantics channel` (~84 s fine-tune per rung).

- **`2026-08-22_channel_ladder_final.json`** — 326.0 MB finishing the job the
  2026-08-20 entry above started. That pass switched `line_seg` to the parameter
  ratio but deliberately spared `deliverables/line_seg_pruning/` and its
  HuggingFace copies, because they were the round-1 handoff. The downstream
  project has since stopped consuming them, and the unit is actively misleading —
  `pruned60pct` on `link_r34` is an 84.1% parameter cut. The five channel rungs are
  now gone from the package, from `outputs/line_seg_eval/`, and from
  `results/line_seg/{results,report,latency}.json`, which no document cited.
  `families/segmentation/line_seg/prune.py` now defaults to `--ratio_semantics param`.
  Regenerate with `families/segmentation/line_seg/prune.py --ratio_semantics channel`.

- **`2026-08-22_superseded_lite1_onnx2tf.json`** — 299.8 MB of `efficientdet_lite1`
  artifacts dropped when the family moved off the onnx2tf export. That path lowered
  effdet's BiFPN fusion to a rank-5 `CONCAT`/`SUM` and timm's SAME-pool to `PADV2`,
  neither of which the Edge TPU compiler can map, so only 83 of 479 operators ran on
  the TPU. The replacement maps 305 of 305. Regeneration is the one entry in this
  directory that does **not** come with a command reachable from a clone: the current
  build is produced by `aiedge-lab/`, which lives outside this repository. See the
  not-reproducible table in [`docs/PRUNING_HAZARDS.md`](../../docs/PRUNING_HAZARDS.md).
