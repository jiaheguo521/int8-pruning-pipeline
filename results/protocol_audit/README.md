# Protocol audit — scripts, summaries and run logs

These are the runs behind §1 and §3 of [`docs/PRUNING_HAZARDS.md`](../../docs/PRUNING_HAZARDS.md).
They lived under `outputs/` (gitignored) until 2026-08-21; a document that cites a
number should let you open the file that produced it, so the text half was promoted
here. The `.pt` checkpoints they wrote (92 MiB across `k1_recovery` and
`pareto_recovery`) stay out of git — model binaries are published to HuggingFace and
fetched with `scripts/fetch_deliverables.sh`.

Two changes were made to the promoted copies, and to nothing else:

- **`REPO` is now derived from the file's own location** instead of a hard-coded
  absolute path, so the scripts run from a clone. Override with `ETPU_REPO`.
- **`k1_run.log` line 880** had the author's absolute repo path in a `written:` line;
  the prefix was replaced with `<repo>`. No measured value was touched. The
  byte-exact original is still at `outputs/k1_recovery/k1_run.log`.

All of these were run on EfficientDet-Lite1/Lite2, COCO val2017 (5000 images),
no fine-tuning unless stated.

## What each file answers

| files | question | result |
|---|---|---|
| `k1_recovery.py` `k1_recovery_summary.json` `k1_run.log` | Does the allocation collapse survive recovery fine-tuning? | No — it recovers, and the pre-FT ordering is wrong. Lite1, backbone −10%, 15 ep minitrain: `lamp g=T` 0.1092 → **0.3027** (94.0% of baseline), `magnitude_l2 g=T` 0.0003 → **0.2043** (63.4%), `magnitude_l2 g=F` 0.0820 → **0.2952** (91.7%). |
| `pareto_recovery.py` `pareto_recovery_summary.json` `pareto_run.log` | How far does that hold as you prune deeper? | Same budget, `lamp g=T`: bb20 0.0690 → 0.2967 (92.1%), bb30 0.0086 → 0.2872 (89.2%), bb50 0.0014 → **0.2555** (79.3%); `magnitude_l2 g=F` bb50 0.0003 → 0.2201 (68.4%). |
| `k2_cap_sweep.py` `k2_cap_sweep.log` | Does `max_pruning_ratio` preserve mAP, or only min-survival? | Only min-survival. cap 0.9 / 0.5 / 0.3 / 0.1 → min survival 8.3 / 37.5 / 56.2 / 76.0%, mAP 0.0003 / **0.0000** / 0.0003 / 0.0091. Non-monotone in the cap: 0.5 is worse than 0.9. |
| `k4_lite2.py` `k4_lite2.log` | Does the collapse transfer to Lite2? (When this ran, Lite2 was the rung deployed downstream; that moved to Lite1 on 2026-08-22.) | Yes. Re-measured baseline **0.3589**; `magnitude_l2 g=T` 0.0001 (min surv 8.3%, `blocks.1.0.conv_dw`), `magnitude_l2 g=F` 0.0848 (91.7%), `lamp g=T` 0.1655 (79.6%). |
| `audit_sizes.py` | Where does the int8 `.tflite` byte budget go? | Produces [`results/tflite_size_audit.json`](../tflite_size_audit.json). No log was kept. **It cannot run from a clone** — it measures `outputs/**/*_int8.tflite`, which is gitignored — and it now refuses to write an empty table rather than silently emptying the committed one. Where it can run it merges: 11 rows whose `.tflite` was deleted after the 2026-08-18 pass are carried forward flagged `source_present: false`, and any row whose bytes moved is reported. The two hand-written top-level fields (`export_path_scope`, `coverage`) are carried too, with a reminder that a re-run may have invalidated the counts they state. |
| `ft_throughput.py` | What does one fine-tune rung actually cost? | **Writes nothing and no log was kept.** Its numbers survive only as `cost_per_criterion` in [`results/detection/full_scope_grid.json`](../detection/full_scope_grid.json). |
| `distill_vs_labelled_cost.py` `distill_vs_labelled_cost.json` | Is distillation recovery the cheap arm, as "no labels needed" suggests? | **No — it is the expensive one.** Same student, same batch, same optimizer: a distillation step costs **1.31x** a labelled step on the dense model and **2.63x** at 90% pruned. The whole difference is the frozen teacher's forward pass (91.9 / 91.2 ms), which does not shrink with the student — 61.3% of the step at 90%. What distillation buys is the absence of annotations, not GPU time. Per-step only: it says nothing about epochs to converge. |
| `k5_narrow_ignore.py` `k5_run.log` | Does narrowing `ignored_layers` preserve accuracy? | **Unanswered — the run crashed** (`AttributeError: 'list' object has no attribute 'shape'`). Kept as negative evidence: the log does show the pruned allocation reached backbone −50.3% with `fpn`, `class_net` and `box_net` all cut 0.0%. |
| `allocation_stats.py` `../detection/allocation_stats.json` | Why does global ranking gut the narrow early layers, if `normalizer='mean'` already cancels cross-layer scale? | Within-layer skew, not cross-layer scale. Produces the four statistics §1 quotes: per-group normalised mean, raw L2, the fraction of channels in the long left tail, and the coefficient of variation. Needs `torch`; no forward pass. |
| `criterion_census.py` `../criterion_census.json` | How many rungs and checkpoints exist per importance criterion, and under which prune mode? | 121 rungs, `magnitude_l2` 101 and `lamp` 20, the other three never run; 0 rungs iterative. Recounted over every worker log, so **it cannot run from a clone** — the logs are under `outputs/`, which is gitignored. That is why the JSON is committed and this is not in tier 1. |

## One number in here is not evidence for itself

`k2_cap_sweep.log` prints

```
  global=True  random    minsurv 87.5%   mAP 0.0674
```

**That line is a hard-coded string, not a measurement.** It sits in a block headed
"reference points from the audit" (`k2_cap_sweep.py:86–89`) whose three entries are all
typed literals. The script's only pruner uses `get_importance("magnitude_l2")` with
`global_pruning=True` (`:56–58`) and its loop variable is `cap`, not the criterion;
`k4_lite2.py:42` likewise has no `random` arm.

So the mAP **0.0674** quoted for `global + random` in `docs/PRUNING_HAZARDS.md` §1 and §3
came from an ad-hoc control script that was never committed. Promoting this directory
does **not** make it reproducible — it only adds a second copy of the same unsourced
literal. Treat it as on-record but unverifiable, as
[`results/detection/full_scope_grid.json`](../detection/full_scope_grid.json)
(`lamp_selection.caveat`) already does.

## Added 2026-08-22

| | |
|---|---|
| [`channel_census.py`](channel_census.py) | Per-rung Conv2d channel census and the tile-width fit behind section 2's "below 64 channels" subsection. Writes [`../detection/channel_census_lite1.json`](../detection/channel_census_lite1.json). Needs `torch` and the gitignored checkpoints; one forward pass per rung. `--check` asserts the published peak and R². |
| [`export_litert/`](export_litert/) | The scripts that produced the full-mapping detection ladder, committed as run. **Not part of the supported pipeline** — `scripts/convert.sh` is. See that directory's README for why they will rot and what to check before trusting a re-run. |

## Added 2026-08-27

| | |
|---|---|
| [`assemble_ladder_lite2.py`](assemble_ladder_lite2.py) | Builds [`../detection/full_mapping_ladder_lite2.json`](../detection/full_mapping_ladder_lite2.json) from the measured artifacts, mirroring the lite1 template field for field. Latency columns are written only where a bench JSON is present; without a Coral attached they stay null and the file's `open` block says so. |

## Added 2026-08-28

Both read tracked files under `results/` only, so unlike the audits above they run from a
clone, and both sit in tier 1 of `scripts/check.sh`.

| | |
|---|---|
| [`edgetpu_census.py`](edgetpu_census.py) | How many models and compiled artifacts have been timed on the Coral, counted over every latency file under `results/`. Writes [`../edgetpu_census.json`](../edgetpu_census.json) and fails if either README stops printing the two totals it counts, which is how the old hand-carried pair went stale twice. |
| [`pipeline_provenance.py`](pipeline_provenance.py) | Which branches of the pipeline produced each results file: ratio scope, criteria, recovery arm, prune mode, export path. Writes [`../pipeline_provenance.json`](../pipeline_provenance.json), rendered into the "Which part of the pipeline" table of [`../README.md`](../README.md). |
