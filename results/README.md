# Results

Every measurement this repository publishes, plus the code that produced it.

The rule is that **a number in a document must be reachable in one click, and
re-derivable in one command**. That is why `.py` files live here next to `.json`
rather than in `src/` or `families/`: these are one-shot analysis and audit
scripts, not library code and not pipeline stages. Nothing here is imported.

```
clip_rn50/         open-vocabulary zero-shot (+ what per-channel int8 rescues)
detection/         effdet + ssdlite (COCO mAP, Coral latency, the pruning matrix)
line_seg/          lane segmentation (the 30-model latency panel lives here)
reid/              person Re-ID (the only on-device measurements this family has)
protocol_audit/    the audit runs behind PRUNING_HAZARDS sections 1-3: scripts,
                   summaries and captured stdout, plus export_litert/ (the detection
                   export, committed as run and NOT a supported pipeline)
latency_law/       the weight-transfer latency fit and the script that refits it
figures/           the README figures drawn from the measurements here, and the
                   script that draws them
dsd2026_reference/ the DSD 2026 ResNet-50 appendix table, for the cross-check
capabilities.json       what each family DECLARES it implements (recovery arm,
                        ladder modes, int8 configs, delivery) -- both READMEs
                        quote its `rendered` lines
criterion_census.json   what each importance criterion has actually been run on
edgetpu_census.json     what has actually been timed on the Coral, and under which
                        export path -- both READMEs quote its two totals
pipeline_provenance.json  which branches of the pipeline produced each file here;
                        rendered into "Which part of the pipeline" below
tflite_size_audit.json  where the bytes of an int8 .tflite actually go
deliverables.sha256     integrity manifest for the binaries published to HuggingFace
```

A `produced_by` or `note` in these files points at the script or family file behind the
number. Those pointers are kept resolvable: when `families/` was refiled under task buckets
on 2026-08-28, the paths inside these JSONs were updated with them. A pointer is a reference,
not a measurement, and a dead one costs the reader the click this directory exists to give.
What is never rewritten is captured output: the `.log` files under `protocol_audit/` and the
scripts under `protocol_audit/export_litert/` still name the module paths they ran under.

Model binaries are never stored here. They go to HuggingFace and are fetched
with [`scripts/fetch_deliverables.sh`](../scripts/fetch_deliverables.sh).

## Which file is which

Nine files here are pruning ladders under six different names, and the name does not say
which one is current. This does. Read a superseded file only to see what was run at the
time; quote the current one.

| ladder | covers | status |
|---|---|---|
| [`detection/full_mapping_ladder_lite1.json`](detection/full_mapping_ladder_lite1.json) | EfficientDet-Lite1, 19 rungs | **current**: fp32 + int8 mAP, 305/305 operators mapped, on-device latency |
| [`detection/full_mapping_ladder_lite2.json`](detection/full_mapping_ladder_lite2.json) | EfficientDet-Lite2, 19 rungs | **current**: fp32 + int8 mAP, 340/340 operators mapped, on-device latency |
| [`detection/full_scope_grid.json`](detection/full_scope_grid.json) | both EfficientDets, 36 rungs | marks itself superseded for Lite1; still the only file with the **pre-recovery** mAP column and the fine-tune cost |
| [`detection/target_reachability.json`](detection/target_reachability.json) | both EfficientDets | **superseded** by `pruning_matrix.json`'s `reachable_ceilings_full_model`, which measures the same ceilings |
| [`line_seg/param_ladder_oncar.json`](line_seg/param_ladder_oncar.json) | 3 line-seg backbones, 30 compiled models | **current**: the panel section 2's latency law is fitted on |
| [`line_seg/w96_deep_rungs.json`](line_seg/w96_deep_rungs.json) | 4 deep `line_seg_w96` rungs | **current**: where that law stops applying, below ~1 MiB of weights |
| [`line_seg/prune_sweep_10_90pct.json`](line_seg/prune_sweep_10_90pct.json) | same 3 backbones, 27 rungs | **retired** round-1 ladder. Its ratios are CHANNEL ratios, not parameter ratios; params fall as (1−r)². Reproduced by `--ratio_semantics channel` |
| [`reid/sweep_results.json`](reid/sweep_results.json) | `reid_youtu_lite`, 11 rungs | **current**: embedding-head widths and pruning ratios together |
| [`classification/litert_stage0.json`](classification/litert_stage0.json) | 5 ImageNet backbones | **current**, and the only classification measurement: export path and bytes, no accuracy |

[`detection/pruning_matrix.json`](detection/pruning_matrix.json) looks like a tenth but is
not a ladder. It is the audit itself: the four-way allocation control, the per-layer floor
sweep, the reachable ceilings, and the matched-pair comparisons against the cluster runs.

The rest fall into three groups, and those names are consistent already:

* **On-device, Coral USB**: `clip_rn50/`, `detection/` and `reid/coral_latency.json`, one
  per family, plus [`latency_law/latency_law_fit.json`](latency_law/latency_law_fit.json)
  (the fit over all of them) and
  [`detection/io_floor_control.json`](detection/io_floor_control.json) (the control that
  separates the fixed I/O cost from everything else). What those files add up to is counted
  in [`edgetpu_census.json`](edgetpu_census.json).
* **Bytes and structure**: [`tflite_size_audit.json`](tflite_size_audit.json),
  [`detection/compiled_size.json`](detection/compiled_size.json),
  [`detection/channel_census_lite1.json`](detection/channel_census_lite1.json),
  [`detection/allocation_stats.json`](detection/allocation_stats.json),
  [`clip_rn50/global_allocation.json`](clip_rn50/global_allocation.json).
* **Task metrics**: [`clip_rn50/zero_shot.json`](clip_rn50/zero_shot.json),
  [`criterion_census.json`](criterion_census.json).

Five of these files are **published to HuggingFace under their exact names** and pinned by
the 202 hashes in [`deliverables.sha256`](deliverables.sha256):
`detection/channel_census_lite{1,2}.json`, `detection/full_mapping_ladder_lite{1,2}.json`
and `detection/full_scope_grid.json`, which both packages carry. Renaming one means
re-cutting that release.

Every file opens with a `what` or `note` saying what it measures, and names what produced
it: `produced_by` where a script did, and `pipeline`, `protocol` or `note` where a run did.
[`detection/pruning_matrix.json`](detection/pruning_matrix.json) is the exception. It is a
hand-assembled audit record and only two of its sixteen sections name their source, so the
run behind any given row there is unrecorded.

## Which part of the pipeline

`capabilities.json` says what a family declares it implements, `criterion_census.json` says
which criterion has been run, `edgetpu_census.json` says which export path has been timed.
This table is the per-file view: for one results file, which branches of the pipeline
produced the numbers in it.

<!-- generated: pipeline -->
| file | scope | criteria | recovery | mode | export path |
|---|---|---|---|---|---|
| [`detection/full_mapping_ladder_lite1.json`](detection/full_mapping_ladder_lite1.json) | full-model | `lamp`, `magnitude_l2` | labelled | independent | litert |
| [`detection/full_mapping_ladder_lite2.json`](detection/full_mapping_ladder_lite2.json) | full-model | `lamp`, `magnitude_l2` | labelled | independent | litert |
| [`detection/full_scope_grid.json`](detection/full_scope_grid.json) | full-model | `lamp`, `magnitude_l2` | labelled | independent | n/a |
| [`detection/coral_latency.json`](detection/coral_latency.json) | backbone | `magnitude_l2` | labelled | independent | litert, onnx2tf |
| [`detection/compiled_size.json`](detection/compiled_size.json) | backbone | `magnitude_l2` | labelled | independent | onnx2tf |
| [`detection/target_reachability.json`](detection/target_reachability.json) | full-model | `lamp`, `magnitude_l2` | labelled | independent | n/a |
| [`line_seg/param_ladder_oncar.json`](line_seg/param_ladder_oncar.json) | full-model | `magnitude_l2` | labelled | independent | litert, onnx2tf |
| [`line_seg/w96_deep_rungs.json`](line_seg/w96_deep_rungs.json) | full-model | `magnitude_l2` | labelled | independent | litert, onnx2tf |
| [`line_seg/prune_sweep_10_90pct.json`](line_seg/prune_sweep_10_90pct.json) | channel | `magnitude_l2` | labelled | independent | onnx2tf |
| [`reid/coral_latency.json`](reid/coral_latency.json) | full-model | `magnitude_l2` | distill | independent | litert, onnx2tf |
| [`reid/sweep_results.json`](reid/sweep_results.json) | full-model | `magnitude_l2` | distill | independent | onnx2tf |
| [`clip_rn50/coral_latency.json`](clip_rn50/coral_latency.json) | full-model | `magnitude_l2` | distill | independent | litert, onnx2tf |
| [`clip_rn50/zero_shot.json`](clip_rn50/zero_shot.json) | full-model | `magnitude_l2` | distill | independent | onnx2tf |
| [`classification/litert_stage0.json`](classification/litert_stage0.json) | full-model | n/a | labelled | n/a | litert |
<!-- /generated -->

Where each cell came from is in
[`pipeline_provenance.json`](pipeline_provenance.json): `scope_from` and `export_paths_from`
name the key the value was read from, and `by family` marks a value derived from the family
rather than recorded in the file. `n/a` means the stage was never reached, so a
pruning-stage file has no export path. A `—` marks a file that reached a stage without
recording which branch it took. No file does now, and the two whose value had to be
recovered rather than read say where from:
[`line_seg/prune_sweep_10_90pct.json`](line_seg/prune_sweep_10_90pct.json) off the
channel-era stems in `tflite_size_audit.json`, and
[`reid/sweep_results.json`](reid/sweep_results.json) by matching its `int8_mib` column
against the two export paths carried side by side in `reid/coral_latency.json`.

Recovery is derived throughout, from `capabilities.json`: the arm is a property of the
family, not a per-run knob. Mode comes from
[`criterion_census.json`](criterion_census.json), which tallies the label each worker wrote
on the rung: **0 of 121 rungs are iterative**. 79 carry a label and none of them is an
iterative one; the other 42 carry none, and those are effdet and ssdlite rungs, whose
families declare `independent` alone in `family.yaml` while `scripts/pruning.sh` refuses a
mode a family has not declared. `prune_mode_from` says which of the two settled each row.

Three names in this repository mean something else. `iterative_steps` in
[`detection/pruning_matrix.json`](detection/pruning_matrix.json) and
[`clip_rn50/global_allocation.json`](clip_rn50/global_allocation.json) is `torch_pruning`'s
step count within a single pruning call, and `structured_incremental_param` is `line_seg`'s
parameter-target walk inside one call. Neither is `PRUNE_MODE=iterative`, which feeds a
recovered model back into the next round of pruning.

## Reproducing

```bash
./scripts/check.sh                # every tier this environment supports
./scripts/check.sh --clean-clone  # only the tier that needs nothing installed
```

`--clean-clone` is what CI runs: those checks import the standard library only and
read tracked files under `results/`, so they pass on a fresh clone with no venv, no
torch and no `outputs/` tree. The rest are skipped, loudly, when a prerequisite is
missing: a skip is never a pass. Several checks regenerate a tracked `.json` in
place, so `check.sh` also asserts that nothing changed while re-deriving it: a byte
that moved means a published number moved.

The individual commands, if you want one on its own:

```bash
python3 results/latency_law/fit_latency_law.py --check   # stdlib only, no deps
python3 results/figures/make_figures.py --check          # are the inputs complete?
python3 results/capabilities.py                         # needs pyyaml, nothing else
python3 results/protocol_audit/edgetpu_census.py         # stdlib only, no deps
python3 results/protocol_audit/pipeline_provenance.py    # stdlib only, no deps
python3 results/protocol_audit/criterion_census.py       # needs the full outputs/ tree
python3 results/protocol_audit/channel_census.py --check # needs torch + the checkpoints
python3 results/protocol_audit/allocation_stats.py       # needs torch

python3 results/figures/make_figures.py                  # needs matplotlib
```

Redrawing the figures is the one thing `check.sh` leaves out: it is a build step, not
a check, because matplotlib renders to different bytes on a different version. Reproducing a
*claim* must not require an environment; redrawing a picture may.

`fig_pipeline.svg` and `fig_pipeline_en.svg` are the exception to all of the above,
and that is why they live in [`figures/`](../figures) at the repository root rather
than here: they are hand-written SVG, not projections of any measurement, so no
script owns them and `--check` says nothing about them. Edit the SVG, then
re-rasterize it at 2x the `1115x300` viewBox to get the committed `2230x600` PNG:

```bash
google-chrome --headless --disable-gpu --hide-scrollbars \
    --force-device-scale-factor=2 --window-size=1115,300 \
    --screenshot=figures/fig_pipeline.png \
    "file://$PWD/figures/fig_pipeline.svg"
```

Both READMEs embed the SVG. The matching `.png` is referenced by nothing in this
repository and `sync_to_public.sh` excludes it from the public mirror. It is a raster
copy kept in this repository alone, for pasting into slides and papers, and nothing
checks that the two agree, so change both or neither. Keep the node order the same
in the two languages, and keep it matching the alt text: the figure and the sentence
describing it are the same claim twice.

Read [`docs/PRUNING_HAZARDS.md`](../docs/PRUNING_HAZARDS.md) before drawing
conclusions from any of it; its closing Provenance table says which claims rest
on weaker evidence than the rest.
