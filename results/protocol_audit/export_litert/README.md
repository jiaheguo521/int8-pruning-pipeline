# The full-mapping detection export

These are the scripts that produced
[`results/detection/full_mapping_ladder_lite1.json`](../../detection/full_mapping_ladder_lite1.json),
committed exactly as they were run.

> **They are not part of the supported pipeline.** `scripts/convert.sh` is, and it goes
> through `onnx2tf`. This path exists because `onnx2tf` cannot map EfficientDet: it lowers
> the BiFPN fusion to a rank-5 `CONCAT`/`SUM` and timm's SAME-pool to `PADV2`, neither of
> which `edgetpu_compiler` accepts, so the TPU subgraph stops at the backbone — **83 of 479
> operators**. This path reaches **305 of 305, none on the CPU**.
>
> They are here so the repository's newest headline result is traceable to code, not so
> that it becomes a second maintained pipeline. Nothing else in the repo calls them.

## Why they will rot

Two of the three steps monkeypatch third-party internals:

```python
E.FpnCombine.forward     = _fwd        # --patch-fpn   (effdet)
_P.MaxPool2dSame.forward = _pool_fwd   # --pool-ceil   (timm)
```

`--patch-fpn` is arithmetically identical to what it replaces: `weight_method` is `sum` for
the lite configs, so `torch.stack(nodes,-1).sum(-1)` and a chain of adds are the same value.

**`--pool-ceil` is only identical on EfficientDet-Lite1, and it was applied to Lite2 as
well. That was a defect.** `ceil_mode=True` with no padding matches timm's SAME-pool
exactly when the pool input is **even**, and diverges when it is odd:

| pool input | SAME-pad out | `ceil_mode=True` out | equal |
|---:|---|---|---|
| 56, 48, 28, 24, 14, 12, 6, 4 | same | same | ✅ |
| **7** | **4×4** | **3×3** | ❌ |
| **3** | **2×2** | **1×1** | ❌ |

Lite1 at 384 pools 48→24→12→6→3 — every pool input is even, so the rewrite is bit-exact
and the Lite1 ladder is sound. Lite2 at 448 pools 56→28→14→**7**→4, so its P7 level comes
out 3×3 instead of 4×4. Confirmed on the artifact as it stood on 2026-08-22:
`efficientdet_lite2_coco-train2017_pruned0pct_int8.tflite` emitted levels
56 / 28 / 14 / 7 / **3**, where EfficientDet-Lite2 is 56 / 28 / 14 / 7 / **4**. (That file has
since been rebuilt through the supported pipeline and now emits the correct 4; the
artifacts *this* directory produced, under `outputs/edgetpu/litert_20260821_231846/`, were
not.)

**The five Lite2 rungs this directory produced, under `outputs/edgetpu/litert_20260821_231846/`,
are therefore not EfficientDet-Lite2.** Do not quote any Lite2 number from this path.
`run_ladder.sh` gates `--pool-ceil` to the Lite1 branch, so a re-run does not reproduce it.

**Fixed in the supported pipeline, 2026-08-27 — and dropping the flag was not the fix.**
Without the rewrite the SAME-pool `F.pad(-inf)` survives as a `PADV2` whose fill never
equals the tensor zero point; `edgetpu_compiler` declines it and the graph shatters —
measured on a correct Lite2 build: **83 of 375 operators on the TPU, 292 on the CPU**.
The fix is to choose the rewrite per site from the extent's parity, because TF SAME pads
by 1 (asymmetric → `ceil_mode`) on an even extent and by 2 (symmetric → `padding=1`) on an
odd one. `src/etpu/convert/export_litert.py:_same_pool_args` does that, and all 19 Lite2 rungs now
build with FPN levels 56/28/14/7/**4** and **340/340** operators mapped in one subgraph.
Lite1 rebuilds byte-identical under the new rule — verified against the pre-fix patch on
`pruned30pct_lamp_full`. The rows in `results/tflite_size_audit.json` are re-measured and
their `defect` fields cleared; this directory's own artifacts were **not** rebuilt, since it
is the retired audit path, not the supported one.

Both rewrites bind to attributes upstream is free to move, **and a patch that stops
applying fails silently**: the export still succeeds, and the model simply goes back to
mapping 83 operators. If you re-run this, check the operator count before trusting
anything.

## Versions this was run on

| | |
|---|---|
| `litert-torch` | 0.9.3 |
| `ai-edge-quantizer` | 0.8.0 |
| `torch` | 2.10.0 |
| `timm` | 1.0.28 |
| `effdet` | 0.4.1 |
| `edgetpu_compiler` | 16.0.384591198 |

`pruning-env` carries older `litert-torch` (0.9.1) and `ai-edge-quantizer` (0.7.0); the
ladder was **not** produced with those and they have not been tested here. Point
`$AIEDGE_ENV` at a venv with the versions above.

## Three toolchain workarounds, all load-bearing

1. **`ai-edge-quantizer` writes external buffers.** It emits `(offset, size)` pairs past
   the end of the flatbuffer. `edgetpu_compiler` 16.0's older parser reads them as garbage
   and maps 3 of 345 operators. `inline_buffers.py` re-serializes with the buffers inline.
2. **`ai-edge-quantizer` smooths calibration ranges.** Every tensor's min/max is averaged
   across batches (`smoothing_factor=0.95`), which costs **0.020 mAP**. The live call site
   is `algorithm_manager.get_update_qsv_func()` — *not* the `qsv_utils` module attribute,
   which is what you find first and which does nothing. `QSV_MINMAX=1` applies the fix.
3. **`litert-torch`'s PT2E path does not work here.** `OptimizeLayoutTransposesPass`
   aborts on the q/dq graph (*"Attempting to convert non-NHWC compatible node to NHWC"*).
   Float export followed by `ai-edge-quantizer` is the route that works.

## One virtualenv, not two

`run_ladder.sh` calls two interpreters because `inline_buffers.py` used to need
`tensorflow` for the tflite schema. It no longer does — `ai_edge_litert` ships the same
generated module — so a single venv carrying `litert-torch`, `ai-edge-quantizer`, `timm`,
`effdet` and `torchreid` runs every stage. Verified 2026-08-22: all seven family
checkpoints (`classification`, `clip_rn50`, `effdet`, `line_seg`, both `reid` variants,
`ssdlite`) unpickle in that one environment, and `inline_buffers.py` produces byte-identical
output there and under `tensorflow`.

## Running it

```bash
export AIEDGE_ENV=/path/to/a/venv-with-litert-torch
./results/protocol_audit/export_litert/run_ladder.sh          # skips rungs already built
FORCE=1 ./results/protocol_audit/export_litert/run_ladder.sh  # rebuild everything
```

It reads `outputs/pytorch_pruned/efficientdet_*coco-train2017*.pt` and the calibration
cache under `outputs/tflite_int8/calib_cache/`. Those checkpoints are the output of the
supported pruning launcher — `scripts/pruning.sh` with the ladder invocation documented in
its header (`DET_SCOPE=full`, `CHECKPOINTS="10 … 90"`, `IMPORTANCES="lamp magnitude_l2"`,
`FINAL_EPOCHS=40`). This directory picks up where that leaves off; it does not prune. **Both are gitignored**, so a fresh clone
cannot run this — the checkpoints are published separately to HuggingFace. Float exports
are kept under `outputs/tflite_float_litert/` because re-quantizing a rung takes about
3 minutes while re-exporting takes 5.

Every rung's mapping lands in `mapping_summary.tsv` next to the compiled models. A rung
that fails to reach 0 CPU operators is written to that file as such rather than skipped.

## Files

| | |
|---|---|
| `aiedge_float.py` | `.pt` → float32 tflite via `litert-torch`, with the two model rewrites behind flags |
| `aeq_quantize.py` | float32 → int8 via `ai-edge-quantizer`, `static_wi8_ai8`, with the QSV fix |
| `inline_buffers.py` | re-serialize with buffers inline, for the compiler's parser |
| `run_ladder.sh` | drives all three over every rung, then `edgetpu_compiler`, and tabulates |
| `etpu_fixups.py` | three flatbuffer rewrites the compiler needs (INT64→INT32 paddings, PADV2→PAD, identity GATHER_ND). **Runs AFTER `inline_buffers.py`** — reversed, it silently drops every large constant |
| `cls_float.py` | `.pt` → float32 tflite for a plain classifier; no wrapper, no monkeypatch |
| `run_classification.sh` | Stage 0: the five dense ImageNet backbones end to end, one venv |
