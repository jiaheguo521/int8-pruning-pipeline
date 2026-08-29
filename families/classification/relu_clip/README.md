# `relu_clip`: pruning the relu-clip efflite4 student

**English** | [简体中文](README.zh-CN.md)

The open-vocabulary family's second baseline: `tf_efficientnet_lite4` distilled
from CLIP ViT-L/14 into a 768-d embedding, published at
[jiaheguo521/relu-clip](https://huggingface.co/jiaheguo521/relu-clip). Recovery
after pruning is **cosine feature distillation against the model as it was
before pruning**: there is no label space to fine-tune a head against, so the
dataset method does not apply. See [`int8_pruning/prune/recover.py`](../../../src/int8_pruning/prune/recover.py)
for the two recovery methods and which families can use which.

## Why it is not just another `clip_rn50` rung

Same objective, opposite starting point, and that is the point of having both.

| | `clip_rn50` | `relu_clip` |
|---|--:|--:|
| what it is | CLIP's own RN50 image tower | a CNN distilled *from* CLIP, designed against int8 |
| params | 38.3 M | 12.71 M |
| int8 | 37.31 MiB | 13.58 MiB |
| off-chip / on-chip | 16.09 / 6.60 MiB | 7.50 / 7.60 MiB |
| latency | 51.51 ms | 26.49 ms |
| int8 cost | zero-shot 94.17 per-channel; per-tensor collapses to 18.83 | −0.10 points fp32→int8 |

Both latency figures are **onnx2tf** builds, which is what makes them comparable
That is the export path upstream published on, and the path this repo used
before 2026-08-22. The same `clip_rn50` graph rebuilt through litert-torch
measures 64.40 ms, so do not mix the two columns of
[`results/clip_rn50/coral_latency.json`](../../../results/clip_rn50/coral_latency.json),
which is where the `clip_rn50` row comes from (params counted from
`outputs/models/clip_rn50.pt`). The `relu_clip` row is upstream's, measured on
the same Coral USB Edge TPU and **not** re-measured here.

Pruning a model that already fits the accelerator's shape is a different
question from pruning one that does not, and the on-chip cap is what makes it
sharp: 7.60 MiB cached + 7.50 MiB streamed means the streamed half disappears
somewhere near a 45–50% parameter cut, and this repo's own weight-transfer law
([`results/latency_law/latency_law_fit.json`](../../../results/latency_law/latency_law_fit.json),
R² = 0.9964 over 30 compiled models) says that is where the latency return flattens.

## Use

```bash
# once: fetch the published weights and write the baseline pickle (~51 MB)
python families/classification/relu_clip/download_baseline.py        # -> outputs/models/relu_clip.pt

# prune + distill-recover, via the standard driver
CLS_MODEL=relu_clip CLS_DATASET=imagenet PARTS=A \
CHECKPOINTS="20 30 40 50 60" IMPORTANCES=magnitude_l2 FINAL_EPOCHS=10 \
    ./scripts/pruning.sh

# int8 + compile (unchanged for this family)
GLOB='relu_clip_*.pt' ./scripts/convert.sh
./scripts/compile_edgetpu.sh
```

Optional zero-shot during recovery; otherwise the worker tracks
`cos(student, teacher)` only, which needs no labels:

```bash
RELU_CLIP_TEXT_EMB=weights/text/imagenet1k_text_emb_vitl14.npz \
RELU_CLIP_VAL_DIR=$IMAGENET_ROOT/val \
CLS_MODEL=relu_clip ... ./scripts/pruning.sh
```

The `.npz` is upstream's, with keys `embs` + `labels`. The worker aligns val
folders to embedding rows by name where the names match, else by position when
the counts match, and refuses anything else rather than half-matching. A wnid
subset against a 1000-row matrix scores every image against the wrong class.

## Protocol, and the one place it deviates

`PRUNING_PROTOCOL["global_pruning"]` is **False** here where `clip_rn50` has it
`True`. On the RN50 tower, global ranking × mean-normalised magnitude at only
−30% params left `stem.conv1` at 3 of 32 channels and `stages.0.0.conv2_kxk` at
5 of 64, and zero-shot went 99.0 → 0.0 → 70.75 after recovery
([`results/clip_rn50/global_allocation.json`](../../../results/clip_rn50/global_allocation.json)).
EfficientNet-Lite4's early projections are narrower still. `RELU_CLIP_GLOBAL_PRUNING=1`
flips it back, so the fork can be measured rather than assumed.

`RELU_CLIP_ROUND_TO=N` rounds surviving channel counts up to a multiple of N.
The Edge TPU's systolic array is 64 wide and channels cut below it cost accuracy
while buying no time (see section 2 of [`docs/PRUNING_HAZARDS.md`](../../../docs/PRUNING_HAZARDS.md):
recomputing MACs on channels padded to 64 lifts the EfficientDet latency fit from
R² = 0.937 to 0.992). Nothing in this repo has measured it yet, so it is **off by
default**.

Calibration uses `_calib_imagenet_train_reluclip`, not the loader every other
ImageNet family uses. Upstream's `preprocessing.json` pins
`Resize(224, BICUBIC) + CenterCrop(224)`, with no 256/224 upscale, and names the two
rounding details that "silently cost accuracy if guessed", both of which go the
opposite way from this repo's default loader.

## Status

Bring-up only. Nothing here is a result yet.

What has been run end to end, on a 10-class / 9 469-image local ImageNet subset
with 1 recovery epoch and 32 calibration samples: enough to prove the path, far
too little to quote:

| | dense | −30% params, `magnitude_l2`, per-layer |
|---|--:|--:|
| params | 12 709 376 | 8 869 722 (−30.2%) |
| int8 (litert path) | 13.82 MiB | 9.91 MiB |
| cos(student, teacher) | 1.0 | 0.508 before FT → 0.905 after 1 epoch |

Not measured: zero-shot at any rung, on-device latency at any rung, any
importance criterion other than `magnitude_l2`, and the `round_to` knob. The
dense int8 file lands within 1.8% of upstream's onnx2tf build (13.82 vs 13.58 MiB),
which is a size check, not an accuracy one.
