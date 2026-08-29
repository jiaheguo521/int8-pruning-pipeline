# clip_rn50: CLIP RN50 image tower on the Edge TPU

Open-vocabulary zero-shot classification. The conv trunk maps to the TPU (79 ops)
and the AttentionPool2d head falls back to CPU. **That split has now been measured on
both export paths, and it is not an exporter artifact**: 107 ops with 28 on CPU under
onnx2tf, 112 ops with 33 on CPU under litert-torch + ai-edge-quantizer, and 79 on the
TPU either way. What blocks it is the attention itself: `BATCH_MATMUL` ×2 and
`SOFTMAX`. The Edge TPU has no batch matmul, so the QK and AV products cannot map at
any shape; 6 `TRANSPOSE` and 1 `FULLY_CONNECTED` come back "otherwise supported, but
not mapped", and the remaining 23 cascade from the cut. EfficientDet's CPU fallback
turned out to be the exporter's doing; this one is not. →
[`results/clip_rn50/coral_latency.json`](../../../results/clip_rn50/coral_latency.json),
`export_path_migration`

Quantization is **per-channel**, because per-tensor collapses zero-shot top-1 from ~97% to
~19% on the direction-sensitive 1024-d embedding.

> Prerequisites: the venv and Phase 1–3 drivers in the [root README](../../../README.md).

## 5. Vision-language (CLIP): open-vocabulary zero-shot on the Edge TPU

```bash
# The multimodal axis: a CLIP RN50 IMAGE tower (clip_rn50) on the Coral emits a
# 1024-d image embedding; the CPU scores it by cosine against TEXT embeddings of
# the candidate labels (computed offline). Swap the text matrix -> new vocabulary,
# no recompile. That is the open-vocab "find the <named> object" path for the car.
# Edge-TPU split UNDER ONNX2TF: conv body -> TPU (79 ops), AttentionPool2d head ->
# CPU (28 ops). Not re-measured on the litert path; do not assume the head is
# intrinsically unmappable -- effdet's CPU fallback turned out to be the exporter. clip_rn50.pt is downloaded by
# scripts/download_and_finetune_models.sh (timm resnet50_clip, OpenAI weights).

# Step 0 (optional) — Phase-1 PRUNE the CLIP tower (backbone-only). CLIP has no
# labels, so recovery is cosine FEATURE-DISTILLATION from a frozen teacher on
# unlabeled ImageNet train images (NOT CrossEntropy). Dedicated worker
# families/classification/clip_rn50/prune.py, dispatched by CLS_MODEL=clip_rn50 (imagenet-only).
# Use FINAL_EPOCHS >= 3 — 1 epoch leaves BatchNorm running-stats unconverged
# (eval-mode zero-shot reads 0%). Optional zero-shot eval during FT via
# CLIP_TEXT_EMB + CLIP_VAL_DIR; else it tracks cosine-to-teacher only.
PARTS=A CLS_MODEL=clip_rn50 CLS_DATASET=imagenet \
  CHECKPOINTS="30 50 70" IMPORTANCES="magnitude_l2" FINAL_EPOCHS=10 \
  CLIP_TEXT_EMB=outputs/models/clip_rn50_text_imagenette.npy \
  CLIP_VAL_DIR="$DATASET_DIR/Imagenet_1k/train" ./scripts/pruning.sh
# NOTE: $DATASET_DIR/Imagenet_1k is the 10-class IMAGENETTE stand-in here, not
# ILSVRC2012 -- eval.py:52 maps its wnids to the 10 text labels. It has train/ and
# no val/. Pointed at a real 1000-class ImageNet val this still works, but silently
# scores only the 10 folders that match (list_images prints "[skip] folder ...").
# -> outputs/pytorch_pruned/clip_rn50_imagenet_pruned<P>pct_magnitude_l2.pt
# Pruning shrinks CLIP toward the 8 MiB cache: at 30% the int8 drops 37.34->26.02 MiB,
# off-chip streaming 16.09->8.29 MiB, and measured Coral latency 51.51->31.75 ms (1.62x)
# -- the cost/accuracy thesis, now for a VLM. Both figures are per-channel; the older
# 36.6/15.9 pair quoted the per-tensor build, which is the one whose zero-shot collapses.
# See results/clip_rn50/coral_latency.json.

# Step 1 — int8 convert (clip_rn50 uses PER-CHANNEL quant; per-tensor collapses
# cosine zero-shot, see the cost table below) + compile. The pruned .pt auto-
# detects its family (no --model-family needed); the 0pct baseline below uses it:
python src/int8_pruning/convert/tflite.py --input outputs/models/clip_rn50.pt \
    --model-family clip_rn50                       # -> outputs/tflite_int8_litert/clip_rn50_int8.tflite
GLOB='clip_rn50_int8.tflite' ./scripts/compile_edgetpu.sh
# Co-compile with a detector to share the 8 MiB cache (unpruned CLIP is 37.34 MiB and
# streams 16.09 MiB off-chip on its own; prune it via Step 0 to shrink that):
PAIR="efficientdet_lite1_coco-train2017_pruned0pct clip_rn50" ./scripts/compile_edgetpu.sh

# Step 2 — precompute TEXT embeddings offline (needs open_clip_torch; never runs
# on the TPU). Built-in 10-class Imagenette set, or arbitrary open-vocab targets:
python families/classification/clip_rn50/text_embeddings.py --preset imagenette --ensemble
python families/classification/clip_rn50/text_embeddings.py --name car_targets \
    --classes "red cup,blue backpack,potted plant,person,office chair"

# Step 3 — zero-shot top-1/5 (cosine of image-emb vs text-emb). fp32 / int8(CPU)
# / _edgetpu(Coral) are all comparable. --images-dir is an ImageFolder tree
# (folder name -> label; Imagenette wnids are auto-mapped):
python families/classification/clip_rn50/eval.py \
    --tflite outputs/tflite_int8_litert/clip_rn50_int8.tflite \
    --text-emb outputs/models/clip_rn50_text_imagenette.npy \
    --images-dir "$DATASET_DIR/Imagenet_1k/train"
# fp32 reference: swap --tflite ... for --pt outputs/models/clip_rn50.pt

# Imagenette 10-class zero-shot cost (9469 imgs), why per-channel is the default:
#   fp32                 Top-1 97.29%   Top-5 99.87%
#   int8 per-tensor      Top-1 18.83%   Top-5 61.57%   (collapse — direction-sensitive cosine)
#   int8 per-channel     Top-1 94.17%   Top-5 99.73%   (-3.12 vs fp32, acceptable)
```
