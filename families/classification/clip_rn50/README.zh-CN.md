# clip_rn50：Edge TPU 上的 CLIP RN50 图像塔

开放词表零样本分类。卷积主干映射到 TPU（79 op），AttentionPool2d 头回落到
CPU。**这个切分已经在两条导出路径上各测过一次，结论是它不是导出器造成的**：
onnx2tf 下 107 op / 28 个在 CPU，litert-torch + ai-edge-quantizer 下 112 op /
33 个在 CPU，TPU 侧两边都是 79 个。堵点是注意力本身：`BATCH_MATMUL` ×2 与
`SOFTMAX`，而 Edge TPU 没有 batch matmul，任何形状下都上不去；另外 6 个
`TRANSPOSE` 和 1 个 `FULLY_CONNECTED` 报「otherwise supported, but not mapped」，
其余 23 个是从这一刀级联出来的。EfficientDet 那次的 CPU 回落最后查明是导出器
造成的，这一次不是。→ [`results/clip_rn50/coral_latency.json`](../../../results/clip_rn50/coral_latency.json)
的 `export_path_migration`

量化必须 **per-channel**：输出是方向敏感的 1024 维余弦嵌入，per-tensor
会把零样本 top-1 从约 97% 打到约 19%。

> 前置：venv 与 Phase 1–3 驱动见[根 README](../../../README.zh-CN.md)。

## 5. 视觉-语言（CLIP）：Edge TPU 上的开放词表 zero-shot

```bash
# 多模态轴：Coral 上跑 CLIP RN50 图像塔（clip_rn50），输出 1024 维图像 embedding；
# CPU 用它与候选标签的文本 embedding（离线预计算）算余弦相似度。换文本矩阵 = 换词表，
# 无需重新编译——这就是小车 "find the <说出的物体>" 的开放词表路径。
# Edge-TPU 切分：卷积主体 -> TPU（79 op），AttentionPool2d 头 -> CPU。两条导出
# 路径都测过，都是这个结果（onnx2tf 28 个 CPU 算子，litert 33 个），所以这个头
# 不是导出器造成的——挡路的是 BATCH_MATMUL + SOFTMAX。见本文件开头。
# clip_rn50.pt 由 scripts/download_and_finetune_models.sh 下载
# （timm resnet50_clip，OpenAI 权重）。

# 步骤 0（可选）—— Phase-1 剪枝 CLIP 塔（backbone-only）。CLIP 无标签，恢复用
# cosine 特征蒸馏（冻结 teacher，无标签 ImageNet train 图像上 1-cos，而非 CrossEntropy）。
# 专用 worker families/classification/clip_rn50/prune.py，由 CLS_MODEL=clip_rn50 分发（仅 imagenet）。
# FINAL_EPOCHS 需 >= 3——1 个 epoch 时 BatchNorm running stats 未收敛（eval-mode zero-shot 读 0%）。
# 训练中的 zero-shot 评测可选：CLIP_TEXT_EMB + CLIP_VAL_DIR；缺省只跟踪 cos-to-teacher。
PARTS=A CLS_MODEL=clip_rn50 CLS_DATASET=imagenet \
  CHECKPOINTS="30 50 70" IMPORTANCES="magnitude_l2" FINAL_EPOCHS=10 \
  CLIP_TEXT_EMB=outputs/models/clip_rn50_text_imagenette.npy \
  CLIP_VAL_DIR="$DATASET_DIR/Imagenet_1k/train" ./scripts/pruning.sh
# 注意：这里的 $DATASET_DIR/Imagenet_1k 是 10 类的 IMAGENETTE 替身，不是 ILSVRC2012——
# eval.py:52 把它的 wnid 映射到那 10 个文本标签。它只有 train/，没有 val/。指向真正的
# 1000 类 ImageNet val 也能跑，但会静默地只评那 10 个对得上的文件夹
#（list_images 会打印 "[skip] folder ..."）。
# -> outputs/pytorch_pruned/clip_rn50_imagenet_pruned<P>pct_magnitude_l2.pt
# 剪枝把 CLIP 往 8 MiB 缓存里压：30% 时 int8 37.34->26.02 MiB、off-chip 16.09->8.29 MiB、
# Coral 实测延迟 51.51->31.75 ms（1.62 倍）——cost/accuracy 主线，现在轮到 VLM。两个数都是
# per-channel；旧的 36.6/15.9 那一对报的是 per-tensor 那次构建，而那正是 zero-shot 崩掉的
# 那一个。见 results/clip_rn50/coral_latency.json。

# 步骤 1 —— int8 转换（clip_rn50 用 PER-CHANNEL 量化；per-tensor 会把 cosine zero-shot
# 打崩，见下方 cost 表）+ 编译。剪枝后的 .pt 自动识别 family（无需 --model-family）；
# 下面 0pct 基线演示用 --model-family：
python src/int8_pruning/convert/tflite.py --input outputs/models/clip_rn50.pt \
    --model-family clip_rn50                       # -> outputs/tflite_int8_litert/clip_rn50_int8.tflite
GLOB='clip_rn50_int8.tflite' ./scripts/compile_edgetpu.sh
# 与检测器 co-compile 共享 8 MiB 缓存（未剪枝 CLIP 37.34 MiB、单独编译就要流式 16.09 MiB，
# 用步骤 0 剪枝可缩小 off-chip 占用）：
PAIR="efficientdet_lite1_coco-train2017_pruned0pct clip_rn50" ./scripts/compile_edgetpu.sh

# 步骤 2 —— 离线预计算文本 embedding（需要 open_clip_torch；绝不上 TPU）。
# 内置 10 类 Imagenette，或任意开放词表目标：
python families/classification/clip_rn50/text_embeddings.py --preset imagenette --ensemble
python families/classification/clip_rn50/text_embeddings.py --name car_targets \
    --classes "red cup,blue backpack,potted plant,person,office chair"

# 步骤 3 —— zero-shot top-1/5（图像 emb 与文本 emb 的余弦）。fp32 / int8(CPU) /
# _edgetpu(Coral) 三者可直接对比。--images-dir 为 ImageFolder 目录（文件夹名 -> 标签；
# Imagenette 的 wnid 已自动映射）：
python families/classification/clip_rn50/eval.py \
    --tflite outputs/tflite_int8_litert/clip_rn50_int8.tflite \
    --text-emb outputs/models/clip_rn50_text_imagenette.npy \
    --images-dir "$DATASET_DIR/Imagenet_1k/train"
# fp32 参考：把 --tflite ... 换成 --pt outputs/models/clip_rn50.pt

# Imagenette 10 类 zero-shot cost（9469 张），即为何 per-channel 作默认：
#   fp32                 Top-1 97.29%   Top-5 99.87%
#   int8 per-tensor      Top-1 18.83%   Top-5 61.57%   （崩溃——余弦对方向敏感）
#   int8 per-channel     Top-1 94.17%   Top-5 99.73%   （较 fp32 -3.12，可接受）
```
