# `relu_clip`：剪枝 relu-clip 的 efflite4 学生模型

[English](README.md) | **简体中文**

开放词表这条线的第二个基线：从 CLIP ViT-L/14 蒸馏出来的 `tf_efficientnet_lite4`，
输出 768 维嵌入，权重发布在
[jiaheguo521/relu-clip](https://huggingface.co/jiaheguo521/relu-clip)。剪枝后的恢复走
**对剪枝前模型的余弦特征蒸馏**：这一族没有标签空间可以微调分类头，所以数据集那条恢复路
在这里不成立。两种恢复方法各自适用于哪些族，见
[`int8_pruning/prune/recover.py`](../../../src/int8_pruning/prune/recover.py)。

## 为什么不是 `clip_rn50` 的又一档

目标一样，起点相反，两个都留着的意义就在这里。

| | `clip_rn50` | `relu_clip` |
|---|--:|--:|
| 是什么 | CLIP 自己的 RN50 图像塔 | 从 CLIP 蒸馏出来、对着 int8 设计的 CNN |
| 参数量 | 38.3 M | 12.71 M |
| int8 体积 | 37.31 MiB | 13.58 MiB |
| 片外 / 片内 | 16.09 / 6.60 MiB | 7.50 / 7.60 MiB |
| 延迟 | 51.51 ms | 26.49 ms |
| int8 的代价 | per-channel zero-shot 94.17；per-tensor 塌到 18.83 | fp32→int8 掉 0.10 分 |

两个延迟都是 **onnx2tf** 构建的，可比性正来自这一点：那是上游发布用的导出路径，也是本仓
2026-08-22 之前用的路径。同一个 `clip_rn50` 图换 litert-torch 重建测出来是 64.40 ms，所以
不要把 [`results/clip_rn50/coral_latency.json`](../../../results/clip_rn50/coral_latency.json)
的两列混着读；`clip_rn50` 这一行就出自该文件（参数量数自 `outputs/models/clip_rn50.pt`）。
`relu_clip` 这一行是上游的数，同一块 Coral USB Edge TPU 上测的，本仓**没有**复测。

剪一个本来就合设备形状的模型，和剪一个不合的，是两个不同的问题；片上容量让这个区别变得很锐利：
7.60 MiB 缓存 + 7.50 MiB 流式，意味着流式那一半大约在剪掉 45–50% 参数时消失，而本仓自己拟合的
搬运定律（[`results/latency_law/latency_law_fit.json`](../../../results/latency_law/latency_law_fit.json)，
30 个已编译模型，R² = 0.9964）说那里正是延迟回报转平的位置。

## 用法

```bash
# 一次性：拉发布权重，写成基线 pickle（约 51 MB）
python families/classification/relu_clip/download_baseline.py        # -> outputs/models/relu_clip.pt

# 剪枝 + 蒸馏恢复，走标准驱动脚本
CLS_MODEL=relu_clip CLS_DATASET=imagenet PARTS=A \
CHECKPOINTS="20 30 40 50 60" IMPORTANCES=magnitude_l2 FINAL_EPOCHS=10 \
    ./scripts/pruning.sh

# int8 + 编译（这一族无需任何改动）
GLOB='relu_clip_*.pt' ./scripts/convert.sh
./scripts/compile_edgetpu.sh
```

恢复过程中的 zero-shot 是可选的；不给的话 worker 只跟踪 `cos(student, teacher)`，
那个不需要标签：

```bash
RELU_CLIP_TEXT_EMB=weights/text/imagenet1k_text_emb_vitl14.npz \
RELU_CLIP_VAL_DIR=$IMAGENET_ROOT/val \
CLS_MODEL=relu_clip ... ./scripts/pruning.sh
```

`.npz` 是上游那份，键为 `embs` + `labels`。worker 对齐 val 目录和嵌入行的规则是：名字对得上就
按名字，对不上但数量相等就按位置，两者都不成立就直接报错而不是半匹配。拿 wnid 子集去对
1000 行的矩阵，会让每一张图都跟错误的类别打分。

## 协议，以及唯一偏离的地方

这里 `PRUNING_PROTOCOL["global_pruning"]` 是 **False**，而 `clip_rn50` 是 `True`。在 RN50 塔上，
全局排序 × mean-归一化幅值在只剪 30% 参数时就把 `stem.conv1` 剩到 32 通道里的 3 个、
`stages.0.0.conv2_kxk` 剩到 64 里的 5 个，zero-shot 走了 99.0 → 0.0 → 恢复后 70.75
（[`results/clip_rn50/global_allocation.json`](../../../results/clip_rn50/global_allocation.json)）。
EfficientNet-Lite4 的早期投影层比那还窄。`RELU_CLIP_GLOBAL_PRUNING=1` 可以翻回去，
这样这个分叉是被测出来的，而不是被假定的。

`RELU_CLIP_ROUND_TO=N` 把存活通道数向上取整到 N 的倍数。Edge TPU 的脉动阵列宽 64，剪到它以下的
通道要精度、不买时间（见 [`docs/PRUNING_HAZARDS.md`](../../../docs/PRUNING_HAZARDS.md) 第 2 节：
把 MAC 按补齐到 64 重算，EfficientDet 的延迟拟合从 R² = 0.937 升到 0.992）。本仓还没有测过它，
所以**默认关闭**。

标定用的是 `_calib_imagenet_train_reluclip`，不是其他 ImageNet 族共用的那个加载器。上游的
`preprocessing.json` 钉的是 `Resize(224, BICUBIC) + CenterCrop(224)`，没有 256/224 的放大，
并且点名了两个"猜错就会悄悄掉精度"的取整细节，两个的方向都和本仓默认加载器相反。

## 状态

只是打通，还没有任何结果。

在本地 10 类 / 9 469 张的 ImageNet 子集上、1 个恢复 epoch、32 张标定图跑通了全链路：
足以证明路是通的，远不足以引用：

| | 稠密 | 剪 30% 参数，`magnitude_l2`，逐层 |
|---|--:|--:|
| 参数量 | 12 709 376 | 8 869 722（−30.2%） |
| int8（litert 路径） | 13.82 MiB | 9.91 MiB |
| cos(student, teacher) | 1.0 | FT 前 0.508 → 1 epoch 后 0.905 |

没测的：任何一档的 zero-shot、任何一档的设备延迟、`magnitude_l2` 以外的任何准则、以及
`round_to` 这个旋钮。稠密模型的 int8 文件和上游 onnx2tf 构建相差 1.8%（13.82 对 13.58 MiB），
那是一个体积核对，不是精度核对。
