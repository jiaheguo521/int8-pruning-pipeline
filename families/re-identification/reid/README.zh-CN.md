# reid：Edge TPU 上的行人 Re-ID 外观嵌入器

两条线：三个 Torchreid Market-1501 嵌入器**不剪枝**直接交付，以及 Youtu
`reid_youtu_lite` 的剪枝比例 sweep。两者都用 **per-channel** int8，原因与
CLIP 相同：输出是余弦嵌入。

输入 256x128（H x W），是本仓唯一的非方形家族。

> 前置：venv 与 Phase 1–3 驱动见[根 README](../../../README.zh-CN.md)。

## 6. 行人 Re-ID：Torchreid 外观嵌入器（不剪枝）

```bash
# 三个 Torchreid 行人 Re-ID 嵌入器（Market-1501 权重）作为 int8 Edge-TPU 模型：
# osnet_x0_5、osnet_x0_75、mobilenetv2_x1_0。直接从预训练 checkpoint 走 Phase 2/3——
# 不训练、不剪枝。和 CLIP 一样输出余弦匹配的 EMBEDDING（osnet：512 维），所以用
# PER-CHANNEL int8 —— 这是**比照 clip_rn50** 定的,那边 per-tensor 打崩指标是实测的
# (零样本 top-1 94.17% -> 18.83%)。**但 Re-ID 自己这一侧从未实测**,仓里没有任何
# per-tensor 档;eval_discrim.py:36 把它写成读者可以自己跑一次的实验。
# 不要把「per-tensor 会打崩 Re-ID」当成实测结论转述。输入 256x128（H x W）、ImageNet 归一化、RGB。

# 步骤 0 —— 构建基线（Torchreid 以 --no-deps 安装以保护 torch/TF 环境；Market-1501
# 权重经 gdown 从 Model Zoo 下载）。每个 forward() 都被校验返回 EMBEDDING 而非 751 类
# ID 分类头（断言维度 != 751）：
./scripts/download_and_finetune_models.sh
# -> outputs/models/reid_{osnet_x0_5,osnet_x0_75,mobilenetv2_x1_0}_market1501.pt
MARKET1501_ENABLED=1 ./scripts/download_datasets.sh   # 校准 + 评测裁片（~140 MB）

# 步骤 1 —— int8 转换（per-channel，自动识别）+ 编译。基线在 outputs/models/ 下，
# 所以用 PT_PRUNED_DIR 把 Phase 2 指过去：
PT_PRUNED_DIR=$PWD/outputs/models GLOB='reid_*_market1501.pt' \
  DATASET_DIR=$PWD/data/datasets ./scripts/convert.sh
GLOB='reid_*_market1501_int8.tflite' SHOW_OPS=1 ./scripts/compile_edgetpu.sh
# -> outputs/tflite_int8_litert/reid_*_int8.tflite  ->  *_int8_edgetpu.tflite

# 步骤 2 —— 判别力抽查（同人 vs 不同人余弦，量化前后对比，外加 fp32<->int8 方向闸）。
# 确认 int8 没把度量做没——即 PINTO_model_zoo「同人/不同人分不开」那个坑：
python families/re-identification/reid/eval_discrim.py \
    --pt outputs/models/reid_osnet_x0_5_market1501.pt \
    --tflite outputs/tflite_int8_litert/reid_osnet_x0_5_market1501_int8.tflite \
    --market-dir data/datasets/market1501 --num-ids 60

# 步骤 3 —— SRAM 报告（片上/片外；片外>0 => 参数走 USB 流式读取，即约 51ms 的坑）
# + CPU int8 延迟。SRAM 预测无需 Coral：
python families/re-identification/reid/report.py --glob 'reid_*_market1501_int8.tflite' \
    --out-json outputs/pruning_logs/reid_edgetpu_report.json

# Market-1501 上的 Rank-1 / mAP（Torchreid Model Zoo，供参考）：
#   osnet_x0_75  93.7% / 81.2%     osnet_x0_5  92.5% / 79.8%     mobilenetv2_x1_0  85.6% / 67.3%
#   （OSNet 是 Re-ID 专用架构：参数远少于 MobileNetV2，精度反而更高。）

# 步骤 4（可选）—— Coral USB 真机单次 embed 延迟，用 tflite_runtime + libedgetpu
# delegate 在 Edge-TPU docker 里跑（无需 pycoral）：
docker run --rm --privileged -v /dev/bus/usb:/dev/bus/usb \
    -v $PWD/src/int8_pruning/backends/edgetpu:/src:ro -v $PWD/outputs/edgetpu:/models:ro \
    edgetpu_ros:latest \
    bash -c 'python3 /src/bench.py /models/single_*/reid_*_edgetpu.tflite'
# Coral USB 实测——排序相对 参数量/精度 反转：
#   mobilenetv2_x1_0  2.9 ms   （70 算子；参数最多却最快——全是密集卷积）
#   osnet_x0_5        8.1 ms   （290 算子）    osnet_x0_75  11.2 ms  （290 算子；最慢）
# 三个都 off-chip=0（装进 8 MiB SRAM）=> 都没触发 USB 流式惩罚。按实测延迟选，别按参数：
# OSNet 更准但每次 embed 慢 3-4 倍。
```

## 7. Youtu Re-ID（`reid_youtu_lite`）：剪枝比例 sweep

```bash
# 唯一一个「要剪枝」的 reid_* 家族。腾讯 YouReID youtu_reid_baseline_lite
#（== opencv_zoo 的 person_reid_youtu_2021nov）：ResNet50 + cat(gap,gmp) -> 1x1 卷积
# -> 768 维，26.66M 参数，int8 ≈26.7 MB。在**四个**数据集上训练（market1501 + Duke +
# MSMT17 + CUHK03）——这份多源泛化能力正是它优于单源 OSNet 的原因，也是「值得把它压小
# 而不是换掉」的原因。
#
# 为什么要做 sweep：Coral 上 emb 模型和检测器共享 8 MiB SRAM。SSD 吃掉 5.41 MB 缓存后，
# emb 只分到约 2.4 MiB —— 26.1 MiB 的基线要流式传输约 23.7 MiB。流式单价是上机量出来的，
# 这一族 2.30 ms/MiB（lat = 2.302*片外 + 0.223*片上 + 2.782，六档 R^2 = 0.9976，见
# results/reid/coral_latency.json）=> co-compile 下单次 embed 约 58 ms，standalone 实测
# 46.7 ms。剪枝直接换延迟：80% 档 standalone 实测 5.6 ms。

# 步骤 0 —— 基线（YouReID Model Zoo checkpoint，经 gdown，约 226 MB）。
./scripts/download_and_finetune_models.sh
# -> outputs/models/reid_youtu_lite.pt   （**无后缀**：权重是四源的，标 _market1501 是错的）
MARKET1501_ENABLED=1 ./scripts/download_datasets.sh

# 步骤 1 —— 剪枝 + 蒸馏恢复。Duke/MSMT17/CUHK03 都拿不到（Duke 已被原作者撤下、MSMT17
# 需许可），所以原始四源标签空间根本够不着，CrossEntropy/triplet 恢复无从谈起。蒸馏绕开
# 了这个问题：**teacher 本身就是四源模型**，它的知识无需标签即可传递。
DATASET_DIR=$PWD/data/datasets PARTS=A CLS_MODEL=reid_youtu_lite \
  CLS_DATASET=market1501 IMPORTANCES=magnitude_l2 CHECKPOINTS="20 50 70 80 90" \
  ./scripts/pruning.sh
# -> outputs/pytorch_pruned/reid_youtu_lite_market1501_pruned{20,50,70,80,90}pct_magnitude_l2.pt
#    （外加一个指向基线的 _pruned0pct.pt 符号链接）

# 注意：REID_GLOBAL_PRUNING 在这里默认 0 —— 与**其他所有家族相反**。全局剪枝比较的是各层
# **原始**滤波器范数，而 resnet50 深层的范数系统性地压过 stem，于是浅层被整个抹掉。
# 90% 档、同等 1 epoch 蒸馏后的 cos(student, teacher) 实测：
#   global + magnitude_l2  0.390   stem 仅存 1/64 通道   head 占参数 64.7%
#   global + lamp          0.688   stem 50/64            head  9.2%
#   local  + magnitude_l2  0.766   stem 17/64            head 32.8%
#   local  + lamp          0.766   stem 17/64            head 32.8%
# 两个 local 行小数点后 4 位完全一致，这才是关键：一旦用统一的逐层比例固定了「**分配**」，
# 选哪个 importance 准则（它只决定层**内**哪些通道被砍）就几乎无所谓了。全局 magnitude
# 甚至输给全局 RANDOM。

# 步骤 2 —— int8。per-channel、真实行人裁剪图校准、ImageNet 归一化、图里**不做** BGR<->RGB
# 翻转（负步长 reverse 会让 edgetpu_compiler 在 dwl.reverse 上编不过）。
DATASET_DIR=$PWD/data/datasets GLOB='reid_youtu_lite*_market1501_pruned*.pt' \
  ./scripts/convert.sh
# -> outputs/tflite_int8_litert/..._int8.tflite   <- **CPU 版 int8 就是交付物**。
# 不要交付 *_edgetpu.tflite：它必须和检测器在**同一次** edgetpu_compiler 调用里 CO-COMPILE，
# 否则两个模型会互相踢掉对方的缓存，延迟数据作废。

# 步骤 3 —— 真正的 Market-1501 mAP / rank-1（torchreid 的 evaluate_rank；fp32 vs int8）。
# eval_reid_discriminability.py 只是量化闸门，**无法**给 sweep 排序。
python families/re-identification/reid/eval_map.py \
    --pt outputs/models/reid_youtu_lite.pt \
    --tflite outputs/tflite_int8_litert/reid_youtu_lite_market1501_pruned90pct_magnitude_l2_int8.tflite \
    --market-dir data/datasets/market1501
# 基线复现了官方数字：mAP 87.89 / rank-1 95.22（官方 87.86 / 95.01）——这同时验证了
# 复刻架构、split_bn 折叠、评测协议三件事同时正确。

# 对照组（各有独立 stem：同一个基线，但产出的是不同的模型）：
#   768 维 + simmat 损失  -> reid_youtu_lite_simmat_market1501   （损失对照）
#   256 维 + simmat 损失  -> reid_youtu_lite_e256_market1501     （头部 vs 主干）
# 头部被钉死以保证 768 维输出 drop-in 兼容，但它的 in_channels 绑在 layer4 的存活宽度上，
# 于是占比从 0% 档的 11.8% 涨到 90% 档的 32.8%。e256 组先把头剪掉，把预算让给主干。
REID_EMBED_DIM=256 REID_DISTILL_LOSS=simmat DATASET_DIR=$PWD/data/datasets PARTS=A \
  CLS_MODEL=reid_youtu_lite CLS_DATASET=market1501 IMPORTANCES=magnitude_l2 \
  CHECKPOINTS="80 90 95" ./scripts/pruning.sh
# 95% == 1.3M 参数 == osnet_x0_75 的同等容量：这个点用来把「多源**数据**」和「**容量**」
# 分开——即 Youtu 在没见过的视角上打赢 OSNet，到底是靠哪一个。

# 蒸馏只在**你采样到的地方**钉住 teacher。只用 Market 图像，学生就只在 Market-like 裁剪上
# 贴合 teacher；而在高剪枝率下容量稀缺，它会主动牺牲没采样到的域。把部署相机的无标签裁剪图
# 混进来是零成本的（teacher 自己提供目标向量），也是应对「任何 Re-ID 数据集都没有的视角」
# 最高杠杆的旋钮：
#   REID_EXTRA_IMAGE_DIRS="/path/to/car_crops" ... ./scripts/pruning.sh
```
