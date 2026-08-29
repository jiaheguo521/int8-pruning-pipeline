> 中文版 | [English](SETUP.md) · [← 回到 README](../README.zh-CN.md)

# 环境搭建与运行

搭环境、取数据与基线权重，并把流水线端到端跑一遍：结构化剪枝与恢复，然后 int8 量化。
编译成 Edge TPU 产物、在 Coral 上计时是之后的可选后端，见第 5 节。想知道这个仓库
是什么、测出了什么，请从 [README](../README.zh-CN.md) 开始。

## 1. 创建虚拟环境

```bash
# 进入项目根目录（服务器上替换为实际路径）
cd <project_root>

# 一个脚本，两条导出链路。默认那次运行创建 ./pruning-env：先从 CUDA 匹配的 index
# 装 torch，再 `pip install -e '.[all]'` —— 所有模型族，加上 litert 这条 int8
# 导出链路。这就是完整的 prune -> recover -> int8 流水线，大多数人只需要它。
./scripts/setup_env.sh

# 再加上第二条导出链路。它有自己独立的 venv ./onnx2tf-env，这是约束不是偏好：
# onnx2tf 把 numpy 钉在 1.26.4，tensorflow 要 numpy<2.2，而 litert 这条路跑在
# numpy>=2 上。它是干什么用的见第 4 节 —— 只用来复现 2026-08-22 之前的产物。
# 这一步收尾时会把 numpy 钉在 2.1.3，也就是 tensorflow 的天花板：这违反了 onnx2tf
# 自己的 numpy==1.26.4、带一条 pip 警告、而且能跑；它因此需要的那一个 numpy-2
# workaround 在 int8_pruning.convert.export_onnx2tf 里。
ONNX2TF_ENABLED=1 ./scripts/setup_env.sh

source ./pruning-env/bin/activate

# 快速 sanity check（脚本跑完自己也会打印同样这几个版本）
python -c "import torch, torchvision, torch_pruning, effdet, pycocotools; print(torch.__version__, torch.cuda.is_available())"
```

这个脚本由环境变量驱动，且幂等：venv 已存在就直接复用，pip 就地重新解析依赖，
所以改完 `pyproject.toml` 之后重跑一遍就是更新环境的正规做法。它不激活任何东西，
自己也不需要在某个 venv 里跑：所有安装都按绝对路径走 `<venv>/bin/pip`。

```bash
# 它接受的全部开关：
#   PRUNING_ENV_ENABLED=1|0   建 ./pruning-env                          （默认 1）
#   ONNX2TF_ENABLED=1|0       建 ./onnx2tf-env                          （默认 0）
#   PY=python3.12             用来**创建** venv 的解释器（默认 python3；
#                             需要 >=3.10，而所有结果都是在 3.12 上跑出来的）
#   EXTRAS=all                pruning-env 的 extras，名字就是 families/ 下的目录名
#                             （见下面「只装需要的部分」）
#   TORCH_VERSION=2.10.0      >=2.10 是 litert 链路的硬下界；低于它会坏在哪见第 4 节
#   TORCH_INDEX=...           torch/torchvision 的 wheel index（默认 cu128）
#   VENV_DIR=...              ./pruning-env 建在哪    （按名字 gitignore）
#   ONNX2TF_VENV_DIR=...      ./onnx2tf-env 建在哪    （按名字 gitignore）

# 换一台机器时会用到的是这三个：
PY=python3.12 ./scripts/setup_env.sh                                     # 指定解释器（≥3.10）
TORCH_INDEX=https://download.pytorch.org/whl/cpu ./scripts/setup_env.sh  # 这台机器没有 CUDA
PRUNING_ENV_ENABLED=0 ONNX2TF_ENABLED=1 ./scripts/setup_env.sh           # 只建 onnx2tf 那个 venv
```

**只装需要的部分。** 默认给 `[all]` 是因为它一定能跑通。如果你只想跑通一个模型族，
extra 的名字就是 `families/` 下的目录名，每个只拉自己那一族的第三方代码：

```bash
EXTRAS='effdet,convert' ./scripts/setup_env.sh   # EfficientDet + int8 导出链路
EXTRAS='reid,convert'   ./scripts/setup_env.sh   # person Re-ID + int8 导出链路
EXTRAS='clip-rn50'      ./scripts/setup_env.sh   # 开放词表 zero-shot，只剪枝
```

`imagenet_backbones` 和 `line_seg` 故意没有 extra；除了核心依赖和 torchvision
它们什么都不需要。`reid` 还额外需要 torchreid，它是以「不带依赖」的方式安装的，
这样它就没法把 torch 降级；见第 2 节。

## 2. 下载数据集与基线权重

```bash
# 进入项目根目录并激活虚拟环境
cd <project_root>
source ./pruning-env/bin/activate

# 路径默认值集中定义在 scripts/config.sh：PROJECT_DIR、DATASET_DIR、
# MODELS_DIR、PT_PRUNED_DIR、TFLITE_DIR、EDGETPU_DIR、LOG_DIR。
# 只有需要覆盖时才 export，例如把 DATASET_DIR 指向另一块盘：
# export DATASET_DIR=/datasets
# export DATASET_DIR="$(pwd)/data/datasets"
# export MODELS_DIR="$(pwd)/outputs/models"


# 第 1 步 —— 拉数据（opt-in；默认什么都不下，只打印可用开关）。
#   每个数据集都要用各自的 *_ENABLED 显式开启（全部默认 0）：
#     COCO_ENABLED=1            COCO 2017 train/val 图像 + 标注（~25 GB）
#     COCO_MINITRAIN_ENABLED=1  + coco-minitrain 25k JSON（依赖 COCO 图像；恢复 FT 快 ~5x）
#     FLOWERS102_ENABLED=1      经 torchvision 拉 Oxford Flowers-102（~330 MB；快速 bring-up 分类器）
#     INAT_ENABLED=1            iNaturalist 2017 Plantae（~198 GB 流、不可断点续传，落盘 ~30-40 GB）
#     IMAGENET_ENABLED=1        ImageNet-1k：不下载（需注册账号）—— 仅校验 $DATASET_DIR/Imagenet_1k
#                             注意：没有任何东西保证那个目录里装的是什么。开发机上它装的是
#                             10 类的 Imagenette 子集，而 ImageFolder 照收不误 ——
#                             src/int8_pruning/data/classification.py 现在会拒绝非 1000 类的目录树，
#                             而不是默默拿 1000 路的头去训 10 个类。
#                             开启时只校验 ImageFolder 目录结构并写入其 README。
#     MARKET1501_ENABLED=1      Market-1501 行人 Re-ID 基准（~140 MB Google-Drive zip）
#   另有两个不是开关的旋钮：FORCE_README=1 覆盖已存在的数据集 README；
#   INAT_MIRROR_URL=<url> 改从 Plantae-only 镜像拉，而不是流式过 198 GB 全量档。
#   流式那条不能断点续传，镜像那条可以。
#   通用边缘 paper 默认：ImageNet-1k（分类）+ COCO（检测）：
COCO_ENABLED=1 COCO_MINITRAIN_ENABLED=1 IMAGENET_ENABLED=1 ./scripts/download_datasets.sh
# 第 2 步 —— 拉 baseline 权重 + finetune。下载全部剪枝 baseline：
#   分类（ImageNet head=1000）：mobilenetv2、efficientnet_lite0（timm）、
#                               mnasnet1_0、squeezenet1_1、resnet50
#   检测（COCO）：efficientdet_lite1（effdet）、ssdlite_mobilenetv3（torchvision）
#   默认 finetune MobileNetV2 -> iNat 2017 Plantae（2101 类，~4-5h on A6000）
#     -> outputs/models/mobilenetv2_inat.pt
#   切到快速 bring-up 路径：CLS_DATASET=flowers102 ./scripts/download_and_finetune_models.sh
#   已存在则 skip；想强制重训：FINETUNE_FORCE=1；只想下权重不 finetune：FINETUNE_ENABLED=0
#   FINETUNE_EPOCHS=N 设置 finetune 轮数（默认 20）。
#   命名规则：裸下载只保留 family 名（mobilenetv2.pt、efficientdet_lite1.pt）；
#   finetune 才加数据集后缀（mobilenetv2_inat.pt、mobilenetv2_flowers102.pt）。
#   ImageNet 那条是纯下载，所以它就停在 mobilenetv2.pt。
./scripts/download_and_finetune_models.sh

# (可选) 手动重跑默认的 iNat 2017 Plantae finetune（download 脚本已自动跑过；只在想换 epoch / 强制重训时用）：
# python families/classification/imagenet_backbones/finetune.py \
#     --dataset inat_plant \
#     --imagenet_weights $MODELS_DIR/mobilenetv2.pt \
#     --inat_train_json  $DATASET_DIR/inat2017_plant/train2017.json \
#     --inat_val_json    $DATASET_DIR/inat2017_plant/val2017.json \
#     --inat_image_root  $DATASET_DIR/inat2017_plant \
#     --num_classes 2101 --epochs 20 --amp --force

# (可选) 改走 Oxford Flowers-102 快速 bring-up 路径（~30 min on A6000）：
# python families/classification/imagenet_backbones/finetune.py \
#     --imagenet_weights $MODELS_DIR/mobilenetv2.pt \
#     --data_root $DATASET_DIR/flowers102 \
#     --epochs 20 --amp --force
```
## 3. Phase 1：结构化剪枝

默认配方是每个 part 跑 100 轮恢复 FT × 35 次运行（5 个准则 × 7 个比例）。
在 A6000 上这是多少时间：

| 运行 | 墙钟 |
|---|---|
| Part A，Flowers-102 | ~3-5 小时 |
| Part A，ImageNet-1k（默认） | 数周到数月：128 万张图 × 100 ep × 35 次 |
| Part A，iNat 2017 Plantae | ~6-8 天 |
| Part B，检测 | ~25 天，检测每轮慢 5-10 倍 |

第一遍先跑 `FINAL_EPOCHS=5 PARTS=A ./scripts/pruning.sh`，看过曲线再决定要不要延长。
走 ImageNet 这个默认值时，务必设 FINAL_EPOCHS。

```bash
# 正式 sweep：分两步独立跑 Part A（分类）和 Part B（检测，~5-10× 慢）
# 完整 100 ep 总计 ~30+ 天，建议先 30 ep 看曲线再决定扩
#
# 可调 env vars（出现在 ./scripts/pruning.sh 里）：
#   PARTS         A | B | AB                       (default AB；这里拆分跑用 A / B)
#   CLS_MODEL     mobilenetv2 | efficientnet_lite0 | mnasnet1_0 | squeezenet1_1 | resnet50
#                                                  (Part A 分类家族，default mobilenetv2；
#                                                   非 mobilenetv2 仅支持 imagenet)
#   DET_MODEL     efficientdet_lite1 | ssdlite_mobilenetv3
#                                                  (Part B 检测家族，default efficientdet_lite1)
#   DET_SCOPE     backbone | full                  (仅 effdet, default backbone。CHECKPOINTS
#                                                   以谁为分母; 'full' 是已发布档位用的口径,
#                                                   写出独立的 '_full' 文件名。
#                                                   高档位落不到目标：逐层下限饱和了，
#                                                   CHECKPOINTS=90 在 lamp 下实际是
#                                                   88.5554%、在 magnitude_l2 下是
#                                                   88.7362%，是两个不同的模型。
#                                                   比较两条 arm 之前读日志里的
#                                                   realized_full_pct，不要读 target。
#                                                   见 results/detection/target_reachability.json)
#   CHECKPOINTS   "10 20 30 40 50 60 70"           (pruning 比例 %, 空格分隔)
#   IMPORTANCES   "magnitude_l1 magnitude_l2 fpgm random lamp"  (5 个 data-free 重要性)
#   FINAL_EPOCHS  N                                (override recovery FT 轮数, default 100)
#   PROJECT_DIR   path/to/project                  (默认脚本上一级；见 scripts/config.sh)
#   DATASET_DIR   path/to/datasets                 (默认 /datasets；见 scripts/config.sh)
#   MODELS_DIR    path/to/baseline_weights         (默认 outputs/models；见 scripts/config.sh)
#   CLS_DATASET   imagenet | flowers102 | inat_plant | market1501 | line_seg
#                                                  (Part A 数据集，默认 imagenet；
#                                                   clip_rn50、relu_clip、reid_* 和
#                                                   line_seg_* 各自钉死自己的)
#   COCO_SUBSET   train2017 | minitrain | val2017  (Part B 划分，默认 train2017。
#                                                   minitrain 快 ~5x，需要
#                                                   instances_minitrain2017.json；
#                                                   val2017 只能当 smoke test 且会泄漏，
#                                                   因为 eval loader 读的也是它)
#   PRUNE_MODE    independent | iterative          (默认 independent；见下)
#
# Part-B 恢复 FT，取 EfficientDet 论文默认值（不设 = 用 worker 自己的默认）：
#   DET_LR         1e-3                            (默认 1e-3)
#   DET_WD         4e-5                            (默认 4e-5)
#   DET_SCHED      cosine | multistep              (默认 cosine)
#   DET_WARMUP     1                               (线性 warmup 的轮数；0 关闭)
#   DET_MILESTONES "60 80"                         (仅 multistep；超过 --final_epochs
#                                                   的 milestone 会被丢掉)
#
# 幂等，而且是静默的幂等：已存在的 .pt 会被 Python worker 跳过，它打印
# "Already present, skip" 然后 exit 0 —— 于是什么都没做的一次运行，看起来和全部重建
# 过的那次一模一样。要重做哪些档，就先把对应的 .pt 删掉。输出命名：
#   Part A   <baseline_name>_pruned<P>pct_<imp>.pt
#   Part B   <baseline_name>_coco-<subset>_pruned<P>pct_<imp>.pt
# subset 那一段写进 Part-B 的文件名，是为了让两个 COCO 划分不可能撞名。
#
# 有四个家族有自己的旋钮，只有选中该家族时才读：
#   CLIP_TEXT_EMB / CLIP_VAL_DIR            (clip_rn50) 蒸馏过程中的 zero-shot eval。
#                                           不设 = worker 只跟踪 cosine-to-teacher。
#   RELU_CLIP_TEXT_EMB / RELU_CLIP_VAL_DIR  (relu_clip) 同一对。
#   RELU_CLIP_GLOBAL_PRUNING=1              跨 backbone 排序，而不是逐层。在 clip_rn50 上
#                                           它把 stem 留成了 32 通道里的 3 个，记录在
#                                           results/clip_rn50/global_allocation.json。
#   RELU_CLIP_ROUND_TO=N                    把存活通道数向上取到 N 的倍数（0 = 关）。
#                                           Edge TPU 的阵列是 64 宽，所以低于 64 时更窄的
#                                           层买不到时间。这里没验证过。
#   REID_EMBED_DIM=768                      钉住已发布的头，可直接对上 768 维 gallery。
#                                           更小会先剪头、把预算让给 backbone；这时需要
#                                           REID_DISTILL_LOSS=simmat。
#   REID_DISTILL_LOSS=cosine|simmat         cosine 钉住嵌入、要求维度一致；simmat 匹配的是
#                                           batch 内的余弦相似度结构，不要求维度一致。
#   REID_GLOBAL_PRUNING=1                   跨层比较，会删掉 resnet50 的 stem
#                                           （90% 档只剩 1/64 通道），恢复保真度腰斩。
#                                           实测表在该家族的 prune.py 里。
#   REID_EXTRA_IMAGE_DIRS="d1 d2"           额外掺进蒸馏集的**无标注**行人 crop 目录；
#                                           标签由教师提供。
#   LANE_SEG_UPSTREAM=...                   上游 MicroROS-Pi5_Coral_TPU 的 training/ 目录。
#                                           它提供 lane_seg_data（数据 + 留出集评测协议）
#                                           和自采数据集；没有公开替代品。
#   LANE_SEG_EPOCHS / LANE_SEG_SAMPLES      恢复 FT 的预算。默认值就是上游的 mix stage，
#                                           30 ep × 3000 samples，也就是这些 baseline
#                                           当初训练用的配方。
#   LANE_SEG_NO_FINETUNE=1                  只剪枝、不训练。字节数和延迟只取决于**结构**，
#                                           所以这条能廉价地把片上/片外的交叉点扫出来。
#                                           这些模型是测量产物、**不是**交付物，因此要把
#                                           LANE_SEG_OUT_DIR 指到 PT_PRUNED_DIR 之外。
# CHECKPOINTS 对 line_seg_* 是另一个含义：逐层**通道比**，不是参数比。交接文档里那张实测
# 字节表就是按通道比排的，而参数量大致按 ratio^2 掉。

# Part A —— MobileNetV2 在 ImageNet-1k 上。这是默认值（CLS_DATASET=imagenet），
# 且**不需要 finetune**：下载的 head=1000 预训练模型本身就是剪枝基线。
# 想跑快速 bring-up 曲线用 CLS_DATASET=flowers102（102 类）—— 那条路径
# **需要**先把 baseline finetune 出来，见 scripts/download_and_finetune_models.sh。
# PARTS=A FINAL_EPOCHS=30 CHECKPOINTS="50 60 70 80" IMPORTANCES="magnitude_l1 magnitude_l2 fpgm random lamp" ./scripts/pruning.sh
PARTS=A FINAL_EPOCHS=30 CHECKPOINTS="50 60 70 80" IMPORTANCES="magnitude_l2" ./scripts/pruning.sh

# 默认：每一档都从稠密基线独立剪一次（PruningBench 的协议）。PRUNE_MODE=iterative
# 改成走一条轨迹 —— 剪到最小比例、恢复、存档，再把恢复后的模型继续剪到下一档 ——
# 也就是 LTH 式的 iterative magnitude pruning（Frankle & Carbin, ICLR 2019），
# 文献上报告这类做法在高比例档恢复得更好。**本仓没有测过**：results/ 下每一条阶梯
# 都是默认的 independent 模式跑的，所以它在这些家族上是否更好，这里没有数据。
# 已验证的是机制 —— 固定稠密基准、可断点续跑、单一 pruner、产物带 `_iter` 后缀，
# 所以轨迹不可能覆盖 independent 的阶梯：
# PRUNE_MODE=iterative PARTS=A CHECKPOINTS="50 60 70 80" ./scripts/pruning.sh
#
# 哪些家族实现了它，声明在 families/*/family.yaml 的 prune.prune_modes 里，用这个查：
#   python -m int8_pruning.manifest capabilities
# 对没实现的家族要这个模式，运行会在任何 worker 启动前就停下、并告诉你要去实现什么，
# 绝不会被静默丢弃。
# 两个 CLIP 族的恢复是蒸馏而不是标签，它们的教师全程钉死在稠密原模型上：LTH warm-start
# 的是学生、不是监督目标；而且 clip 的 zero-shot 是拿离线算好的文本向量打分的，
# 教师一旦漂移，这个指标就不可比了。

# Part B — EfficientDet-Lite1 backbone（检测），同样 30 ep（~10-15 天）
PARTS=B FINAL_EPOCHS=30 CHECKPOINTS="10 20 30" IMPORTANCES="magnitude_l2" ./scripts/pruning.sh

# 广度轴分类家族（仅 ImageNet；baseline 由 download_and_finetune_models.sh
# 下载，无 per-family finetune）：
CLS_MODEL=resnet50 PARTS=A FINAL_EPOCHS=5 CHECKPOINTS="30 50 70" \
    IMPORTANCES="magnitude_l2" ./scripts/pruning.sh

# 第二个检测家族 — SSDLite320-MobileNetV3-Large（torchvision，输入 320）：
DET_MODEL=ssdlite_mobilenetv3 COCO_SUBSET=minitrain PARTS=B FINAL_EPOCHS=15 \
    CHECKPOINTS="30 50 70" IMPORTANCES="magnitude_l2" ./scripts/pruning.sh

# 已发布的 EfficientDet-Lite1 档位（results/detection/full_mapping_ladder_lite1.json
# 背后的 18 个档）。四个字段都覆盖了默认值，所以裸调用复现不出来。除了上面那条
# 「已存在就跳过」，还有第二个静默的坑：日志一律写 recipe_ft.epochs=100，无论
# FINAL_EPOCHS 是多少 —— 那个字段是配方声明的默认值，不是实际跑的轮数。
# 实际跑的是 len(ft_history)，18 个已发布档位全部是 40。
PARTS=B DET_MODEL=efficientdet_lite1 COCO_SUBSET=train2017 DET_SCOPE=full \
    CHECKPOINTS="10 20 30 40 50 60 70 80 90" IMPORTANCES="lamp magnitude_l2" \
    FINAL_EPOCHS=40 ./scripts/pruning.sh


# 两个更小的预算，用于跑通而不是产出已发布的阶梯。这两条都没有对应的已提交结果：
# classification 在 results/ 下根本没有阶梯（docs/PRUNING_HAZARDS.md 第 3 节），
# 而 coco-minitrain 是缩减过的划分。
PARTS=A FINAL_EPOCHS=100 CHECKPOINTS="40 50 60 70 80" IMPORTANCES="magnitude_l2" ./scripts/pruning.sh
COCO_SUBSET=minitrain PARTS=B FINAL_EPOCHS=30 \
    CHECKPOINTS="10 20 30" IMPORTANCES="magnitude_l2" \
    ./scripts/pruning.sh
```

## 4. Phase 2：PyTorch (.pt) → TFLite int8

这是默认的导出链路，第 1 节的 `pip install -e '.[all]'` 已经把它装好了。
litert-torch + ai-edge-quantizer，per-channel int8，没有 tensorflow 也没有 onnx：
`litert_torch.convert` 直接吃 `nn.Module`，所以没有 ONNX 这一步；
`int8_pruning.convert.flatbuffer` 里的两处改写从 `ai_edge_litert` 读 schema。

```bash
source ./pruning-env/bin/activate
# 只有第 1 节装的是窄集合、不是 '.[all]' 时才需要这一行：
EXTRAS='convert' ./scripts/setup_env.sh

# Sanity check：
python -c "import torch, litert_torch, ai_edge_quantizer, ai_edge_litert; print('torch', torch.__version__, '| litert_torch', litert_torch.__version__)"
# 期望 torch 2.10.x+cu128、litert-torch 0.9.1 或更高。torch >= 2.10 是硬下界：
# 在 2.9.1 上，torchvision SSDLite 的导出会死在 torch 自己的 functionalization
# 里，报 "tensor does not have a device"。

# 单模型 smoke：产物落在 outputs/tflite_int8_litert/：
#   <name>_int8.tflite             最终 artefact（喂给 Edge TPU 用）
#   <name>_int8.size.json          大小 + i/o dtype/shape + calib 元信息
#   calib_cache/<family>_n100_seed42.npy   可复用的 calibration 样本
python src/int8_pruning/convert/tflite.py \
    --input outputs/pytorch_pruned/mobilenetv2_flowers102_pruned50pct_magnitude_l2.pt

# 批量转 outputs/pytorch_pruned/ 下所有 .pt（calibration 每个 family
# 只算一次，剪枝比例之间复用）：
./scripts/convert.sh

# 常用 override（env-var 驱动，风格和 pruning.sh 一致）：
#   GLOB='mobilenetv2_*pct_*.pt' ./scripts/convert.sh     # 按 family 过滤
#   EXTRA_ARGS='--skip-existing' ./scripts/convert.sh     # 幂等重跑
#   EXTRA_ARGS='--keep-intermediate' ./scripts/convert.sh # 保留 fp32 .tflite
#   NUM_CALIB=200 SEED=7 EXTRA_ARGS='--rebuild-calib' ./scripts/convert.sh
# 一个输入目录，每条导出路径一个输出目录，默认值都在 scripts/config.sh：
#   PT_PRUNED_DIR       outputs/pytorch_pruned        （输入，三条路径共用）
#   TFLITE_DIR          outputs/tflite_int8_litert    （litert）
#   TFLITE_ONNX2TF_DIR  outputs/tflite_int8_onnx2tf   （onnx2tf）
#   ONNX_FP32_DIR       outputs/onnx_fp32             （onnx）

# 另一条导出路径。上面的 litert-torch 是默认，也是当前每一个产物的来路；
# torch.onnx.export -> onnx2tf 是 2026-08-22 之前所有产物的来路，保留它是为了
# 能复现旧产物。用它复现，不要用它产出：走 onnx2tf，EfficientDet 只有 83/479
# 个算子映射到 Edge TPU，而 litert-torch 是 305/305。
# 它需要自己独立的 venv —— 它的 tensorflow/onnx 版本钉法和 pruning-env 共存不了
# （第 1 节写了那几条 numpy 的钉法，是硬约束）。同一个 setup 脚本就能建它，
# 挂在一个开关后面：
#   ONNX2TF_ENABLED=1 ./scripts/setup_env.sh
#   . ./onnx2tf-env/bin/activate && EXPORT_PATH=onnx2tf ./scripts/convert.sh
#   QUANT_TYPE=per-channel|per-tensor 是这条路径独有的一个旋钮（默认 per-channel）。
#   只有它读这个变量；两种设置各自的代价见 docs/PRUNING_HAZARDS.md 第 4 节。
# -> outputs/tflite_int8_onnx2tf/（单独一个目录：两条路径的字节数和
#    packaging_frac 不可比）。

# 第三条导出路径写出 fp32 ONNX 就结束。它不量化，因而不需要校准集、也不需要数据集，
# results/ 下没有任何数字经由它产生。它面向那些不读 TFLite 的工具链：TensorRT、
# OpenVINO、Core ML、RKNN。它就在这个 venv 里跑，只多一个包（'.[all]' 已经装了；
# 窄安装用 EXTRAS='onnx'）。
EXPORT_PATH=onnx ./scripts/convert.sh
# -> outputs/onnx_fp32/<name>.onnx，外加一个 <name>.onnx.json 记录 opset、输入形状
#    和输出名。检测模型保留全部十个头张量，命名 cls_0..4 / box_0..4，与 onnx2tf 一致。

# 用 Coral host 上的 runtime 验证 int8 模型能正确加载
# （Feranick build：tflite_runtime 2.16.2 + Python 3.10）：
python -c "
from tflite_runtime.interpreter import Interpreter
it = Interpreter('outputs/tflite_int8_litert/mobilenetv2_flowers102_pruned50pct_magnitude_l2_int8.tflite')
it.allocate_tensors()
print(it.get_input_details()[0])
print(it.get_output_details()[0])
"
# 期望：输入/输出 dtype 都是 int8 或 uint8；分类模型 shape 为 (1, 224, 224, 3) NHWC。

# (可选) 本地装了 edgetpu_compiler 的话可以顺手验证一下 Edge TPU 映射：
edgetpu_compiler --show_operations \
    outputs/tflite_int8_litert/mobilenetv2_flowers102_pruned50pct_magnitude_l2_int8.tflite
# 期望：**全部** op 都 map 到 Edge TPU，1 个子图，日志里没有 CPU 回落那一段。
# 编译器需要的三处改写（INT64 paddings、PADV2 -> PAD、恒等 GATHER_ND）转换器
# 已经做掉了 —— 见 src/int8_pruning/convert/flatbuffer.py。

# 检测类 int8 模型暴露的是 per-level 原始 head 张量（EfficientDet-Lite1 为 10
# 个输出，SSDLite 为 12 个）；anchor 解码 + NMS 在 CPU 上跑，与 PyTorch
# baseline 的解码逐位一致。测 COCO mAP / 单图 demo：
python families/detection/effdet/eval.py eval \
    --tflite outputs/tflite_int8_litert/efficientdet_lite1_coco-minitrain_pruned0pct_int8.tflite \
    --coco-root "$DATASET_DIR/coco"
python families/detection/ssdlite/eval.py eval \
    --tflite outputs/tflite_int8_litert/ssdlite_mobilenetv3_coco-minitrain_pruned0pct_int8.tflite \
    --coco-root "$DATASET_DIR/coco"
# 单图 demo：把 `eval ...` 换成 `predict --image some.jpg --draw out.jpg`
```

## 5. Phase 3（可选）：TFLite int8 → Edge TPU (Coral USB)

到这里为止的每一步都是流水线本体，既不需要 Coral 也不需要 `edgetpu_compiler`：
第 3、4 节在一台普通机器上就能把一个 checkpoint 走到成品 int8 `.tflite`。本节是可选
后端，没有任何阶段会串进来。`compile_edgetpu.sh` 是需要你自己敲的驱动，
编译器缺席时它 exit 1，因为编译就是它唯一的职责。而在那些「Edge TPU 只是顺带一步」
的地方，缺席会被明说出来：`scripts/deliver.sh` 打印 `[SKIPPED] edgetpu/` 并只打包
int8 产物，`families/re-identification/reid/report.py` 给编译那条腿记 `skipped`。

没有这块硬件就可以在这里停下。需要它的代码全部收在
`src/int8_pruning/backends/edgetpu/` 里，包内其余部分都不 import 它。


```bash
# 一次性安装 Edge TPU 编译器（见 https://coral.ai/docs/edgetpu/compiler/）：
#   curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
#   echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
#       | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
#   sudo apt-get update && sudo apt-get install edgetpu-compiler
edgetpu_compiler --version

# 批量编译 outputs/tflite_int8_litert/ 下所有 *_int8.tflite。每次调用 compile_edgetpu.sh
# 会在 outputs/edgetpu/ 下新建一个 run 目录，里面放：
# *_edgetpu.tflite 产物、编译器 .log，以及对应的 *_int8.size.json sidecar
# （这样不用重新打开 .tflite 就能知道 I/O dtype/shape）。
./scripts/compile_edgetpu.sh
# -> outputs/edgetpu/single_<TS>/<stem>_int8_edgetpu.tflite

# 子集过滤（例如只编译 MobileNetV2 magnitude_l2 那批）：
GLOB='mobilenetv2_*pct_magnitude_l2_int8.tflite' ./scripts/compile_edgetpu.sh

# 只想看 op 映射、不要真的产出？加 SHOW_OPS=1，会在产物旁多输出一个
# <stem>.show_ops.txt：
SHOW_OPS=1 GLOB='mobilenetv2_flowers102_pruned50pct_magnitude_l2_int8.tflite' \
    ./scripts/compile_edgetpu.sh

# 「2 个模型组合」路径：一次 invocation 同时编两个模型，让它们在 Coral USB 上
# 共享 on-chip 参数 cache。实测过的组合与各自的片上/片外划分见
# docs/edgetpu_cocompile.md。stem 不带 _int8.tflite 后缀，两个文件都必须已经在
# TFLITE_DIR 里。
PAIR="mobilenetv2_flowers102_pruned50pct_magnitude_l2 efficientdet_lite1_coco-train2017_pruned10pct_magnitude_l2" \
    ./scripts/compile_edgetpu.sh
# -> outputs/edgetpu/pair_<stemA>__<stemB>_<TS>/

# 常用 override（env-var 驱动，风格和 convert.sh 一致）：
#   TFLITE_DIR=path/to/int8_models ./scripts/compile_edgetpu.sh
#   EXTRA_ARGS='-m 13' ./scripts/compile_edgetpu.sh       # 锁定 runtime 版本
#   EXTRA_ARGS='-a'    ./scripts/compile_edgetpu.sh       # 打开 multi-subgraph
#   EDGETPU_COMPILER=/opt/edgetpu/bin/edgetpu_compiler ./scripts/compile_edgetpu.sh

# 上机 sanity check（在 Coral host 上跑，不是这台机器）：
#   from pycoral.utils.edgetpu import make_interpreter
#   it = make_interpreter('mobilenetv2_..._int8_edgetpu.tflite')
#   it.allocate_tensors()
```

## 6. 不自己跑，直接取已发布的产物

上面每一节都是在造模型。`scripts/fetch_deliverables.sh` 下载已经发布出去的那些，
下游 MicroROS-Pi5_Coral_TPU 仓走的就是这条路。模型二进制永远不进 git：它们放在一个
HuggingFace 仓里，每个文件在落盘时都对着
[`results/deliverables.sha256`](../results/deliverables.sha256) 校验一遍。
sha256 已经对上的文件会被跳过，所以下载中断后重跑一次就继续。只需要 curl（或 wget）
和 sha256sum，没有别的依赖。

两个 tier，因为 `.pt` checkpoint 占了 54% 的体积，而只想**跑**模型的场景一个都用不上：

| tier | 内容 | 体积 |
|---|---|---|
| `deploy`（默认） | `models_int8_tflite/` + `edgetpu/` + `reference/` | 162 件，354 MB |
| `full` | 全部，多出 `checkpoints_pytorch/` | 202 件，763 MB |

想重新剪枝、继续微调、或者查看剪完的结构，取 `full`。它的 0% 那一行是未剪枝的基线，
留着是为了让这个包自己就能画出一条完整的比例曲线。两个 tier 都覆盖两个已发布家族。

```bash
./scripts/fetch_deliverables.sh                 # deploy tier，两个家族
./scripts/fetch_deliverables.sh --tier full     # 加上 .pt checkpoint
./scripts/fetch_deliverables.sh effdet_lite1_pruning   # 只取一个家族
./scripts/fetch_deliverables.sh --list
./scripts/fetch_deliverables.sh --verify        # 检查本地已有的
./scripts/fetch_deliverables.sh --force         # 即使校验通过也重下
```

HuggingFace 仓还是私有期间，还需要一个 token，来自 `hf auth login` 或环境变量
`HF_TOKEN=...`。没有它每个请求都返回 401。

checkpoint 是整模块 pickle，`torch.load(weights_only=False)` 重建的是剪完的模块图，
所以那些类必须能 import 到。它们来自 pip（`effdet==0.4.1`、`timm==1.0.27`、
`omegaconf==2.3.0`），不来自本仓。精确的版本钉法和加载调用写在
`checkpoints_pytorch/READ_THIS_FIRST.md` 里，脚本每次都会把它一起拉下来。

manifest 是 `deliverables/` 的子集，不是它的哈希：那棵树里其余的包是下游交接件，
不发布。哪些够格发布、凭什么够格，在 `scripts/manifest_deliverables.sh` 里。
