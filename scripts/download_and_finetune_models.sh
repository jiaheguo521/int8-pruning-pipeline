#!/usr/bin/env bash
# download_and_finetune_models.sh — pull every prune baseline, then finetune the
# MobileNetV2 classifier onto CLS_DATASET. Run download_datasets.sh first.
#
# Usage:
#   ./scripts/download_and_finetune_models.sh                        # imagenet: no finetune
#   CLS_DATASET=flowers102 ./scripts/download_and_finetune_models.sh
#   FINETUNE_ENABLED=0 ./scripts/download_and_finetune_models.sh     # weights only
# Env vars and the file-naming rule: docs/SETUP.md section 2.

set -euo pipefail

# Shared paths + require_venv live in scripts/config.sh.
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
require_venv

CLS_DATASET="${CLS_DATASET:-imagenet}"      # classification dataset; same toggle as pruning.sh
FINETUNE_ENABLED="${FINETUNE_ENABLED:-1}"   # auto-run the finetune at the end (no-op for imagenet)
FINETUNE_FORCE="${FINETUNE_FORCE:-0}"       # pass --force to the finetune script (re-train even if .pt exists)
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-20}"    # epochs for the auto finetune step

# Derive baseline name + class count from CLS_DATASET. The ImageNet path has no user-driven
# finetune, so its baseline is the unsuffixed `mobilenetv2.pt`; the other two carry a suffix.
case "$CLS_DATASET" in
    imagenet)    CLS_BASELINE_NAME="mobilenetv2";            CLS_NUM_CLASSES=1000 ;;
    flowers102)  CLS_BASELINE_NAME="mobilenetv2_flowers102"; CLS_NUM_CLASSES=102 ;;
    inat_plant)  CLS_BASELINE_NAME="mobilenetv2_inat";       CLS_NUM_CLASSES=2101 ;;
    *) echo "[ERROR] CLS_DATASET must be imagenet, flowers102 or inat_plant (got: $CLS_DATASET)" >&2; exit 1 ;;
esac

echo "DATASET_DIR      = $DATASET_DIR"
echo "MODELS_DIR       = $MODELS_DIR"
echo "CLS_DATASET      = $CLS_DATASET  (baseline=$CLS_BASELINE_NAME, num_classes=$CLS_NUM_CLASSES)"
echo "FINETUNE_ENABLED = $FINETUNE_ENABLED   (0 = skip finetune, 1 = run after downloads)"
echo "FINETUNE_FORCE   = $FINETUNE_FORCE   (1 = retrain even if $CLS_BASELINE_NAME.pt exists)"
echo "FINETUNE_EPOCHS  = $FINETUNE_EPOCHS"
echo

mkdir -p "$MODELS_DIR"

# ---------- 1. Pre-trained baseline weights ----------
# torchvision and effdet download from their own caches; copies land in $MODELS_DIR so the
# pipeline resolves them as outputs/models/<family>.pt.
echo "=== Pre-trained weights ==="

MODELS_DIR="$MODELS_DIR" PROJECT_DIR="$PROJECT_DIR" python - <<'PY'
import os, sys
from pathlib import Path
import torch

models_dir = Path(os.environ["MODELS_DIR"])
models_dir.mkdir(parents=True, exist_ok=True)

# Saved as the FULL nn.Module, NOT a state_dict: that is what the pruning workers
# torch.load(weights_only=False) back. Re-loading one elsewhere needs torchvision / effdet.

# --- MobileNetV2 (ImageNet pretrained from torchvision) ---
# The raw download (head 1280->1000) IS the prune baseline for CLS_DATASET=imagenet, and the
# starting point for finetune.py on flowers102 (->102) and inat_plant (->2101).
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
out = models_dir / "mobilenetv2.pt"
if out.exists():
    print(f"[skip] {out}")
else:
    print("Downloading MobileNetV2 ImageNet weights ...")
    m = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
    torch.save(m, out)
    print(f"  -> {out}  (full nn.Module)")

# --- Extra classification baselines (ImageNet pretrained, head=1000) ---
# Pure downloads, so they keep the bare family name; Phase-1 outputs carry _imagenet because
# recovery FT runs on ImageNet.

# MNASNet-1.0, SqueezeNet1.1, ResNet-50 ship in torchvision.
from torchvision.models import (
    mnasnet1_0, MNASNet1_0_Weights,
    squeezenet1_1, SqueezeNet1_1_Weights,
    resnet50, ResNet50_Weights,
)
_torchvision_cls = [
    ("mnasnet1_0",    lambda: mnasnet1_0(weights=MNASNet1_0_Weights.IMAGENET1K_V1)),
    ("squeezenet1_1", lambda: squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)),
    ("resnet50",      lambda: resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)),
]
for stem, ctor in _torchvision_cls:
    out = models_dir / f"{stem}.pt"
    if out.exists():
        print(f"[skip] {out}")
    else:
        print(f"Downloading {stem} ImageNet weights ...")
        torch.save(ctor(), out)
        print(f"  -> {out}  (full nn.Module)")

# EfficientNet-Lite0 lives in timm, already a dep of effdet; mirror effdet's pip-on-ImportError guard.
out = models_dir / "efficientnet_lite0.pt"
if out.exists():
    print(f"[skip] {out}")
else:
    try:
        import timm
    except ImportError:
        import subprocess
        print("'timm' not found — installing via pip ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "timm"])
        import timm  # retry
    print("Downloading EfficientNet-Lite0 ImageNet weights (timm) ...")
    m = timm.create_model("tf_efficientnet_lite0", pretrained=True)
    torch.save(m, out)
    print(f"  -> {out}  (full nn.Module)")

# --- CLIP RN50 image tower (OpenAI CLIP visual tower, timm) — VLM family ---
# Keep the default num_classes=1024: it retains the projection into the 1024-dim CLIP
# embedding space. num_classes=0 drops it and returns 2048-dim pre-projection features that
# do NOT align with the CLIP text tower (default head gives 98% Imagenette zero-shot vs
# open_clip RN50 text; num_classes=0 breaks alignment). forward(x) returns the RAW
# embedding, L2-norm being applied CPU-side so the int8 TPU graph stays clean.
out = models_dir / "clip_rn50.pt"
if out.exists():
    print(f"[skip] {out}")
else:
    import timm  # confirmed dep (used by the effnet-lite0 block above + effdet)
    print("Downloading CLIP RN50 image tower (timm resnet50_clip, OpenAI) ...")
    m = timm.create_model("resnet50_clip", pretrained=True)  # default num_classes=1024
    torch.save(m, out)
    print(f"  -> {out}  (full nn.Module, 1024-dim CLIP embedding)")

# --- SSDLite-MobileNetV3-Large (COCO pretrained, torchvision detection) ---
# The second prunable detection family, consumed directly by families/detection/ssdlite/prune.py.
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights,
)
out = models_dir / "ssdlite_mobilenetv3.pt"
if out.exists():
    print(f"[skip] {out}")
else:
    print("Downloading SSDLite-MobileNetV3-Large COCO weights ...")
    det = ssdlite320_mobilenet_v3_large(
        weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1)
    torch.save(det, out)
    print(f"  -> {out}  (full torchvision SSD nn.Module)")

# --- EfficientDet-Lite1/Lite2 (COCO pretrained, rwightman/efficientdet-pytorch) ---
# Raw download, consumed directly by families/detection/effdet/prune.py. Lite1 (384) is what
# the downstream robot deploys, since 2026-08-22; Lite2 (448) has a complete float ladder,
# int8 / Edge TPU artifacts and int8 mAP column, with only its on-device latency unmeasured.
# Same upstream checkpoints Coral's *_ptq.tflite came from.
for _stem in ("efficientdet_lite1", "efficientdet_lite2"):
    out = models_dir / f"{_stem}.pt"
    if out.exists():
        print(f"[skip] {out}")
        continue
    try:
        from effdet import create_model
    except ImportError:
        import subprocess
        print("'effdet' not found — installing via pip ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "effdet"])
        from effdet import create_model  # retry
    print(f"Downloading {_stem} COCO weights ...")
    det = create_model(f"tf_{_stem}", pretrained=True)
    torch.save(det, out)
    print(f"  -> {out}  (full EfficientDet nn.Module)")

# --- Person Re-ID embedders (Torchreid, Market-1501 Model Zoo) — reid_* family ---
# Torchreid's setup pins old torch, so it MUST be installed --no-deps; a plain install would
# downgrade the torch/torchvision/tensorflow stack. The "wrong tensor" trap: in EVAL mode a
# torchreid model's forward() returns the POOLED FEATURE EMBEDDING (osnet 512-d,
# mobilenetv2_x1_0 1280-d), NOT the 751-way ID classifier logits, so the assert below
# refuses anything that is not a single [1, D] tensor with D != 751.
_REID_MARKET1501 = [   # BatchNorm2d-only variants; the _ibn/_ain InstanceNorm ones are
                       # excluded on purpose, InstanceNorm not being Edge-TPU safe
    # (output stem,                       torchreid name,     Market-1501 gdrive id)
    ("reid_osnet_x0_5_market1501",        "osnet_x0_5",       "1PLB9rgqrUM7blWrg4QlprCuPT7ILYGKT"),
    ("reid_osnet_x0_75_market1501",       "osnet_x0_75",      "1ozRaDSQw_EQ8_93OUmjDbvLXw9TnfPer"),
    ("reid_mobilenetv2_x1_0_market1501",  "mobilenetv2_x1_0", "18DgHC2ZJkjekVoqBWszD8_Xiikz-fewp"),
]
if all((models_dir / f"{stem}.pt").exists() for stem, _, _ in _REID_MARKET1501):
    for stem, _, _ in _REID_MARKET1501:
        print(f"[skip] {models_dir / (stem + '.pt')}")
else:
    try:
        import torchreid
    except ImportError:
        import subprocess
        print("'torchreid' not found — installing --no-deps (protects the torch/TF stack) ...")
        # Cython is a BUILD-time dep for torchreid's rank_cylib extension (source install).
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "Cython"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
                               "git+https://github.com/KaiyangZhou/deep-person-reid.git"])
        # torchreid's light runtime imports (cv2/yacs/gdown/... — never touch torch).
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                               "gdown", "yacs", "opencv-python-headless", "h5py",
                               "imageio", "tabulate"])
        import torchreid  # retry
    try:
        import gdown
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "gdown"])
        import gdown  # retry

    weights_cache = models_dir / "reid_weights_cache"
    weights_cache.mkdir(parents=True, exist_ok=True)

    for stem, reid_name, gdrive_id in _REID_MARKET1501:
        out = models_dir / f"{stem}.pt"
        if out.exists():
            print(f"[skip] {out}")
            continue
        print(f"Building {reid_name} + loading Market-1501 weights ...")
        # num_classes=751 = Market-1501 train identities, the shape the checkpoint's classifier needs.
        model = torchreid.models.build_model(
            name=reid_name, num_classes=751, loss="softmax", pretrained=False)
        wpath = weights_cache / f"{stem}.pth"
        if not wpath.exists():
            gdown.download("https://drive.google.com/uc?id=" + gdrive_id,
                           str(wpath), quiet=False)
        torchreid.utils.load_pretrained_weights(model, str(wpath))
        model.eval()

        # Verify we are exporting the embedding, not the ID-classifier head.
        with torch.no_grad():
            probe = model(torch.randn(1, 3, 256, 128))
        if isinstance(probe, (tuple, list)):
            raise RuntimeError(
                f"{stem}: eval forward returned {type(probe).__name__} of len "
                f"{len(probe)}, expected a single embedding tensor — the model is "
                f"likely in training mode or has a triplet head.")
        if probe.ndim != 2 or probe.shape[0] != 1:
            raise RuntimeError(f"{stem}: unexpected embedding shape {tuple(probe.shape)}")
        emb_dim = int(probe.shape[1])
        if emb_dim == 751:
            raise RuntimeError(
                f"{stem}: forward returned [1, 751] — that is the ID CLASSIFIER "
                f"head, not the embedding. Refusing to save the wrong tensor.")
        torch.save(model, out)
        print(f"  -> {out}  (full nn.Module, {emb_dim}-d Re-ID embedding)")

# --- Youtu Re-ID baseline_lite (TencentYoutuResearch/PersonReID-YouReID) — reid_youtu_lite ---
# The ONLY reid_* baseline that gets PRUNED; the torchreid embedders above are convert-only.
# youtu_reid_baseline_lite == opencv_zoo's person_reid_youtu_2021nov. Trained on FOUR
# datasets (market1501 + Duke + MSMT17 + CUHK03; the released fc_layer is Linear(768, 3246)
# over the merged ID space), so the file is deliberately UNSUFFIXED -- `_market1501` would be
# a lie about the weights, while Phase-1 outputs DO carry it because the recovery
# distillation runs on Market crops. Architecture and the checkpoint's two upstream quirks
# (split_bn's 4-way bn_list, the DataParallel `module.` prefix) are in youreid_model.py;
# cos(vendored fp32, official opencv_zoo ONNX) = 0.9999999 on real Market crops.
_YOUTU_STEM = "reid_youtu_lite"
_YOUTU_GDRIVE_ID = "1l-8Lj9OPs4D6qKGAljbJgZuxGvENkDjl"
out = models_dir / f"{_YOUTU_STEM}.pt"
if out.exists():
    print(f"[skip] {out}")
else:
    try:
        import gdown
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "gdown"])
        import gdown  # retry
    # youreid_model is a family-local module, and int8_pruning.convert.common owns the list
    # of those: it puts the same three directories on sys.path to unpickle full-module .pt
    # files. Deriving the path here instead is what broke when families/ was restructured.
    from int8_pruning.convert.common import enable_full_module_unpickling
    enable_full_module_unpickling()
    from youreid_model import build_youtu_lite

    weights_cache = models_dir / "reid_weights_cache"
    weights_cache.mkdir(parents=True, exist_ok=True)
    wpath = weights_cache / f"{_YOUTU_STEM}.pth"
    if not wpath.exists():
        print("Downloading YouReID youtu_reid_baseline_lite weights (~226 MB) ...")
        gdown.download("https://drive.google.com/uc?id=" + _YOUTU_GDRIVE_ID,
                       str(wpath), quiet=False)
    print("Building reid_youtu_lite + loading four-source weights ...")
    model = build_youtu_lite(wpath)  # strict load; collapses split_bn -> plain BN

    # Same wrong-tensor gate as the torchreid embedders, plus an InstanceNorm check (IBN would not compile).
    with torch.no_grad():
        probe = model(torch.randn(1, 3, 256, 128))
    if isinstance(probe, (tuple, list)):
        raise RuntimeError(
            f"{_YOUTU_STEM}: eval forward returned {type(probe).__name__}, expected "
            f"a single embedding tensor.")
    if tuple(probe.shape) != (1, 768):
        raise RuntimeError(
            f"{_YOUTU_STEM}: expected a [1, 768] embedding, got {tuple(probe.shape)}")
    if any(isinstance(m, torch.nn.InstanceNorm2d) for m in model.modules()):
        raise RuntimeError(
            f"{_YOUTU_STEM}: InstanceNorm present — that is the resnet101_ibn_a "
            f"(medium/large) path and it does NOT compile for Edge TPU.")
    torch.save(model, out)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  -> {out}  (full nn.Module, 768-d Re-ID embedding, {n_params/1e6:.2f}M params)")
PY

# ---------- 2. Auto-finetune MobileNetV2 -> CLS_DATASET ----------
# The finetune script skips when the output .pt exists (without --force); FINETUNE_ENABLED=0 skips all.
OUT_PT="$MODELS_DIR/$CLS_BASELINE_NAME.pt"

# Per-dataset finetune args (mirrors scripts/pruning.sh Part A dispatch).
cls_finetune_args=(--dataset "$CLS_DATASET")
case "$CLS_DATASET" in
    flowers102)
        cls_finetune_args+=(--data_root "$DATASET_DIR/flowers102")
        ;;
    inat_plant)
        inat="$DATASET_DIR/inat2017_plant"
        cls_finetune_args+=(
            --inat_train_json "$inat/train2017.json"
            --inat_val_json   "$inat/val2017.json"
            --inat_image_root "$inat"
        )
        ;;
esac

if [[ "$CLS_DATASET" == "imagenet" ]]; then
    # ImageNet needs no finetune: the pretrained download IS the prune baseline, and step 1 wrote it.
    echo "=== ImageNet: no finetune (baseline = pretrained MobileNetV2, head=1000) ==="
    if [[ ! -f "$OUT_PT" ]]; then
        echo "[ERROR] $OUT_PT missing — the weights download step above should have created it." >&2
        exit 1
    fi
    echo "[skip] imagenet baseline = pretrained MobileNetV2 ($OUT_PT); no finetune."
elif [[ "$FINETUNE_ENABLED" == "1" ]]; then
    echo "=== Auto-finetune: MobileNetV2 -> $CLS_DATASET ==="
    if [[ -f "$OUT_PT" && "$FINETUNE_FORCE" != "1" ]]; then
        echo "[skip] $OUT_PT already exists (FINETUNE_FORCE=1 to re-train)"
    else
        # For iNat, fail early with a clear hint rather than letting the dataset loader crash mid-run.
        if [[ "$CLS_DATASET" == "inat_plant" ]]; then
            if [[ ! -f "$DATASET_DIR/inat2017_plant/train2017.json" || ! -f "$DATASET_DIR/inat2017_plant/val2017.json" ]]; then
                echo "[ERROR] CLS_DATASET=inat_plant but train2017.json/val2017.json are missing under" >&2
                echo "        $DATASET_DIR/inat2017_plant/." >&2
                echo "        Run (INAT_ENABLED=1) ./scripts/download_datasets.sh first." >&2
                exit 1
            fi
        fi
        force_flag=()
        [[ "$FINETUNE_FORCE" == "1" ]] && force_flag=(--force)
        echo "[run ] families/classification/imagenet_backbones/finetune.py ${cls_finetune_args[*]} --epochs $FINETUNE_EPOCHS --amp ${force_flag[*]}"
        python "$PROJECT_DIR/families/classification/imagenet_backbones/finetune.py" \
            --imagenet_weights "$MODELS_DIR/mobilenetv2.pt" \
            "${cls_finetune_args[@]}" \
            --epochs "$FINETUNE_EPOCHS" --amp "${force_flag[@]}"
    fi
else
    echo "=== Auto-finetune SKIPPED (FINETUNE_ENABLED=0) ==="
    echo "      To run manually:"
    echo "        python families/classification/imagenet_backbones/finetune.py \\"
    echo "            --imagenet_weights $MODELS_DIR/mobilenetv2.pt \\"
    echo "            ${cls_finetune_args[*]} \\"
    echo "            --epochs $FINETUNE_EPOCHS --amp"
fi

echo
echo "Done."
echo "  Weights  : $MODELS_DIR"
if [[ -f "$OUT_PT" ]]; then
    echo "  $CLS_DATASET baseline : $OUT_PT"
fi
echo
echo "Next steps:"
if [[ ! -f "$OUT_PT" ]]; then
    echo "  1. Fine-tune MobileNetV2 head to $CLS_DATASET ($CLS_NUM_CLASSES classes):"
    echo "       python families/classification/imagenet_backbones/finetune.py \\"
    echo "           --imagenet_weights $MODELS_DIR/mobilenetv2.pt \\"
    echo "           ${cls_finetune_args[*]} \\"
    echo "           --epochs $FINETUNE_EPOCHS --amp"
    echo "     Output: outputs/models/$CLS_BASELINE_NAME.pt (full nn.Module)"
    echo
fi
echo "  - Verify EfficientDet-Lite1 arch matches Coral's Lite1 before pruning"
echo "     (anchors, input size 384, num_classes=90 must match the deployed model)."
echo
echo "  - (iNat path) Fine-tune MobileNetV2 head to iNat 2017 Plantae (2101 species, ~4-5h):"
echo "     (requires INAT_ENABLED=1 ./scripts/download_datasets.sh first)"
echo "       python families/classification/imagenet_backbones/finetune.py \\"
echo "           --dataset inat_plant \\"
echo "           --imagenet_weights $MODELS_DIR/mobilenetv2.pt \\"
echo "           --inat_train_json  $DATASET_DIR/inat2017_plant/train2017.json \\"
echo "           --inat_val_json    $DATASET_DIR/inat2017_plant/val2017.json \\"
echo "           --inat_image_root  $DATASET_DIR/inat2017_plant \\"
echo "           --num_classes 2101 --epochs 20"
echo "     Output: outputs/models/mobilenetv2_inat.pt (full nn.Module)"
echo
echo "Notes:"
echo "  - Flowers-102 has 102 classes (Nilsback & Zisserman 2008). The class count"
echo "    comes from DATASET_NUM_CLASSES in src/int8_pruning/data/classification.py."
echo "  - iNat 2017 Plantae has 2,101 species in the official split (per the"
echo "    iNaturalist 2017 dataset card). Override with --num_classes if your"
echo "    annotation subset has a different category count."
