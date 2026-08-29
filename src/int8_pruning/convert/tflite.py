#!/usr/bin/env python3
"""Generic PyTorch -> Edge TPU int8 TFLite converter, over two export paths.

    --export-path litert    (default)  .pt -> float tflite -> int8
                            litert_torch.convert + ai-edge-quantizer, per-channel.
                            No ONNX step, no tensorflow. See int8_pruning.convert.export_litert.

    --export-path onnx2tf              .pt -> .onnx -> int8 tflite
    --export-path onnx                 .pt -> fp32 .onnx, and stops there:
                                       the interchange format for the
                                       toolchains that do not read TFLite.
                                       Not quantized, and not measured.
                            torch.onnx.export + onnx2tf. The path every artifact
                            in this repo dated before 2026-08-22 was built with,
                            kept so those can be reproduced and compared against.
                            Needs `pip install -e '.[onnx2tf]'`. See
                            int8_pruning.convert.export_onnx2tf.

Both paths share this file's calibration loaders, model registry and per-file
orchestration; each writes its own output directory so the two never interleave
in one folder (scripts/config.sh: TFLITE_DIR vs TFLITE_ONNX2TF_DIR).

litert is the default and should stay it. Through onnx2tf, EfficientDet mapped
only **83 of 479 operators** to the Edge TPU -- onnx2tf lowered effdet's BiFPN
fusion to a rank-5 CONCAT/SUM and timm's SAME-pool to PADV2, and neither is
mappable, so the TPU subgraph stopped at the backbone. Through litert-torch the
same network maps **305 of 305**, every rung, nothing on the CPU. onnx2tf also
defaulted this repo's classification and ssdlite families to per-tensor weights
while the DSD 2026 protocol specifies per-channel; both paths here default to
per-channel now, and `--quant-type per-tensor` (onnx2tf only) is what reproduces
those retired artifacts. See docs/PRUNING_HAZARDS.md.

Model family is auto-detected from filename via MODEL_REGISTRY, which is built
from families/*/family.yaml. Add a family by adding a convert entry there plus a
calibration-loader function below.
"""
import argparse
import fnmatch
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from int8_pruning import manifest
from int8_pruning.convert.common import ModelFamily
from int8_pruning.report.tflite_size import measure as measure_tflite_size

# Calibration loaders

def _normalize_nhwc(arr_uint8: np.ndarray, mean, std) -> np.ndarray:
    """(N, H, W, 3) uint8 -> (N, H, W, 3) float32, ImageNet-normalized."""
    x = arr_uint8.astype(np.float32) / 255.0
    x = (x - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return x.astype(np.float32)


def _calib_flowers102_train(num, input_h, input_w, mean, std, seed, dataset_root):
    """100 random Flowers-102 train images, NHWC float32, normalized.

    Uses the eval transform (Resize + CenterCrop, no random augment) so the
    calibration set is deterministic given (num, seed).
    """
    input_size = input_h  # square family: input_w == input_h
    from PIL import Image
    from torchvision.datasets import Flowers102

    ds = Flowers102(root=str(dataset_root), split="train", transform=None,
                    download=False)
    rng = random.Random(seed)
    idx = rng.sample(range(len(ds)), k=min(num, len(ds)))

    resize_target = int(round(input_size * 256 / 224))
    out = np.zeros((len(idx), input_size, input_size, 3), dtype=np.uint8)
    for i, j in enumerate(idx):
        img, _ = ds[j]  # PIL.Image
        if img.mode != "RGB":
            img = img.convert("RGB")
        # Resize shorter side to resize_target, then center crop.
        w, h = img.size
        if w < h:
            new_w, new_h = resize_target, int(round(h * resize_target / w))
        else:
            new_w, new_h = int(round(w * resize_target / h)), resize_target
        img = img.resize((new_w, new_h), Image.BILINEAR)
        left = (new_w - input_size) // 2
        top = (new_h - input_size) // 2
        img = img.crop((left, top, left + input_size, top + input_size))
        out[i] = np.asarray(img, dtype=np.uint8)
    return _normalize_nhwc(out, mean, std)


def _imagenet_train_files(dataset_root, num, seed):
    """`num` deterministic random JPEG paths from <dataset_root>/train/<wnid>/.

    A direct glob, not a full ImageFolder build. Shared by the two ImageNet
    calibration loaders; what they do NOT share is the resize/crop, which is a
    per-family fact and stays visible in each loader.
    """
    train_dir = Path(dataset_root) / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"ImageNet train directory not found: {train_dir}\n"
            f"Expected layout: <dataset-dir>/Imagenet_1k/train/<wnid>/*.JPEG"
        )
    files = []
    for ext in ("*.JPEG", "*.jpeg", "*.jpg"):
        files.extend(train_dir.glob(f"*/{ext}"))
    files.sort()
    if not files:
        raise FileNotFoundError(f"No JPEG files under {train_dir}/<wnid>/")
    rng = random.Random(seed)
    return rng.sample(files, k=min(num, len(files)))


def _calib_imagenet_train(num, input_h, input_w, mean, std, seed, dataset_root):
    """`num` random ImageNet-1k train images, NHWC float32, normalized.

    Uses the eval transform this repo's ImageNet families train and evaluate
    with: resize the shorter side to 256/224 of the input size (BILINEAR), then
    center crop. Deterministic given (num, seed). The train-tree glob runs once
    and the result is cached by the caller (build_calibration), so the listing
    cost is paid at most once.
    """
    input_size = input_h  # square family: input_w == input_h
    from PIL import Image

    picks = _imagenet_train_files(dataset_root, num, seed)
    resize_target = int(round(input_size * 256 / 224))
    out = np.zeros((len(picks), input_size, input_size, 3), dtype=np.uint8)
    for i, p in enumerate(picks):
        with Image.open(p) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            # Resize shorter side to resize_target, then center crop.
            w, h = img.size
            if w < h:
                new_w, new_h = resize_target, int(round(h * resize_target / w))
            else:
                new_w, new_h = int(round(w * resize_target / h)), resize_target
            img = img.resize((new_w, new_h), Image.BILINEAR)
            left = (new_w - input_size) // 2
            top = (new_h - input_size) // 2
            img = img.crop((left, top, left + input_size, top + input_size))
            out[i] = np.asarray(img, dtype=np.uint8)
    return _normalize_nhwc(out, mean, std)


def _calib_imagenet_train_reluclip(num, input_h, input_w, mean, std, seed,
                                   dataset_root):
    """As above, but with the relu_clip family's OWN eval transform.

    Not a stylistic difference. The published relu-clip preprocessing is
    `Resize(224, BICUBIC) + CenterCrop(224)` -- no 256/224 upscale -- and its
    `preprocessing.json` calls out the two rounding details that "silently cost
    accuracy if guessed", both of which go the opposite way from the loader
    above: torchvision TRUNCATES the resized long side (`int`, not `round`) and
    ROUNDS the crop offset (`int(round(...))`, not floor). Calibrating on a
    tighter, bilinear crop would set activation ranges from pixels the deployed
    model never sees.
    """
    input_size = input_h
    from PIL import Image

    picks = _imagenet_train_files(dataset_root, num, seed)
    out = np.zeros((len(picks), input_size, input_size, 3), dtype=np.uint8)
    for i, p in enumerate(picks):
        with Image.open(p) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            w, h = img.size
            if w < h:
                new_w, new_h = input_size, int(h * input_size / w)
            else:
                new_w, new_h = int(w * input_size / h), input_size
            img = img.resize((new_w, new_h), Image.BICUBIC)
            left = int(round((new_w - input_size) / 2.0))
            top = int(round((new_h - input_size) / 2.0))
            img = img.crop((left, top, left + input_size, top + input_size))
            out[i] = np.asarray(img, dtype=np.uint8)
    return _normalize_nhwc(out, mean, std)


def _calib_coco_val2017(num, input_h, input_w, mean, std, seed, dataset_root):
    """100 random COCO val2017 images, NHWC float32, normalized.

    Direct JPG glob — no pycocotools. val2017 over train2017: calibration
    needs pixel statistics only, val2017 is ~1 GB vs ~18 GB.
    """
    input_size = input_h  # square family: input_w == input_h
    from PIL import Image

    img_dir = Path(dataset_root) / "val2017"
    if not img_dir.is_dir():
        raise FileNotFoundError(
            f"COCO val2017 directory not found: {img_dir}\n"
            f"Expected layout: <dataset-dir>/coco/val2017/*.jpg"
        )
    jpgs = sorted(img_dir.glob("*.jpg"))
    if not jpgs:
        raise FileNotFoundError(f"No .jpg files in {img_dir}")

    rng = random.Random(seed)
    picks = rng.sample(jpgs, k=min(num, len(jpgs)))

    out = np.zeros((len(picks), input_size, input_size, 3), dtype=np.uint8)
    for i, p in enumerate(picks):
        with Image.open(p) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = img.resize((input_size, input_size), Image.BILINEAR)
            out[i] = np.asarray(img, dtype=np.uint8)
    return _normalize_nhwc(out, mean, std)


def _calib_market1501(num, input_h, input_w, mean, std, seed, dataset_root):
    """`num` random Market-1501 bounding_box_train crops, NHWC float32, normalized.

    Re-ID uses a DIRECT resize to (H, W) = (256, 128) — no center crop (the
    pedestrian crops are already tight). RGB, ImageNet mean/std. Deterministic
    given (num, seed). Tolerates the archive's nested Market-1501-v15.09.15/ dir.
    """
    from PIL import Image

    root = Path(dataset_root)
    train_dirs = sorted(root.glob("**/bounding_box_train"))
    if not train_dirs:
        raise FileNotFoundError(
            f"Market-1501 bounding_box_train not found under {root}\n"
            f"Expected layout: <dataset-dir>/market1501/**/bounding_box_train/*.jpg"
        )
    # -1 / 0000 prefixes are junk/distractor crops; skip them for calibration.
    jpgs = sorted(p for p in train_dirs[0].glob("*.jpg")
                  if not p.name.startswith(("-1", "0000")))
    if not jpgs:
        raise FileNotFoundError(f"No usable .jpg files in {train_dirs[0]}")

    rng = random.Random(seed)
    picks = rng.sample(jpgs, k=min(num, len(jpgs)))

    out = np.zeros((len(picks), input_h, input_w, 3), dtype=np.uint8)
    for i, p in enumerate(picks):
        with Image.open(p) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = img.resize((input_w, input_h), Image.BILINEAR)  # PIL takes (W, H)
            out[i] = np.asarray(img, dtype=np.uint8)
    return _normalize_nhwc(out, mean, std)


# Composition of the upstream-delivered line-seg calibration set, in order.
_LINE_SEG_N_TRACK, _LINE_SEG_N_FLOOR = 140, 60


def _calib_line_seg(num, input_h, input_w, mean, std, seed, dataset_root):
    """The line-seg calibration set delivered with each upstream model.

    Lane following has no public dataset, so there is nothing to sample from: the
    upstream build writes `cal.npy` (N=200, raw RGB float [0,255], already
    ROI-cropped at roi_top=0.45 and resized) next to the model, drawn from the
    TRAIN half of the held-out split only. `dataset_subdir` therefore points at one
    specific model's delivery directory, which is what `dataset_root` resolves to.

    Composition is 140 track + 60 floor, in that order, and it is load-bearing:
    seam negatives are ~half the deployed input distribution and are exactly what
    the `floor_fire` acceptance metric measures, so calibrating on track frames
    alone skews the int8 range. Sub-sampling below 200 therefore keeps the 140:60
    proportion instead of slicing the head (which would be all-track).
    """
    p = Path(dataset_root) / "cal.npy"
    if not p.is_file():
        raise FileNotFoundError(
            f"line-seg calibration set not found: {p}\n"
            f"Expected layout: <dataset-dir>/line_seg/<variant>/cal.npy "
            f"(run families/segmentation/line_seg/setup.sh to link the upstream delivery)")
    arr = np.load(p)
    n_total = _LINE_SEG_N_TRACK + _LINE_SEG_N_FLOOR
    if arr.shape != (n_total, input_h, input_w, 3):
        raise RuntimeError(
            f"{p}: expected ({n_total},{input_h},{input_w},3), got {arr.shape} — "
            f"the calibration set does not match this family's input geometry")
    if num > n_total:
        raise ValueError(f"--num-calib {num} exceeds the delivered set ({n_total}); "
                         f"use NUM_CALIB<={n_total}")
    if num < n_total:
        k = round(num * _LINE_SEG_N_TRACK / n_total)
        idx = list(range(k)) + list(range(_LINE_SEG_N_TRACK,
                                          _LINE_SEG_N_TRACK + (num - k)))
        arr = arr[idx]
    return _normalize_nhwc(arr, mean, std)



# Model registry. The ModelFamily record lives in int8_pruning.convert.common so both export
# paths can annotate against it; the registry is built here because it needs the calib loaders.

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# OpenAI CLIP normalization (timm resnet50_clip pretrained_cfg). Required for the embedding to
# land in the CLIP text-embedding space; ImageNet stats here silently mis-align zero-shot.
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
# TF-style "lite" preprocessing (timm tf_efficientnet_lite0, torchvision SSDLite): mean=std=0.5.
NORM_05 = [0.5, 0.5, 0.5]


# Built from families/<name>/family.yaml -- see int8_pruning.manifest. The 18 entries used
# to be hand-written here, with their normalization duplicated in
# int8_pruning.data.classification.FAMILY_NORMS and the family list duplicated again in
# three `case` blocks in scripts/pruning.sh; nothing kept those in sync and they had
# drifted, clip_rn50 missing here while the README documented it. Calibration loaders
# stay Python because they read datasets; the manifest names one and we resolve it here.
_CALIB_LOADERS = {name: obj for name, obj in list(globals().items())
                  if name.startswith("_calib_") and callable(obj)}

MODEL_REGISTRY: List[ModelFamily] = manifest.build_model_registry(
    ModelFamily, _CALIB_LOADERS)


def resolve_family(pt_path: Path, override: Optional[str]) -> ModelFamily:
    if override:
        for f in MODEL_REGISTRY:
            if f.name == override:
                return f
        raise ValueError(f"--model-family={override} not in registry. "
                         f"Known: {[f.name for f in MODEL_REGISTRY]}")
    basename = pt_path.name
    matches = [f for f in MODEL_REGISTRY if f.matches(basename)]
    if not matches:
        raise ValueError(
            f"No registry entry matches '{basename}'. Add a ModelFamily to "
            f"MODEL_REGISTRY or pass --model-family explicitly."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Filename '{basename}' is ambiguous; matches: "
            f"{[f.name for f in matches]}. Tighten the filename_pattern regex."
        )
    return matches[0]



def build_calibration(family: ModelFamily, num: int, seed: int,
                      dataset_dir: Path, cache_dir: Path,
                      rebuild: bool) -> Path:
    """Return path to a .npy of shape (N, S, S, 3) float32, ImageNet-normalized."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{family.name}_n{num}_seed{seed}.npy"
    if cache_path.exists() and not rebuild:
        print(f"  [calib] reusing cached {cache_path.name}")
        return cache_path

    dataset_root = dataset_dir / family.dataset_subdir
    input_h = family.input_size
    input_w = family.input_w or family.input_size
    print(f"  [calib] building {num} samples from {dataset_root} (seed={seed})")
    arr = family.calib_loader(num, input_h, input_w, family.mean, family.std,
                              seed, dataset_root)
    if arr.shape != (num, input_h, input_w, 3) or arr.dtype != np.float32:
        raise RuntimeError(
            f"calib_loader returned shape={arr.shape} dtype={arr.dtype}; "
            f"expected ({num},{input_h},{input_w},3) float32"
        )
    np.save(cache_path, arr)
    print(f"  [calib] cached at {cache_path}")
    return cache_path





def inspect_tflite(int8_path: Path) -> dict:
    """Read input/output dtype + shape via tflite_runtime if available, else tf.lite."""
    info = {"input_dtype": None, "output_dtype": None,
            "input_shape": None, "output_shape": None}
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:  # older env
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            return info
    it = Interpreter(model_path=str(int8_path))
    it.allocate_tensors()
    i = it.get_input_details()[0]
    outs = it.get_output_details()
    info["input_dtype"] = np.dtype(i["dtype"]).name
    info["input_shape"] = list(map(int, i["shape"]))
    # Keep the single-output fields (first output) for backward compatibility, then record them all.
    o0 = outs[0]
    info["output_dtype"] = np.dtype(o0["dtype"]).name
    info["output_shape"] = list(map(int, o0["shape"]))
    info["num_outputs"] = len(outs)
    if len(outs) > 1:
        info["outputs"] = [
            {"name": o["name"],
             "dtype": np.dtype(o["dtype"]).name,
             "shape": list(map(int, o["shape"]))}
            for o in outs
        ]
    return info




# Per-file orchestration

EXPORT_PATHS = ("litert", "onnx2tf", "onnx")

# One output dir per export path, kept apart on purpose: the two paths produce a different
# file for the same .pt, and `packaging_frac` is not comparable across them.
_OUTPUT_SUBDIR = {"litert": "tflite_int8_litert", "onnx2tf": "tflite_int8_onnx2tf",
                  "onnx": "onnx_fp32"}

# The two int8 paths quantize; `onnx` writes an fp32 graph and stops, as the interchange
# format the other vendor toolchains read (TensorRT, OpenVINO, Core ML, RKNN, QNN). TFLite is
# terminal, and no number under results/ came through the onnx path.
QUANTIZING_PATHS = ("litert", "onnx2tf")

# Sidecar tag. The litert string is what every sidecar on disk carries, and audit_sizes.py
# derives the same label from the output directory, so the two agree without reading each other.
_EXPORT_PATH_TAG = {"litert": "litert-torch + ai-edge-quantizer",
                    "onnx2tf": "onnx2tf",
                    "onnx": "torch.onnx.export (fp32)"}


def convert_one(pt_path: Path, family: ModelFamily, args, calib_npy: Path) -> dict:
    project = Path(args.project_dir)
    int8_dir = Path(args.output_dir)
    int8_dir.mkdir(parents=True, exist_ok=True)

    stem = pt_path.stem  # e.g. mobilenetv2_flowers102_pruned50pct_magnitude_l2

    if args.export_path == "onnx":
        # Returns here: everything below inspects a TFLite flatbuffer, which this path never produces.
        from int8_pruning.convert import export_onnx

        out_path = int8_dir / f"{stem}.onnx"
        if args.skip_existing and out_path.exists():
            print(f"[skip] {out_path.name} exists (--skip-existing)")
            return {"skipped": True, "path": str(out_path)}
        info = export_onnx.export_fp32_onnx(pt_path, out_path, family)
        sidecar = {"name": stem, "model_family": family.name,
                   "export_path": _EXPORT_PATH_TAG["onnx"],
                   "input_size": family.input_size, **info}
        (int8_dir / f"{stem}.onnx.json").write_text(json.dumps(sidecar, indent=2))
        print(f"  [done] {out_path.name}  {info['size_mib']:.2f} MiB  fp32")
        return sidecar

    int8_path = int8_dir / f"{stem}_int8.tflite"
    sidecar_path = int8_dir / f"{stem}_int8.size.json"

    if args.skip_existing and int8_path.exists():
        print(f"[skip] {int8_path.name} exists (--skip-existing)")
        return {"skipped": True, "path": str(int8_path)}

    tmp_dir = int8_dir / f"_tmp_{stem}_pid{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = project / "outputs" / "onnx_models" / f"{stem}.onnx"
    try:
        if args.export_path == "litert":
            from int8_pruning.convert import export_litert

            # Re-exporting the float graph costs minutes and re-quantizing does not, so keep it when asked.
            fp32_dir = project / "outputs" / "tflite_float_litert"
            f32_path = (fp32_dir / f"{stem}_f32.tflite" if args.keep_intermediate
                        else tmp_dir / f"{stem}_f32.tflite")
            path_info = export_litert.export_float_tflite(pt_path, f32_path, family)
            path_info.update(export_litert.quantize_int8(f32_path, calib_npy, int8_path,
                                                  args.num_calib, tmp_dir))
            path_info["quant_type"] = "per-channel"
        else:
            from int8_pruning.convert import export_onnx2tf

            input_name = export_onnx2tf.export_onnx(pt_path, onnx_path, family)
            produced = export_onnx2tf.run_onnx2tf(onnx_path, calib_npy, input_name,
                                                  tmp_dir, args.quant_type)
            shutil.move(str(produced), str(int8_path))
            if args.keep_intermediate:
                fp32_dir = project / "outputs" / "tflite_float32"
                fp32_dir.mkdir(parents=True, exist_ok=True)
                for f32 in sorted(tmp_dir.glob("*_float32.tflite"))[:1]:
                    shutil.move(str(f32), str(fp32_dir / f"{stem}_float32.tflite"))
            path_info = {"onnx_opset": family.onnx_opset,
                         "quant_type": args.quant_type,
                         "num_calib": args.num_calib}

        info = inspect_tflite(int8_path)
        # Raw file size is not a weight proxy, and it gets worse as you prune: on the litert path graph
        # packaging is 24% of an effdet file at the dense rung and 55% at pruned90 (it was 65% throughout
        # on the retired onnx2tf one). Record the constant-data bytes -- see int8_pruning.report.tflite_size.
        budget = measure_tflite_size(int8_path)
        size_mib = int8_path.stat().st_size / (1024 * 1024)
        sidecar = {
            "name": stem,
            "model_family": family.name,
            "size_mib": round(size_mib, 4),
            "size_bytes": int8_path.stat().st_size,
            **budget.as_dict(),
            "export_path": _EXPORT_PATH_TAG[args.export_path],
            "calib_seed": args.seed,
            "input_size": family.input_size,
            **path_info,
            **info,
        }
        sidecar_path.write_text(json.dumps(sidecar, indent=2))
        frac = budget.packaging_frac
        note = ""
        if frac is not None and frac > 0.5:
            note = (f"  [!] {100 * frac:.0f}% of this file is graph packaging, "
                    f"not weights -- report const_mib ({budget.const_mib:.2f}) "
                    f"or the compiled artifact, not size_mib")
        print(f"  [done] {int8_path.name}  {size_mib:.2f} MiB  "
              f"in={info['input_dtype']} out={info['output_dtype']}{note}")
        return sidecar
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        if not args.keep_intermediate and onnx_path.exists():
            onnx_path.unlink()


# CLI

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path,
                     help="convert a single .pt file")
    src.add_argument("--input-dir", type=Path,
                     help="convert every .pt under this directory")

    # src/int8_pruning/convert/tflite.py -> convert -> int8_pruning -> src -> project root
    project_default = Path(__file__).resolve().parents[3]
    p.add_argument("--project-dir", type=Path, default=project_default,
                   help="project root (default: %(default)s)")
    p.add_argument("--export-path", choices=EXPORT_PATHS, default="litert",
                   help="litert: litert-torch + ai-edge-quantizer, no ONNX step "
                        "(default). onnx2tf: torch.onnx.export + onnx2tf, the "
                        "pre-2026-08-22 path, kept for reproducing those "
                        "artifacts; needs `pip install -e '.[onnx2tf]'`")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="where to write *_int8.tflite (default: "
                        "outputs/tflite_int8_<export-path>/)")
    p.add_argument("--dataset-dir", type=Path,
                   default=project_default / "data" / "datasets",
                   help="parent of per-dataset folders (default: %(default)s)")
    p.add_argument("--glob", default="*.pt",
                   help="filter when using --input-dir (default: %(default)s)")
    p.add_argument("--num-calib", type=int, default=100,
                   help="calibration sample count (default: %(default)s)")
    p.add_argument("--seed", type=int, default=42,
                   help="random seed for calibration sampling (default: %(default)s)")
    p.add_argument("--model-family", default=None,
                   help="override filename auto-detect (one of: "
                        + ", ".join(f.name for f in MODEL_REGISTRY) + ")")
    p.add_argument("--quant-type", default="per-channel",
                   choices=["per-channel", "per-tensor"],
                   help="int8 weight granularity, ONNX2TF PATH ONLY (the litert "
                        "path is per-channel with no knob). per-tensor is what "
                        "the retired classification/ssdlite artifacts used; read "
                        "docs/PRUNING_HAZARDS.md section 4 first (default: "
                        "%(default)s)")
    p.add_argument("--keep-intermediate", action="store_true",
                   help="keep the float .tflite (outputs/tflite_float_litert/ or "
                        "outputs/tflite_float32/) and, on the onnx2tf path, the "
                        ".onnx under outputs/onnx_models/")
    p.add_argument("--rebuild-calib", action="store_true",
                   help="ignore calib_cache, regenerate")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip .pt files whose _int8.tflite already exists")
    args = p.parse_args(argv)

    if args.output_dir is None:
        args.output_dir = (Path(args.project_dir) / "outputs"
                           / _OUTPUT_SUBDIR[args.export_path])
    if args.export_path == "onnx" and args.quant_type != "per-channel":
        p.error("--quant-type does not apply to --export-path onnx: that path "
                "writes an fp32 graph and does not quantize.")
    if args.export_path == "litert" and args.quant_type != "per-channel":
        p.error("--quant-type is an onnx2tf-path knob; the litert path quantizes "
                "per-channel and has no per-tensor mode. Pass "
                "--export-path onnx2tf, or drop --quant-type.")
    return args


def list_inputs(args) -> List[Path]:
    # Keep symlink paths un-resolved: the link name carries the semantics MODEL_REGISTRY matches
    # on (e.g. `efficientdet_lite1_coco-train2017_pruned0pct.pt`), and `.resolve()` would collapse
    # to the storage filename and break family detection.
    if args.input is not None:
        if not args.input.is_file():
            raise FileNotFoundError(args.input)
        return [args.input.absolute()]
    if not args.input_dir.is_dir():
        raise FileNotFoundError(args.input_dir)
    out = []
    for p in sorted(args.input_dir.iterdir()):
        if p.is_symlink() and not p.exists():
            print(f"  [skip] broken symlink {p.name}")
            continue
        if not fnmatch.fnmatch(p.name, args.glob):
            continue
        if p.suffix != ".pt":
            continue
        out.append(p.absolute())
    return out


def main(argv=None):
    args = parse_args(argv)

    inputs = list_inputs(args)
    if not inputs:
        print("No .pt inputs matched. Nothing to do.")
        return 0

    cache_dir = Path(args.output_dir) / "calib_cache"

    # Calibration .npy is built at most once per family, so all prune ratios share the same set.
    family_to_calib = {}
    for pt in inputs:
        fam = resolve_family(pt, args.model_family)
        if fam.name not in family_to_calib:
            # No calibration on the fp32 path, so no dataset is needed to run it -- the one practical
            # reason to reach for it on a machine that has the checkpoints but not the data.
            calib = (None if args.export_path not in QUANTIZING_PATHS else
                     build_calibration(fam, args.num_calib, args.seed,
                                       Path(args.dataset_dir), cache_dir,
                                       args.rebuild_calib))
            family_to_calib[fam.name] = (fam, calib)

    print(f"\n[convert] {len(inputs)} model(s) across "
          f"{len(family_to_calib)} family/families  "
          f"export-path={args.export_path} -> {args.output_dir}")
    for name in family_to_calib:
        n = sum(1 for pt in inputs
                if resolve_family(pt, args.model_family).name == name)
        print(f"  - {name}: {n} model(s)")
    print()

    results = []
    t0 = time.time()
    for i, pt in enumerate(inputs, 1):
        fam = resolve_family(pt, args.model_family)
        calib_npy = family_to_calib[fam.name][1]
        print(f"[{i}/{len(inputs)}] {pt.name}  (family={fam.name})")
        try:
            r = convert_one(pt, fam, args, calib_npy)
            results.append(r)
        except Exception as e:
            import traceback
            print(f"  [FAIL] {type(e).__name__}: {e}")
            traceback.print_exc()
            results.append({"error": f"{type(e).__name__}: {e}",
                            "path": str(pt)})
    dt = time.time() - t0
    ok = sum(1 for r in results if "error" not in r and not r.get("skipped"))
    skipped = sum(1 for r in results if r.get("skipped"))
    failed = sum(1 for r in results if "error" in r)
    print(f"\n[convert] done in {dt:.1f}s  "
          f"ok={ok} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
