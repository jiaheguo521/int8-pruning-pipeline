#!/usr/bin/env python3
"""Zero-shot top-1/5 of a CLIP image tower.

Open-vocabulary **zero-shot** top-1/5 of a CLIP image tower against precomputed
text embeddings — the accuracy leg of the Edge-TPU Scenario-1 cascade.

Unlike `families/classification/imagenet_backbones/eval.py` (argmax over a fixed-class logit head), this
worker reads the model's **image embedding** output and scores it by cosine
similarity against a `[num_classes, D]` text-embedding matrix produced offline by
`families/classification/clip_rn50/text_embeddings.py`. Swapping that matrix changes the label set
with no recompile — that is the "open vocabulary" the robot car needs.

Runs three back-ends so the int8 cost is directly comparable:
  --pt   <clip_rn50.pt>            fp32 reference (torch, the upper bound)
  --tflite <..._int8.tflite>       int8 on CPU (tflite_runtime / tf.lite)
  --tflite <..._int8_edgetpu.tflite>  int8 on the Coral (pycoral), if present

Image input: a directory of class sub-folders (ImageFolder-style). Folder names
map to label rows via the built-in Imagenette wnid map, else the folder name is
taken as the label name. Preprocessing matches the int8 calibration (resize
shorter side to 256, center-crop 224) so fp32 and int8 numbers are apples-to-apples.

Usage:
    # int8 (CPU) zero-shot on the 10 local Imagenette classes:
    python families/classification/clip_rn50/eval.py \
        --tflite outputs/tflite_int8_litert/clip_rn50_int8.tflite \
        --text-emb outputs/models/clip_rn50_text_imagenette.npy \
        --images-dir data/datasets/Imagenet_1k/train

    # fp32 reference (same images / text):
    python families/classification/clip_rn50/eval.py \
        --pt outputs/models/clip_rn50.pt \
        --text-emb outputs/models/clip_rn50_text_imagenette.npy \
        --images-dir data/datasets/Imagenet_1k/train
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from int8_pruning.data.classification import norm_for_model
# Reuse the int8 interpreter + input-quant helpers from the classifier eval.
from int8_pruning.runtime.interpreter import (
    _input_quant, embed_tflite, make_interpreter, quantize_input,
)

# Imagenette wnid -> CLIP label name (the local data/datasets/Imagenet_1k layout).
WNID_TO_NAME = {
    "n01440764": "tench", "n02102040": "English springer",
    "n02979186": "cassette player", "n03000684": "chain saw",
    "n03028079": "church", "n03394916": "French horn",
    "n03417042": "garbage truck", "n03425413": "gas pump",
    "n03445777": "golf ball", "n03888257": "parachute",
}


def preprocess_224(pil, input_size, mean, std):
    """Resize shorter side to round(input*256/224), center-crop, normalize.

    Matches `int8_pruning.convert.tflite._calib_imagenet_train` so the eval pixels
    follow the same pipeline the int8 quant ranges were calibrated on. Returns
    HWC float32 (normalized)."""
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    resize_target = int(round(input_size * 256 / 224))
    w, h = pil.size
    if w < h:
        new_w, new_h = resize_target, int(round(h * resize_target / w))
    else:
        new_w, new_h = int(round(w * resize_target / h)), resize_target
    pil = pil.resize((new_w, new_h), Image.BILINEAR)
    left = (new_w - input_size) // 2
    top = (new_h - input_size) // 2
    pil = pil.crop((left, top, left + input_size, top + input_size))
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return (arr - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)


def list_images(images_dir, labels):
    """-> list of (path, label_index). Sub-folder name -> label via WNID_TO_NAME
    (Imagenette) or the folder name itself; folders with no matching label row
    are skipped with a warning."""
    name_to_idx = {n: i for i, n in enumerate(labels)}
    items = []
    for sub in sorted(p for p in Path(images_dir).iterdir() if p.is_dir()):
        label_name = WNID_TO_NAME.get(sub.name, sub.name)
        if label_name not in name_to_idx:
            print(f"  [skip] folder '{sub.name}' -> '{label_name}' not in text labels")
            continue
        idx = name_to_idx[label_name]
        for ext in ("*.JPEG", "*.jpeg", "*.jpg", "*.png"):
            for f in sorted(sub.glob(ext)):
                items.append((f, idx))
    return items


def run(args):
    text = np.load(args.text_emb).astype(np.float32)          # [C, D], L2-normalized
    text = text / np.linalg.norm(text, axis=1, keepdims=True)  # belt-and-braces
    labels_path = Path(args.text_emb).with_suffix("").as_posix() + ".labels.json"
    labels = json.loads(Path(labels_path).read_text())
    assert len(labels) == text.shape[0], "labels.json / .npy row count mismatch"

    backend = "pt" if args.pt else "tflite"
    src = args.pt or args.tflite
    mean, std = norm_for_model(Path(src).name)  # clip_rn50 -> CLIP norm

    if backend == "tflite":
        it = make_interpreter(args.tflite)
        input_size = int(it.get_input_details()[0]["shape"][1])
    else:
        import torch
        model = torch.load(args.pt, map_location="cpu", weights_only=False).eval()
        input_size = 224

    items = list_images(args.images_dir, labels)
    if args.num_images:
        items = items[: args.num_images]
    print(f"[zs-eval] backend={backend}  src={Path(src).name}  classes={len(labels)}  "
          f"images={len(items)}  input={input_size}px  norm(mean={mean})")

    c1 = c5 = 0
    for i, (path, lbl) in enumerate(items):
        hwc = preprocess_224(Image.open(path), input_size, mean, std)
        if backend == "tflite":
            emb = embed_tflite(it, hwc)
        else:
            import torch
            x = torch.from_numpy(hwc).permute(2, 0, 1).unsqueeze(0).float()
            with torch.no_grad():
                emb = model(x).flatten().numpy().astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        sims = text @ emb                       # cosine (both L2-normalized)
        top5 = np.argsort(sims)[-5:][::-1]
        c1 += int(top5[0] == lbl)
        c5 += int(lbl in top5)

    n = len(items)
    top1 = 100.0 * c1 / n if n else 0.0
    top5 = 100.0 * c5 / n if n else 0.0
    summary = {"backend": backend, "src": str(src), "text_emb": str(args.text_emb),
               "n_images": n, "num_classes": len(labels),
               "input_size": input_size, "top1": top1, "top5": top5}
    print(f"[zs-eval] n={n}  Top-1: {top1:.2f}%  Top-5: {top5:.2f}%")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, indent=2))
        print(f"[zs-eval] wrote {args.out_json}")
    print(json.dumps(summary, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pt", help="fp32 torch baseline (clip_rn50.pt)")
    src.add_argument("--tflite", help="int8 tflite (CPU) or *_edgetpu.tflite (Coral)")
    p.add_argument("--text-emb", required=True,
                   help="text-embedding .npy from families/classification/clip_rn50/text_embeddings.py "
                        "(sibling .labels.json auto-loaded)")
    p.add_argument("--images-dir", required=True,
                   help="dir of class sub-folders (ImageFolder-style)")
    p.add_argument("--num-images", type=int, default=0,
                   help="cap total images (0 = all); small N for a smoke test")
    p.add_argument("--out-json", default=None, help="write the accuracy summary here")
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
