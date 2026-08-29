#!/usr/bin/env python3
"""Offline text-tower step for the CLIP Edge-TPU cascade (Scenario 1).

The Edge-TPU model (clip_rn50) only carries the *image* tower — it emits a
1024-dim image embedding. Zero-shot / open-vocabulary classification then
compares that image embedding against **text** embeddings of the candidate
labels. Those text embeddings are computed here, once, on the CPU/GPU, and saved
to disk. They are NEVER needed on the robot car at runtime beyond a cheap matmul
(the text tower itself does not run on the device).

Alignment: timm `resnet50_clip` (the image tower we compile to the TPU) and
open_clip `RN50 / pretrained=openai` (the text tower used here) are the same
OpenAI CLIP checkpoint, so their embeddings share one space — verified at
bring-up: timm-image x open_clip-text scored 98% Imagenette zero-shot.

Dependency: `open_clip_torch` (offline-only; not imported on the TPU/runtime
path). `pip install open_clip_torch` into the pruning-env.

Outputs (next to the baselines, in outputs/models/):
    clip_rn50_text_<name>.npy          float32 [num_classes, 1024], L2-normalized
    clip_rn50_text_<name>.labels.json  list[str] of the class names (row order)

Usage:
    # built-in 10-class Imagenette set (matches data/datasets/Imagenet_1k):
    python families/classification/clip_rn50/text_embeddings.py --preset imagenette

    # arbitrary open-vocabulary targets for the car:
    python families/classification/clip_rn50/text_embeddings.py --name car_targets \
        --classes "red cup,blue backpack,potted plant,person,office chair"

    # classes one-per-line from a file, with template ensembling:
    python families/classification/clip_rn50/text_embeddings.py --name myset \
        --classes-file labels.txt --ensemble
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

# 10-class Imagenette label set; row order is the sorted-wnid order the eval uses.
IMAGENETTE = [
    "tench", "English springer", "cassette player", "chain saw", "church",
    "French horn", "garbage truck", "gas pump", "golf ball", "parachute",
]

# A compact CLIP prompt-template ensemble (subset of the OpenAI set). "a photo of a {}." alone
# already gives strong zero-shot; averaging a handful of normalized text embeddings adds a point or two.
TEMPLATES = [
    "a photo of a {}.",
    "a bad photo of a {}.",
    "a photo of the {}.",
    "a cropped photo of a {}.",
    "a photo of one {}.",
    "a close-up photo of a {}.",
    "a bright photo of a {}.",
]


def encode_text(class_names, model_name, pretrained, ensemble, device):
    """-> float32 [num_classes, D], L2-normalized; rows align with class_names."""
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(model_name,
                                                         pretrained=pretrained)
    model = model.eval().to(device)
    tok = open_clip.get_tokenizer(model_name)
    templates = TEMPLATES if ensemble else ["a photo of a {}."]

    embs = []
    with torch.no_grad():
        for name in class_names:
            prompts = [t.format(name) for t in templates]
            te = model.encode_text(tok(prompts).to(device)).float()
            te = te / te.norm(dim=-1, keepdim=True)   # per-prompt normalize
            te = te.mean(dim=0)                        # ensemble average
            te = te / te.norm()                        # re-normalize the mean
            embs.append(te.cpu())
    return torch.stack(embs).numpy().astype(np.float32)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--preset", choices=["imagenette"],
                     help="built-in class set (imagenette = the 10 local classes)")
    src.add_argument("--classes", help="comma-separated class names")
    src.add_argument("--classes-file", type=Path,
                     help="text file, one class name per line")

    p.add_argument("--name", default=None,
                   help="output set name (default: the preset name, else 'custom')")
    p.add_argument("--model", default="RN50", help="open_clip model (default RN50)")
    p.add_argument("--pretrained", default="openai",
                   help="open_clip pretrained tag (default openai — matches timm resnet50_clip)")
    p.add_argument("--ensemble", action="store_true",
                   help="average over a 7-template prompt ensemble (else single template)")
    p.add_argument("--out-dir", type=Path,
                   default=Path(__file__).resolve().parents[3] / "outputs" / "models")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    if args.preset == "imagenette":
        class_names = list(IMAGENETTE)
        name = args.name or "imagenette"
    elif args.classes:
        class_names = [c.strip() for c in args.classes.split(",") if c.strip()]
        name = args.name or "custom"
    else:
        class_names = [ln.strip() for ln in args.classes_file.read_text().splitlines()
                       if ln.strip()]
        name = args.name or args.classes_file.stem

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[clip-text] {len(class_names)} classes, model={args.model}/{args.pretrained}, "
          f"ensemble={args.ensemble}, device={device}")

    emb = encode_text(class_names, args.model, args.pretrained, args.ensemble, device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = args.out_dir / f"clip_rn50_text_{name}.npy"
    json_path = args.out_dir / f"clip_rn50_text_{name}.labels.json"
    np.save(npy_path, emb)
    json_path.write_text(json.dumps(class_names, indent=2))
    print(f"[clip-text] wrote {npy_path}  shape={emb.shape} dtype={emb.dtype}")
    print(f"[clip-text] wrote {json_path}  ({len(class_names)} labels)")


if __name__ == "__main__":
    main()
