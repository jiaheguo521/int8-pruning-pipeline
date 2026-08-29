#!/usr/bin/env python3
"""Top-1 / top-5 of a classification int8 or Edge-TPU tflite.

Top-1 / top-5 accuracy of a classification *int8 / Edge-TPU* tflite on the
matching validation split (iNat-2017 Plantae, Flowers-102, or ImageNet-1k).

Why this exists:
A generic Imagenette-shaped benchmark script only knows the 10-class folder
layout, so it cannot score the cascade's classifier (MobileNetV2 finetuned on iNat
Plantae / Flowers-102). This worker reuses the *same* val datasets and label
maps the finetune trained on — `int8_pruning.data.classification.build_classification_dataloaders`
— so the ΔAcc it reports (pruned vs unpruned, int8 vs the float baseline) is
directly comparable to the training-time numbers.

How the input is fed:
The val loader is built with **identity normalization** (mean=0, std=1), so it
yields the ToTensor() image in [0,1] with the correct Resize/CenterCrop and
label. This worker then applies the model's real per-family normalization
(`int8_pruning.data.classification.norm_for_model`, derived from the tflite stem) and
quantizes with the tflite's own input scale/zero-point — exactly the path the
int8 model was calibrated for.

Usage:
    # iNat Plantae val (the cascade classifier):
    python families/classification/imagenet_backbones/eval.py \
        --tflite outputs/edgetpu/<run>/mobilenetv2_inat_pruned30pct_magnitude_l2_int8_edgetpu.tflite \
        --dataset inat_plant \
        --inat-val-json data/datasets/inat2017/val2017.json \
        --inat-image-root data/datasets/inat2017

    # Flowers-102 val (small-head ablation):
    python families/classification/imagenet_backbones/eval.py \
        --tflite outputs/tflite_int8_litert/mobilenetv2_flowers102_pruned0pct_int8.tflite \
        --dataset flowers102 --data-root data/datasets/flowers102

A path ending in `_edgetpu.tflite` runs on the Coral via pycoral; any other int8
tflite runs on CPU via tflite_runtime / tf.lite (so the same number is reachable
without the device).

Prerequisites: the int8 CPU path needs `tflite_runtime` (or `tensorflow`); the
`_edgetpu.tflite` path additionally needs `pycoral` + a plugged-in Coral.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from int8_pruning.data.classification import build_classification_dataloaders, norm_for_model


from int8_pruning.runtime.interpreter import (  # noqa: F401  (re-exported for callers)
    make_interpreter, quantize_input, _input_quant,
)


# eval
def run(args):
    it = make_interpreter(args.tflite)
    inp = it.get_input_details()[0]
    out = it.get_output_details()[0]
    input_size = int(inp["shape"][1])
    num_out = int(out["shape"][-1])

    mean, std = norm_for_model(Path(args.tflite).name)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)

    # Identity-normalized loader keeps images in [0,1], so mean/std and tflite quant apply exactly once here.
    _, val_loader, _, num_classes = build_classification_dataloaders(
        args.dataset, data_root=args.data_root, batch_size=args.batch_size,
        image_size=input_size, download=args.download,
        inat_val_json=args.inat_val_json, inat_image_root=args.inat_image_root,
        inat_train_json=args.inat_val_json,  # train unused here; reuse val path
        mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))

    print(f"[eval] {Path(args.tflite).name}")
    print(f"[eval] dataset={args.dataset}  input={input_size}px  "
          f"model_outputs={num_out}  dataset_classes={num_classes}  "
          f"norm(mean={mean.tolist()}, std={std.tolist()})  "
          f"label_offset={args.label_offset}")
    if num_out not in (num_classes, num_classes + 1):
        print(f"[warn] model has {num_out} outputs but dataset has {num_classes} "
              f"classes — check --label-offset / the right tflite is paired.")

    c1 = c5 = n = 0
    for images, labels in val_loader:
        imgs = images.permute(0, 2, 3, 1).numpy()  # NHWC, [0,1]
        labels = labels.numpy()
        for img01, lbl in zip(imgs, labels):
            arr = (img01 - mean) / std
            it.set_tensor(inp["index"], quantize_input(arr, inp))
            it.invoke()
            logits = it.get_tensor(out["index"]).flatten()
            top5 = np.argsort(logits)[-5:][::-1] - args.label_offset
            if top5[0] == lbl:
                c1 += 1
            if lbl in top5:
                c5 += 1
            n += 1
            if args.num_images and n >= args.num_images:
                break
        if args.num_images and n >= args.num_images:
            break

    top1 = 100.0 * c1 / n if n else 0.0
    top5 = 100.0 * c5 / n if n else 0.0
    summary = {
        "tflite": str(args.tflite),
        "dataset": args.dataset,
        "n_images": n,
        "input_size": input_size,
        "num_outputs": num_out,
        "top1": top1,
        "top5": top5,
    }
    print(f"[eval] n={n}  Top-1: {top1:.2f}%  Top-5: {top5:.2f}%")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, indent=2))
        print(f"[eval] wrote {args.out_json}")
    print(json.dumps(summary, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tflite", required=True,
                   help="int8 tflite (CPU) or *_edgetpu.tflite (Coral)")
    p.add_argument("--dataset", required=True,
                   choices=("inat_plant", "flowers102", "imagenet"))
    p.add_argument("--data-root", default=None,
                   help="flowers102 / imagenet root (ignored for inat_plant)")
    p.add_argument("--inat-val-json", default=None, help="iNat val2017.json")
    p.add_argument("--inat-image-root", default=None, help="iNat image root")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-images", type=int, default=0,
                   help="cap val images (0 = all); use a small N for a smoke test")
    p.add_argument("--label-offset", type=int, default=0,
                   help="subtract from argmax for background-class (1001-out) models")
    p.add_argument("--download", action="store_true",
                   help="allow torchvision to download flowers102 if missing")
    p.add_argument("--out-json", default=None, help="write the accuracy summary here")
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
