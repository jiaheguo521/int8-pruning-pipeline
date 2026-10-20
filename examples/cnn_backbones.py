"""Torchvision / timm image classifiers: prune and recover, then export and score in int8.

    python examples/cnn_backbones.py fetch-baseline --model mobilenetv2 --dataset flowers102
    python examples/cnn_backbones.py prune --model mobilenetv2 --dataset flowers102 --ratios 0.3 0.5
    python examples/cnn_backbones.py int8 --pt outputs/pytorch_pruned/<stem>.pt --dataset flowers102
    python examples/cnn_backbones.py int8 --tflite <stem>_int8_edgetpu.tflite --dataset flowers102

Models: mobilenetv2, efficientnet_lite0, mnasnet1_0, squeezenet1_1, resnet50 on ImageNet
(<dataset-dir>/Imagenet_1k); mobilenetv2 also on flowers102 and inat_plant, from a baseline
finetuned onto that dataset. Recovery is the PruningBench ImageNet recipe (Li et al. 2024):
SGD-Nesterov lr 0.1, multistep at 30 and 60, 90 epochs, the best validation epoch kept.

Outputs, under --out-dir (default outputs/):
    pytorch_pruned/<stem>_pruned<P>pct_<criterion>[_iter].pt     full nn.Module
    pruning_logs/<stem>_<criterion>_<P>pct[_iter].json
    tflite_int8_litert/<.pt stem>_int8.tflite
    edgetpu/<.pt stem>_int8_edgetpu.tflite    with --edgetpu
"""
import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch_pruning as tp
import torchvision.models as tvm
from torchvision.datasets import Flowers102
from PIL import Image, ImageFile

import int8_pruning
from example_data import (DATASET_NUM_CLASSES, IMPORTANCES,
                          build_classification_dataloaders, imagenet_center_crop)

ImageFile.LOAD_TRUNCATED_IMAGES = True

MODELS = ("mobilenetv2", "efficientnet_lite0", "mnasnet1_0", "squeezenet1_1", "resnet50")
DATA_DIR = {"imagenet": "Imagenet_1k", "flowers102": "flowers102", "inat_plant": "inat2017_plant"}
FINETUNED = {"flowers102": "mobilenetv2_flowers102", "inat_plant": "mobilenetv2_inat"}
IMAGENET_NORM = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
NORM = {"efficientnet_lite0": ([0.5] * 3, [0.5] * 3)}       # timm's tf_ port; the rest ImageNet

# torchvision constructor, weights enum, weights
TORCHVISION = {
    "mobilenetv2": ("mobilenet_v2", "MobileNet_V2_Weights", "IMAGENET1K_V2"),
    "mnasnet1_0": ("mnasnet1_0", "MNASNet1_0_Weights", "IMAGENET1K_V1"),
    "squeezenet1_1": ("squeezenet1_1", "SqueezeNet1_1_Weights", "IMAGENET1K_V1"),
    "resnet50": ("resnet50", "ResNet50_Weights", "IMAGENET1K_V2"),
}

# PruningBench: global ranking, 400 steps, no layer loses more than 90%.
PRUNING = {"global_pruning": True, "iterative_steps": 400, "max_pruning_ratio": 0.9}


def recovery(model, epochs):
    """PruningBench's ImageNet recipe: SGD-Nesterov lr 0.1, x0.1 at epochs 30 and 60."""
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True,
                          weight_decay=1e-4)
    return opt, torch.optim.lr_scheduler.MultiStepLR(opt, [m for m in (30, 60) if m < epochs])


def norm(model_name):
    return NORM.get(model_name, IMAGENET_NORM)


def baseline_name(model, dataset):
    """The .pt the ladder starts from, and the stem its outputs are written under."""
    if dataset == "imagenet":
        return model, f"{model}_imagenet"
    if model != "mobilenetv2":
        raise SystemExit(f"{dataset} has a mobilenetv2 baseline only")
    return FINETUNED[dataset], FINETUNED[dataset]


def loaders(dataset, dataset_dir, model_name, batch_size, seed, mean_std=None):
    root = Path(dataset_dir) / DATA_DIR[dataset]
    mean, std = mean_std or norm(model_name)
    train, val, test, _ = build_classification_dataloaders(
        dataset, data_root=str(root), batch_size=batch_size, seed=seed,
        inat_train_json=str(root / "train2017.json"), inat_val_json=str(root / "val2017.json"),
        inat_image_root=str(root), mean=mean, std=std)
    return train, val, test


def ignored_layers(model):
    """The classifier head: every Linear / Conv2d in `model.classifier`, else the last one."""
    head = getattr(model, "classifier", None)
    layers = [m for m in (head if isinstance(head, nn.Sequential) else [head])
              if isinstance(m, (nn.Linear, nn.Conv2d))]
    if not layers:
        layers = [[m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))][-1]]
    return layers


def evaluate_top1(model, val_loader, device):
    """Top-1 accuracy on val_loader, in %."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total


@torch.no_grad()
def top1_top5_ci(model, loader, device, n_bootstrap=1000, seed=0):
    """Top-1 and top-5 in %, with bootstrap 95% intervals (Efron 1979)."""
    model.eval()
    c1, c5 = [], []
    for images, labels in loader:
        labels = labels.to(device)
        top5 = model(images.to(device)).topk(5, dim=1).indices
        c1.append((top5[:, 0] == labels).cpu().numpy())
        c5.append((top5 == labels.unsqueeze(1)).any(dim=1).cpu().numpy())
    c1, c5 = np.concatenate(c1).astype(np.uint8), np.concatenate(c5).astype(np.uint8)
    rng = np.random.RandomState(seed)
    boot = [rng.randint(0, len(c1), size=len(c1)) for _ in range(n_bootstrap)]
    m1, m5 = np.array([c1[i].mean() for i in boot]), np.array([c5[i].mean() for i in boot])
    return {"top1_pct": 100 * c1.mean(), "top5_pct": 100 * c5.mean(),
            "top1_ci95": [100 * np.percentile(m1, 2.5), 100 * np.percentile(m1, 97.5)],
            "top5_ci95": [100 * np.percentile(m5, 2.5), 100 * np.percentile(m5, 97.5)],
            "n_eval": len(c1)}


# fetch-baseline ---------------------------------------------------------------------------

def fetch_baseline(args):
    """Write the dense baseline `prune` starts from into --models-dir."""
    name, _ = baseline_name(args.model, args.dataset)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    start = models_dir / f"{args.model}.pt"
    if not start.exists():
        if args.model == "efficientnet_lite0":
            model = timm.create_model("tf_efficientnet_lite0", pretrained=True)
        else:
            ctor, enum, weights = TORCHVISION[args.model]
            model = getattr(tvm, ctor)(weights=getattr(getattr(tvm, enum), weights))
        torch.save(model, start)
        print(f"-> {start}")
    if args.dataset == "imagenet":
        return
    # Finetune the ImageNet MobileNetV2 onto the dataset: new head, SGD lr 0.01, cosine.
    int8_pruning.seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val, _ = loaders(args.dataset, args.dataset_dir, args.model, 64, args.seed)
    model = torch.load(start, map_location="cpu", weights_only=False)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features,
                                     DATASET_NUM_CLASSES[args.dataset])
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    int8_pruning.finetune(model, train, args.epochs, optimizer=opt,
                          scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs),
                          amp=device.type == "cuda", select="val_acc",
                          evaluate=lambda: {"val_acc": evaluate_top1(model, val, device)})
    out = models_dir / f"{name}.pt"
    torch.save(model.cpu(), out)
    print(f"-> {out}")


# prune ------------------------------------------------------------------------------------

def prune(args):
    """Prune the baseline to each of --ratios, recover, and save each rung with its log."""
    name, stem = baseline_name(args.model, args.dataset)
    src = Path(args.models_dir) / f"{name}.pt"
    if not src.exists():
        raise SystemExit(f"{src} is missing: run `python {sys.argv[0]} fetch-baseline "
                         f"--model {args.model} --dataset {args.dataset}` first")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    pruned_dir, log_dir = Path(args.out_dir) / "pytorch_pruned", Path(args.out_dir) / "pruning_logs"
    pruned_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    mode = "_iter" if args.iterative else ""

    def dense():
        int8_pruning.seed_everything(args.seed)
        model = torch.load(src, map_location="cpu", weights_only=False).to(device)
        data = loaders(args.dataset, args.dataset_dir, args.model, args.batch_size, args.seed)
        ref = {"params": int8_pruning.count_params(model),
               "val_acc": evaluate_top1(model, data[1], device),
               "macs": tp.utils.count_ops_and_params(
                   model, torch.randn(1, 3, 224, 224, device=device))[0]}
        return model, data, ref

    def recover_and_save(model, data, ref, ratio, info):
        train, val, test = data
        pct = round(100 * ratio)
        log = {"model": stem, "importance": args.importance, "ratio": ratio,
               "iterative": args.iterative, "dense": ref, **info._asdict(),
               "macs": tp.utils.count_ops_and_params(
                   model, torch.randn(1, 3, 224, 224, device=device))[0],
               "pruned_val_acc": evaluate_top1(model, val, device)}
        opt, sched = recovery(model, args.epochs)
        history = int8_pruning.finetune(
            model, train, args.epochs, optimizer=opt, scheduler=sched, select="val_acc",
            label=f"FT @{pct}%",
            evaluate=lambda: {"val_acc": evaluate_top1(model, val, device)})
        log["history"] = [r._asdict() for r in history]
        log["final"] = top1_top5_ci(model, test, device)
        out = pruned_dir / f"{stem}_pruned{pct}pct_{args.importance}{mode}.pt"
        torch.save(model, out)
        (log_dir / f"{stem}_{args.importance}_{pct}pct{mode}.json").write_text(
            json.dumps(log, indent=2, default=float))
        print(f"-> {out}  top-1 {ref['val_acc']:.2f}% dense, {log['final']['top1_pct']:.2f}% "
              f"after ({log['final']['n_eval']} test images)")

    kw = dict(importance=args.importance, **PRUNING)
    if args.iterative:
        model, data, ref = dense()
        example = torch.randn(1, 3, 224, 224).to(device)
        for ratio, info in int8_pruning.prune_trajectory(
                model, example, args.ratios, ignored_layers=ignored_layers(model), **kw):
            recover_and_save(model, data, ref, ratio, info)
        return
    for ratio in args.ratios:
        model, data, ref = dense()
        example = torch.randn(1, 3, 224, 224).to(device)
        info = int8_pruning.prune(model, example, ratio, ignored_layers=ignored_layers(model), **kw)
        recover_and_save(model, data, ref, ratio, info)


# int8 -------------------------------------------------------------------------------------

def flowers102_calibration(root, num, size, mean, std, seed):
    """`num` Flowers-102 train images, resized and center-cropped like the eval transform."""
    ds = Flowers102(root=str(root), split="train", download=False)
    idx = random.Random(seed).sample(range(len(ds)), k=min(num, len(ds)))
    target = int(round(size * 256 / 224))
    out = np.zeros((len(idx), size, size, 3), dtype=np.uint8)
    for i, j in enumerate(idx):
        img = ds[j][0].convert("RGB")
        w, h = img.size
        new = ((target, int(round(h * target / w))) if w < h
               else (int(round(w * target / h)), target))
        img = img.resize(new, Image.BILINEAR)
        left, top = (new[0] - size) // 2, (new[1] - size) // 2
        out[i] = np.asarray(img.crop((left, top, left + size, top + size)), dtype=np.uint8)
    return int8_pruning.export.normalize_nhwc(out, mean, std)


def int8(args):
    """Export --pt to int8 (or take --tflite as it is), then score it on the validation split."""
    model_name = next(m for m in MODELS if Path(args.pt or args.tflite).name.startswith(m))
    mean, std = norm(model_name)
    root = Path(args.dataset_dir) / DATA_DIR[args.dataset]
    tflite = args.tflite
    if args.pt:
        calib = (flowers102_calibration(root, args.num_calib, 224, mean, std, args.seed)
                 if args.dataset == "flowers102" else
                 imagenet_center_crop(args.num_calib, 224, 224, mean, std, args.seed, root))
        model = torch.load(args.pt, map_location="cpu", weights_only=False)
        tflite = Path(args.out_dir) / "tflite_int8_litert" / f"{Path(args.pt).stem}_int8.tflite"
        info = int8_pruning.export_int8(model, calib, tflite)
        print(f"-> {tflite}  ({info.size_bytes:,} bytes)")
    if args.edgetpu:
        tflite = int8_pruning.compile_edgetpu(tflite, Path(args.out_dir) / "edgetpu")

    it = int8_pruning.make_interpreter(tflite)
    # Loaded at identity normalization, so the model's own mean/std is applied once, here.
    _, val, _ = loaders(args.dataset, args.dataset_dir, model_name, 64, args.seed,
                        mean_std=([0.0] * 3, [1.0] * 3))
    c1 = c5 = n = 0
    for images, labels in val:
        for img, label in zip(images.permute(0, 2, 3, 1).numpy(), labels.numpy()):
            logits = int8_pruning.run_tflite(it, (img - mean) / std)[0].flatten()
            top5 = np.argsort(logits)[-5:][::-1]
            c1, c5, n = c1 + (top5[0] == label), c5 + (label in top5), n + 1
            if n == args.num_images:
                break
        if n == args.num_images:
            break
    print(f"{Path(tflite).name}: int8 top-1 {100 * c1 / n:.2f}%  top-5 {100 * c5 / n:.2f}%  "
          f"({n} {args.dataset} validation images)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    for name, fn in (("fetch-baseline", fetch_baseline), ("prune", prune), ("int8", int8)):
        s = sub.add_parser(name, help=fn.__doc__.splitlines()[0])
        s.set_defaults(fn=fn)
        s.add_argument("--dataset", choices=sorted(DATA_DIR), default="imagenet")
        s.add_argument("--dataset-dir", type=Path, default=Path("data/datasets"))
        s.add_argument("--seed", type=int, default=42)
        if name != "int8":
            s.add_argument("--model", choices=MODELS, required=True)
            s.add_argument("--models-dir", type=Path, default=Path("outputs/models"))
        if name != "fetch-baseline":
            s.add_argument("--out-dir", type=Path, default=Path("outputs"))
    f = sub.choices["fetch-baseline"]
    f.add_argument("--epochs", type=int, default=20, help="finetune epochs (default: 20)")
    r = sub.choices["prune"]
    r.add_argument("--ratios", type=float, nargs="+", required=True,
                   help="fractions of the parameters to remove, e.g. 0.3 0.5 0.7")
    r.add_argument("--importance", default="magnitude_l2",
                   choices=IMPORTANCES)
    r.add_argument("--iterative", action="store_true",
                   help="one trajectory through ascending --ratios instead of each from dense")
    r.add_argument("--epochs", type=int, default=90, help="recovery epochs (default: 90)")
    r.add_argument("--batch-size", type=int, default=64)
    r.add_argument("--device", default=None)
    q = sub.choices["int8"]
    src = q.add_mutually_exclusive_group(required=True)
    src.add_argument("--pt", type=Path, help="a pruned .pt to export to int8 and score")
    src.add_argument("--tflite", type=Path, help="an int8 or _edgetpu.tflite to score")
    q.add_argument("--num-calib", type=int, default=100)
    q.add_argument("--num-images", type=int, default=0, help="cap on scored images (0 = all)")
    q.add_argument("--edgetpu", action="store_true",
                   help="compile for the Edge TPU into <out-dir>/edgetpu/ and score the "
                        "compiled model on an attached Coral")
    args = p.parse_args(argv)
    logging.basicConfig(format="%(message)s")
    logging.getLogger("int8_pruning").setLevel(logging.INFO)
    args.fn(args)


if __name__ == "__main__":
    main()
