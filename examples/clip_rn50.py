"""CLIP RN50 image tower: prune, recover by distillation, then export and score zero-shot in int8.

    python examples/clip_rn50.py fetch-baseline
    python examples/clip_rn50.py text-embeddings --preset imagenette
    python examples/clip_rn50.py prune --ratios 0.3 0.5 \
        --text-emb outputs/models/clip_rn50_text_imagenette.npz \
        --val-images-dir data/datasets/Imagenet_1k/train
    python examples/clip_rn50.py int8 --pt outputs/pytorch_pruned/<stem>.pt \
        --text-emb outputs/models/clip_rn50_text_imagenette.npz \
        --images-dir data/datasets/Imagenet_1k/train

CLIP has no class labels: zero-shot scores the image embedding by cosine against text
embeddings of the candidate labels, computed offline by `text-embeddings` (open_clip's RN50
text tower, the same OpenAI checkpoint as timm's `resnet50_clip`). So recovery is
distillation: 1 - cos(student, teacher) against a frozen copy of the unpruned tower, on
unlabelled ImageNet train images. The attention-pool head is protected, which keeps the
1024-d output and the shared embedding space; the conv body prunes.

Outputs, under --out-dir (default outputs/):
    pytorch_pruned/clip_rn50_imagenet_pruned<P>pct_<criterion>[_iter].pt
    pruning_logs/clip_rn50_imagenet_<criterion>_<P>pct[_iter].json
    tflite_int8_litert/<.pt stem>_int8.tflite
    edgetpu/<.pt stem>_int8_edgetpu.tflite    with --edgetpu
"""
import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torchvision.datasets import ImageFolder

import int8_pruning
from example_data import (IMPORTANCES, build_transforms, imagenet_center_crop,
                          load_text_embeddings)

ImageFile.LOAD_TRUNCATED_IMAGES = True

STEM = "clip_rn50_imagenet"
MEAN, STD = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
# Global ranking, as for the other backbones here; the head is what is protected.
PRUNING = {"global_pruning": True, "iterative_steps": 200, "max_pruning_ratio": 0.9}

# Imagenette wnid -> CLIP label name, for the 10-class ImageNet subset.
WNID_TO_NAME = {
    "n01440764": "tench", "n02102040": "English springer", "n02979186": "cassette player",
    "n03000684": "chain saw", "n03028079": "church", "n03394916": "French horn",
    "n03417042": "garbage truck", "n03425413": "gas pump", "n03445777": "golf ball",
    "n03888257": "parachute",
}
# A subset of OpenAI's prompt templates; averaging them adds a point or two of zero-shot.
TEMPLATES = ["a photo of a {}.", "a bad photo of a {}.", "a photo of the {}.",
             "a cropped photo of a {}.", "a photo of one {}.", "a close-up photo of a {}.",
             "a bright photo of a {}."]


def ignored_layers(model):
    """Every Linear in the AttentionPool2d head (q, k, v, proj)."""
    return [m for m in model.head.modules() if isinstance(m, nn.Linear)]


def load_text(path):
    """The [classes, D] text embeddings, L2-normalized, and their label names."""
    text, labels = load_text_embeddings(path)
    return text / np.linalg.norm(text, axis=1, keepdims=True), labels


def labelled_images(images_dir, labels):
    """(path, label row) for every image under a class folder whose name is a label."""
    row = {n: i for i, n in enumerate(labels)}
    items = []
    for sub in sorted(p for p in Path(images_dir).iterdir() if p.is_dir()):
        name = WNID_TO_NAME.get(sub.name, sub.name)
        if name in row:
            for ext in ("*.JPEG", "*.jpeg", "*.jpg", "*.png"):
                items += [(f, row[name]) for f in sorted(sub.glob(ext))]
    return items


def zero_shot(embed, items, text, preprocess):
    """Top-1 and top-5 in %: each image's embedding against every label's, by cosine."""
    c1 = c5 = 0
    for path, label in items:
        emb = embed(preprocess(Image.open(path).convert("RGB")))
        top5 = np.argsort(text @ (emb / (np.linalg.norm(emb) + 1e-12)))[-5:][::-1]
        c1, c5 = c1 + (top5[0] == label), c5 + (label in top5)
    n = max(1, len(items))
    return {"top1": 100 * c1 / n, "top5": 100 * c5 / n, "n_images": len(items)}


# fetch-baseline / text-embeddings --------------------------------------------------------

def fetch_baseline(args):
    """Write timm's resnet50_clip (OpenAI weights, 1024-d CLIP embedding) as the baseline."""
    out = Path(args.models_dir) / "clip_rn50.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    # num_classes left at 1024 keeps the projection into the space the text tower shares.
    torch.save(timm.create_model("resnet50_clip", pretrained=True), out)
    print(f"-> {out}")


def text_embeddings(args):
    """Encode label names with open_clip's RN50 text tower, offline."""
    # Imported here, not at the top: open_clip is only needed for this offline step.
    import open_clip

    if args.preset:
        names = [WNID_TO_NAME[w] for w in sorted(WNID_TO_NAME)]
    elif args.classes:
        names = [c.strip() for c in args.classes.split(",") if c.strip()]
    else:
        names = [ln.strip() for ln in args.classes_file.read_text().splitlines() if ln.strip()]
    name = args.name or args.preset or (args.classes_file.stem if args.classes_file else "custom")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms("RN50", pretrained="openai")
    model, tok = model.eval().to(device), open_clip.get_tokenizer("RN50")
    rows = []
    with torch.no_grad():
        for label in names:
            prompts = [t.format(label) for t in (TEMPLATES if args.ensemble else TEMPLATES[:1])]
            te = model.encode_text(tok(prompts).to(device)).float()
            te = (te / te.norm(dim=-1, keepdim=True)).mean(dim=0)
            rows.append((te / te.norm()).cpu())
    out = Path(args.models_dir) / f"clip_rn50_text_{name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    # The layout relu-clip publishes its text embeddings in, so both examples read one format.
    np.savez(out, embs=torch.stack(rows).numpy().astype(np.float32), labels=np.array(names))
    print(f"-> {out}  ({len(names)} labels)")


# prune ------------------------------------------------------------------------------------

def prune(args):
    """Prune to each of --ratios, distil against the unpruned tower, save each rung."""
    src = Path(args.models_dir) / "clip_rn50.pt"
    if not src.exists():
        raise SystemExit(f"{src} is missing: run `python {sys.argv[0]} fetch-baseline` first")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    pruned_dir, log_dir = Path(args.out_dir) / "pytorch_pruned", Path(args.out_dir) / "pruning_logs"
    pruned_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    mode = "_iter" if args.iterative else ""
    train_tf, eval_tf = build_transforms(224, mean=MEAN, std=STD)
    text = labels = None
    if args.text_emb and args.val_images_dir:
        text, labels = load_text(args.text_emb)
        items = labelled_images(args.val_images_dir, labels)[:args.eval_images or None]

    def evaluate(model):
        if text is None:
            return None
        model.eval()
        with torch.no_grad():
            return zero_shot(lambda x: model(x.unsqueeze(0).to(device)).float().flatten().cpu().numpy(),
                             items, text, eval_tf)

    def dense():
        int8_pruning.seed_everything(args.seed)
        model = torch.load(src, map_location="cpu", weights_only=False).to(device)
        # The teacher is the dense model, pinned for a whole iterative trajectory: zero-shot
        # is scored against text vectors computed offline, so a teacher that moved with the
        # student would make the metric incomparable.
        teacher = copy.deepcopy(model).to(device)
        g = torch.Generator()
        g.manual_seed(args.seed)
        loader = torch.utils.data.DataLoader(
            ImageFolder(str(Path(args.dataset_dir) / "Imagenet_1k" / "train"), transform=train_tf),
            batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
            drop_last=True, generator=g)
        return model, teacher, loader, {"params": int8_pruning.count_params(model),
                                        "zero_shot": evaluate(model)}

    def recover_and_save(model, teacher, loader, ref, ratio, info):
        pct = round(100 * ratio)
        log = {"model": STEM, "importance": args.importance, "ratio": ratio,
               "iterative": args.iterative, "dense": ref, **info._asdict(),
               "pruned": evaluate(model)}
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        history = int8_pruning.distill(
            model, teacher, loader, args.epochs, loss="cosine", optimizer=opt,
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs),
            evaluate=(lambda: {"zeroshot_top1": evaluate(model)["top1"]}) if text is not None else None)
        log["history"] = [r._asdict() for r in history]
        log["final"] = evaluate(model)
        out = pruned_dir / f"{STEM}_pruned{pct}pct_{args.importance}{mode}.pt"
        torch.save(model, out)
        (log_dir / f"{STEM}_{args.importance}_{pct}pct{mode}.json").write_text(
            json.dumps(log, indent=2, default=float))
        print(f"-> {out}  zero-shot {log['final'] or 'not scored'}")

    kw = dict(importance=args.importance, **PRUNING)
    if args.iterative:
        model, teacher, loader, ref = dense()
        example = torch.randn(1, 3, 224, 224).to(device)
        for ratio, info in int8_pruning.prune_trajectory(
                model, example, args.ratios, ignored_layers=ignored_layers(model), **kw):
            recover_and_save(model, teacher, loader, ref, ratio, info)
        return
    for ratio in args.ratios:
        model, teacher, loader, ref = dense()
        example = torch.randn(1, 3, 224, 224).to(device)
        info = int8_pruning.prune(model, example, ratio, ignored_layers=ignored_layers(model), **kw)
        recover_and_save(model, teacher, loader, ref, ratio, info)


# int8 -------------------------------------------------------------------------------------

def center_crop(pil, size=224):
    """Resize the shorter side to 256/224 of `size` and center-crop: the calibration's transform."""
    target = int(round(size * 256 / 224))
    w, h = pil.size
    new = (target, int(round(h * target / w))) if w < h else (int(round(w * target / h)), target)
    pil = pil.resize(new, Image.BILINEAR)
    left, top = (new[0] - size) // 2, (new[1] - size) // 2
    arr = np.asarray(pil.crop((left, top, left + size, top + size)), dtype=np.float32) / 255.0
    return (arr - np.asarray(MEAN, np.float32)) / np.asarray(STD, np.float32)


def int8(args):
    """Export --pt to int8 (or take --tflite as it is), then score it zero-shot."""
    tflite = args.tflite
    if args.pt:
        root = Path(args.dataset_dir) / "Imagenet_1k"
        calib = imagenet_center_crop(args.num_calib, 224, 224, MEAN, STD, args.seed, root)
        model = torch.load(args.pt, map_location="cpu", weights_only=False)
        tflite = Path(args.out_dir) / "tflite_int8_litert" / f"{Path(args.pt).stem}_int8.tflite"
        info = int8_pruning.export_int8(model, calib, tflite)
        print(f"-> {tflite}  ({info.size_bytes:,} bytes)")
    if args.edgetpu:
        tflite = int8_pruning.compile_edgetpu(tflite, Path(args.out_dir) / "edgetpu")
    text, labels = load_text(args.text_emb)
    items = labelled_images(args.images_dir, labels)[:args.num_images or None]
    it = int8_pruning.make_interpreter(tflite)
    score = zero_shot(lambda hwc: int8_pruning.run_tflite(it, hwc)[0].flatten(),
                      items, text, center_crop)
    print(f"{Path(tflite).name}: int8 zero-shot top-1 {score['top1']:.2f}%  "
          f"top-5 {score['top5']:.2f}%  ({score['n_images']} images, {len(labels)} labels)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    for name, fn in (("fetch-baseline", fetch_baseline), ("text-embeddings", text_embeddings),
                     ("prune", prune), ("int8", int8)):
        s = sub.add_parser(name, help=fn.__doc__.splitlines()[0])
        s.set_defaults(fn=fn)
        s.add_argument("--seed", type=int, default=42)
        if name in ("fetch-baseline", "text-embeddings", "prune"):
            s.add_argument("--models-dir", type=Path, default=Path("outputs/models"))
        if name in ("prune", "int8"):
            s.add_argument("--dataset-dir", type=Path, default=Path("data/datasets"),
                           help="holds Imagenet_1k/train: distillation and calibration images")
            s.add_argument("--out-dir", type=Path, default=Path("outputs"))
    t = sub.choices["text-embeddings"]
    src = t.add_mutually_exclusive_group(required=True)
    src.add_argument("--preset", choices=["imagenette"])
    src.add_argument("--classes", help="comma-separated label names")
    src.add_argument("--classes-file", type=Path, help="one label name per line")
    t.add_argument("--name", help="output set name (default: the preset or file name)")
    t.add_argument("--ensemble", action="store_true", help="average the 7 prompt templates")
    r = sub.choices["prune"]
    r.add_argument("--ratios", type=float, nargs="+", required=True,
                   help="fractions of the parameters to remove, e.g. 0.3 0.5 0.7")
    r.add_argument("--importance", default="magnitude_l2",
                   choices=IMPORTANCES)
    r.add_argument("--iterative", action="store_true",
                   help="one trajectory through ascending --ratios instead of each from dense")
    r.add_argument("--epochs", type=int, default=10, help="distillation epochs (default: 10)")
    r.add_argument("--batch-size", type=int, default=64)
    r.add_argument("--text-emb", help="text embeddings for zero-shot scoring; without them "
                                      "and --val-images-dir the run is not scored")
    r.add_argument("--val-images-dir", help="labelled class folders to score zero-shot on")
    r.add_argument("--eval-images", type=int, default=0, help="cap on scored images (0 = all)")
    r.add_argument("--device", default=None)
    q = sub.choices["int8"]
    src = q.add_mutually_exclusive_group(required=True)
    src.add_argument("--pt", type=Path, help="a pruned .pt to export to int8 and score")
    src.add_argument("--tflite", type=Path, help="an int8 or _edgetpu.tflite to score")
    q.add_argument("--text-emb", required=True)
    q.add_argument("--images-dir", required=True, help="labelled class folders")
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
