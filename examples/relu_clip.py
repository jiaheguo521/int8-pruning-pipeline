"""relu-clip's efflite4 CLIP student: prune, recover by distillation, score it in int8.

    python examples/relu_clip.py fetch-baseline
    python examples/relu_clip.py prune --ratios 0.3 0.5
    python examples/relu_clip.py int8 --pt outputs/pytorch_pruned/<stem>.pt \
        --text-emb <published text embeddings>.npz --val-images-dir <class folders>

The baseline is `tf_efficientnet_lite4` distilled from CLIP ViT-L/14 into a 768-d
embedding (jiaheguo521/relu-clip). The model already fits the Edge TPU's on-chip memory,
so pruning it measures how far it can shrink before the embedding stops agreeing with the
dense one.

Recovery is distillation, 1 - cos(student, teacher), against a frozen copy of the
unpruned student on unlabelled ImageNet train images. `teacher_cos`, the mean cosine to
that teacher over a few batches, is scored after pruning, every epoch and at the end;
zero-shot top-1 too when --text-emb (the published .npz) and --val-images-dir are given,
in float during `prune` and in int8 by `int8`.
Channels are ranked within each layer by default (--global-pruning ranks across them):
global ranking with a magnitude criterion can strip this backbone's narrow early layers.

Outputs, under --out-dir (default outputs/):
    pytorch_pruned/relu_clip_imagenet_pruned<P>pct_<criterion>[_iter].pt
    pruning_logs/relu_clip_imagenet_<criterion>_<P>pct[_iter].json
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
import torch
import torch.nn.functional as F
import torch_pruning as tp
from huggingface_hub import hf_hub_download
from PIL import Image, ImageFile
from safetensors.torch import load_file
from torchvision.datasets import ImageFolder

import int8_pruning
from example_data import IMPORTANCES, imagenet_train_files, load_text_embeddings
from relu_clip_model import (CLIP_MEAN, CLIP_STD, DEFAULT_BACKBONE, DEFAULT_EMBED_DIM,
                             INPUT_SIZE, build_student, student_transform)

ImageFile.LOAD_TRUNCATED_IMAGES = True

STEM = "relu_clip_imagenet"
HF_REPO, HF_WEIGHTS = "jiaheguo521/relu-clip", "efflite4-vitl14/student.safetensors"


@torch.no_grad()
def teacher_cos(student, teacher, loader, device, max_batches=8):
    """Mean cos(student, teacher) over the first `max_batches` batches of `loader`."""
    student.eval()
    total, n = 0.0, 0
    for i, (images, _) in enumerate(loader):
        if i >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        se = F.normalize(student(images).float(), dim=-1)
        te = F.normalize(teacher(images).float(), dim=-1)
        total, n = total + (se * te).sum(-1).sum().item(), n + images.size(0)
    return total / max(1, n)


def folder_rows(val_dir, labels):
    """Each class folder with its text-embedding row: by name when every folder name is a
    label, else by position when the counts match (sorted wnids are class-index order)."""
    subs = sorted(p for p in Path(val_dir).iterdir() if p.is_dir())
    row = {n: i for i, n in enumerate(labels)}
    if all(p.name in row for p in subs):
        return [(p, row[p.name]) for p in subs]
    if len(subs) == len(labels):
        return list(zip(subs, range(len(subs))))
    raise SystemExit(f"cannot align {len(subs)} folders under {val_dir} with "
                     f"{len(labels)} text-embedding rows")


@torch.no_grad()
def zero_shot_top1(embed, val_dir, embs, labels, device, num_images=0):
    """Zero-shot top-1 in %, and the number of images scored. `embed` takes one image as
    the published preprocessing makes it, a CHW tensor, and returns its [1, D] embedding
    on `device`: the float model's, or the int8 model's."""
    tf = student_transform(INPUT_SIZE)
    text = torch.from_numpy(embs / np.linalg.norm(embs, axis=1, keepdims=True)).to(device)
    correct = n = 0
    for sub, row in folder_rows(val_dir, labels):
        files = [f for ext in ("*.JPEG", "*.jpeg", "*.jpg", "*.png") for f in sorted(sub.glob(ext))]
        for f in files:
            with Image.open(f) as im:
                emb = embed(tf(im.convert("RGB")))
            correct += int(int((text @ (emb / (emb.norm() + 1e-12)).squeeze(0)).argmax()) == row)
            n += 1
            if n == num_images:
                return 100 * correct / n, n
    return (100 * correct / n if n else 0.0), n


# fetch-baseline ---------------------------------------------------------------------------

def fetch_baseline(args):
    """Rebuild the published student from its Hugging Face weights, as a full nn.Module."""
    model = build_student(DEFAULT_BACKBONE, DEFAULT_EMBED_DIM, pretrained=False)
    model.load_state_dict(load_file(hf_hub_download(repo_id=HF_REPO, filename=HF_WEIGHTS)),
                          strict=True)
    model.eval()
    # A mis-shaped head survives every later stage and shows up only as collapsed zero-shot.
    with torch.no_grad():
        emb = model(torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE))
    if tuple(emb.shape) != (1, DEFAULT_EMBED_DIM):
        raise SystemExit(f"expected a [1, {DEFAULT_EMBED_DIM}] embedding, got {list(emb.shape)}")
    out = Path(args.models_dir) / "relu_clip.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, out)
    print(f"-> {out}  ({int8_pruning.count_params(model):,} parameters)")


# prune ------------------------------------------------------------------------------------

def prune(args):
    """Prune to each of --ratios, distil against the unpruned student, save each rung."""
    src = Path(args.models_dir) / "relu_clip.pt"
    if not src.exists():
        raise SystemExit(f"{src} is missing: run `python {sys.argv[0]} fetch-baseline` first")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    pruned_dir, log_dir = Path(args.out_dir) / "pytorch_pruned", Path(args.out_dir) / "pruning_logs"
    pruned_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    mode = "_iter" if args.iterative else ""
    embs = labels = None
    if args.text_emb and args.val_images_dir:
        embs, labels = load_text_embeddings(args.text_emb)

    def evaluate(model, teacher, loader, stage):
        out = {}
        if embs is not None:
            model.eval()
            out["zeroshot_top1"], out["n_eval"] = zero_shot_top1(
                lambda x: model(x.unsqueeze(0).to(device)).float(), args.val_images_dir,
                embs, labels, device, args.eval_images)
        if stage != "dense":
            out["teacher_cos"] = teacher_cos(model, teacher, loader, device)
        return out or None

    def dense():
        int8_pruning.seed_everything(args.seed)
        model = torch.load(src, map_location="cpu", weights_only=False).to(device)
        # Pinned to the dense student for a whole iterative trajectory, so teacher_cos keeps
        # meaning cosine to the unpruned model at every rung.
        teacher = copy.deepcopy(model).to(device)
        g = torch.Generator()
        g.manual_seed(args.seed)
        loader = torch.utils.data.DataLoader(
            ImageFolder(str(Path(args.dataset_dir) / "Imagenet_1k" / "train"),
                        transform=student_transform(INPUT_SIZE)),
            batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
            drop_last=True, generator=g)
        ref = {"params": int8_pruning.count_params(model),
               "eval": evaluate(model, teacher, loader, "dense"),
               "macs": tp.utils.count_ops_and_params(
                   model, torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device))[0]}
        return model, teacher, loader, ref

    def recover_and_save(model, teacher, loader, ref, ratio, info):
        pct = round(100 * ratio)
        log = {"model": STEM, "importance": args.importance, "ratio": ratio,
               "iterative": args.iterative, "global_pruning": bool(args.global_pruning),
               "round_to": args.round_to or None, "dense": ref, **info._asdict(),
               "macs": tp.utils.count_ops_and_params(
                   model, torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device))[0],
               "pruned": evaluate(model, teacher, loader, "pruned")}
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        history = int8_pruning.distill(
            model, teacher, loader, args.epochs, loss="cosine", optimizer=opt,
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs),
            evaluate=lambda: {"teacher_cos": teacher_cos(model, teacher, loader, device)})
        log["history"] = [r._asdict() for r in history]
        log["final"] = evaluate(model, teacher, loader, "final")
        out = pruned_dir / f"{STEM}_pruned{pct}pct_{args.importance}{mode}.pt"
        torch.save(model, out)
        (log_dir / f"{STEM}_{args.importance}_{pct}pct{mode}.json").write_text(
            json.dumps(log, indent=2, default=float))
        print(f"-> {out}  {log['final']}")

    kw = dict(importance=args.importance, global_pruning=bool(args.global_pruning),
              iterative_steps=200, max_pruning_ratio=0.9, round_to=args.round_to or None)
    head = lambda model: [model.head]      # pins the 768-d output; the backbone prunes
    if args.iterative:
        model, teacher, loader, ref = dense()
        example = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(device)
        for ratio, info in int8_pruning.prune_trajectory(model, example, args.ratios,
                                                         ignored_layers=head(model), **kw):
            recover_and_save(model, teacher, loader, ref, ratio, info)
        return
    for ratio in args.ratios:
        model, teacher, loader, ref = dense()
        example = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(device)
        info = int8_pruning.prune(model, example, ratio, ignored_layers=head(model), **kw)
        recover_and_save(model, teacher, loader, ref, ratio, info)


# int8 -------------------------------------------------------------------------------------

def published_calibration(root, num, seed):
    """ImageNet train images under the published preprocessing (`student_transform`):
    Resize(224, BICUBIC) truncating the long side, CenterCrop rounding its offset. The
    usual 256/224 center crop would calibrate on pixels the deployed model never sees."""
    s = INPUT_SIZE
    picks = imagenet_train_files(root, num, seed)
    out = np.zeros((len(picks), s, s, 3), dtype=np.uint8)
    for i, p in enumerate(picks):
        with Image.open(p) as img:
            img = img.convert("RGB")
            w, h = img.size
            new = (s, int(h * s / w)) if w < h else (int(w * s / h), s)
            img = img.resize(new, Image.BICUBIC)
            left, top = int(round((new[0] - s) / 2.0)), int(round((new[1] - s) / 2.0))
            out[i] = np.asarray(img.crop((left, top, left + s, top + s)), dtype=np.uint8)
    return int8_pruning.export.normalize_nhwc(out, CLIP_MEAN, CLIP_STD)


def int8(args):
    """Export --pt to int8 (or take --tflite as it is), then score it zero-shot."""
    tflite = args.tflite
    if args.pt:
        calib = published_calibration(Path(args.dataset_dir) / "Imagenet_1k", args.num_calib,
                                      args.seed)
        model = torch.load(args.pt, map_location="cpu", weights_only=False)
        tflite = Path(args.out_dir, "tflite_int8_litert", f"{Path(args.pt).stem}_int8.tflite")
        info = int8_pruning.export_int8(model, calib, tflite)
        print(f"-> {tflite}  ({info.size_bytes:,} bytes)")
    if args.edgetpu:
        tflite = int8_pruning.compile_edgetpu(tflite, Path(args.out_dir) / "edgetpu")
    if not (args.text_emb and args.val_images_dir):
        return
    embs, labels = load_text_embeddings(args.text_emb)
    it = int8_pruning.make_interpreter(tflite)

    def embed(x):
        out = int8_pruning.run_tflite(it, x.permute(1, 2, 0).numpy())[0]
        return torch.from_numpy(out.reshape(1, -1))

    top1, n = zero_shot_top1(embed, args.val_images_dir, embs, labels, torch.device("cpu"),
                             args.num_images)
    print(f"{Path(tflite).name}: int8 zero-shot top-1 {top1:.2f}%  ({n} images, "
          f"{len(labels)} labels)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    for name, fn in (("fetch-baseline", fetch_baseline), ("prune", prune), ("int8", int8)):
        s = sub.add_parser(name, help=fn.__doc__.splitlines()[0])
        s.set_defaults(fn=fn)
        s.add_argument("--seed", type=int, default=42)
        if name != "int8":
            s.add_argument("--models-dir", type=Path, default=Path("outputs/models"))
        if name != "fetch-baseline":
            s.add_argument("--dataset-dir", type=Path, default=Path("data/datasets"),
                           help="holds Imagenet_1k/train: distillation and calibration images")
            s.add_argument("--out-dir", type=Path, default=Path("outputs"))
    r = sub.choices["prune"]
    r.add_argument("--ratios", type=float, nargs="+", required=True,
                   help="fractions of the parameters to remove, e.g. 0.3 0.5 0.7")
    r.add_argument("--importance", default="magnitude_l2",
                   choices=IMPORTANCES)
    r.add_argument("--iterative", action="store_true",
                   help="one trajectory through ascending --ratios instead of each from dense")
    r.add_argument("--global-pruning", action="store_true",
                   help="rank channels across layers instead of within each")
    r.add_argument("--round-to", type=int, default=0,
                   help="round surviving channel counts up to a multiple of N (0 = off); "
                        "the Edge TPU's array is 64 wide. Untested here")
    r.add_argument("--epochs", type=int, default=10, help="distillation epochs (default: 10)")
    r.add_argument("--batch-size", type=int, default=64)
    r.add_argument("--text-emb", help="the published text-embedding .npz, for zero-shot scoring")
    r.add_argument("--val-images-dir", help="labelled class folders to score zero-shot on")
    r.add_argument("--eval-images", type=int, default=0, help="cap on scored images (0 = all)")
    r.add_argument("--device", default=None)
    q = sub.choices["int8"]
    src = q.add_mutually_exclusive_group(required=True)
    src.add_argument("--pt", type=Path, help="a pruned .pt to export to int8 and score")
    src.add_argument("--tflite", type=Path, help="an int8 or _edgetpu.tflite to score")
    q.add_argument("--text-emb", help="the published text-embedding .npz; without it and "
                                      "--val-images-dir the model is exported, not scored")
    q.add_argument("--val-images-dir", help="labelled class folders to score zero-shot on")
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
