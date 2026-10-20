"""Youtu Re-ID (reid_youtu_lite): prune, recover by distillation, export and score mAP in int8.

    python examples/youreid.py fetch-baseline
    python examples/youreid.py prune --ratios 0.5 0.7 0.9
    python examples/youreid.py int8 --pt outputs/pytorch_pruned/<stem>.pt

The baseline is Tencent YouReID's `baseline_lite` (== opencv_zoo person_reid_youtu_2021nov):
ResNet-50 -> cat(gap, gmp) -> 1x1 conv -> 768-d, trained on four datasets (Market-1501,
Duke, MSMT17, CUHK03). Only Market-1501 is available, so recovery distils against a frozen
copy of the unpruned model on unlabelled Market-1501 crops (--extra-image-dirs adds crops from the deployment camera; the teacher
supplies the targets). Input is 256x128 (H x W), resized directly: the crops are already
tight.

The 768-d embedding layer is protected, so its input width follows the backbone while its
output stays a drop-in for a 768-d gallery. That layer is a fixed cost the body cannot shed,
and past roughly 80% it is most of the model: --embed-dim prunes it first and hands the
budget to the backbone, with --distill-loss simmat, which matches the batch's cosine
similarity structure and so works across embedding widths. Channels are ranked within each
layer by default: global ranking compares raw norms across layers, and ResNet-50's deep
filters out-norm the stem's, so it can delete the stem.

Outputs, under --out-dir (default outputs/):
    pytorch_pruned/reid_youtu_lite[_e<D>|_simmat]_market1501_pruned<P>pct_<criterion>.pt
    pruning_logs/<same stem>_<criterion>_<P>pct.json
    tflite_int8_litert/<.pt stem>_int8.tflite
    edgetpu/<.pt stem>_int8_edgetpu.tflite    with --edgetpu
"""
import argparse
import copy
import json
import logging
import random
import re
import sys
from pathlib import Path

import gdown
import numpy as np
import torch
import torch.nn.functional as F
import torch_pruning as tp
from PIL import Image, ImageFile
from torchvision import transforms

import int8_pruning
from example_data import IMPORTANCES
from youreid_model import build_youtu_lite

ImageFile.LOAD_TRUNCATED_IMAGES = True

H, W = 256, 128
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
EMBED_DIM = 768
GDRIVE_ID = "1l-8Lj9OPs4D6qKGAljbJgZuxGvENkDjl"
# A per-layer cap of 0.97: with ranking inside each layer, 0.9 would stop the deep rungs short.
PRUNING = {"iterative_steps": 400, "max_pruning_ratio": 0.97}


def stem(embed_dim, loss):
    """768 + cosine is the drop-in mainline; the others are the head and loss ablations."""
    if embed_dim != EMBED_DIM:
        return f"reid_youtu_lite_e{embed_dim}_market1501"
    return "reid_youtu_lite_market1501" if loss == "cosine" else f"reid_youtu_lite_{loss}_market1501"


def market_dir(dataset_dir, split):
    found = sorted(Path(dataset_dir, "market1501").glob(f"**/{split}"))
    if not found:
        raise SystemExit(f"no {split}/ under {dataset_dir}/market1501")
    return found[0]


class Crops(torch.utils.data.Dataset):
    """Unlabelled crops: direct resize to 256x128, horizontal flip, ImageNet normalization."""

    def __init__(self, paths):
        self.paths = paths
        self.tf = transforms.Compose([
            transforms.Resize((H, W), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


@torch.no_grad()
def teacher_cos(student, teacher, loader, device, max_batches=8):
    """Mean cos(student, teacher) over a few batches; None once the widths differ."""
    if student.embedding_layer.out_channels != teacher.embedding_layer.out_channels:
        return None
    student.eval()
    total, n = 0.0, 0
    for i, images in enumerate(loader):
        if i >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        se = F.normalize(student(images).float(), dim=-1)
        te = F.normalize(teacher(images).float(), dim=-1)
        total, n = total + (se * te).sum(-1).sum().item(), n + images.size(0)
    return total / max(1, n)


def shrink_head(model, embed_dim, example):
    """Prune the embedding layer's output channels to `embed_dim`, keeping the
    highest-L2-norm filters, before the body is pruned."""
    conv = model.embedding_layer
    norms = conv.weight.detach().flatten(1).norm(p=2, dim=1)
    drop = torch.argsort(norms)[: conv.out_channels - embed_dim].tolist()
    dg = tp.DependencyGraph().build_dependency(model, example_inputs=example)
    dg.get_pruning_group(conv, tp.prune_conv_out_channels, idxs=drop).prune()


# fetch-baseline ---------------------------------------------------------------------------

def fetch_baseline(args):
    """Download the YouReID checkpoint (~226 MB) and save it as a full nn.Module."""
    models_dir = Path(args.models_dir)
    cache = models_dir / "reid_weights_cache" / "reid_youtu_lite.pth"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        gdown.download("https://drive.google.com/uc?id=" + GDRIVE_ID, str(cache), quiet=False)
    model = build_youtu_lite(cache)          # strict load; collapses split_bn to plain BN
    with torch.no_grad():
        out = model(torch.randn(1, 3, H, W))
    if tuple(out.shape) != (1, EMBED_DIM):
        raise SystemExit(f"expected a [1, {EMBED_DIM}] embedding, got {tuple(out.shape)}")
    dst = models_dir / "reid_youtu_lite.pt"
    torch.save(model, dst)
    print(f"-> {dst}  ({int8_pruning.count_params(model):,} parameters)")


# prune ------------------------------------------------------------------------------------

def prune(args):
    """Prune to each of --ratios, distil against the unpruned model, save each rung."""
    if args.embed_dim > EMBED_DIM:
        raise SystemExit(f"--embed-dim can only shrink the {EMBED_DIM}-d head")
    if args.embed_dim != EMBED_DIM and args.distill_loss == "cosine":
        raise SystemExit(f"--embed-dim {args.embed_dim} needs --distill-loss simmat: a "
                         f"{args.embed_dim}-d embedding has no cosine to the teacher's 768-d one")
    src = Path(args.models_dir) / "reid_youtu_lite.pt"
    if not src.exists():
        raise SystemExit(f"{src} is missing: run `python {sys.argv[0]} fetch-baseline` first")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    name = stem(args.embed_dim, args.distill_loss)
    pruned_dir, log_dir = Path(args.out_dir) / "pytorch_pruned", Path(args.out_dir) / "pruning_logs"
    pruned_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Market's train crops (junk and distractor ids left out), plus any extra unlabelled dirs.
    paths = sorted(p for p in market_dir(args.dataset_dir, "bounding_box_train").glob("*.jpg")
                   if not p.name.startswith(("-1", "0000")))
    for d in args.extra_image_dirs or []:
        paths += sorted(p for ext in ("*.jpg", "*.jpeg", "*.png") for p in Path(d).rglob(ext))

    for ratio in args.ratios:
        int8_pruning.seed_everything(args.seed)
        model = torch.load(src, map_location="cpu", weights_only=False).to(device)
        dense_params = int8_pruning.count_params(model)
        teacher = copy.deepcopy(model).to(device)
        g = torch.Generator()
        g.manual_seed(args.seed)
        loader = torch.utils.data.DataLoader(
            Crops(paths), batch_size=args.batch_size, shuffle=True, num_workers=4,
            pin_memory=True, drop_last=True, generator=g)
        example = torch.randn(1, 3, H, W).to(device)
        if args.embed_dim != EMBED_DIM:
            shrink_head(model, args.embed_dim, example)
        # The ratio is of the dense model, head shrinking included.
        info = int8_pruning.prune(model, example, ratio, dense_params=dense_params,
                                  importance=args.importance,
                                  ignored_layers=[model.embedding_layer],
                                  global_pruning=bool(args.global_pruning), **PRUNING)
        head = model.embedding_layer
        pct = round(100 * ratio)
        log = {"model": name, "importance": args.importance, "ratio": ratio,
               "dense_params": dense_params, **info._asdict(), "n_distill_images": len(paths),
               "head_params": head.in_channels * head.out_channels,
               "pruned_teacher_cos": teacher_cos(model, teacher, loader, device)}
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        history = int8_pruning.distill(
            model, teacher, loader, args.epochs, loss=args.distill_loss, optimizer=opt,
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs),
            evaluate=lambda: {"teacher_cos": teacher_cos(model, teacher, loader, device)})
        log["history"] = [r._asdict() for r in history]
        log["final_teacher_cos"] = teacher_cos(model, teacher, loader, device)
        out = pruned_dir / f"{name}_pruned{pct}pct_{args.importance}.pt"
        torch.save(model.eval(), out)
        (log_dir / f"{name}_{args.importance}_{pct}pct.json").write_text(
            json.dumps(log, indent=2, default=float))
        print(f"-> {out}  teacher_cos {log['pruned_teacher_cos']} -> {log['final_teacher_cos']}")


# int8 -------------------------------------------------------------------------------------

def calibration(dataset_dir, num, seed):
    """`num` random Market-1501 train crops, resized directly to 256x128."""
    jpgs = sorted(p for p in market_dir(dataset_dir, "bounding_box_train").glob("*.jpg")
                  if not p.name.startswith(("-1", "0000")))
    picks = random.Random(seed).sample(jpgs, k=min(num, len(jpgs)))
    out = np.zeros((len(picks), H, W, 3), dtype=np.uint8)
    for i, p in enumerate(picks):
        with Image.open(p) as img:
            out[i] = np.asarray(img.convert("RGB").resize((W, H), Image.BILINEAR), dtype=np.uint8)
    return int8_pruning.export.normalize_nhwc(out, MEAN, STD)


def split(dataset_dir, name):
    """(paths, person ids, camera ids) of a Market split; id -1 (junk) dropped, 0 kept."""
    paths, pids, cams = [], [], []
    for p in sorted(market_dir(dataset_dir, name).glob("*.jpg")):
        m = re.match(r"^(-?\d+)_c(\d+)", p.name)
        if m and int(m.group(1)) != -1:
            paths.append(p)
            pids.append(int(m.group(1)))
            cams.append(int(m.group(2)))
    return paths, np.asarray(pids, np.int64), np.asarray(cams, np.int64)


def embed_all(it, paths):
    """[N, D] int8-model embeddings, one 256x128 crop at a time."""
    mean, std = np.asarray(MEAN, np.float32), np.asarray(STD, np.float32)
    rows = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((W, H), Image.BILINEAR)
        hwc = (np.asarray(img, dtype=np.float32) / 255.0 - mean) / std
        rows.append(int8_pruning.run_tflite(it, hwc)[0].flatten())
    return np.stack(rows)


def int8(args):
    """Export --pt to int8 (or take --tflite as it is), then score Market-1501 mAP / CMC."""
    tflite = args.tflite
    if args.pt:
        model = torch.load(args.pt, map_location="cpu", weights_only=False)
        tflite = Path(args.out_dir) / "tflite_int8_litert" / f"{Path(args.pt).stem}_int8.tflite"
        # litert-torch cannot lower AdaptiveMaxPool2d; the patch rewrites the gmp branch.
        info = int8_pruning.export_int8(
            model, calibration(args.dataset_dir, args.num_calib, args.seed), tflite,
            graph_patches=[int8_pruning.export.adaptive_max_pool])
        print(f"-> {tflite}  ({info.size_bytes:,} bytes)")
    if args.edgetpu:
        tflite = int8_pruning.compile_edgetpu(tflite, Path(args.out_dir) / "edgetpu")
    # Imported here, not at the top: torchreid is installed separately with --no-deps, and
    # fetch-baseline and prune do not need it.
    from torchreid.metrics.distance import compute_distance_matrix
    from torchreid.metrics.rank import evaluate_rank

    it = int8_pruning.make_interpreter(tflite)
    q_paths, q_pids, q_cams = split(args.dataset_dir, "query")
    g_paths, g_pids, g_cams = split(args.dataset_dir, "bounding_box_test")
    qf, gf = embed_all(it, q_paths), embed_all(it, g_paths)
    dist = compute_distance_matrix(torch.from_numpy(qf), torch.from_numpy(gf), metric="cosine")
    cmc, m_ap = evaluate_rank(dist.numpy(), q_pids, g_pids, q_cams, g_cams, max_rank=10)
    print(f"{Path(tflite).name}: int8 mAP {100 * m_ap:.2f}%  rank-1 {100 * cmc[0]:.2f}%  "
          f"rank-5 {100 * cmc[4]:.2f}%  ({len(q_paths)} query, {len(g_paths)} gallery)")


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
                           help="holds market1501/")
            s.add_argument("--out-dir", type=Path, default=Path("outputs"))
    r = sub.choices["prune"]
    r.add_argument("--ratios", type=float, nargs="+", required=True,
                   help="fractions of the parameters to remove, e.g. 0.5 0.7 0.9")
    r.add_argument("--importance", default="magnitude_l2",
                   choices=IMPORTANCES)
    r.add_argument("--embed-dim", type=int, default=EMBED_DIM,
                   help="below 768, prune the embedding layer first (needs --distill-loss simmat)")
    r.add_argument("--distill-loss", default="cosine", choices=("cosine", "simmat"))
    r.add_argument("--global-pruning", action="store_true",
                   help="rank channels across layers instead of within each")
    r.add_argument("--extra-image-dirs", nargs="+", help="unlabelled crops to distil on as well")
    r.add_argument("--epochs", type=int, default=30, help="distillation epochs (default: 30)")
    r.add_argument("--batch-size", type=int, default=64)
    r.add_argument("--device", default=None)
    q = sub.choices["int8"]
    src = q.add_mutually_exclusive_group(required=True)
    src.add_argument("--pt", type=Path, help="a pruned .pt to export to int8 and score")
    src.add_argument("--tflite", type=Path, help="an int8 or _edgetpu.tflite to score")
    q.add_argument("--num-calib", type=int, default=100)
    q.add_argument("--edgetpu", action="store_true",
                   help="compile for the Edge TPU into <out-dir>/edgetpu/ and score the "
                        "compiled model on an attached Coral")
    args = p.parse_args(argv)
    logging.basicConfig(format="%(message)s")
    logging.getLogger("int8_pruning").setLevel(logging.INFO)
    args.fn(args)


if __name__ == "__main__":
    main()
