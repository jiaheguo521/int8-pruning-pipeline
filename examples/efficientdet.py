"""EfficientDet-Lite on COCO: prune the backbone, fine-tune, export and score mAP in int8.

    python examples/efficientdet.py fetch-baseline --model efficientdet_lite1
    python examples/efficientdet.py prune --model efficientdet_lite1 --ratios 0.3 0.5 --coco-subset minitrain
    python examples/efficientdet.py int8 --pt outputs/pytorch_pruned/<stem>.pt

Lite1 runs at 384, Lite2 at 448; the checkpoints are the ones Coral's own *_ptq.tflite
came from (rwightman's effdet). Backbone-only pruning: the BiFPN, both heads and the stem
conv are protected, and --ratios is measured over `model.backbone` (--scope full measures
it over the whole model instead, written under a `_full` stem; a high target can then land
short, so read `actual_ratio` from the log).

Recovery fine-tunes on COCO through effdet's DetBenchTrain, SGD lr 1e-3, cosine after a
one-epoch warmup, the best val2017 mAP epoch kept. The int8 model ends at the 10 raw head
tensors (5 levels x class / box); anchor decoding and NMS run on the CPU, as in Coral's
own detection examples. Two graph rewrites (the BiFPN's stacked sum, timm's SAME max-pool)
keep the whole graph on the Edge TPU.

Outputs, under --out-dir (default outputs/):
    pytorch_pruned/<model>_coco-<subset>_pruned<P>pct_<criterion>[_full].pt
    pruning_logs/<model>_coco-<subset>_<criterion>_<P>pct[_full].json
    tflite_int8_litert/<.pt stem>_int8.tflite
    edgetpu/<.pt stem>_int8_edgetpu.tflite    with --edgetpu
"""
import argparse
import contextlib
import json
import logging
import sys
from pathlib import Path

import effdet.efficientdet as E
import numpy as np
import timm.layers.pool2d_same as P
import torch
import torch.nn.functional as F
import torch_pruning as tp
from effdet import DetBenchPredict, DetBenchTrain, create_model, get_efficientdet_config
from effdet.anchors import Anchors
# Private to effdet, and what DetBenchPredict itself decodes with: the int8 heads go
# through the float model's own post-processing. Pinned below 0.5 in pyproject.
from effdet.bench import _batch_detection, _post_process
from effdet.data import create_loader
from effdet.data.dataset import DetectionDatset
from effdet.data.parsers import create_parser
from effdet.data.parsers.parser_config import CocoParserCfg
from PIL import Image
from pycocotools.coco import COCO
from tqdm.auto import tqdm

import int8_pruning
from example_data import (COCO_TRAIN_SUBSETS, IMPORTANCES, coco_map, coco_val2017,
                          get_num_workers, warmup_cosine)

SIZE = {"efficientdet_lite1": 384, "efficientdet_lite2": 448}
NUM_CLASSES = 90
PRUNING = {"global_pruning": True, "iterative_steps": 400, "max_pruning_ratio": 0.9}


def norm(model_name):
    """(mean, std) of the config the checkpoint was trained with: 0.5 / 0.5, not ImageNet's."""
    cfg = get_efficientdet_config(f"tf_{model_name}")
    return tuple(cfg["mean"]), tuple(cfg["std"])


def ignored_layers(model):
    """Everything outside the backbone, and the backbone's stem conv."""
    return [model.fpn, model.class_net, model.box_net, model.backbone.conv_stem]


# Graph rewrites for the int8 export -------------------------------------------------------

@contextlib.contextmanager
def fpn_sum():
    """FpnCombine's torch.stack(nodes, -1).sum(-1) as a chain of adds.

    `weight_method` is 'sum' for every lite config, so this is arithmetically identical,
    and it removes the rank-5 CONCAT / RESHAPE / SUM the Edge TPU cannot map; without it
    the TPU subgraph stops at the backbone.
    """
    orig = E.FpnCombine.forward

    def fwd(self, x):
        nodes = [rs(x[o]) for o, rs in zip(self.inputs_offsets, self.resample.values())]
        if self.weight_method != "sum":
            return orig(self, x)
        out = nodes[0]
        for n in nodes[1:]:
            out = out + n
        return out

    E.FpnCombine.forward = fwd
    try:
        yield
    finally:
        E.FpnCombine.forward = orig


def _same_pool_args(n, k, s):
    """(padding, ceil_mode) of a torch max_pool2d equal to TF SAME on extent n.

    TF SAME pads pad_total = (ceil(n/s) - 1) * s + k - n, split (pad_total // 2, the rest).
    Even pad_total is symmetric already; odd puts one extra on the right, which is what
    ceil_mode adds. Lite1 pools 48 -> 24 -> 12 -> 6, all even, so ceil_mode at every site;
    Lite2 pools 56 -> 28 -> 14 -> 7 -> 4, and the odd 7 needs padding 1 instead. One site
    wrong changes the network: P7 comes out 3x3 instead of 4x4. None when torch cannot
    express the padding (it must be at most k / 2).
    """
    pad_total = max((-(-n // s) - 1) * s + k - n, 0)
    pad = pad_total // 2
    return None if pad * 2 > k else (pad, bool(pad_total % 2))


@contextlib.contextmanager
def pool_same():
    """timm's SAME max-pool (F.pad with -inf, then a VALID pool) as one padded pool.

    litert-torch keeps that PAD in NCHW behind three transposes per site, and the PAD is a
    PADV2 the compiler declines because its -inf fill is never the zero point. The form is
    decided per site from the actual input extent (`_same_pool_args`).
    """
    orig = P.MaxPool2dSame.forward

    def fwd(self, x):
        k = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size,) * 2
        s = self.stride if isinstance(self.stride, tuple) else (self.stride,) * 2
        args = [_same_pool_args(n, ki, si) for n, ki, si in zip(x.shape[-2:], k, s)]
        if any(a is None for a in args) or args[0][1] != args[1][1]:
            return orig(self, x)
        (ph, ceil), (pw, _) = args
        return F.max_pool2d(x, k, s, (ph, pw), self.dilation, ceil_mode=ceil)

    P.MaxPool2dSame.forward = fwd
    try:
        yield
    finally:
        P.MaxPool2dSame.forward = orig


# Data and mAP -----------------------------------------------------------------------------

def coco_dataset(coco_root, ann, img_dir):
    return DetectionDatset(data_dir=coco_root / img_dir, parser=create_parser(
        "coco", cfg=CocoParserCfg(ann_filename=coco_root / ann, has_labels=True)))


def loaders(coco_root, subset, size, batch_size, model_name):
    """effdet's own COCO loaders (resize-pad, flip; images arrive on the GPU, normalized),
    training on `subset`, scoring on val2017."""
    train = coco_dataset(coco_root, *COCO_TRAIN_SUBSETS[subset])
    val = coco_dataset(coco_root, *COCO_TRAIN_SUBSETS["val2017"])
    mean, std = norm(model_name)
    kw = dict(input_size=(3, size, size), batch_size=batch_size, use_prefetcher=True,
              num_workers=get_num_workers(), pin_mem=True, mean=mean, std=std)
    return create_loader(train, is_training=True, **kw), create_loader(val, is_training=False, **kw), val


def to_device(target, device):
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in target.items()}


def as_coco(image_id, det):
    """[x1, y1, x2, y2, score, category id] rows in COCO's submission format."""
    out = []
    for x1, y1, x2, y2, score, cls in det:
        w, h = max(0.0, float(x2 - x1)), max(0.0, float(y2 - y1))
        if score > 0 and w > 0 and h > 0:
            out.append({"image_id": int(image_id), "category_id": int(cls),
                        "bbox": [float(x1), float(y1), w, h], "score": float(score)})
    return out


@torch.no_grad()
def evaluate(model, val, val_ds, device):
    """val2017 mAP through DetBenchPredict. Its class ids are COCO's (head index + 1)."""
    # Moved to the device explicitly: its anchor boxes are built on the CPU.
    bench = DetBenchPredict(model).to(device).eval()
    ids, dets = val_ds.parser.img_ids, []
    for images, targets in tqdm(val, desc="eval", leave=False, dynamic_ncols=True,
                                mininterval=1.0):
        targets = to_device(targets, device)
        out = bench(images.to(device, non_blocking=True), img_info=targets).cpu().numpy()
        for idx, det in zip(targets["img_idx"].cpu().numpy(), out):
            dets += as_coco(ids[int(idx)], det)
    return coco_map(val_ds.parser.coco, dets)


# fetch-baseline / prune ---------------------------------------------------------------------

def fetch_baseline(args):
    """Write the COCO-trained EfficientDet-Lite checkpoint as a full nn.Module."""
    out = Path(args.models_dir) / f"{args.model}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(create_model(f"tf_{args.model}", pretrained=True), out)
    print(f"-> {out}")


def prune(args):
    """Prune the backbone to each of --ratios, fine-tune on COCO, save each rung."""
    src = Path(args.models_dir) / f"{args.model}.pt"
    if not src.exists():
        raise SystemExit(f"{src} is missing: run `python {sys.argv[0]} fetch-baseline "
                         f"--model {args.model}` first")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    size = SIZE[args.model]
    stem = f"{args.model}_coco-{args.coco_subset}"
    sfx = "_full" if args.scope == "full" else ""
    pruned_dir, log_dir = Path(args.out_dir) / "pytorch_pruned", Path(args.out_dir) / "pruning_logs"
    pruned_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    for ratio in args.ratios:
        int8_pruning.seed_everything(args.seed)
        model = torch.load(src, map_location="cpu", weights_only=False).to(device)
        train, val, val_ds = loaders(Path(args.dataset_dir) / "coco", args.coco_subset, size,
                                     args.batch_size, args.model)
        log = {"model": stem, "importance": args.importance, "ratio": ratio, "scope": args.scope,
               "dense_backbone_params": int8_pruning.count_params(model.backbone),
               "dense_map": evaluate(model, val, val_ds, device),
               "dense_macs": tp.utils.count_ops_and_params(
                   model, torch.randn(1, 3, size, size, device=device))[0]}
        example = torch.randn(1, 3, size, size).to(device)
        log.update(int8_pruning.prune(model, example, ratio, importance=args.importance,
                                      ignored_layers=ignored_layers(model),
                                      scope=(None if args.scope == "full"
                                             else (lambda m: m.backbone)),
                                      **PRUNING)._asdict())
        log["macs"] = tp.utils.count_ops_and_params(
            model, torch.randn(1, 3, size, size, device=device))[0]
        log["backbone_params"] = int8_pruning.count_params(model.backbone)
        log["pruned_map"] = evaluate(model, val, val_ds, device)
        bench = DetBenchTrain(model, create_labeler=True).to(device)   # shares `model`

        def train_step(_, batch):
            images, targets = batch
            return bench(images.to(device, non_blocking=True), to_device(targets, device))["loss"]

        # lr 1e-3 is about a tenth of the paper's scaled from-scratch lr, the usual
        # fine-tune of a pretrained detector at batch 16; weight decay is Tan et al. 2020's.
        opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=4e-5)
        history = int8_pruning.finetune(
            model, train, args.epochs, optimizer=opt, scheduler=warmup_cosine(opt, args.epochs),
            train_step=train_step,
            evaluate=lambda: evaluate(model, val, val_ds, device), select="mAP",
            eval_every=args.eval_every, label=f"FT @{round(100 * ratio)}%")
        log["history"] = [r._asdict() for r in history]
        log["final_map"] = evaluate(model, val, val_ds, device)
        pct = round(100 * ratio)
        out = pruned_dir / f"{stem}_pruned{pct}pct_{args.importance}{sfx}.pt"
        torch.save(model, out)
        (log_dir / f"{stem}_{args.importance}_{pct}pct{sfx}.json").write_text(
            json.dumps(log, indent=2, default=float))
        print(f"-> {out}  mAP {log['dense_map']['mAP']:.4f} dense, "
              f"{log['final_map']['mAP']:.4f} after")


# int8 ---------------------------------------------------------------------------------------

class Decoder:
    """Anchor decoding and NMS on the CPU for the int8 model's 10 head tensors, with
    effdet's own post-processing and anchors, so only the head tensors differ from the
    float model's detections."""

    def __init__(self, it, model_name, size):
        cfg = get_efficientdet_config(f"tf_{model_name}")
        cfg.num_classes, cfg.image_size = NUM_CLASSES, (size, size)
        self.cfg, self.size = cfg, size
        self.anchors = Anchors.from_config(cfg).boxes
        self.it = it
        self.cls_ch = cfg.num_scales * len(cfg.aspect_ratios) * NUM_CLASSES
        self.mean, self.std = (np.asarray(v, np.float32) for v in norm(model_name))

    def _heads(self, maps):
        """The dequantized outputs split by channel count, largest grid first, as NCHW."""
        nchw = lambda arrs: [torch.from_numpy(np.ascontiguousarray(a.transpose(0, 3, 1, 2)))
                             for a in sorted(arrs, key=lambda a: a.shape[1], reverse=True)]
        return (nchw([a for a in maps if a.shape[-1] == self.cls_ch]),
                nchw([a for a in maps if a.shape[-1] != self.cls_ch]))

    @torch.no_grad()
    def __call__(self, pil):
        """[max_det, 6] detections in the image's own pixels for one PIL image. The input is
        effdet's resize-pad: scaled to fit, pasted at the top left of a black square."""
        pil = pil.convert("RGB")
        w, h = pil.size
        scale = min(self.size / h, self.size / w)
        canvas = Image.new("RGB", (self.size, self.size), (0, 0, 0))
        canvas.paste(pil.resize((int(w * scale), int(h * scale)), Image.BILINEAR))
        arr = (np.asarray(canvas, dtype=np.float32) / 255.0 - self.mean) / self.std
        cls, box = self._heads(int8_pruning.run_tflite(self.it, arr))
        c = self.cfg
        cls_out, box_out, indices, classes = _post_process(
            cls, box, num_levels=c.num_levels, num_classes=NUM_CLASSES,
            max_detection_points=c.max_detection_points)
        return _batch_detection(1, cls_out, box_out, self.anchors, indices, classes,
                                torch.tensor([1.0 / scale]), torch.tensor([[w, h]], dtype=torch.float32),
                                max_det_per_image=c.max_det_per_image, soft_nms=c.soft_nms)[0]


def int8(args):
    """Export --pt to int8 (or take --tflite as it is), then score val2017 mAP with CPU decoding."""
    model_name = next(m for m in SIZE if Path(args.pt or args.tflite).name.startswith(m))
    size, tflite = SIZE[model_name], args.tflite
    coco_root = Path(args.dataset_dir) / "coco"
    if args.pt:
        model = torch.load(args.pt, map_location="cpu", weights_only=False)
        calib = coco_val2017(args.num_calib, size, size, *norm(model_name), args.seed, coco_root)
        tflite = Path(args.out_dir) / "tflite_int8_litert" / f"{Path(args.pt).stem}_int8.tflite"
        info = int8_pruning.export_int8(model, calib, tflite, graph_patches=[fpn_sum, pool_same])
        print(f"-> {tflite}  ({info.size_bytes:,} bytes, {info.n_outputs} outputs)")
    if args.edgetpu:
        tflite = int8_pruning.compile_edgetpu(tflite, Path(args.out_dir) / "edgetpu")
    coco_gt = COCO(str(coco_root / "annotations" / "instances_val2017.json"))
    img_ids = sorted(coco_gt.getImgIds())[:args.num_images or None]
    decode = Decoder(int8_pruning.make_interpreter(tflite), model_name, size)
    dets = []
    for iid in tqdm(img_ids, desc="eval", dynamic_ncols=True, mininterval=1.0):
        det = decode(Image.open(coco_root / "val2017" / coco_gt.loadImgs(iid)[0]["file_name"]))
        dets += as_coco(iid, [d for d in det.cpu().numpy() if d[4] > 0.001])
    score = coco_map(coco_gt, dets, img_ids)
    print(f"{Path(tflite).name}: int8 mAP {score['mAP']:.4f}  mAP@50 {score['mAP_50']:.4f}  "
          f"({len(img_ids)} val2017 images)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    for name, fn in (("fetch-baseline", fetch_baseline), ("prune", prune), ("int8", int8)):
        s = sub.add_parser(name, help=fn.__doc__.splitlines()[0])
        s.set_defaults(fn=fn)
        s.add_argument("--seed", type=int, default=42)
        if name != "int8":
            s.add_argument("--model", choices=sorted(SIZE), default="efficientdet_lite1")
            s.add_argument("--models-dir", type=Path, default=Path("outputs/models"))
        if name != "fetch-baseline":
            s.add_argument("--dataset-dir", type=Path, default=Path("data/datasets"),
                           help="holds coco/ (annotations/, train2017/, val2017/)")
            s.add_argument("--out-dir", type=Path, default=Path("outputs"))
    r = sub.choices["prune"]
    r.add_argument("--ratios", type=float, nargs="+", required=True,
                   help="fractions of the parameters to remove, e.g. 0.3 0.5")
    r.add_argument("--importance", default="magnitude_l2",
                   choices=IMPORTANCES)
    r.add_argument("--scope", choices=("backbone", "full"), default="backbone",
                   help="what --ratios is measured over (default: backbone)")
    r.add_argument("--coco-subset", default="train2017", choices=sorted(COCO_TRAIN_SUBSETS))
    r.add_argument("--epochs", type=int, default=100, help="fine-tune epochs (default: 100)")
    r.add_argument("--eval-every", type=int, default=5, help="score mAP every N epochs")
    r.add_argument("--batch-size", type=int, default=16)
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
