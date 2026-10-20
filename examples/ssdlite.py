"""SSDLite320-MobileNetV3-Large on COCO: prune the backbone, fine-tune, export and score mAP in int8.

    python examples/ssdlite.py fetch-baseline
    python examples/ssdlite.py prune --ratios 0.3 0.5 --coco-subset minitrain
    python examples/ssdlite.py int8 --pt outputs/pytorch_pruned/<stem>.pt

Backbone-only pruning: the per-level head convs and the stem conv are protected, so every
channel that feeds a head stays, and --ratios is measured over `model.backbone`. The
torchvision SSD cannot be traced as it stands (its forward resizes a list of images and,
in eval mode, decodes and runs NMS), so the pruner traces `PerLevelHeads`, which shares
the SSD's backbone and head convs; pruning it prunes the SSD. The same wrapper is what
is exported, with anchor decoding and NMS left to the CPU, as on a Coral.

Recovery fine-tunes on COCO with torchvision's own detection loss, SGD lr 1e-3, cosine
after a one-epoch warmup, the best val2017 mAP epoch kept. --coco-subset minitrain (25k
images) is about 5x faster than train2017; val2017 is a smoke test only, since it is also
what mAP is scored on.

Outputs, under --out-dir (default outputs/):
    pytorch_pruned/ssdlite_mobilenetv3_coco-<subset>_pruned<P>pct_<criterion>.pt
    pruning_logs/ssdlite_mobilenetv3_coco-<subset>_<criterion>_<P>pct.json
    tflite_int8_litert/<.pt stem>_int8.tflite
    edgetpu/<.pt stem>_int8_edgetpu.tflite    with --edgetpu
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch_pruning as tp
from PIL import Image, ImageFile
from pycocotools.coco import COCO
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large)
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.transform import resize_boxes
from torchvision.transforms.functional import to_tensor
from tqdm.auto import tqdm

import int8_pruning
from example_data import (COCO_TRAIN_SUBSETS, IMPORTANCES, coco_map, coco_val2017,
                          get_num_workers, warmup_cosine)

ImageFile.LOAD_TRUNCATED_IMAGES = True

SIZE = 320
MEAN, STD = [0.5] * 3, [0.5] * 3            # the SSD's own transform normalizes this way
PRUNING = {"global_pruning": True, "iterative_steps": 400, "max_pruning_ratio": 0.9}


class PerLevelHeads(nn.Module):
    """The traceable core of the SSD: feature extractor + per-level head convs.

    forward(x [N, 3, 320, 320], normalized) -> (6 class maps, 6 box maps), raw NCHW.
    Shares its modules with the SSD it wraps.
    """

    def __init__(self, ssd):
        super().__init__()
        self.backbone = ssd.backbone
        self.cls_heads = ssd.head.classification_head.module_list
        self.box_heads = ssd.head.regression_head.module_list

    def forward(self, x):
        feats = list(self.backbone(x).values())
        return ([m(f) for m, f in zip(self.cls_heads, feats)],
                [m(f) for m, f in zip(self.box_heads, feats)])


class _HardSigmoid(nn.Module):
    """relu6(x + 3) / 6 as ADD + RELU6 + MUL, which the Edge TPU maps."""

    def forward(self, x):
        return torch.clamp(x + 3.0, 0.0, 6.0) * (1.0 / 6.0)


class _HardSwish(nn.Module):
    """x * relu6(x + 3) / 6, spelled so that neither converter hazard fires.

    The TFLite converter pattern-matches ADD/RELU6/MUL(1/6), and even MIN/MAX(0, 6),
    back into HARD_SWISH, which edgetpu_compiler cannot map; the (0, 3) range below has
    no RELU6 lowering, so the matcher never fires. And the upper clamp is not a MINIMUM,
    which ai-edge-quantizer leaves in float (DEQUANTIZE -> MINIMUM -> QUANTIZE at each of
    the 20 sites cuts the Edge TPU subgraph): min(t, 3) is written t - relu(t - 3).
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("three", torch.full((), 3.0))

    def forward(self, x):
        t = torch.relu(x * 0.5 + 1.5)
        return x * ((t - torch.relu(t - self.three)) * (1.0 / 3.0))


def decompose_hard_activations(module):
    """Replace Hardswish / Hardsigmoid in place, for export only: the native modules
    become HARD_SWISH and RELU_0_TO_1, which the Edge TPU compiler cannot map or parse."""
    for name, child in module.named_children():
        if isinstance(child, nn.Hardswish):
            setattr(module, name, _HardSwish())
        elif isinstance(child, nn.Hardsigmoid):
            setattr(module, name, _HardSigmoid())
        else:
            decompose_hard_activations(child)
    return module


def ignored_layers(wrapper):
    """Both head stacks, so the channels feeding them stay, and the stem conv."""
    stem = next(m for m in wrapper.backbone.modules() if isinstance(m, nn.Conv2d))
    return [wrapper.cls_heads, wrapper.box_heads, stem]


# Data ---------------------------------------------------------------------------------------

class CocoDetection(torch.utils.data.Dataset):
    """COCO images at their own size, unnormalized: the SSD resizes and normalizes itself.

    train: crowd and degenerate boxes dropped, images left without boxes dropped, random
    horizontal flip. eval: every image, target carrying only its image_id.
    Labels are COCO category ids (1..90): the COCO_V1 head has background at column 0.
    """

    def __init__(self, img_dir, ann_file, train):
        self.coco, self.img_dir, self.train = COCO(str(ann_file)), Path(img_dir), train
        ids = sorted(self.coco.imgs.keys())
        if train:
            ids = [i for i in ids if any(
                a["bbox"][2] > 0 and a["bbox"][3] > 0
                for a in self.coco.loadAnns(self.coco.getAnnIds(imgIds=i, iscrowd=False)))]
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        img = to_tensor(Image.open(self.img_dir / info["file_name"]).convert("RGB"))
        if not self.train:
            return img, {"image_id": img_id}
        boxes, labels = [], []
        for a in self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, iscrowd=False)):
            x, y, w, h = a["bbox"]
            if w > 0 and h > 0:
                boxes.append([x, y, x + w, y + h])
                labels.append(a["category_id"])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        if torch.rand(1).item() < 0.5:
            img = torch.flip(img, dims=[2])
            boxes[:, [0, 2]] = img.shape[2] - boxes[:, [2, 0]]
        return img, {"boxes": boxes, "labels": torch.as_tensor(labels, dtype=torch.int64),
                     "image_id": img_id}


def loaders(coco_root, subset, batch_size, seed):
    """Train loader on `subset`, val loader on val2017, and val2017's COCO ground truth.
    drop_last: a trailing batch of 1 crashes the BatchNorm over [1, C, 1, 1] in the
    SSD's extra blocks."""
    train_ann, train_img = COCO_TRAIN_SUBSETS[subset]
    val_ann, val_img = COCO_TRAIN_SUBSETS["val2017"]
    train = CocoDetection(coco_root / train_img, coco_root / train_ann, train=True)
    val = CocoDetection(coco_root / val_img, coco_root / val_ann, train=False)
    nw, g = get_num_workers(), torch.Generator()
    g.manual_seed(seed)
    collate = lambda batch: tuple(zip(*batch))
    return (torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,
                                        drop_last=True, num_workers=nw, pin_memory=True,
                                        collate_fn=collate, generator=g),
            torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False,
                                        num_workers=nw, pin_memory=True, collate_fn=collate),
            val.coco)


def as_coco(image_id, boxes, scores, labels):
    """Detections in COCO's submission format, zero-area and zero-score ones dropped."""
    out = []
    for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels):
        w, h = max(0.0, float(x2 - x1)), max(0.0, float(y2 - y1))
        if w > 0 and h > 0 and score > 0:
            out.append({"image_id": int(image_id), "category_id": int(label),
                        "bbox": [float(x1), float(y1), w, h], "score": float(score)})
    return out


@torch.no_grad()
def evaluate(model, val, coco_gt, device):
    """The full SSD in eval mode, its own decode and NMS, scored on val2017."""
    model.eval()
    dets = []
    for images, targets in tqdm(val, desc="eval", leave=False, dynamic_ncols=True,
                                mininterval=1.0):
        for t, out in zip(targets, model([im.to(device, non_blocking=True) for im in images])):
            dets += as_coco(t["image_id"], out["boxes"].cpu().numpy(),
                            out["scores"].cpu().numpy(), out["labels"].cpu().numpy())
    return coco_map(coco_gt, dets)


def train_step(model, batch):
    """torchvision's detection loss: smooth-L1 box regression + hard-negative-mined CE."""
    images, targets = batch
    device = next(model.parameters()).device
    images = [im.to(device, non_blocking=True) for im in images]
    targets = [{k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in t.items()} for t in targets]
    return sum(model(images, targets).values())


# fetch-baseline / prune ---------------------------------------------------------------------

def fetch_baseline(args):
    """Write torchvision's COCO-trained SSDLite320-MobileNetV3-Large as a full nn.Module."""
    out = Path(args.models_dir) / "ssdlite_mobilenetv3.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1),
               out)
    print(f"-> {out}")


def prune(args):
    """Prune the backbone to each of --ratios, fine-tune on COCO, save each rung."""
    src = Path(args.models_dir) / "ssdlite_mobilenetv3.pt"
    if not src.exists():
        raise SystemExit(f"{src} is missing: run `python {sys.argv[0]} fetch-baseline` first")
    if args.batch_size < 2:
        raise SystemExit("--batch-size must be at least 2 (see `loaders`)")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    stem = f"ssdlite_mobilenetv3_coco-{args.coco_subset}"
    pruned_dir, log_dir = Path(args.out_dir) / "pytorch_pruned", Path(args.out_dir) / "pruning_logs"
    pruned_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    for ratio in args.ratios:
        int8_pruning.seed_everything(args.seed)
        model = torch.load(src, map_location="cpu", weights_only=False).to(device)
        train, val, coco_gt = loaders(Path(args.dataset_dir) / "coco", args.coco_subset,
                                      args.batch_size, args.seed)
        log = {"model": stem, "importance": args.importance, "ratio": ratio,
               "dense_backbone_params": int8_pruning.count_params(model.backbone),
               "dense_map": evaluate(model, val, coco_gt, device)}
        heads = PerLevelHeads(model).eval()
        log["dense_macs"] = tp.utils.count_ops_and_params(
            heads, torch.randn(1, 3, SIZE, SIZE, device=device))[0]
        example = torch.randn(1, 3, SIZE, SIZE).to(device)
        log.update(int8_pruning.prune(model, example, ratio, importance=args.importance,
                                      ignored_layers=ignored_layers(heads), traced=heads,
                                      scope=lambda m: m.backbone, **PRUNING)._asdict())
        log["macs"] = tp.utils.count_ops_and_params(
            heads, torch.randn(1, 3, SIZE, SIZE, device=device))[0]
        log["pruned_map"] = evaluate(model, val, coco_gt, device)
        # A pretrained detector's fine-tune at batch 16; torchvision's own SSDLite recipe
        # trains from scratch (lr 0.15, 660 epochs), too high an lr for a fine-tune.
        opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=4e-5)
        history = int8_pruning.finetune(
            model, train, args.epochs, optimizer=opt, scheduler=warmup_cosine(opt, args.epochs),
            train_step=train_step,
            evaluate=lambda: evaluate(model, val, coco_gt, device), select="mAP",
            eval_every=args.eval_every, label=f"FT @{round(100 * ratio)}%")
        log["history"] = [r._asdict() for r in history]
        log["final_map"] = evaluate(model, val, coco_gt, device)
        pct = round(100 * ratio)
        out = pruned_dir / f"{stem}_pruned{pct}pct_{args.importance}.pt"
        torch.save(model, out)
        (log_dir / f"{stem}_{args.importance}_{pct}pct.json").write_text(
            json.dumps(log, indent=2, default=float))
        print(f"-> {out}  mAP {log['dense_map']['mAP']:.4f} dense, "
              f"{log['final_map']['mAP']:.4f} after")


# int8 ---------------------------------------------------------------------------------------

class Decoder:
    """Anchor decoding and NMS on the CPU for the 12 head tensors of the int8 model.

    Everything comes from a weightless torchvision SSDLite of the COCO_V1 architecture:
    the default boxes, the box coder (10, 10, 5, 5) and its score threshold, top-k and NMS.
    So the only difference from the float model's own postprocessing is where the head
    tensors were computed.
    """

    def __init__(self, it):
        self.skel = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None).eval()
        self.it = it
        self.num_classes = 91
        anchors = self.skel.anchor_generator.num_anchors_per_location()[0]
        self.cls_ch = anchors * self.num_classes
        grids = sorted({(int(o["shape"][1]), int(o["shape"][2])) for o in it.get_output_details()},
                       reverse=True)
        feats = [torch.zeros(1, 1, h, w) for h, w in grids]
        il = ImageList(torch.zeros(1, 3, SIZE, SIZE), [(SIZE, SIZE)])
        self.anchors = self.skel.anchor_generator(il, feats)[0]

    def _heads(self, maps):
        """The dequantized outputs split into class / box maps by channel count, largest grid
        first, each NHWC [1, H, W, A*K] reshaped to [1, H*W*A, K]: torchvision's order."""
        cat = lambda arrs, k: torch.cat([torch.from_numpy(np.ascontiguousarray(a)).reshape(1, -1, k)
                                         for a in sorted(arrs, key=lambda a: a.shape[1], reverse=True)], 1)
        return (cat([a for a in maps if a.shape[-1] == self.cls_ch], self.num_classes),
                cat([a for a in maps if a.shape[-1] != self.cls_ch], 4))

    @torch.no_grad()
    def __call__(self, pil):
        """Detections (boxes in the image's own pixels, scores, labels) for one PIL image."""
        w, h = pil.size
        arr = np.asarray(pil.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR), np.float32)
        cls, box = self._heads(
            int8_pruning.run_tflite(self.it, (arr / 255.0 - np.float32(0.5)) / np.float32(0.5)))
        det = self.skel.postprocess_detections({"cls_logits": cls, "bbox_regression": box},
                                               [self.anchors], [(SIZE, SIZE)])[0]
        return resize_boxes(det["boxes"], (SIZE, SIZE), (h, w)), det["scores"], det["labels"]


def int8(args):
    """Export --pt to int8 (or take --tflite as it is), then score val2017 mAP with CPU decoding."""
    tflite = args.tflite
    coco_root = Path(args.dataset_dir) / "coco"
    if args.pt:
        ssd = torch.load(args.pt, map_location="cpu", weights_only=False)
        model = PerLevelHeads(decompose_hard_activations(ssd))
        calib = coco_val2017(args.num_calib, SIZE, SIZE, MEAN, STD, args.seed, coco_root)
        tflite = Path(args.out_dir) / "tflite_int8_litert" / f"{Path(args.pt).stem}_int8.tflite"
        # SE blocks pool with AdaptiveAvgPool2d, which litert-torch wraps in transposes.
        info = int8_pruning.export_int8(model, calib, tflite,
                                        graph_patches=[int8_pruning.export.se_mean_pool])
        print(f"-> {tflite}  ({info.size_bytes:,} bytes, {info.n_outputs} outputs)")
    if args.edgetpu:
        tflite = int8_pruning.compile_edgetpu(tflite, Path(args.out_dir) / "edgetpu")
    coco_gt = COCO(str(coco_root / "annotations" / "instances_val2017.json"))
    img_ids = sorted(coco_gt.getImgIds())[:args.num_images or None]
    decode = Decoder(int8_pruning.make_interpreter(tflite))
    dets = []
    for iid in tqdm(img_ids, desc="eval", dynamic_ncols=True, mininterval=1.0):
        boxes, scores, labels = decode(Image.open(coco_root / "val2017" /
                                                  coco_gt.loadImgs(iid)[0]["file_name"]))
        keep = scores > 0.001
        dets += as_coco(iid, boxes[keep].numpy(), scores[keep].numpy(), labels[keep].numpy())
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
            s.add_argument("--models-dir", type=Path, default=Path("outputs/models"))
        if name != "fetch-baseline":
            s.add_argument("--dataset-dir", type=Path, default=Path("data/datasets"),
                           help="holds coco/ (annotations/, train2017/, val2017/)")
            s.add_argument("--out-dir", type=Path, default=Path("outputs"))
    r = sub.choices["prune"]
    r.add_argument("--ratios", type=float, nargs="+", required=True,
                   help="fractions of the backbone's parameters to remove, e.g. 0.3 0.5")
    r.add_argument("--importance", default="magnitude_l2",
                   choices=IMPORTANCES)
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
