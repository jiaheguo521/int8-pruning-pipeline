#!/usr/bin/env python3
"""Default-box decode and NMS on the CPU for the SSDLite320 int8 tflite.

CPU-side post-processing for the SSDLite320-MobileNetV3-Large int8 tflite
produced by int8_pruning.convert.tflite. Sibling of families/detection/effdet/eval.py.

Why this exists:
The Edge TPU can only run dense conv/elementwise ops. SSD's detection output
requires **default-box decoding + non-max-suppression**, which are not TPU
compilable. So the TPU runs ``feature extractor + per-level head convs`` and
emits the 12 raw head tensors (6 class maps + 6 box maps), and the host CPU
turns those into boxes.

This module reproduces the *identical* decode used by the PyTorch baseline
(torchvision ``SSD.postprocess_detections``): same default boxes
(``DefaultBoxGenerator``), same box coder (weights 10,10,5,5), same per-class
threshold/top-k/NMS — all taken from a weightless
``ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)``
skeleton (that constructor reproduces the COCO_V1 reduce-tail architecture
offline and carries score_thresh=0.001 / nms=0.55 / topk=300 / det_per_img=300).
The only thing that changes vs the float baseline is that the head tensors
come from the quantized tflite — so a COCO mAP measured here is directly
comparable to the float baseline. **Pruning ratio is the only variable.**

Preprocessing matches the model's internal GeneralizedRCNNTransform: plain
(non-aspect-preserving) resize to 320x320 + mean=std=0.5 normalization —
also exactly what the conversion calibration data used. Boxes are mapped
back to original-image pixels with torchvision's own ``resize_boxes``.

Head-tensor layout: each NHWC output [1,H,W,A*K] reshaped to [1,H*W*A,K]
reproduces torchvision's SSDScoringHead (view->permute->reshape, H,W,anchor-
major) and the DefaultBoxGenerator grid order, so rows align with anchors.

Usage:
    # COCO val2017 mAP:
    python families/detection/ssdlite/eval.py eval \\
        --tflite outputs/tflite_int8_litert/ssdlite_mobilenetv3_coco-minitrain_pruned0pct_int8.tflite \\
        --coco-root data/datasets/coco

    # single-image detection demo:
    python families/detection/ssdlite/eval.py predict \\
        --tflite outputs/tflite_int8_litert/..._int8.tflite \\
        --image some.jpg --draw out.jpg
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
INPUT_SIZE_DEFAULT = 320
NUM_CLASSES_DEFAULT = 91  # COCO-91 head incl. background = column 0


# tflite interpreter (light runtime, then ai_edge_litert, then full TF)
def _load_interpreter_cls():
    try:
        from tflite_runtime.interpreter import Interpreter   # the docker bench image
        return Interpreter
    except ImportError:
        pass
    try:
        from ai_edge_litert.interpreter import Interpreter   # the convert venv
        return Interpreter
    except ImportError:
        import tensorflow as tf                              # neither: older env
        return tf.lite.Interpreter


# Preprocessing (GeneralizedRCNNTransform parity: fixed-size resize, 0.5/0.5)
def preprocess(pil_img, input_size=INPUT_SIZE_DEFAULT):
    """PIL RGB image -> (nhwc_float [1,S,S,3] 0.5/0.5-normalized, (orig_w, orig_h)).

    Plain BILINEAR resize to S x S (the model's transform uses
    fixed_size=(320,320), NOT aspect-preserving padding). Box remapping back
    to the original image is handled by resize_boxes in detect().
    """
    from PIL import Image

    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    w, h = pil_img.size
    resized = pil_img.resize((input_size, input_size), Image.BILINEAR)

    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    nhwc = arr[None, ...].astype(np.float32)  # [1,S,S,3]
    return nhwc, (w, h)


# Predictor: tflite head tensors -> decoded detections
class SSDLiteTFLitePredictor:
    def __init__(self, tflite_path, num_classes=NUM_CLASSES_DEFAULT,
                 input_size=INPUT_SIZE_DEFAULT):
        from torchvision.models.detection import ssdlite320_mobilenet_v3_large
        from torchvision.models.detection.image_list import ImageList

        # weights=None AND weights_backbone=None reproduces the COCO_V1 reduce-tail architecture offline
        # (reduce_tail = weights_backbone is None) and ships the decode params: anchor_generator,
        # box_coder (10,10,5,5), score_thresh / nms_thresh / topk / detections_per_img.
        self.skel = ssdlite320_mobilenet_v3_large(
            weights=None, weights_backbone=None,
            num_classes=num_classes).eval()
        self.num_classes = num_classes
        self.num_anchors = self.skel.anchor_generator.num_anchors_per_location()[0]
        self.input_size = input_size

        Interpreter = _load_interpreter_cls()
        self.it = Interpreter(model_path=str(tflite_path))
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.outs = self.it.get_output_details()

        cls_ch = self.num_anchors * self.num_classes  # 546 for COCO-91
        box_ch = self.num_anchors * 4                 # 24
        cls_shapes = sorted([tuple(int(d) for d in o["shape"])
                             for o in self.outs
                             if int(o["shape"][-1]) == cls_ch],
                            key=lambda s: s[1], reverse=True)
        n_box = sum(1 for o in self.outs if int(o["shape"][-1]) == box_ch)
        if len(cls_shapes) != 6 or n_box != 6:
            raise RuntimeError(
                f"tflite has {len(self.outs)} outputs (cls={len(cls_shapes)}, "
                f"box={n_box}); expected 6 of each (cls_ch={cls_ch}, "
                f"box_ch={box_ch}). Re-run convert.sh — a graph without the "
                f"per-level head outputs will trip this check."
            )

        # Default boxes, generated once. DefaultBoxGenerator only reads the feature-map grid sizes,
        # taken from the tflite cls outputs ([1,H,W,C] NHWC -> (H,W) per level, sorted like _heads).
        il = ImageList(torch.zeros(1, 3, input_size, input_size),
                       [(input_size, input_size)])
        feats = [torch.zeros(1, 1, s[1], s[2]) for s in cls_shapes]
        self.anchors = self.skel.anchor_generator(il, feats)[0]  # [3234,4] xyxy

    def _dequant(self, detail):
        arr = self.it.get_tensor(detail["index"])
        scale, zp = detail["quantization"]
        if scale and scale != 0:
            arr = (arr.astype(np.float32) - zp) * scale
        return arr.astype(np.float32)  # NHWC [1,H,W,C]

    def _quant_input(self, nhwc_float):
        scale, zp = self.inp["quantization"]
        if not scale or scale == 0:
            return nhwc_float.astype(self.inp["dtype"])
        q = np.round(nhwc_float / scale + zp)
        info = np.iinfo(self.inp["dtype"])
        q = np.clip(q, info.min, info.max)
        return q.astype(self.inp["dtype"])

    def _heads(self):
        """Collect dequantized outputs, split cls/box by channel count, order
        by feature-map size descending (20,10,5,3,2,1), reshape each NHWC
        [1,H,W,A*K] to [1,H*W*A,K] (torchvision's H,W,anchor-major order) and
        concat levels -> cls_logits [1,3234,91], bbox_regression [1,3234,4]."""
        cls_ch = self.num_anchors * self.num_classes
        collected = [self._dequant(o) for o in self.outs]
        cls = [a for a in collected if a.shape[-1] == cls_ch]
        box = [a for a in collected if a.shape[-1] != cls_ch]
        cls.sort(key=lambda a: a.shape[1], reverse=True)
        box.sort(key=lambda a: a.shape[1], reverse=True)

        def levels_cat(arrs, k):
            per = [torch.from_numpy(np.ascontiguousarray(a)).reshape(1, -1, k)
                   for a in arrs]
            return torch.cat(per, dim=1)

        return levels_cat(cls, self.num_classes), levels_cat(box, 4)

    @torch.no_grad()
    def detect(self, nhwc_float, orig_wh):
        """nhwc_float [1,S,S,3] -> detections tensor [n, 6]
        as [x1, y1, x2, y2, score, coco_category_id] in *original* image
        pixels (torchvision labels ARE native COCO category ids; column 0 =
        background is dropped by postprocess_detections).
        """
        from torchvision.models.detection.transform import resize_boxes

        self.it.set_tensor(self.inp["index"], self._quant_input(nhwc_float))
        self.it.invoke()
        cls_logits, bbox_regression = self._heads()

        head_outputs = {"cls_logits": cls_logits,
                        "bbox_regression": bbox_regression}
        det = self.skel.postprocess_detections(
            head_outputs, [self.anchors],
            [(self.input_size, self.input_size)])[0]
        boxes = resize_boxes(det["boxes"],
                             (self.input_size, self.input_size),
                             (orig_wh[1], orig_wh[0]))  # (h, w)
        return torch.cat([boxes,
                          det["scores"][:, None],
                          det["labels"][:, None].float()], dim=1)  # [n,6]


# eval: COCO val2017 mAP
def run_eval(args):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from PIL import Image
    from tqdm.auto import tqdm

    coco_root = Path(args.coco_root)
    ann_path = coco_root / args.ann
    img_dir = coco_root / args.img_dir
    if not ann_path.is_file():
        sys.exit(f"[ERROR] annotations not found: {ann_path}")
    if not img_dir.is_dir():
        sys.exit(f"[ERROR] image dir not found: {img_dir}")

    coco_gt = COCO(str(ann_path))
    img_ids = sorted(coco_gt.getImgIds())
    if args.num_images:
        img_ids = img_ids[: args.num_images]

    pred = SSDLiteTFLitePredictor(args.tflite, num_classes=args.num_classes,
                                  input_size=args.input_size)
    print(f"[eval] {Path(args.tflite).name}  on {len(img_ids)} val images "
          f"(score_thresh={args.score_thresh})")

    results = []
    for iid in tqdm(img_ids, desc="eval", dynamic_ncols=True, mininterval=1.0):
        meta = coco_gt.loadImgs(iid)[0]
        img = Image.open(img_dir / meta["file_name"])
        nhwc, orig_wh = preprocess(img, args.input_size)
        det = pred.detect(nhwc, orig_wh).cpu().numpy()
        for x1, y1, x2, y2, score, cls in det:
            if score <= args.score_thresh:
                continue
            w = max(0.0, float(x2 - x1))
            h = max(0.0, float(y2 - y1))
            if w <= 0 or h <= 0:
                continue
            results.append({
                "image_id": int(iid),
                "category_id": int(cls),
                "bbox": [float(x1), float(y1), w, h],
                "score": float(score),
            })

    if not results:
        print("[eval] no detections -> mAP = 0")
        return

    coco_dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.params.imgIds = img_ids
    ev.evaluate(); ev.accumulate(); ev.summarize()
    summary = {
        "tflite": str(args.tflite),
        "n_images": len(img_ids),
        "n_detections": len(results),
        "mAP": float(ev.stats[0]),
        "mAP_50": float(ev.stats[1]),
        "mAP_75": float(ev.stats[2]),
        "mAP_small": float(ev.stats[3]),
        "mAP_medium": float(ev.stats[4]),
        "mAP_large": float(ev.stats[5]),
    }
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, indent=2))
        print(f"[eval] wrote {args.out_json}")
    print(json.dumps(summary, indent=2))


# predict: single-image detection demo
def run_predict(args):
    from PIL import Image

    pred = SSDLiteTFLitePredictor(args.tflite, num_classes=args.num_classes,
                                  input_size=args.input_size)
    img = Image.open(args.image)
    nhwc, orig_wh = preprocess(img, args.input_size)
    det = pred.detect(nhwc, orig_wh).cpu().numpy()

    kept = [d for d in det if d[4] > args.score_thresh]
    print(f"[predict] {Path(args.image).name}  "
          f"({orig_wh[0]}x{orig_wh[1]})  -> {len(kept)} detections "
          f"(score > {args.score_thresh})")
    for x1, y1, x2, y2, score, cls in kept:
        print(f"  cls={int(cls):3d}  score={score:.3f}  "
              f"box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

    if args.draw:
        from PIL import ImageDraw
        if img.mode != "RGB":
            img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        for x1, y1, x2, y2, score, cls in kept:
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
            draw.text((x1 + 2, y1 + 2), f"{int(cls)}:{score:.2f}",
                      fill=(255, 255, 0))
        img.save(args.draw)
        print(f"[predict] annotated image -> {args.draw}")


# CLI
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--tflite", required=True, help="int8 SSDLite tflite")
    common.add_argument("--num-classes", type=int, default=NUM_CLASSES_DEFAULT)
    common.add_argument("--input-size", type=int, default=INPUT_SIZE_DEFAULT)
    common.add_argument("--score-thresh", type=float, default=0.001,
                        help="eval: keep low (0.001) for mAP; predict: raise "
                             "(e.g. 0.3) for clean boxes")

    pe = sub.add_parser("eval", parents=[common], help="COCO val2017 mAP")
    pe.add_argument("--coco-root", default="data/datasets/coco")
    pe.add_argument("--ann", default="annotations/instances_val2017.json")
    pe.add_argument("--img-dir", default="val2017")
    pe.add_argument("--num-images", type=int, default=0,
                    help="limit images (0 = all 5000); use a small N for a smoke test")
    pe.add_argument("--out-json", default=None, help="write mAP summary JSON here")
    pe.set_defaults(func=run_eval)

    pp = sub.add_parser("predict", parents=[common], help="single-image demo")
    pp.add_argument("--image", required=True)
    pp.add_argument("--draw", default=None, help="save annotated image to this path")
    pp.set_defaults(func=run_predict)

    args = p.parse_args(argv)
    # predict wants a higher default threshold for readable output
    if args.cmd == "predict" and args.score_thresh == 0.001:
        args.score_thresh = 0.3
    args.func(args)


if __name__ == "__main__":
    main()
