#!/usr/bin/env python3
"""Anchor decode and NMS on the CPU for an EfficientDet-Lite int8 tflite.

CPU-side post-processing for the EfficientDet-Lite int8 tflite produced by
int8_pruning.convert.tflite. Defaults are Lite1's; Lite2 needs only ``--input-size 448``
(see "Lite2" below).

Why this exists:
The Edge TPU can only run dense conv/elementwise ops. EfficientDet's detection
output requires **anchor decoding + non-max-suppression**, which are not TPU
compilable. So — exactly like Google/Coral's own ``efficientdet-lite*.tflite``
— the TPU runs ``backbone + BiFPN + class_net + box_net`` and emits the raw
per-level head tensors (5 class maps + 5 box maps for Lite1), and the host CPU
turns those into boxes.

This module reproduces the *identical* decode used by the PyTorch baseline
(``effdet.DetBenchPredict``): same anchors (``Anchors.from_config``), same
top-k (``_post_process``), same NMS (``_batch_detection``). The only thing that
changes between the float baseline and this path is that the head tensors come
from the quantized tflite instead of the float model — so a COCO mAP measured
here is directly comparable to the float baseline and to Coral's published
EfficientDet-Lite1 number. **Pruning ratio is the only variable.**

Preprocessing matches how the model was trained: effdet's ``ResizePad`` to
``--input-size`` (aspect-preserving, top-left paste) + mean=std=0.5 normalization
(the ``tf_efficientdet_lite*`` config value, NOT ImageNet), then quantized to int8
with the tflite's own input scale/zero-point.

Lite2:
``--input-size 448`` is the only flag Lite2 needs; the hardcoded
``tf_efficientdet_lite1`` config is still correct for it. The two configs differ
**only** in ``image_size`` -- ``min_level``/``max_level`` 3/7, ``num_scales`` 3,
``aspect_ratios``, ``anchor_scale`` 3.0 and ``max_detection_points`` 5000 are
identical -- and ``image_size`` is overwritten from ``--input-size`` below, so
``Anchors.from_config`` builds Lite2's anchors. Verified 2026-08-27.

Usage:
    # COCO val2017 mAP. This is the unpruned rung of the published ladder; it scores
    # int8 mAP 0.31748 (results/detection/full_mapping_ladder_lite1.json).
    python families/detection/effdet/eval.py eval \\
        --tflite outputs/tflite_int8_litert/efficientdet_lite1_coco-train2017_pruned0pct_int8.tflite \\
        --coco-root data/datasets/coco
    # NOTE the directory: outputs/tflite_int8/ holds the retired onnx2tf build, which maps
    # only 83 of 479 operators to the Edge TPU. See results/protocol_audit/export_litert/.

    # Lite2: same command, one extra flag. No int8 mAP is published for Lite2 yet -- the
    # artifacts exist and map 340/340, the evaluation has not been run (see todo.md).
    python families/detection/effdet/eval.py eval \\
        --tflite outputs/tflite_int8_litert/efficientdet_lite2_coco-train2017_pruned0pct_int8.tflite \\
        --input-size 448 --coco-root data/datasets/coco

    # single-image detection demo (proves the end-to-end use case):
    python families/detection/effdet/eval.py predict \\
        --tflite outputs/tflite_int8_litert/..._int8.tflite \\
        --image some.jpg --draw out.jpg
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# tf_efficientdet_lite* weights are trained with mean=std=0.5 (get_efficientdet_config()['mean'/'std']),
# NOT ImageNet. It must match train/eval/calibration time or mAP drops ~3 points.
EFFDET_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
EFFDET_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
DEFAULT_MODEL = "tf_efficientdet_lite1"


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


# Preprocessing (effdet ResizePad, aspect-preserving, top-left paste)
def preprocess(pil_img, input_size=384):
    """PIL RGB image -> (nhwc_float [1,S,S,3] normalized with mean=std=0.5,
    (orig_w, orig_h)).

    Mirrors effdet.data.transforms.ResizePad: scale = min(S/h, S/w), resize,
    paste at the upper-left of an S x S canvas. img_scale = 1/scale maps boxes
    back to original-image pixels.
    """
    from PIL import Image

    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    w, h = pil_img.size
    scale = min(input_size / h, input_size / w)
    sw, sh = int(w * scale), int(h * scale)
    canvas = Image.new("RGB", (input_size, input_size), (0, 0, 0))
    canvas.paste(pil_img.resize((sw, sh), Image.BILINEAR))  # upper-left

    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = (arr - EFFDET_MEAN) / EFFDET_STD
    nhwc = arr[None, ...].astype(np.float32)  # [1,S,S,3]
    return nhwc, 1.0 / scale, (w, h)


# Predictor: tflite head tensors -> decoded detections
class EffDetTFLitePredictor:
    def __init__(self, tflite_path, num_classes=90, input_size=384,
                 model_name=DEFAULT_MODEL):
        from effdet import get_efficientdet_config
        from effdet.anchors import Anchors

        cfg = get_efficientdet_config(model_name)
        cfg.num_classes = num_classes
        cfg.image_size = (input_size, input_size)
        self.config = cfg
        self.num_levels = cfg.num_levels
        self.num_classes = num_classes
        self.num_anchors = cfg.num_scales * len(cfg.aspect_ratios)
        self.max_detection_points = cfg.max_detection_points
        self.max_det_per_image = cfg.max_det_per_image
        self.soft_nms = cfg.soft_nms
        self.anchors = Anchors.from_config(cfg)
        self.anchor_boxes = self.anchors.boxes  # CPU tensor
        self.input_size = input_size

        Interpreter = _load_interpreter_cls()
        self.it = Interpreter(model_path=str(tflite_path))
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.outs = self.it.get_output_details()

        cls_ch = self.num_anchors * self.num_classes  # 810 for lite1
        box_ch = self.num_anchors * 4                 # 36
        n_cls = sum(1 for o in self.outs if int(o["shape"][-1]) == cls_ch)
        n_box = sum(1 for o in self.outs if int(o["shape"][-1]) == box_ch)
        if n_cls != self.num_levels or n_box != self.num_levels:
            raise RuntimeError(
                f"tflite has {len(self.outs)} outputs (cls={n_cls}, box={n_box}); "
                f"expected {self.num_levels} of each (cls_ch={cls_ch}, "
                f"box_ch={box_ch}). Re-run convert.sh after the export fix — a "
                f"backbone-only graph (single output) will trip this check."
            )

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
        by feature-map size descending (level min..max = 48->3 for Lite1), and
        return as NCHW torch tensors (what effdet._post_process expects)."""
        cls_ch = self.num_anchors * self.num_classes
        collected = [self._dequant(o) for o in self.outs]
        cls = [a for a in collected if a.shape[-1] == cls_ch]
        box = [a for a in collected if a.shape[-1] != cls_ch]
        cls.sort(key=lambda a: a.shape[1], reverse=True)
        box.sort(key=lambda a: a.shape[1], reverse=True)
        to_nchw = lambda a: torch.from_numpy(
            np.ascontiguousarray(a.transpose(0, 3, 1, 2)))
        return [to_nchw(a) for a in cls], [to_nchw(a) for a in box]

    @torch.no_grad()
    def detect(self, nhwc_float, img_scale, orig_wh):
        """nhwc_float [1,S,S,3] -> detections tensor [max_det, 6]
        as [x1, y1, x2, y2, score, coco_category_id] in *original* image pixels.
        """
        from effdet.bench import _post_process, _batch_detection

        self.it.set_tensor(self.inp["index"], self._quant_input(nhwc_float))
        self.it.invoke()
        cls_t, box_t = self._heads()

        class_out, box_out, indices, classes = _post_process(
            cls_t, box_t, num_levels=self.num_levels,
            num_classes=self.num_classes,
            max_detection_points=self.max_detection_points)
        img_scale_t = torch.tensor([img_scale], dtype=torch.float32)
        img_size_t = torch.tensor([[orig_wh[0], orig_wh[1]]], dtype=torch.float32)
        det = _batch_detection(
            1, class_out, box_out, self.anchor_boxes, indices, classes,
            img_scale_t, img_size_t,
            max_det_per_image=self.max_det_per_image, soft_nms=self.soft_nms)
        return det[0]  # [max_det, 6]


# per-category AP (the in-domain "potted plant" number for the small-car cascade)
def _category_ap(ev, cat_id, iou_idx=None):
    """Pull one category's AP out of an already-accumulated COCOeval.

    ev.eval['precision'] is [T(iou) x R(recall) x K(cat) x A(area) x M(maxDet)].
    We take area='all' (index 0) and maxDet=100 (index -1); iou_idx=None averages
    [.5:.95] (the primary AP), iou_idx=0 gives AP@0.5. Returns NaN if the category
    was never evaluated (e.g. no GT/detections for it on the chosen image set).
    """
    prec = ev.eval["precision"]
    cat_ids = list(ev.params.catIds)
    if cat_id not in cat_ids:
        return float("nan")
    k = cat_ids.index(cat_id)
    s = prec[:, :, k, 0, -1] if iou_idx is None else prec[iou_idx, :, k, 0, -1]
    s = s[s > -1]
    return float(s.mean()) if s.size else float("nan")


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

    pred = EffDetTFLitePredictor(args.tflite, num_classes=args.num_classes,
                                 input_size=args.input_size)
    print(f"[eval] {Path(args.tflite).name}  on {len(img_ids)} val images "
          f"(score_thresh={args.score_thresh})")

    results = []
    for iid in tqdm(img_ids, desc="eval", dynamic_ncols=True, mininterval=1.0):
        meta = coco_gt.loadImgs(iid)[0]
        img = Image.open(img_dir / meta["file_name"])
        nhwc, img_scale, orig_wh = preprocess(img, args.input_size)
        det = pred.detect(nhwc, img_scale, orig_wh).cpu().numpy()
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
    # In-domain single-class number for the small-car cascade (COCO "potted plant" = 64), beside overall mAP.
    if args.category_id:
        cat_ap = _category_ap(ev, args.category_id)
        cat_ap50 = _category_ap(ev, args.category_id, iou_idx=0)
        summary[f"AP_{args.category_name}"] = cat_ap
        summary[f"AP50_{args.category_name}"] = cat_ap50
        print(f"[eval] {args.category_name} (cat {args.category_id})  "
              f"AP={cat_ap:.4f}  AP50={cat_ap50:.4f}")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, indent=2))
        print(f"[eval] wrote {args.out_json}")
    print(json.dumps(summary, indent=2))


# predict: single-image detection demo
def run_predict(args):
    from PIL import Image

    pred = EffDetTFLitePredictor(args.tflite, num_classes=args.num_classes,
                                 input_size=args.input_size)
    img = Image.open(args.image)
    nhwc, img_scale, orig_wh = preprocess(img, args.input_size)
    det = pred.detect(nhwc, img_scale, orig_wh).cpu().numpy()

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
    common.add_argument("--tflite", required=True, help="int8 EfficientDet tflite")
    common.add_argument("--num-classes", type=int, default=90)
    common.add_argument("--input-size", type=int, default=384)
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
    pe.add_argument("--category-id", type=int, default=64,
                    help="also report per-category AP for this COCO id "
                         "(default 64 = potted plant, the cascade target; 0 = off)")
    pe.add_argument("--category-name", default="potted_plant",
                    help="label used in the per-category AP summary keys")
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
