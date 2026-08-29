#!/usr/bin/env python3
"""The pieces both export paths need: ModelFamily, full-module unpickling, the output flattener, the detection probe.

What deliberately does NOT live here is the per-path graph work. litert-torch
needs five torch monkeypatches that onnx2tf never needed -- the two lower
SE-pool, SAME-pool and adaptive-max-pool differently -- and onnx2tf needs a
*name* per leaf output where litert-torch indexes them positionally. Those stay
with their path in `int8_pruning.convert.export_litert` / `int8_pruning.convert.export_onnx2tf`,
so neither path pays for the other's workarounds.

The registry itself stays in `int8_pruning.convert.tflite`: it is built from the
manifests plus the calibration loaders, which read datasets. Keeping the
dataclass here and the construction there is what lets both path modules
annotate against `ModelFamily` without importing the CLI.
"""
import dataclasses
import re
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch


# Model family record

@dataclasses.dataclass
class ModelFamily:
    name: str
    filename_pattern: str  # regex matched against .pt basename
    input_size: int
    input_channels: int
    mean: list
    std: list
    dataset_subdir: str  # appended to --dataset-dir to get this family's data root
    calib_loader: Callable
    # input_size is the HEIGHT, input_w the WIDTH; None means square, which every classifier
    # and detector here is. Re-ID embedders are 256x128 (H x W).
    input_w: Optional[int] = None
    # Detection models return a LIST of per-FPN-level head tensors, not a single logits
    # tensor; both export paths collapse such nested outputs, so they are flattened before
    # export (see FlattenOutputs). The anchor decode + NMS that turns those raw head tensors
    # into boxes is NOT Edge-TPU compilable and runs on CPU, exactly like Coral's own
    # efficientdet-lite tflite.
    detection: bool = False
    # onnx2tf path only; the litert path never builds an ONNX graph. 18 everywhere except the
    # lane-seg families, which pin 13 to match the opset their upstream export chain was gated at.
    onnx_opset: int = 18

    def matches(self, basename: str) -> bool:
        return re.match(self.filename_pattern, basename) is not None


# Loading a full-module pickle

# Full-module pickles bake the defining module's import name into the file, so
# torch.load(weights_only=False) needs it importable under exactly that name. A scan of
# every family .pt found three such modules: `lane_seg_models`, `youreid_model`,
# `student`. Appended, never prepended: families/ is full of generic basenames
# (export.py, eval.py, prune.py, model.py) that would shadow each other.
UNPICKLE_MODULE_DIRS = ("segmentation/line_seg", "re-identification/reid",
                        "classification/relu_clip")

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def enable_full_module_unpickling() -> None:
    root = _PROJECT_ROOT / "families"
    for name in UNPICKLE_MODULE_DIRS:
        p = str(root / name)
        if p not in sys.path:
            sys.path.append(p)


def load_pt_module(pt_path: Path) -> torch.nn.Module:
    """Load a full-module .pt pickle, unwrap DataParallel, put it in eval mode."""
    enable_full_module_unpickling()
    model = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    model = (model.module if hasattr(model, "module")
             and hasattr(model.module, "parameters") else model)
    model.eval()
    if hasattr(model, "aux_logits"):
        model.aux_logits = False
    return model


# Nested outputs

class FlattenOutputs(torch.nn.Module):
    """Wrap a model whose forward returns nested list/tuple outputs (e.g. an
    EfficientDet returning ``(class_out_list, box_out_list)``) so the exported
    graph exposes *every* leaf tensor as a distinct output.

    ``litert_torch.to_channel_last_io`` indexes outputs positionally, so the
    forward has to hand it a flat tuple. The nested form also collapses under
    ``torch.onnx.export`` — that is how the EfficientDet int8 tflite once ended
    up with a lone ``[1,3,3,36]`` output instead of the full 10-tensor pyramid.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        flat: List[torch.Tensor] = []

        def _rec(o):
            if isinstance(o, (list, tuple)):
                for e in o:
                    _rec(e)
            else:
                flat.append(o)

        _rec(out)
        return tuple(flat)


# Detection models

def wrap_ssd_if_needed(model) -> Tuple[torch.nn.Module, bool]:
    """torchvision SSD -> per-level-heads wrapper. Returns (model, wrapped).

    torchvision SSD modules are not tensor-traceable (their forward runs
    GeneralizedRCNNTransform on a list of images plus postprocessing); the
    wrapper returns (cls_list, box_list) like effdet. No-op for anything that is
    not a torchvision SSD. The wrapper lives with the family that needs it;
    this module does not import families at import time, so this stays a lazy,
    detection-only path.
    """
    sys.path.insert(0, str(_PROJECT_ROOT / "families" / "detection" / "ssdlite"))
    from export import maybe_wrap_ssd
    wrapped = maybe_wrap_ssd(model)
    if wrapped is not model:
        return wrapped.eval(), True
    return model, False


def probe_detection_levels(model, dummy: torch.Tensor, family_name: str) -> int:
    """Number of FPN levels a detection forward returns.

    EfficientDet returns (class_out[0..L-1], box_out[0..L-1]); both paths keep
    that ordering so the CPU post-processor can map outputs back to levels.
    """
    with torch.no_grad():
        raw = model(dummy)
    if not (isinstance(raw, (list, tuple)) and len(raw) == 2):
        raise RuntimeError(
            f"{family_name}: expected detection forward to return "
            f"(class_out, box_out); got {type(raw).__name__}")
    return len(raw[0])
