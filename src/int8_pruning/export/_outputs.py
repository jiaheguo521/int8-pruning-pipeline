"""Nested model outputs, for every export path: flattening them, and naming them."""
import torch


class FlattenOutputs(torch.nn.Module):
    """Wrap a model whose forward returns nested list/tuple outputs (e.g. an
    EfficientDet returning ``(class_out_list, box_out_list)``) so the exported
    graph exposes *every* leaf tensor as a distinct output.

    ``litert_torch.to_channel_last_io`` indexes outputs positionally, so the
    forward has to hand it a flat tuple. The nested form can also collapse under
    ``torch.onnx.export``, leaving one output instead of the full pyramid.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return tuple(_leaves(self.model(x)))


def head_names(outputs):
    """ONNX output names for a forward's outputs.

    A pair of lists, the `(class_out[0..L-1], box_out[0..L-1])` a detector's per-level
    heads return, becomes cls_0.. / box_0.. so the post-processor can map an output back
    to its level; any other nesting is output_0..; a single tensor is "output".
    """
    if (isinstance(outputs, (list, tuple)) and len(outputs) == 2
            and all(isinstance(o, (list, tuple)) for o in outputs)):
        return ([f"cls_{i}" for i in range(len(outputs[0]))]
                + [f"box_{i}" for i in range(len(outputs[1]))])
    if isinstance(outputs, (list, tuple)):
        return [f"output_{i}" for i in range(len(_leaves(outputs)))]
    return ["output"]


def _leaves(o):
    return [x for e in o for x in _leaves(e)] if isinstance(o, (list, tuple)) else [o]
