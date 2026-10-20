"""Graph patches not tied to one model; a model's own live with it, as in
examples/efficientdet.py. Each is a class-level monkeypatch, so it applies whether the model
was built before or after `torch.load`, and it restores the class on exit.
"""
import contextlib

import torch
import torch.nn.functional as F

try:
    from torchvision.ops.misc import SqueezeExcitation
except ImportError as e:
    raise ImportError(
        "int8_pruning.export needs torchvision, installed alongside torch from the same "
        "index, e.g.\n"
        "    pip install torch==2.10.0 torchvision --index-url "
        "https://download.pytorch.org/whl/cu128"
    ) from e

__all__ = ["se_mean_pool", "adaptive_max_pool"]


@contextlib.contextmanager
def se_mean_pool():
    """Pool torchvision's SqueezeExcitation with torch.mean, not AdaptiveAvgPool2d(1).

    litert-torch wraps AdaptiveAvgPool2d in a transpose pair the Edge TPU declines, cutting
    the subgraph at every SE block; the mean computes the same and stays channel-last.
    """

    def _scale(self, input):
        scale = input.mean(dim=(2, 3), keepdim=True)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    original = SqueezeExcitation._scale
    SqueezeExcitation._scale = _scale
    try:
        yield
    finally:
        SqueezeExcitation._scale = original


@contextlib.contextmanager
def adaptive_max_pool():
    """Lower nn.AdaptiveMaxPool2d(1) as a max_pool over the whole spatial extent.

    litert-torch has no lowering for aten.adaptive_max_pool2d; with output_size=1 the
    max_pool is bit-exact. Any other output_size raises NotImplementedError.
    """

    def fwd(self, x):
        if not (self.output_size == 1 or tuple(self.output_size) == (1, 1)):
            raise NotImplementedError(
                f"AdaptiveMaxPool2d(output_size={self.output_size}) has no static "
                f"equivalent here; only output_size=1 is rewritten")
        return F.max_pool2d(x, kernel_size=(x.shape[-2], x.shape[-1]))

    original = torch.nn.AdaptiveMaxPool2d.forward
    torch.nn.AdaptiveMaxPool2d.forward = fwd
    try:
        yield
    finally:
        torch.nn.AdaptiveMaxPool2d.forward = original
