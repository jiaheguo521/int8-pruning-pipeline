"""The relu-clip student: timm backbone -> GAP -> Linear to the teacher's width.

Kept structurally identical to the upstream definition (jiaheguo521/relu-clip,
`sources/students.py`) so the published `student.safetensors` loads with no key
renaming: same attribute names (`backbone` / `pool` / `head`), same forward, same
`num_classes=0, global_pool=""` backbone configuration.

A pruned model is saved as a whole-module pickle, which records this class as
`relu_clip_model.CnnStudent`; loading one needs this file importable under that name.
`examples/relu_clip.py` is, because Python puts a script's own directory on sys.path.

Only the deployable rung is here: `tf_efficientnet_lite4` distilled from CLIP ViT-L/14.
Its published parameter count, zero-shot top-1 and latency are upstream's. Upstream's
ReLU->ReLU6 swap and padded-MaxPool->AvgPool stem rewrite for its ResNet students are
no-ops here: EfficientNet-Lite is natively ReLU6 and has no padded MaxPool.
"""

import timm
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# The published rung. Backbone is a timm name; embed dim is the teacher's width (ViT-L/14 768, RN50 1024).
DEFAULT_BACKBONE = "tf_efficientnet_lite4"
DEFAULT_EMBED_DIM = 768
INPUT_SIZE = 224

# OpenAI CLIP normalization, not ImageNet: the student lands in the CLIP text embedding space, and
# ImageNet stats here silently mis-align zero-shot.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class CnnStudent(nn.Module):
    """backbone feature map -> GAP -> Linear(feat_dim, embed_dim). Output is the raw
    (unnormalized) embedding; L2-norm happens on the eval/deploy side."""

    def __init__(self, backbone: nn.Module, feat_dim: int, embed_dim: int):
        super().__init__()
        self.backbone = backbone        # timm model, num_classes=0 + global_pool="" -> [B,C,H,W]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(feat_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)            # [B, feat_dim, H, W]
        x = self.pool(x).flatten(1)     # [B, feat_dim]
        return self.head(x)             # [B, embed_dim], unnormalized


def build_student(backbone_id: str = DEFAULT_BACKBONE,
                  embed_dim: int = DEFAULT_EMBED_DIM,
                  pretrained: bool = False) -> CnnStudent:
    """Empty student with the published shape. `pretrained` pulls timm's ImageNet
    weights for the backbone, useful only as a control; the deployed weights come
    from the Hub checkpoint (`relu_clip.py fetch-baseline`)."""
    backbone = timm.create_model(backbone_id, pretrained=pretrained,
                                 num_classes=0, global_pool="")
    return CnnStudent(backbone, backbone.num_features, embed_dim)


def student_transform(image_size: int = INPUT_SIZE):
    """The published preprocessing, verbatim.

    Upstream's `preprocessing.json` pins `Resize(224, BICUBIC) + CenterCrop(224)`, unlike
    the usual ImageNet eval transform's 256/224 resize, and names the two details that
    "silently cost accuracy if guessed": `Resize` truncates the resized long side and
    `CenterCrop` rounds the offset. torchvision's own implementations do both, so this is
    expressed as the transform rather than as arithmetic.

    The transform is deterministic, so training, calibration and evaluation see the same
    pixels.
    """
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])
