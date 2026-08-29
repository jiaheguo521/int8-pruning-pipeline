"""The relu-clip student: timm backbone -> GAP -> Linear to the teacher's width.

Kept structurally identical to the upstream definition (jiaheguo521/relu-clip,
`sources/students.py`) so the published `student.safetensors` loads with no key
renaming: same attribute names (`backbone` / `pool` / `head`), same forward, same
`num_classes=0, global_pool=""` backbone configuration.

Why the class lives here at all: the pruning workers in this repo save FULL
MODULE pickles, which bake the defining module's import name into the file. A
pickle written against upstream's `students.CnnStudent` cannot be loaded here.
`int8_pruning.convert.common.UNPICKLE_MODULE_DIRS` puts this directory on sys.path so
`student.CnnStudent` resolves for the converter -- the same arrangement
`line_seg` (`lane_seg_models`) and `reid` (`youreid_model`) already use.

Only the deployable rung is here. Upstream's registry carries 9 backbones x 2
teachers, of which this repo prunes the one that is published with weights and
measured on the device: `tf_efficientnet_lite4` distilled from CLIP ViT-L/14,
12.71 M parameters, 68.25% int8 ImageNet-1k zero-shot at 26.49 ms. The
ReLU->ReLU6 swap and the padded-MaxPool->AvgPool stem rewrite that upstream
applies to its ResNet students are not reproduced here: EfficientNet-Lite is
natively ReLU6 and has no padded MaxPool, so both are no-ops for this backbone,
and carrying them would be carrying untested code.
"""

import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# The published rung. Backbone is a timm name; embed dim is the teacher's width (ViT-L/14 768, RN50 1024).
DEFAULT_BACKBONE = "tf_efficientnet_lite4"
DEFAULT_EMBED_DIM = 768
INPUT_SIZE = 224

# OpenAI CLIP normalization, not ImageNet: the student lands in the CLIP text embedding space, and
# ImageNet stats here silently mis-align zero-shot. Keep in sync with this family's family.yaml.
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
    weights for the backbone, which is only useful as a control -- the weights
    that matter come from the Hub checkpoint (see download_baseline.py)."""
    import timm
    backbone = timm.create_model(backbone_id, pretrained=pretrained,
                                 num_classes=0, global_pool="")
    return CnnStudent(backbone, backbone.num_features, embed_dim)


def student_transform(image_size: int = INPUT_SIZE):
    """The published preprocessing, verbatim.

    NOT `int8_pruning.data.classification.build_transforms`, whose eval branch resizes to
    256/224 of the input first. Upstream's `preprocessing.json` pins
    `Resize(224, BICUBIC) + CenterCrop(224)` and names the two details that
    "silently cost accuracy if guessed": `Resize` TRUNCATES the resized long side
    and `CenterCrop` ROUNDS the offset. Both come free from torchvision's own
    implementations, which is why this is expressed as the transform rather than
    as arithmetic.

    There is no augmentation branch. Upstream distils against a per-image CACHE
    of teacher embeddings, so an augmented student view would no longer match the
    view its target was computed from; the recovery FT here runs the teacher live
    and could augment, but keeping the deterministic transform is what makes the
    training, calibration and evaluation pixels the same pixels.
    """
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])
