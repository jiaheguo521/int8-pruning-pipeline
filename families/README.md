# Model families

Seven families, filed under four task buckets. The pipeline is three device-neutral
stages, prune and recover, convert to int8, evaluate, plus an optional Edge TPU compile,
run across every family; putting the task and the family in the **directory** and the
stage in the **filename** is what keeps that problem readable. (It used to be the other
way round: both axes encoded in filenames under a flat `sources/`.)

| family | task | metric | notes |
|---|---|---|---|
| [`imagenet_backbones`](classification/imagenet_backbones/) | ImageNet-1k / Flowers-102 | top-1 / top-5 | 5 architectures (mobilenetv2, efficientnet_lite0, mnasnet1_0, squeezenet1_1, resnet50) |
| [`effdet`](detection/effdet/) | COCO detection | mAP | EfficientDet-Lite1; backbone-only pruning (BiFPN + heads protected) |
| [`ssdlite`](detection/ssdlite/) | COCO detection | mAP | SSDLite320-MobileNetV3; needs `export.py`'s traceable wrapper |
| [`clip_rn50`](classification/clip_rn50/README.md) | open-vocabulary zero-shot | top-1 / top-5 | per-channel int8 is mandatory |
| [`relu_clip`](classification/relu_clip/README.md) | open-vocabulary zero-shot | top-1 / top-5 | `tf_efficientnet_lite4` distilled from CLIP ViT-L/14; per-layer allocation, unlike `clip_rn50`'s global |
| [`reid`](re-identification/reid/README.md) | person Re-ID | mAP / CMC rank-k | 256×128 input, the only non-square family |
| [`line_seg`](segmentation/line_seg/) | lane segmentation | IoU / floor-fire | full-model **parameter** reduction; stems carry a `_param` suffix |

## What a family directory holds

```
families/<task>/<name>/
├── family.yaml     the manifest, single source of truth (see below)
├── prune.py        the prune + recover worker
├── eval.py         the evaluation worker
├── <arch>.py       model definition, only when it is not importable from a library
└── README.md       only when the family needs more than the root README gives
```

The matrix is deliberately **ragged**: a file exists only when that family
genuinely needs it. `line_seg` has four extra shell stages because it is the
only family with an external truth source and an external eval protocol;
`effdet` has no README because the root one covers it.

## `family.yaml`: the single source of truth

Family knowledge used to live in **five** hand-maintained registries that
nothing kept in sync: `MODEL_REGISTRY` in the converter, `FAMILY_NORMS` in the
data layer, and three `case` blocks in `scripts/pruning.sh`. They had already
drifted: the converter was missing `clip_rn50` entirely while the README
documented `--model-family clip_rn50`, which made the published CLIP numbers
irreproducible from a clean clone.

Now one file per family feeds all of them, via [`int8_pruning.manifest`](../src/int8_pruning/manifest.py):

```yaml
family: clip_rn50
prune:                     # -> scripts/pruning.sh dispatch + validity checks
  worker: families/classification/clip_rn50/prune.py
  recovery: distill        # labelled | distill -- WHAT this family recovers against
  prune_modes: [independent, iterative]   # which modes the worker implements
  models: [clip_rn50]
  datasets: [imagenet]
deliver:                   # -> scripts/deliver.sh (only for families that ship)
  ...
convert:                   # -> int8_pruning.convert.tflite.MODEL_REGISTRY + FAMILY_NORMS
  - name: clip_rn50
    input_size: 224
    mean: [0.48145466, 0.4578275, 0.40821073]
    quant_type: per-channel
    ...
```

Query it from the shell:

```bash
python -m int8_pruning.manifest models                 # every prunable model
python -m int8_pruning.manifest worker <model>         # which worker runs it
python -m int8_pruning.manifest check <model> <ds>     # exit 0 if the pair is valid
python -m int8_pruning.manifest capabilities           # which legs each family implements
python -m int8_pruning.manifest capabilities <family>  # ... for one family
```

### Which legs a family implements

Most of the pipeline is self-describing: a `convert:` block **is** the int8 leg, a
`deliver:` block **is** the packaging leg, and `deliver.ship_edgetpu` **is** the Edge
TPU leg. Two things cannot be read off the YAML, so they are declared under `prune:`:

| key | values | meaning |
|---|---|---|
| `recovery` | `labelled` \| `distill` | `labelled` fine-tunes on the task's own labels; `distill` minimises `1 - cos(student, teacher)` against a frozen copy of the **unpruned** model, which is what families with no usable label space use |
| `prune_modes` | list | `independent`, every ratio pruned from the dense baseline on its own; the protocol default, and the mode **every ladder under `results/` was run in**. `iterative`, one warm-started trajectory (`PRUNE_MODE=iterative`); implemented and mechanism-tested, but no published ladder uses it, so its effect is unmeasured here |

`manifest capabilities` prints all of it together. Declaring a mode the worker does
not implement fails loudly at argparse; implementing one without declaring it means
`scripts/pruning.sh` refuses the run and tells you to add it here. Both are safe
failures, and neither is silent.

**Wiring a family into a leg it does not implement is refused, not ignored.**
`PRUNE_MODE=iterative` with a family whose `prune_modes` lacks it stops the run
before any worker starts:

```
[ERROR] families/re-identification/reid does not implement prune_mode 'iterative' (implements: independent).
        Implement it in families/re-identification/reid/prune.py first -- see the run_iterative in
        families/classification/clip_rn50/prune.py for the shape, and int8_pruning.prune.ladder for
        the contracts -- then add 'iterative' to prune_modes in families/re-identification/reid/family.yaml.
```

## Adding a family

1. `families/<task>/<name>/family.yaml`, the manifest.
2. `families/<task>/<name>/prune.py`: import the shared helpers from
   [`int8_pruning.prune.core`](../src/int8_pruning/prune/core.py); keep `PRUNING_PROTOCOL`
   local, because `global_pruning` genuinely forks between families and that
   choice should be visible at the point of use. **The fork is not stylistic**:
   `global_pruning=True` combined with a mean-normalised magnitude criterion is
   measured to guts the narrowest early layers and take the task metric to zero on
   three architectures. Read [`../docs/PRUNING_HAZARDS.md`](../docs/PRUNING_HAZARDS.md)
   before picking a default for a new family.
3. Declare what you implemented: `prune.recovery` and `prune.prune_modes` in the
   manifest. A new family starts at `prune_modes: [independent]`; add `iterative`
   only once the worker has a trajectory function
   ([`int8_pruning.prune.ladder`](../src/int8_pruning/prune/ladder.py) holds the
   naming and resume contracts, and the three that cannot be enforced from there).
4. `families/<task>/<name>/eval.py`: the metric is family-specific by nature; reuse
   [`int8_pruning.runtime.interpreter`](../src/int8_pruning/runtime/interpreter.py) for the
   TFLite/Edge TPU plumbing.

Nothing else needs editing.
