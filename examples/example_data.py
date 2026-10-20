"""The datasets the examples train, calibrate and score on.

Classification loaders (ImageNet-1k, Flowers-102, iNat 2017 Plantae), the COCO splits the
detection examples train on, their mAP and LR schedule, the zero-shot examples' text
embeddings, and the ImageNet / COCO calibration sets. These belong to the examples;
the library itself takes no dataset.

Classification:
    train_loader, val_loader, test_loader, num_classes = build_classification_dataloaders(
        dataset, data_root=..., batch_size=..., seed=..., image_size=224,
        inat_train_json=..., inat_val_json=..., inat_image_root=...)

`test_loader` aliases `val_loader` for datasets without a public test split.

Datasets:
imagenet    ILSVRC2012 ImageNet-1k, consumed straight from a server-provided
            ImageFolder tree (no download). torchvision's MobileNetV2 ImageNet
            head (1280->1000) already matches; torchvision class order equals
            ImageFolder's sorted-WNID order, so the pretrained weights are the
            prune baseline.
              layout: <data_root>/train/<wnid>/*.JPEG  (~1.28M imgs, 1000 dirs)
                      <data_root>/val/<wnid>/*.JPEG     (50k imgs, 1000 dirs)
              splits: train -> recovery FT; val -> per-epoch acc AND final eval
                      (ImageNet has no public test split; test aliases val).

flowers102  Oxford Flowers-102 (Nilsback & Zisserman, 2008), via torchvision.
              splits: train 1020 / val 1020 / test 6149.
              test is a real held-out split (used for FP32 Top-1/Top-5 + CI95).

inat_plant  iNaturalist 2017 Plantae subset, read from the official COCO-style
            JSON (train2017.json / val2017.json) filtered by supercategory.
            num_classes is dynamic (nominally 2101). test aliases val.
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import Flowers102, ImageFolder

import int8_pruning

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_num_workers():
    """DataLoader workers: the CPUs Slurm granted, else the ones this process may run on."""
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        return max(1, int(slurm_cpus))
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return 4


# The criteria `int8_pruning.prune` takes by name; every example's --importance offers these.
IMPORTANCES = ("magnitude_l1", "magnitude_l2", "fpgm", "random", "lamp")

# The default normalization for the transforms below; each example passes its model's own.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Nominal class counts. iNat is dynamic; 2101 is the full Plantae count and the wiring default.
DATASET_NUM_CLASSES = {
    "imagenet": 1000,
    "flowers102": 102,
    "inat_plant": 2101,
}

# Training split -> (annotation file, image dir) under the COCO root. minitrain is the 25k
# stratified sample of Samet et al., BMVC 2020, over the train2017 images; val2017 as a
# training split is a smoke test only, since it is also what mAP is scored on.
COCO_TRAIN_SUBSETS = {
    "train2017": ("annotations/instances_train2017.json", "train2017"),
    "minitrain": ("annotations/instances_minitrain2017.json", "train2017"),
    "val2017":   ("annotations/instances_val2017.json",   "val2017"),
}


def build_transforms(image_size, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """ImageNet-style train/eval transforms. Resize target scales with image_size
    (image_size * 256/224), so 224 -> 256 as in the standard recipe."""
    resize_target = int(round(image_size * 256 / 224))
    train_tf = transforms.Compose([
        transforms.Resize(resize_target),
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(resize_target),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


def _make_train_loader(dataset, batch_size, seed):
    nw = get_num_workers()
    g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=nw, pin_memory=True, drop_last=True,
                      persistent_workers=(nw > 0), generator=g)


def _make_eval_loader(dataset, batch_size):
    nw = get_num_workers()
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=nw, pin_memory=True,
                      persistent_workers=(nw > 0))


# iNaturalist 2017 Plantae Dataset (JSON annotation)
class INatPlantaeDataset(Dataset):
    """iNat 2017 Plantae subset, reads the official JSON and filters by supercategory.

    iNat 2017 JSON format (https://github.com/visipedia/inat_comp/tree/master/2017):
        {
          "images":      [{"id": int, "file_name": str, "width": int, "height": int}, ...],
          "annotations": [{"id": int, "image_id": int, "category_id": int}, ...],
          "categories":  [{"id": int, "name": str, "supercategory": str}, ...]
        }

    Behavior:
      1. Filters `categories` by `supercategory == "Plantae"`; a pre-filtered JSON
         with no category of that name keeps all.
      2. Maps `category_id` -> contiguous class index [0, N).
      3. Iterates over `annotations`, keeping only those whose category is in
         Plantae AND whose image file physically exists
         (the user may have downloaded only a subset).
      4. Skip statistics exposed via `self._skipped`.

    Args:
        json_path     : train2017.json or val2017.json (or pre-filtered version).
        image_root    : root for resolving relative `file_name`.
        transform     : torchvision transform applied to each PIL.Image.
        supercategory : filter by supercategory (default "Plantae"; None = no filter).
        strict_files  : True -> raises FileNotFoundError on missing image.
                        False (default) -> silent skip, counted in _skipped.
    """
    def __init__(self, json_path, image_root, transform=None,
                 supercategory="Plantae", strict_files=False):
        self.transform = transform
        self.image_root = Path(image_root)

        with open(json_path, "r") as f:
            ann = json.load(f)

        if supercategory is not None:
            cats = [c for c in ann["categories"]
                    if c.get("supercategory") == supercategory]
            if not cats:
                # JSON already pre-filtered without the supercategory tag -> keep everything
                cats = list(ann["categories"])
        else:
            cats = list(ann["categories"])
        cats.sort(key=lambda c: c["id"])

        self.cat_id_to_class = {c["id"]: i for i, c in enumerate(cats)}
        self.class_names = [c["name"] for c in cats]
        self._num_classes = len(cats)

        img_id_to_file = {im["id"]: im["file_name"] for im in ann["images"]}

        samples = []
        n_skipped_cat = 0
        n_skipped_file = 0
        for a in ann["annotations"]:
            if a["category_id"] not in self.cat_id_to_class:
                n_skipped_cat += 1
                continue
            file_name = img_id_to_file.get(a["image_id"])
            if file_name is None:
                continue
            full_path = self.image_root / file_name
            if not full_path.exists():
                if strict_files:
                    raise FileNotFoundError(f"Missing image: {full_path}")
                n_skipped_file += 1
                continue
            samples.append((str(full_path),
                            self.cat_id_to_class[a["category_id"]]))

        self.samples = samples
        self._skipped = {"category": n_skipped_cat, "file": n_skipped_file}

        if not samples:
            raise RuntimeError(
                f"INatPlantaeDataset: 0 samples after filtering. "
                f"JSON={json_path}, image_root={image_root}, "
                f"supercategory={supercategory}. "
                f"Skipped {n_skipped_cat} out-of-category, "
                f"{n_skipped_file} missing files."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    @property
    def num_classes(self):
        return self._num_classes


# Per-dataset builders -> (train_loader, val_loader, test_loader, num_classes)
def _build_imagenet(*, data_root, batch_size, seed, image_size,
                    mean=IMAGENET_MEAN, std=IMAGENET_STD, **_):
    """ImageNet-1k ImageFolder. Read-only, never downloads; test aliases val."""
    if not data_root:
        raise SystemExit("--data-root is required for --dataset imagenet")
    data_root = Path(data_root)
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    for d in (train_dir, val_dir):
        if not d.is_dir():
            raise FileNotFoundError(
                f"ImageNet ImageFolder dir not found: {d}\n"
                f"Expected layout: {data_root}/train/<wnid>/*.JPEG and "
                f"{data_root}/val/<wnid>/*.JPEG"
            )

    train_tf, eval_tf = build_transforms(image_size, mean, std)
    train_ds = ImageFolder(root=str(train_dir), transform=train_tf)
    val_ds = ImageFolder(root=str(val_dir), transform=eval_tf)

    train_loader = _make_train_loader(train_ds, batch_size, seed)
    val_loader = _make_eval_loader(val_ds, batch_size)
    num_classes = len(train_ds.classes)
    # A directory named Imagenet_1k may hold any ImageFolder, the 10-class Imagenette subset
    # included, so the class count is checked where the data is consumed.
    if num_classes != 1000:
        raise SystemExit(
            f"--dataset imagenet expects a 1000-class ImageFolder at {data_root}, "
            f"found {num_classes} class dirs in {train_dir}.\n"
            f"If this is the 10-class Imagenette stand-in, it is not ImageNet-1k: use it "
            f"with the clip_rn50 example (which maps its wnids explicitly) or point "
            f"--dataset-dir at a real ILSVRC2012 tree."
        )
    print(f"  [Dataset/ImageNet] train: {len(train_ds)} samples, "
          f"val: {len(val_ds)} samples, num_classes={num_classes}", flush=True)
    return train_loader, val_loader, val_loader, num_classes


def _build_flowers102(*, data_root, batch_size, seed, image_size, download,
                      mean=IMAGENET_MEAN, std=IMAGENET_STD, **_):
    """Oxford Flowers-102 via torchvision. Real 6149-image test split returned."""
    if not data_root:
        raise SystemExit("--data-root is required for --dataset flowers102")
    data_root = Path(data_root)
    if download:
        data_root.mkdir(parents=True, exist_ok=True)

    train_tf, eval_tf = build_transforms(image_size, mean, std)
    try:
        train_ds = Flowers102(root=str(data_root), split="train",
                              transform=train_tf, download=download)
        val_ds = Flowers102(root=str(data_root), split="val",
                            transform=eval_tf, download=download)
        test_ds = Flowers102(root=str(data_root), split="test",
                             transform=eval_tf, download=download)
    except RuntimeError:  # torchvision's "Dataset not found or corrupted"
        if download:
            raise
        raise SystemExit(
            f"[ERROR] Flowers-102 is missing or incomplete under {data_root}.\n"
            f"        Download it first: python examples/download_datasets.py --dataset flowers102 "
            f"--dataset-dir {data_root.parent}") from None

    train_loader = _make_train_loader(train_ds, batch_size, seed)
    val_loader = _make_eval_loader(val_ds, batch_size)
    test_loader = _make_eval_loader(test_ds, batch_size)
    num_classes = DATASET_NUM_CLASSES["flowers102"]
    print(f"  [Dataset/Flowers102] train: {len(train_ds)} samples, "
          f"val: {len(val_ds)} samples, test: {len(test_ds)} samples, "
          f"num_classes={num_classes}", flush=True)
    return train_loader, val_loader, test_loader, num_classes


def _build_inat_plant(*, batch_size, seed, image_size,
                      inat_train_json, inat_val_json, inat_image_root,
                      mean=IMAGENET_MEAN, std=IMAGENET_STD, **_):
    """iNat 2017 Plantae from official JSON. num_classes dynamic; test aliases val."""
    missing = [name for name, val in (("inat_train_json", inat_train_json),
                                       ("inat_val_json", inat_val_json),
                                       ("inat_image_root", inat_image_root))
               if not val]
    if missing:
        raise SystemExit(
            f"--dataset inat_plant requires: {', '.join('--' + m for m in missing)}")

    train_tf, eval_tf = build_transforms(image_size, mean, std)
    train_ds = INatPlantaeDataset(inat_train_json, inat_image_root, transform=train_tf)
    val_ds = INatPlantaeDataset(inat_val_json, inat_image_root, transform=eval_tf)

    train_loader = _make_train_loader(train_ds, batch_size, seed)
    val_loader = _make_eval_loader(val_ds, batch_size)
    print(f"  [Dataset/iNat] train: {len(train_ds)} samples, val: {len(val_ds)} samples, "
          f"num_classes={train_ds.num_classes}, "
          f"skipped_train={train_ds._skipped}, skipped_val={val_ds._skipped}", flush=True)
    return train_loader, val_loader, val_loader, train_ds.num_classes


_REGISTRY = {
    "imagenet": _build_imagenet,
    "flowers102": _build_flowers102,
    "inat_plant": _build_inat_plant,
}


# Dispatcher
def build_classification_dataloaders(
        dataset, *, data_root=None, batch_size=64, seed=0, image_size=224,
        download=False, inat_train_json=None, inat_val_json=None,
        inat_image_root=None, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Dispatch on `dataset`. Returns (train_loader, val_loader, test_loader, num_classes).

    `test_loader` aliases `val_loader` for datasets without a public test split
    (imagenet, inat_plant). Flowers-102 returns its real 6149-image test split.
    `mean`/`std` select the input normalization.
    `download=True` lets torchvision fetch a missing Flowers-102; by default a missing
    dataset stops the run with the command that downloads it.
    """
    builder = _REGISTRY.get(dataset)
    if builder is None:
        raise SystemExit(f"unknown dataset: {dataset}")
    return builder(
        data_root=data_root, batch_size=batch_size, seed=seed,
        image_size=image_size, download=download,
        inat_train_json=inat_train_json, inat_val_json=inat_val_json,
        inat_image_root=inat_image_root, mean=mean, std=std)


# Text embeddings, for the two zero-shot examples.

def load_text_embeddings(path):
    """[classes, D] float32 text embeddings and their label names, from an .npz holding
    `embs` and `labels` (relu-clip's published file names the first `text_emb`)."""
    z = np.load(path, allow_pickle=False)
    embs = z["embs" if "embs" in z.files else "text_emb"].astype(np.float32)
    return embs, [str(x) for x in z["labels"]]


# COCO mAP and the LR schedule, for the detection examples.

def coco_map(coco_gt, detections, img_ids=None):
    """COCO bbox mAP from [{image_id, category_id, bbox (xywh), score}]."""
    # Imported here: pycocotools comes with the detection examples' extras only.
    from pycocotools.cocoeval import COCOeval

    if not detections:
        return {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0, "n_det": 0}
    ev = COCOeval(coco_gt, coco_gt.loadRes(detections), iouType="bbox")
    if img_ids is not None:
        ev.params.imgIds = img_ids
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    names = ("mAP", "mAP_50", "mAP_75", "mAP_small", "mAP_medium", "mAP_large")
    return {**{n: float(v) for n, v in zip(names, ev.stats)}, "n_det": len(detections)}


def warmup_cosine(optimizer, epochs, warmup_epochs=1, start_factor=0.01):
    """`warmup_epochs` of linear warmup from `start_factor` x lr, then cosine decay to 0,
    stepped once per epoch. The warmup gives way to at least one cosine epoch."""
    sched = torch.optim.lr_scheduler
    warmup = min(warmup_epochs, max(0, epochs - 1))
    cosine = sched.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup))
    if warmup == 0:
        return cosine
    return sched.SequentialLR(
        optimizer, [sched.LinearLR(optimizer, start_factor, total_iters=warmup), cosine],
        milestones=[warmup])


# Calibration sets: (num, H, W, 3) float32, normalized, deterministic given (num, seed).

def imagenet_train_files(dataset_root, num, seed):
    """`num` deterministic random JPEG paths from <dataset_root>/train/<wnid>/.

    A direct glob, cheaper than an ImageFolder build. Shared by the two ImageNet
    calibration loaders, each of which keeps its own resize/crop.
    """
    train_dir = Path(dataset_root) / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"ImageNet train directory not found: {train_dir}\n"
            f"Expected layout: <dataset-dir>/Imagenet_1k/train/<wnid>/*.JPEG"
        )
    files = []
    for ext in ("*.JPEG", "*.jpeg", "*.jpg"):
        files.extend(train_dir.glob(f"*/{ext}"))
    files.sort()
    if not files:
        raise FileNotFoundError(f"No JPEG files under {train_dir}/<wnid>/")
    rng = random.Random(seed)
    return rng.sample(files, k=min(num, len(files)))


def imagenet_center_crop(num, input_h, input_w, mean, std, seed, dataset_root):
    """`num` random ImageNet-1k train images, NHWC float32, normalized.

    Uses the eval transform the ImageNet examples train and evaluate with: resize the shorter side to 256/224 of the input size (BILINEAR), then
    center crop. Deterministic given (num, seed). The train-tree glob runs once
    and the result is cached by the caller (build_calibration), so the listing
    cost is paid at most once.
    """
    input_size = input_h  # square family: input_w == input_h

    picks = imagenet_train_files(dataset_root, num, seed)
    resize_target = int(round(input_size * 256 / 224))
    out = np.zeros((len(picks), input_size, input_size, 3), dtype=np.uint8)
    for i, p in enumerate(picks):
        with Image.open(p) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            # Resize shorter side to resize_target, then center crop.
            w, h = img.size
            if w < h:
                new_w, new_h = resize_target, int(round(h * resize_target / w))
            else:
                new_w, new_h = int(round(w * resize_target / h)), resize_target
            img = img.resize((new_w, new_h), Image.BILINEAR)
            left = (new_w - input_size) // 2
            top = (new_h - input_size) // 2
            img = img.crop((left, top, left + input_size, top + input_size))
            out[i] = np.asarray(img, dtype=np.uint8)
    return int8_pruning.export.normalize_nhwc(out, mean, std)


def coco_val2017(num, input_h, input_w, mean, std, seed, dataset_root):
    """100 random COCO val2017 images, NHWC float32, normalized.

    A direct JPG glob, without pycocotools. val2017 (~1 GB) rather than train2017
    (~18 GB): calibration needs pixel statistics only.
    """
    input_size = input_h  # square family: input_w == input_h

    img_dir = Path(dataset_root) / "val2017"
    if not img_dir.is_dir():
        raise FileNotFoundError(
            f"COCO val2017 directory not found: {img_dir}\n"
            f"Expected layout: <dataset-dir>/coco/val2017/*.jpg"
        )
    jpgs = sorted(img_dir.glob("*.jpg"))
    if not jpgs:
        raise FileNotFoundError(f"No .jpg files in {img_dir}")

    rng = random.Random(seed)
    picks = rng.sample(jpgs, k=min(num, len(jpgs)))

    out = np.zeros((len(picks), input_size, input_size, 3), dtype=np.uint8)
    for i, p in enumerate(picks):
        with Image.open(p) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = img.resize((input_size, input_size), Image.BILINEAR)
            out[i] = np.asarray(img, dtype=np.uint8)
    return int8_pruning.export.normalize_nhwc(out, mean, std)
