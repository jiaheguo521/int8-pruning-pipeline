"""Download the datasets into --dataset-dir, and write a provenance README into each directory.

`--dataset` names the datasets to fetch. The downloads themselves are
wget / unzip / tar, run as subprocesses: wget for resumable fetches, and tar so the
198 GB iNat archive can be streamed through it with only the Plantae subtree kept.

    python examples/download_datasets.py --dataset flowers102 --dataset-dir data/datasets
    python examples/download_datasets.py --dataset coco coco_minitrain imagenet

Sizes: coco ~25 GB, coco_minitrain one ~108 MB JSON (needs coco's train2017 images),
flowers102 ~330 MB, inat_plant a ~198 GB stream that cannot resume (~30-40 GB on disk),
market1501 ~140 MB. imagenet verifies the ImageFolder layout under
<dataset-dir>/Imagenet_1k; ILSVRC2012 itself needs a registered account and a manual download.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from torchvision.datasets import Flowers102


DATASETS = ("coco", "coco_minitrain", "flowers102", "imagenet", "inat_plant", "market1501")
PROVISIONED = "**Provisioned by:** int8-pruning-pipeline (examples/download_datasets.py)"


# ---------- helpers ----------
def fetch(url, out):
    """Resume-friendly wget; skips if the final file is already there *and non-empty*. An empty
    file is the tombstone of a failed download (wget -O creates it before the response body
    arrives), so removing it is what lets a re-run actually retry."""
    if out.is_file() and out.stat().st_size > 0:
        print(f"[skip] {out} already present")
        return
    if out.is_file():
        print(f"[warn] {out} is empty -- previous download failed, retrying")
        out.unlink()
    print(f"[get ] {url}")
    subprocess.run(["wget", "--continue", "--tries=3", "--timeout=60", "-O", str(out), url],
                   check=True)


def fetch_gdrive(file_id, out):
    """Google Drive: >~25 MB hits a virus-scan page; bypass with usercontent.google.com + confirm=t."""
    if out.is_file() and out.stat().st_size > 0:
        print(f"[skip] {out} already present")
        return
    if out.is_file():
        print(f"[warn] {out} is empty -- previous download failed, retrying")
        out.unlink()
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    print(f"[gdrive] {url}")
    subprocess.run(["wget", "--tries=3", "--timeout=120", "-O", str(out), url], check=True)


def extract_once(archive, dest_dir):
    """Extract once; a hidden marker file keeps a re-run from extracting again."""
    marker = dest_dir / f".{archive.name}.extracted"
    if marker.is_file():
        print(f"[skip] {archive} already extracted")
        return
    if archive.suffix == ".zip":
        print(f"[unzip] {archive} -> {dest_dir}")
        subprocess.run(["unzip", "-q", str(archive), "-d", str(dest_dir)], check=True)
    else:
        print(f"[untar] {archive} -> {dest_dir} (large, takes a while)")
        subprocess.run(["tar", "-xzf", str(archive), "-C", str(dest_dir)], check=True)
    marker.touch()


def write_readme(path, text, force):
    """A dataset's provenance README. Skips if it exists unless --force-readme."""
    if path.is_file() and not force:
        print(f"[skip] {path} already exists (--force-readme to overwrite)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"[write] {path}")


# ---------- 1. COCO 2017 (~25 GB) ----------
COCO_README = f"""# COCO 2017

{PROVISIONED}
**Date Added:** 21/05/2026

**Description:** Microsoft COCO (Common Objects in Context) 2017 is a
large-scale object-detection, segmentation, and captioning dataset of
natural images sourced from Flickr. The 2017 split contains 118,287
training images, 5,000 validation images, and 40,670 test-dev images,
with bounding-box and instance-segmentation annotations for 80 "thing"
categories and 91 "stuff" categories. Images vary in resolution and
aspect ratio; for this project they are letterboxed to 384x384 and
normalized with the conventional COCO statistics. Annotations are
distributed as JSON files consumed by `pycocotools` (`instances_train2017
.json`, `instances_val2017.json`, ...).

**Source:** https://cocodataset.org/ (Lin et al., 2014; v2017 split)
**License:** Annotations CC BY 4.0; images under their original Flickr
terms of use (mostly CC, varies per image).

**Purpose:** Detection-side training, recovery fine-tuning, and mAP
evaluation for the EfficientDet-Lite1 branch of the cascade. Also serves
as the calibration source (100 random train images) for TFLite int8
post-training quantization of the detector.

**Subsets available for training:**
- `train2017` (default): full 118k images, the canonical target.
- `minitrain`: 25k stratified subsample from Samet et al., BMVC 2020
  (https://github.com/giddyyupp/coco-minitrain). Reuses train2017 images;
  only `annotations/instances_minitrain2017.json` (~5 MB) is added.
  Used via `--coco-subset minitrain` in the detection examples for the
  recovery-FT sweep (~5x faster per epoch). Class/size distribution of
  train2017 is preserved by construction.
- `val2017`: smoke-test only (5k images; leaky because eval also reads
  val2017).
"""


def coco(root, args):
    coco = root / "coco"
    coco.mkdir(parents=True, exist_ok=True)
    for name, url in (("train2017.zip", "http://images.cocodataset.org/zips/train2017.zip"),
                      ("val2017.zip", "http://images.cocodataset.org/zips/val2017.zip"),
                      ("annotations_trainval2017.zip",
                       "http://images.cocodataset.org/annotations/annotations_trainval2017.zip")):
        fetch(url, coco / name)
    for name in ("train2017.zip", "val2017.zip", "annotations_trainval2017.zip"):
        extract_once(coco / name, coco)
    write_readme(coco / "README.md", COCO_README, args.force_readme)


# ---------- 1b. COCO minitrain (25k stratified subset, ~108 MB JSON) ----------
# Stratified 25k subsample of train2017 by Samet et al., BMVC 2020
# (https://github.com/giddyyupp/coco-minitrain, source link
# https://drive.google.com/file/d/1lezhgY4M_Ag13w0dEzQ7x_zQ_w0ohjin). The JSON references
# existing train2017 image_ids, so train2017/ must already be present. Unlocks --coco-subset minitrain, a ~5x faster recovery-FT sweep.
def coco_minitrain(root, args):
    coco = root / "coco"
    if not (coco / "train2017").is_dir():
        print(f"[warn] {coco}/train2017 not found — minitrain JSON references those image_ids.\n"
              f"       Pass --dataset coco too, or the subset will have no images to read.",
              file=sys.stderr)
    (coco / "annotations").mkdir(parents=True, exist_ok=True)
    fetch_gdrive("1lezhgY4M_Ag13w0dEzQ7x_zQ_w0ohjin",
                 coco / "annotations" / "instances_minitrain2017.json")


# ---------- 2. Oxford Flowers-102 (~330 MB) ----------
FLOWERS_README = f"""# Oxford Flowers-102

{PROVISIONED}
**Date Added:** 22/05/2026

**Description:** Oxford Flowers-102 (Nilsback & Zisserman, 2008) — 102
flower categories with significant intra-class variation and inter-class
similarity. Standard splits: 1,020 train / 1,020 val / 6,149 test images.
JPEGs of varying resolution and aspect ratio; for this project they are
resized to 224x224 and normalized with ImageNet mean/std. Downloaded and
cached automatically via `torchvision.datasets.Flowers102` (label .mat
files + jpg archive).

**Source:** https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
(Nilsback & Zisserman, ICVGIP 2008)
**License:** GNU GPL v3 (mirrored via torchvision).

**Purpose:** Fast bring-up dataset for the classification branch of the
Edge TPU cascade. Same input shape (224x224) and normalization as the
ImageNet-1k target, so the pruning and quantization pipeline behaves
identically and Phase 1 (per-model sweep) + Phase 2 (TFLite int8
conversion) can be validated end-to-end on a 330 MB download. Absolute
Top-1 numbers WILL NOT transfer to ImageNet (102 vs 1000 classes); the
size-vs-accuracy *curve shape* is what's portable.
"""


def flowers102(root, args):
    # Through torchvision, so the Python layer manages the cache layout (.mat label files +
    # jpgs). `download=True` is a no-op once the files are in place.
    flowers_root = root / "flowers102"
    flowers_root.mkdir(parents=True, exist_ok=True)
    print(f"[flowers102] resolving cache under {flowers_root}")
    for split in ("train", "val", "test"):
        Flowers102(root=str(flowers_root), split=split, download=True)
        print(f"  split={split}: OK")
    write_readme(flowers_root / "README.md", FLOWERS_README, args.force_readme)


# ---------- 3. ImageNet-1k (verify only) ----------
IMAGENET_README = f"""# ImageNet-1k (ILSVRC2012)

{PROVISIONED}
**Date Added:** 27/05/2026

**Description:** ILSVRC2012 ImageNet-1k — 1,000-class natural-image
classification dataset (~1,281,167 train + 50,000 val images), the
canonical large-scale classification benchmark. Consumed straight from a
server-provided `ImageFolder` tree (this command does not download it —
ImageNet requires a registered account at image-net.org). For this project
images are resized to 224x224 and normalized with ImageNet mean/std.

**Expected on-disk layout** (`<dataset-dir>/Imagenet_1k`):
    train/<wnid>/*.JPEG   1000 dirs, ~1,281,167 images
    val/<wnid>/*.JPEG     1000 dirs,    50,000 images

**Source:** https://www.image-net.org/ (Russakovsky et al., 2015;
ILSVRC2012 split)
**License:** ImageNet terms of access (non-commercial research use;
per-image copyright retained by the original owners).

**Purpose:** Default classification branch of the general-edge cascade
(paired with COCO detection). torchvision's pretrained MobileNetV2
ImageNet head (1280->1000) already matches this dataset's 1000 classes and
torchvision's class order equals ImageFolder's sorted-WNID order, so the
pretrained baseline is itself the prune baseline. val/ doubles as the eval
set (ImageNet has no public test split). Also the TFLite int8 calibration
source (100 random train images) for the classifier.
"""


def imagenet(root, args):
    # ILSVRC2012 is gated behind a registered-account download at image-net.org and cannot be
    # fetched non-interactively, so this only verifies the expected ImageFolder layout and
    # writes the README; otherwise it prints acquisition instructions.
    imagenet_root = root / "Imagenet_1k"
    train, val = imagenet_root / "train", imagenet_root / "val"
    if not (train.is_dir() and val.is_dir()):
        print(f"[warn] ImageNet ImageFolder layout not found under {imagenet_root}.\n"
              f"       ImageNet-1k cannot be downloaded by this command (registered access only).\n"
              f"       Acquire ILSVRC2012 from https://www.image-net.org/ and arrange it as:\n"
              f"         {imagenet_root}/train/<wnid>/*.JPEG   (1000 dirs)\n"
              f"         {imagenet_root}/val/<wnid>/*.JPEG     (1000 dirs)\n"
              f"       Re-run with --dataset imagenet to verify + write the README.",
              file=sys.stderr)
        return
    n_train = sum(1 for d in train.iterdir() if d.is_dir())
    n_val = sum(1 for d in val.iterdir() if d.is_dir())
    print(f"[ok] ImageNet ImageFolder found: train/ ({n_train} class dirs), val/ ({n_val} class dirs)")
    if (n_train, n_val) != (1000, 1000):
        print(f"[warn] expected 1000 class dirs in each of train/ and val/ (got {n_train} / {n_val})",
              file=sys.stderr)
    write_readme(imagenet_root / "README.md", IMAGENET_README, args.force_readme)


# ---------- 4. iNaturalist 2017 — Plantae-only ----------
INAT_README = f"""# iNaturalist 2017 — Plantae subset

{PROVISIONED}
**Date Added:** 21/05/2026

**Description:** Super-category "Plantae" subset of the iNaturalist 2017
species-classification challenge dataset. The full iNat 2017 release
contains 13 super-categories, 5,089 species, and ~675k citizen-science
photographs (~190 GB); this project keeps only the Plantae super-category
(2,101 species, 158,407 train + 38,206 val images, ~30-40 GB on disk —
per the visipedia 2017 README). Images are crowd-sourced from the
iNaturalist platform at varying resolutions and aspect ratios; for this
project they are resized to 224x224 and normalized with ImageNet
mean/std. The Plantae subtree is extracted by streaming
`train_val_images.tar.gz` through `tar --wildcards
'train_val_images/Plantae/*'` so the full archive never lands on disk;
bbox annotations come from `train_2017_bboxes.zip` / `val_2017_bboxes.zip`
(cover a subset of images). A `category_id -> 0..2100` remapping table
is generated once by `example_data.INatPlantaeDataset` to match
the classifier output dimension.

**Source:** https://github.com/visipedia/inat_comp/tree/master/2017
(Van Horn et al., 2017)
**License:** Per-image Creative Commons licenses via iNaturalist (mix of
CC0, CC BY, CC BY-NC, etc.); commercial use varies — check individual
images. Dataset compilation released for research use.

**Purpose:** Optional fine-grained classification alternative to the
ImageNet-1k default. Trains a fresh 2,101-class MobileNetV2-iNat-plant
head, recovery fine-tuning during structured pruning, Top-1/Top-5
evaluation, and TFLite int8 calibration source (100 random train images,
ImageNet normalization) for the classifier.
"""

_INAT = "https://ml-inat-competition-datasets.s3.amazonaws.com/2017"


def inat_plant(root, args):
    # The official train_val_images.tar.gz is ~198 GB (all super-categories) and there is no
    # per-supercategory archive, so it is streamed through tar and only the 'Plantae' subtree
    # is extracted; the .tar.gz never lands on disk. Bandwidth is still ~198 GB, disk is the
    # ~30-40 GB extract, and the stream pipe cannot resume: an interruption restarts from 0.
    # --inat-mirror-url fetches a Plantae-only mirror instead, with resume support.
    inat = root / "inat2017_plant"
    inat.mkdir(parents=True, exist_ok=True)
    marker = inat / ".plantae.extracted"

    # Annotations: the classification JSONs live in train_val2017.zip (26.8 MB); bounding boxes are
    # split across train_2017_bboxes.zip (22.4 MB) and val_2017_bboxes.zip (3.1 MB).
    for name in ("train_val2017.zip", "train_2017_bboxes.zip", "val_2017_bboxes.zip"):
        fetch(f"{_INAT}/{name}", inat / name)
    for name in ("train_val2017.zip", "train_2017_bboxes.zip", "val_2017_bboxes.zip"):
        extract_once(inat / name, inat)

    # Images — Plantae subset only.
    if marker.is_file():
        print("[skip] Plantae images already extracted")
    elif args.inat_mirror_url:
        # Mirror path: the URL is assumed to point at a Plantae-only archive.
        mirror = inat / Path(args.inat_mirror_url).name
        fetch(args.inat_mirror_url, mirror)
        if str(mirror).endswith((".tar.gz", ".tgz")):
            subprocess.run(["tar", "-xzf", str(mirror), "-C", str(inat)], check=True)
        elif str(mirror).endswith(".tar"):
            subprocess.run(["tar", "-xf", str(mirror), "-C", str(inat)], check=True)
        elif str(mirror).endswith(".zip"):
            subprocess.run(["unzip", "-q", str(mirror), "-d", str(inat)], check=True)
        else:
            raise SystemExit(f"unknown mirror archive type: {mirror}")
        marker.touch()
    else:
        print("[stream] piping train_val_images.tar.gz -> tar (Plantae images only)\n"
              "         interrupting this kills the whole download — no resume.")
        # The archive is train_val_images.tar.gz (~198 GB); train_val2017.tar.gz returns 404,
        # and it holds images only. The marker is touched only on a clean pipe exit, so an
        # aborted stream is re-attempted next run.
        wget = subprocess.Popen(["wget", "-O", "-", "--tries=3", "--timeout=60",
                                 f"{_INAT}/train_val_images.tar.gz"], stdout=subprocess.PIPE)
        tar = subprocess.run(["tar", "-xz", "--wildcards", "--wildcards-match-slash",
                              "-C", str(inat), "train_val_images/Plantae/*"], stdin=wget.stdout)
        wget.stdout.close()
        if wget.wait() == 0 and tar.returncode == 0:
            marker.touch()
        else:
            print("[FAIL] iNat stream pipe aborted — marker NOT written. Re-run "
                  "--dataset inat_plant to retry from 0.", file=sys.stderr)

    # Post-check: warn if the classification annotations cnn_backbones needs are missing.
    if not ((inat / "train2017.json").is_file() and (inat / "val2017.json").is_file()):
        print(f"[WARN] train2017.json / val2017.json not found under {inat}/.\n"
              f"       These are required by examples/cnn_backbones.py.\n"
              f"       Options:\n"
              f"         (a) The official train_val2017.tar.gz layout may have changed; inspect\n"
              f"             its top-level entries with: tar -tzf <archive> | head -20\n"
              f"         (b) Find a separate iNat 2017 classification annotations download and\n"
              f"             place train2017.json / val2017.json under {inat}/.\n"
              f"         (c) Fall back to the bboxes JSON (subset coverage only) by pointing\n"
              f"             --inat-train-json / --inat-val-json at the extracted bboxes file.",
              file=sys.stderr)
    write_readme(inat / "README.md", INAT_README, args.force_readme)


# ---------- 5. Market-1501 person Re-ID (~140 MB) ----------
MARKET_README = f"""# Market-1501

{PROVISIONED}
**Date Added:** 10/07/2026

**Description:** Market-1501 (Zheng et al., ICCV 2015) — the standard person
re-identification benchmark. 1,501 identities captured by 6 cameras (5 HD +
1 SD) in front of a supermarket, 32,668 auto-detected (DPM) pedestrian
bounding boxes, split into 12,936 train / 3,368 query / 19,732 gallery crops.
Each crop is a tight pedestrian image; filenames encode identity and camera
as `PID_cCAM_sSEQ_FRAME_BBOX.jpg` (e.g. `0002_c1s1_000451_03.jpg` = person
0002, camera 1). Identity `-1` marks junk / `0000` distractor crops. For this
project crops are resized to 256x128 (H x W) and normalized with ImageNet
mean/std — matching Torchreid's default test transform.

**Source:** https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html
(Zheng, Shen, Tian, Wang, Wang & Tian, ICCV 2015)
**License:** Research use only (per the dataset's original terms).

**Purpose:** The youreid family (`reid_youtu_lite`). `bounding_box_train` is the
TFLite int8 calibration source (100 random crops); `query` + `bounding_box_test`
are the query and gallery of the Market-1501 mAP / CMC evaluation
(examples/youreid.py int8).
"""


def market1501(root, args):
    # Pedestrian appearance-ReID benchmark (Zheng et al., ICCV 2015): 32,668 crops of 1,501
    # identities over 6 cameras. Two roles here: calibration source for the youreid int8
    # export (bounding_box_train), and the query / gallery of its mAP / CMC evaluation
    # (query + bounding_box_test filenames encode identity + camera, e.g. 0002_c1s1_000451_03.jpg).
    # Distributed as one Google-Drive zip, whose link is quota-limited and can 403 or return an
    # HTML page; on failure, drop Market-1501-v15.09.15.zip from a mirror into
    # <dataset-dir>/market1501/ and re-run, extract-only being idempotent.
    market = root / "market1501"
    market.mkdir(parents=True, exist_ok=True)
    zip_path = market / "Market-1501-v15.09.15.zip"
    fetch_gdrive("0B8-rUzbwVRk0c054eEozWG9COHM", zip_path)
    extract_once(zip_path, market)
    if not any(market.glob("**/bounding_box_train")):
        print(f"[warn] bounding_box_train/ not found under {market} after extract —\n"
              f"       the Drive link may have served a quota/HTML page. See the note in "
              f"this module.", file=sys.stderr)
    write_readme(market / "README.md", MARKET_README, args.force_readme)


STEPS = {"coco": coco, "coco_minitrain": coco_minitrain, "flowers102": flowers102,
         "imagenet": imagenet, "inat_plant": inat_plant, "market1501": market1501}
# What each one needs on the PATH; checked before anything runs.
_TOOLS = {"coco": ("wget", "unzip"), "coco_minitrain": ("wget",), "flowers102": (),
          "imagenet": (), "inat_plant": ("wget", "unzip", "tar"), "market1501": ("wget", "unzip")}


def main(argv=None):
    p = argparse.ArgumentParser(prog="python examples/download_datasets.py",
                                description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.splitlines()[2:]))
    p.add_argument("--dataset", nargs="+", choices=DATASETS, metavar="NAME", default=[],
                   help="the datasets to fetch: " + " ".join(DATASETS))
    p.add_argument("--dataset-dir", type=Path, default=Path("data/datasets"),
                   help="root the datasets go under; must already exist (default: %(default)s)")
    p.add_argument("--force-readme", action="store_true",
                   help="overwrite an existing dataset README")
    p.add_argument("--inat-mirror-url", default=None,
                   help="fetch a Plantae-only iNat mirror instead of streaming the 198 GB "
                        "archive; unlike the stream, a mirror fetch can resume")
    args = p.parse_args(argv)
    if not args.dataset:
        p.print_help()
        return 2

    root = args.dataset_dir
    # Must already exist: tens of GB land inside, so the caller picks and mounts it.
    if not root.is_dir():
        print(f"[ERROR] --dataset-dir {root} does not exist.\n"
              f"        Create it yourself (or mount the intended drive) before re-running.",
              file=sys.stderr)
        return 1
    missing = sorted({t for d in args.dataset for t in _TOOLS[d] if shutil.which(t) is None})
    if missing:
        print(f"[ERROR] missing tool(s): {' '.join(missing)}", file=sys.stderr)
        return 1

    for d in DATASETS:  # a fixed order: coco before coco_minitrain, which reads its images
        if d in args.dataset:
            print(f"=== {d} ===")
            STEPS[d](root, args)

    ex = "python examples/"
    nxt = {"flowers102": f"{ex}cnn_backbones.py fetch-baseline --model mobilenetv2 --dataset flowers102 --dataset-dir {root}",
           "inat_plant": f"{ex}cnn_backbones.py fetch-baseline --model mobilenetv2 --dataset inat_plant --dataset-dir {root}",
           "imagenet": f"{ex}cnn_backbones.py fetch-baseline --model <name>   (or clip_rn50.py, relu_clip.py)",
           "coco": f"{ex}efficientdet.py fetch-baseline   (or ssdlite.py)",
           "market1501": f"{ex}youreid.py fetch-baseline"}
    print(f"\nDone: {root}\nNext step: the baselines for what was fetched")
    for d in DATASETS:
        if d in args.dataset and d in nxt:
            print(f"  {nxt[d]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
