#!/usr/bin/env bash
# download_datasets.sh — fetch the datasets, and write a provenance README into each directory.
# Opt-in: with no flag set it downloads NOTHING and prints the toggles.
#
# Usage:
#   ./scripts/download_datasets.sh                                   # prints the toggles
#   COCO_ENABLED=1 COCO_MINITRAIN_ENABLED=1 ./scripts/download_datasets.sh
# Toggles: COCO_ENABLED COCO_MINITRAIN_ENABLED FLOWERS102_ENABLED INAT_ENABLED
#   IMAGENET_ENABLED MARKET1501_ENABLED. Sizes and the two extra knobs: docs/SETUP.md section 2.

set -euo pipefail

# Shared PROJECT_DIR / DATASET_DIR defaults live in scripts/config.sh.
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

# Per-dataset opt-in flags — all default OFF so the default run downloads nothing.
COCO_ENABLED="${COCO_ENABLED:-0}"
COCO_MINITRAIN_ENABLED="${COCO_MINITRAIN_ENABLED:-0}"
FLOWERS102_ENABLED="${FLOWERS102_ENABLED:-0}"
INAT_ENABLED="${INAT_ENABLED:-0}"
IMAGENET_ENABLED="${IMAGENET_ENABLED:-0}"
MARKET1501_ENABLED="${MARKET1501_ENABLED:-0}"
FORCE_README="${FORCE_README:-0}"

echo "DATASET_DIR             = $DATASET_DIR"
echo "COCO_ENABLED            = $COCO_ENABLED"
echo "COCO_MINITRAIN_ENABLED  = $COCO_MINITRAIN_ENABLED"
echo "FLOWERS102_ENABLED      = $FLOWERS102_ENABLED"
echo "INAT_ENABLED            = $INAT_ENABLED"
echo "IMAGENET_ENABLED        = $IMAGENET_ENABLED"
echo "MARKET1501_ENABLED      = $MARKET1501_ENABLED"
echo "FORCE_README            = $FORCE_README"
echo

# Nothing to do? Print the toggle help and exit cleanly (default path).
if [[ "$COCO_ENABLED" != "1" && "$COCO_MINITRAIN_ENABLED" != "1" \
      && "$FLOWERS102_ENABLED" != "1" && "$INAT_ENABLED" != "1" \
      && "$IMAGENET_ENABLED" != "1" && "$MARKET1501_ENABLED" != "1" ]]; then
    cat <<'HELP'
No dataset enabled — downloading nothing.

Enable datasets explicitly via their *_ENABLED flags, e.g.:
  COCO_ENABLED=1                       COCO 2017 images + annotations (~25 GB)
  COCO_ENABLED=1 COCO_MINITRAIN_ENABLED=1   + the 25k minitrain JSON
  FLOWERS102_ENABLED=1                 Oxford Flowers-102 (~330 MB)
  INAT_ENABLED=1                       iNaturalist 2017 Plantae (~198 GB stream)
  IMAGENET_ENABLED=1                   verify a local ImageNet-1k ImageFolder tree
  MARKET1501_ENABLED=1                 Market-1501 person Re-ID (~140 MB)

Example:
  DATASET_DIR=$PWD/data/datasets COCO_ENABLED=1 ./scripts/download_datasets.sh
HELP
    exit 0
fi

# DATASET_DIR must already exist: tens of GB land inside, so the caller picks and mounts it.
if [[ ! -d "$DATASET_DIR" ]]; then
    echo "error: DATASET_DIR '$DATASET_DIR' does not exist." >&2
    echo "       Create it yourself (or mount the intended drive) before re-running." >&2
    exit 1
fi

# Flowers-102 comes through torchvision, so a venv is required, but only when it is enabled.
[[ "$FLOWERS102_ENABLED" == "1" ]] && require_venv

# Required tools
for tool in wget unzip tar python3; do
    command -v "$tool" >/dev/null 2>&1 || { echo "missing tool: $tool" >&2; exit 1; }
done

# ---------- helpers ----------
# Resume-friendly wget; skips if the final file is already there *and non-empty*. An empty
# file is the tombstone of a failed download (wget -O creates it before the response body
# arrives), so removing it is what lets a re-run actually retry.
fetch() {
    local url="$1" out="$2"
    if [[ -s "$out" ]]; then
        echo "[skip] $out already present"
        return 0
    fi
    if [[ -f "$out" ]]; then
        echo "[warn] $out is empty -- previous download failed, retrying"
        rm -f "$out"
    fi
    echo "[get ] $url"
    wget --continue --tries=3 --timeout=60 -O "$out" "$url"
}

# Google Drive: >~25 MB hits a virus-scan page; bypass with usercontent.google.com + confirm=t.
fetch_gdrive() {
    local file_id="$1" out="$2"
    if [[ -s "$out" ]]; then
        echo "[skip] $out already present"
        return 0
    fi
    if [[ -f "$out" ]]; then
        echo "[warn] $out is empty -- previous download failed, retrying"
        rm -f "$out"
    fi
    local url="https://drive.usercontent.google.com/download?id=${file_id}&export=download&confirm=t"
    echo "[gdrive] $url"
    wget --tries=3 --timeout=120 -O "$out" "$url"
}

# Extract once; uses a hidden marker file so we don't re-extract.
extract_zip_once() {
    local archive="$1" dest_dir="$2"
    local marker="$dest_dir/.$(basename "$archive").extracted"
    if [[ -f "$marker" ]]; then
        echo "[skip] $archive already extracted"
        return 0
    fi
    echo "[unzip] $archive -> $dest_dir"
    unzip -q "$archive" -d "$dest_dir"
    touch "$marker"
}

extract_tar_once() {
    local archive="$1" dest_dir="$2"
    local marker="$dest_dir/.$(basename "$archive").extracted"
    if [[ -f "$marker" ]]; then
        echo "[skip] $archive already extracted"
        return 0
    fi
    echo "[untar] $archive -> $dest_dir (large, takes a while)"
    tar -xzf "$archive" -C "$dest_dir"
    touch "$marker"
}

# Write a dataset provenance README from stdin. Skips if it exists unless FORCE_README=1.
write_readme() {
    local path="$1"
    if [[ -f "$path" && "$FORCE_README" != "1" ]]; then
        echo "[skip] $path already exists (FORCE_README=1 to overwrite)"
        cat >/dev/null  # drain stdin so the heredoc does not error
        return 0
    fi
    mkdir -p "$(dirname "$path")"
    cat > "$path"
    echo "[write] $path"
}

# ---------- 1. COCO 2017 (~25 GB) ----------
coco="$DATASET_DIR/coco"
if [[ "$COCO_ENABLED" == "1" ]]; then
    echo "=== COCO 2017 — COCO_ENABLED=1 ==="
    mkdir -p "$coco"
    fetch http://images.cocodataset.org/zips/train2017.zip                      "$coco/train2017.zip"
    fetch http://images.cocodataset.org/zips/val2017.zip                        "$coco/val2017.zip"
    fetch http://images.cocodataset.org/annotations/annotations_trainval2017.zip "$coco/annotations_trainval2017.zip"

    extract_zip_once "$coco/train2017.zip"                "$coco"
    extract_zip_once "$coco/val2017.zip"                  "$coco"
    extract_zip_once "$coco/annotations_trainval2017.zip" "$coco"

    write_readme "$coco/README.md" <<'EOF'
# COCO 2017

**Provisioned by:** int8-pruning-pipeline (scripts/download_datasets.sh)
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
  Used via `COCO_SUBSET=minitrain` in `scripts/pruning.sh` for the
  recovery-FT sweep (~5x faster per epoch). Class/size distribution of
  train2017 is preserved by construction.
- `val2017`: smoke-test only (5k images; leaky because eval also reads
  val2017).
EOF
else
    echo "=== COCO 2017 — SKIPPED (COCO_ENABLED=0) ==="
fi

# ---------- 1b. COCO minitrain (25k stratified subset, ~108 MB JSON) ----------
# Stratified 25k subsample of train2017 by Samet et al., BMVC 2020
# (https://github.com/giddyyupp/coco-minitrain, source link
# https://drive.google.com/file/d/1lezhgY4M_Ag13w0dEzQ7x_zQ_w0ohjin). Images are NOT
# re-downloaded: the JSON references existing train2017 image_ids, so train2017/ must
# already be present. Unlocks COCO_SUBSET=minitrain, a ~5x faster recovery-FT sweep.
if [[ "$COCO_MINITRAIN_ENABLED" == "1" ]]; then
    echo "=== COCO minitrain JSON — COCO_MINITRAIN_ENABLED=1 ==="
    if [[ ! -d "$coco/train2017" ]]; then
        echo "[warn] $coco/train2017 not found — minitrain JSON references those image_ids." >&2
        echo "       Enable COCO_ENABLED=1 too, or the subset will have no images to read." >&2
    fi
    mkdir -p "$coco/annotations"
    fetch_gdrive 1lezhgY4M_Ag13w0dEzQ7x_zQ_w0ohjin \
                 "$coco/annotations/instances_minitrain2017.json"
else
    echo "=== COCO minitrain JSON — SKIPPED (COCO_MINITRAIN_ENABLED=0) ==="
fi

# ---------- 2. Oxford Flowers-102 (~330 MB) ----------
# Bring-up classification dataset, via torchvision so the Python layer manages the cache layout
# (.mat label files + jpgs). `download=True` is a no-op once the files are in place.
flowers_root="$DATASET_DIR/flowers102"
if [[ "$FLOWERS102_ENABLED" == "1" ]]; then
    echo "=== Oxford Flowers-102 — FLOWERS102_ENABLED=1 ==="
    mkdir -p "$flowers_root"
    python3 - <<PY
from torchvision.datasets import Flowers102
root = "${flowers_root}"
print(f"[flowers102] resolving cache under {root}")
for split in ("train", "val", "test"):
    Flowers102(root=root, split=split, download=True)
    print(f"  split={split}: OK")
PY

    write_readme "$flowers_root/README.md" <<'EOF'
# Oxford Flowers-102

**Provisioned by:** int8-pruning-pipeline (scripts/download_datasets.sh)
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
EOF
else
    echo "=== Oxford Flowers-102 — SKIPPED (FLOWERS102_ENABLED=0) ==="
fi

# ---------- 3. ImageNet-1k (verify only — NO download) ----------
# ILSVRC2012 is gated behind a registered-account download at image-net.org and cannot be
# fetched non-interactively, so when enabled we only verify the expected ImageFolder layout
# and write the README; otherwise we print acquisition instructions. No finetune is needed
# for the baseline: torchvision's pretrained head already matches.
imagenet_root="$DATASET_DIR/Imagenet_1k"
if [[ "$IMAGENET_ENABLED" == "1" ]]; then
    echo "=== ImageNet-1k — IMAGENET_ENABLED=1 (verify only, no download) ==="
    if [[ -d "$imagenet_root/train" && -d "$imagenet_root/val" ]]; then
        n_train=$(find "$imagenet_root/train" -mindepth 1 -maxdepth 1 -type d | wc -l)
        n_val=$(find "$imagenet_root/val" -mindepth 1 -maxdepth 1 -type d | wc -l)
        echo "[ok] ImageNet ImageFolder found: train/ ($n_train class dirs), val/ ($n_val class dirs)"
        [[ "$n_train" == "1000" && "$n_val" == "1000" ]] || \
            echo "[warn] expected 1000 class dirs in each of train/ and val/ (got $n_train / $n_val)" >&2
        write_readme "$imagenet_root/README.md" <<'EOF'
# ImageNet-1k (ILSVRC2012)

**Provisioned by:** int8-pruning-pipeline (scripts/download_datasets.sh)
**Date Added:** 27/05/2026

**Description:** ILSVRC2012 ImageNet-1k — 1,000-class natural-image
classification dataset (~1,281,167 train + 50,000 val images), the
canonical large-scale classification benchmark. Consumed straight from a
server-provided `ImageFolder` tree (this script does not download it —
ImageNet requires a registered account at image-net.org). For this project
images are resized to 224x224 and normalized with ImageNet mean/std.

**Expected on-disk layout** (`$DATASET_DIR/Imagenet_1k`):
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
EOF
    else
        cat >&2 <<EOF
[warn] ImageNet ImageFolder layout not found under $imagenet_root.
       ImageNet-1k cannot be downloaded by this script (registered access only).
       Acquire ILSVRC2012 from https://www.image-net.org/ and arrange it as:
         $imagenet_root/train/<wnid>/*.JPEG   (1000 dirs)
         $imagenet_root/val/<wnid>/*.JPEG     (1000 dirs)
       Re-run with IMAGENET_ENABLED=1 to verify + write the README.
EOF
    fi
else
    echo "=== ImageNet-1k — SKIPPED (IMAGENET_ENABLED=0) ==="
fi

# ---------- 4. iNaturalist 2017 — Plantae-only (opt-in) ----------
# The official train_val_images.tar.gz is ~198 GB (all super-categories) and there is no
# per-supercategory archive, so we stream it through tar and extract ONLY the 'Plantae'
# subtree; the .tar.gz never lands on disk. Bandwidth is still ~198 GB, disk is the ~30-40 GB
# extract, and the stream pipe cannot resume -- an interruption restarts from 0. Set
# INAT_MIRROR_URL=<url> to use a Plantae-only mirror instead, fetched with resume support.
if [[ "$INAT_ENABLED" == "1" ]]; then
    echo "=== iNaturalist 2017 (Plantae only) — INAT_ENABLED=1 ==="
    inat="$DATASET_DIR/inat2017_plant"
    mkdir -p "$inat"
    inat_marker="$inat/.plantae.extracted"

    # Annotations: the classification JSONs live in train_val2017.zip (26.8 MB); bounding boxes are
    # split across train_2017_bboxes.zip (22.4 MB) and val_2017_bboxes.zip (3.1 MB). All three
    # verified against the visipedia/inat_comp 2017 README on 2026-05-21.
    fetch https://ml-inat-competition-datasets.s3.amazonaws.com/2017/train_val2017.zip      "$inat/train_val2017.zip"
    fetch https://ml-inat-competition-datasets.s3.amazonaws.com/2017/train_2017_bboxes.zip  "$inat/train_2017_bboxes.zip"
    fetch https://ml-inat-competition-datasets.s3.amazonaws.com/2017/val_2017_bboxes.zip    "$inat/val_2017_bboxes.zip"
    extract_zip_once "$inat/train_val2017.zip"     "$inat"
    extract_zip_once "$inat/train_2017_bboxes.zip" "$inat"
    extract_zip_once "$inat/val_2017_bboxes.zip"   "$inat"

    # Images — Plantae subset only.
    if [[ -f "$inat_marker" ]]; then
        echo "[skip] Plantae images already extracted"
    elif [[ -n "${INAT_MIRROR_URL:-}" ]]; then
        # Mirror path: assume the URL points to a Plantae-only archive.
        mirror_file="$inat/$(basename "$INAT_MIRROR_URL")"
        fetch "$INAT_MIRROR_URL" "$mirror_file"
        case "$mirror_file" in
            *.tar.gz|*.tgz) tar -xzf "$mirror_file" -C "$inat" ;;
            *.tar)          tar -xf  "$mirror_file" -C "$inat" ;;
            *.zip)          unzip -q "$mirror_file" -d "$inat" ;;
            *) echo "unknown mirror archive type: $mirror_file" >&2; exit 1 ;;
        esac
        touch "$inat_marker"
    else
        echo "[stream] piping train_val_images.tar.gz -> tar (Plantae images only)"
        echo "         interrupting this kills the whole download — no resume."
        # The real archive on S3 is train_val_images.tar.gz (~198 GB), NOT train_val2017.tar.gz (404), and
        # it holds images only. The marker is touched only on a clean pipe exit, so an aborted stream is
        # re-attempted next run.
        if wget -O - --tries=3 --timeout=60 \
                https://ml-inat-competition-datasets.s3.amazonaws.com/2017/train_val_images.tar.gz \
                | tar -xz --wildcards --wildcards-match-slash \
                      -C "$inat" \
                      'train_val_images/Plantae/*'; then
            touch "$inat_marker"
        else
            echo "[FAIL] iNat stream pipe aborted — marker NOT written. Re-run with INAT_ENABLED=1 to retry from 0." >&2
        fi
    fi

    # Post-check: warn if the classification annotations imagenet_backbones needs are missing.
    if [[ ! -f "$inat/train2017.json" || ! -f "$inat/val2017.json" ]]; then
        cat >&2 <<'WARN'
[WARN] train2017.json / val2017.json not found under $DATASET_DIR/inat2017_plant/.
       These are required by families/classification/imagenet_backbones/prune.py and families/classification/imagenet_backbones/finetune.py.
       Options:
         (a) The official train_val2017.tar.gz layout may have changed; inspect
             its top-level entries with: tar -tzf <archive> | head -20
         (b) Find a separate iNat 2017 classification annotations download and
             place train2017.json / val2017.json under $DATASET_DIR/inat2017_plant/.
         (c) Fall back to the bboxes JSON (subset coverage only) by pointing
             --inat_train_json / --inat_val_json at the extracted bboxes file.
WARN
    fi

    write_readme "$inat/README.md" <<'EOF'
# iNaturalist 2017 — Plantae subset

**Provisioned by:** int8-pruning-pipeline (scripts/download_datasets.sh)
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
is generated once by `int8_pruning.data.classification.INatPlantaeDataset` to match
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
EOF
else
    echo "=== iNaturalist 2017 (Plantae only) — SKIPPED (INAT_ENABLED=0) ==="
fi

# ---------- 5. Market-1501 person Re-ID (~140 MB) ----------
# Pedestrian appearance-ReID benchmark (Zheng et al., ICCV 2015): 32,668 crops of 1,501
# identities over 6 cameras. Two roles here: calibration source for the reid_* int8
# embedders (bounding_box_train), and the same/different-person discriminability check
# (query + bounding_box_test filenames encode identity + camera, e.g. 0002_c1s1_000451_03.jpg).
# Distributed as one Google-Drive zip, whose link is quota-limited and can 403 or return an
# HTML page; on failure, drop Market-1501-v15.09.15.zip from a mirror into
# $DATASET_DIR/market1501/ and re-run, extract-only being idempotent.
market="$DATASET_DIR/market1501"
if [[ "$MARKET1501_ENABLED" == "1" ]]; then
    echo "=== Market-1501 — MARKET1501_ENABLED=1 ==="
    mkdir -p "$market"
    fetch_gdrive 0B8-rUzbwVRk0c054eEozWG9COHM "$market/Market-1501-v15.09.15.zip"
    extract_zip_once "$market/Market-1501-v15.09.15.zip" "$market"

    if ! ls -d "$market"/**/bounding_box_train >/dev/null 2>&1 \
         && ! ls -d "$market"/*/bounding_box_train >/dev/null 2>&1; then
        echo "[warn] bounding_box_train/ not found under $market after extract —" >&2
        echo "       the Drive link may have served a quota/HTML page. See note above." >&2
    fi

    write_readme "$market/README.md" <<'EOF'
# Market-1501

**Provisioned by:** int8-pruning-pipeline (scripts/download_datasets.sh)
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

**Purpose:** The reid_* Edge-TPU embedder family (osnet_x0_5 / osnet_x0_75 /
mobilenetv2_x1_0, Torchreid Market-1501 weights). `bounding_box_train` is the
TFLite int8 calibration source (100 random crops); `query` + `bounding_box_test`
supply identity-labeled same-person / different-person pairs for the
before/after-quant cosine discriminability check (families/re-identification/reid/eval_discrim.py).
EOF
else
    echo "=== Market-1501 — SKIPPED (MARKET1501_ENABLED=0) ==="
fi

echo
echo "Done."
echo "  Datasets : $DATASET_DIR"
echo
echo "Next step:"
echo "  - Download baseline weights + run the classification finetune:"
echo "       ./scripts/download_and_finetune_models.sh"
