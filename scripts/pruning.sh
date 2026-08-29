#!/usr/bin/env bash
# pruning.sh — Phase 1 driver: structured pruning + recovery FT, one classification
# family (Part A) and one detection family (Part B).
#
# Usage:
#   ./scripts/pruning.sh
#   PARTS=A FINAL_EPOCHS=5 ./scripts/pruning.sh
# Env vars: PARTS CLS_MODEL CLS_DATASET DET_MODEL DET_SCOPE COCO_SUBSET CHECKPOINTS
#   IMPORTANCES FINAL_EPOCHS PRUNE_MODE DET_LR DET_WD DET_SCHED DET_WARMUP DET_MILESTONES
# Full reference, cost table, published ladder and its two traps: docs/SETUP.md section 3.

set -euo pipefail

# Config: shared paths + require_venv live in scripts/config.sh.
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
require_venv

PARTS="${PARTS:-AB}"
CLS_MODEL="${CLS_MODEL:-mobilenetv2}"   # classification family (Part A); imagenet only for non-mobilenetv2
CLS_DATASET="${CLS_DATASET:-imagenet}"
# `lane_seg` is the historical spelling of this family's dataset, still hard-coded in batch job
# scripts this repo does not own. Normalise here, ahead of the manifest check below.
if [[ "$CLS_DATASET" == "lane_seg" ]]; then
    CLS_DATASET=line_seg
fi
COCO_SUBSET="${COCO_SUBSET:-train2017}"
CHECKPOINTS="${CHECKPOINTS:-10 20 30 40 50 60 70}"
# Remember whether the caller set it, so a family can narrow the DEFAULT without overriding a choice.
_IMPORTANCES_SET="${IMPORTANCES:+1}"
IMPORTANCES="${IMPORTANCES:-magnitude_l1 magnitude_l2 fpgm random lamp}"
FINAL_EPOCHS="${FINAL_EPOCHS:-}"   # empty = Python default (100)

# Part-B FT recipe overrides; empty = Python default (see RECIPE_FT)
DET_LR="${DET_LR:-}"
DET_WD="${DET_WD:-}"
DET_SCHED="${DET_SCHED:-}"
DET_WARMUP="${DET_WARMUP:-}"
DET_SCOPE="${DET_SCOPE:-}"
DET_MILESTONES="${DET_MILESTONES:-}"

# Part-A overrides; empty = Python default. MUST be defaulted: the refs below run under `set -u`.
PRUNE_MODE="${PRUNE_MODE:-}"
CLS_LR="${CLS_LR:-}"
CLS_WD="${CLS_WD:-}"
CLS_SCHED="${CLS_SCHED:-}"
CLS_MILESTONES="${CLS_MILESTONES:-}"
CLS_BATCH="${CLS_BATCH:-}"

# The classification pruner is model-agnostic (torch.loads the baseline, architecture-agnostic
# get_ignored_layers), so any allowlisted model works on ImageNet. The allowlist is in family.yaml.
python -m int8_pruning.manifest check "$CLS_MODEL" "$CLS_DATASET" || exit 1

# CLIP-only knobs (CLS_MODEL=clip_rn50). Empty = the worker tracks cosine-to-teacher only.
CLIP_TEXT_EMB="${CLIP_TEXT_EMB:-}"
CLIP_VAL_DIR="${CLIP_VAL_DIR:-}"

# relu-clip-only knobs (CLS_MODEL=relu_clip); what each does is docs/SETUP.md section 3.
RELU_CLIP_TEXT_EMB="${RELU_CLIP_TEXT_EMB:-}"
RELU_CLIP_VAL_DIR="${RELU_CLIP_VAL_DIR:-}"
RELU_CLIP_GLOBAL_PRUNING="${RELU_CLIP_GLOBAL_PRUNING:-0}"  # 1 left clip_rn50's stem at 3 of 32
RELU_CLIP_ROUND_TO="${RELU_CLIP_ROUND_TO:-0}"   # to a multiple of N; the Edge TPU array is 64 wide

# Re-ID-only knobs (CLS_MODEL=reid_youtu_lite); what each does is docs/SETUP.md section 3.
REID_EMBED_DIM="${REID_EMBED_DIM:-768}"             # 768 pins the released head; smaller needs simmat
REID_DISTILL_LOSS="${REID_DISTILL_LOSS:-cosine}"    # cosine pins the embedding | simmat is dim-agnostic
REID_GLOBAL_PRUNING="${REID_GLOBAL_PRUNING:-0}"     # 1 deletes resnet50's stem (1/64 channels at 90%)
REID_EXTRA_IMAGE_DIRS="${REID_EXTRA_IMAGE_DIRS:-}"  # extra UNLABELED crop dirs; the teacher labels them

# Line-seg-only knobs (CLS_MODEL=line_seg_*); what each does is docs/SETUP.md section 3.
# CHECKPOINTS here are per-layer CHANNEL ratios, not parameter ratios.
# lane_seg_data and the self-collected dataset live upstream; no public substitute exists.
LANE_SEG_UPSTREAM="${LANE_SEG_UPSTREAM:-$PROJECT_DIR/../MicroROS-Pi5_Coral_TPU/training}"
LANE_SEG_EPOCHS="${LANE_SEG_EPOCHS:-}"              # unset = the upstream mix stage's 30 epochs
LANE_SEG_SAMPLES="${LANE_SEG_SAMPLES:-}"            # unset = its 3000 samples
LANE_SEG_NO_FINETUNE="${LANE_SEG_NO_FINETUNE:-0}"   # 1 = prune and stop; bytes/latency need only structure
LANE_SEG_OUT_DIR="${LANE_SEG_OUT_DIR:-}"            # NO_FINETUNE output: artifacts, not deliverables

# CLS_BASELINE_FILE = stem of the baseline .pt on disk; CLS_BASELINE_NAME = prefix for Phase 1
# output filenames. They diverge on the imagenet path, where the download is the unsuffixed
# `<family>.pt` but outputs keep `_imagenet`. flowers102 / inat_plant pin CLS_MODEL=mobilenetv2.
case "$CLS_DATASET" in
    imagenet)
        CLS_BASELINE_FILE="$CLS_MODEL"
        CLS_BASELINE_NAME="${CLS_MODEL}_imagenet"
        CLS_NUM_CLASSES=1000
        ;;
    flowers102|inat_plant)
        if [[ "$CLS_MODEL" != "mobilenetv2" ]]; then
            echo "[ERROR] CLS_DATASET=$CLS_DATASET only has a mobilenetv2 baseline; set CLS_MODEL=mobilenetv2 (got: $CLS_MODEL)." >&2
            echo "        Only the imagenet path supports the other families (no per-family finetune script)." >&2
            exit 1
        fi
        case "$CLS_DATASET" in
            flowers102)  CLS_BASELINE_FILE="mobilenetv2_flowers102"; CLS_BASELINE_NAME="mobilenetv2_flowers102"; CLS_NUM_CLASSES=102 ;;
            inat_plant)  CLS_BASELINE_FILE="mobilenetv2_inat";       CLS_BASELINE_NAME="mobilenetv2_inat";       CLS_NUM_CLASSES=2101 ;;
        esac
        ;;
    market1501)
        # Re-ID embedder. The baseline .pt is unsuffixed, its weights being FOUR-source (market1501 + Duke
        # + MSMT17 + CUHK03), while Phase 1 outputs carry `_market1501` because the recovery distillation
        # runs on Market crops. No class count: distillation uses no labels.
        if [[ "$CLS_MODEL" != "reid_youtu_lite" ]]; then
            echo "[ERROR] CLS_DATASET=market1501 is the Re-ID path; set CLS_MODEL=reid_youtu_lite (got: $CLS_MODEL)." >&2; exit 1
        fi
        CLS_BASELINE_FILE="reid_youtu_lite"
        # Each (embed_dim, distill_loss) is its own family: different models from the same baseline,
        # so they must not share a stem or they overwrite each other.
        #   768 + cosine  -> reid_youtu_lite_market1501         (the deliverable mainline)
        #   256 + simmat  -> reid_youtu_lite_e256_market1501    (head-vs-backbone ablation)
        #   768 + simmat  -> reid_youtu_lite_simmat_market1501  (loss control: without it, the e256
        #                    ablation confounds "head dim" with "loss changed")
        if [[ "$REID_EMBED_DIM" != "768" ]]; then
            CLS_BASELINE_NAME="reid_youtu_lite_e${REID_EMBED_DIM}_market1501"
        elif [[ "$REID_DISTILL_LOSS" != "cosine" ]]; then
            CLS_BASELINE_NAME="reid_youtu_lite_${REID_DISTILL_LOSS}_market1501"
        else
            CLS_BASELINE_NAME="reid_youtu_lite_market1501"
        fi
        CLS_NUM_CLASSES=0
        ;;
    line_seg)
        # Lane-following segmentation. The baselines are produced upstream and symlinked in by the family's
        # setup.sh, so file stem == output stem: the handoff pins the delivered names as
        # `line_seg_w96_pruned10pct_magnitude_l2.pt`, with no dataset segment and no class count.
        case "$CLS_MODEL" in
            line_seg_base128|line_seg_w96|line_seg_link_r34) ;;
            *) echo "[ERROR] CLS_DATASET=line_seg is the line-seg path; set CLS_MODEL=line_seg_base128|line_seg_w96|line_seg_link_r34 (got: $CLS_MODEL)." >&2; exit 1 ;;
        esac
        CLS_BASELINE_FILE="$CLS_MODEL"
        CLS_BASELINE_NAME="$CLS_MODEL"
        CLS_NUM_CLASSES=0
        # The handoff's §E pins magnitude_l2 for this family; the 5-way default would silently produce four
        # extra models nobody asked for. An explicit IMPORTANCES= still wins.
        [[ -z "$_IMPORTANCES_SET" ]] && IMPORTANCES="magnitude_l2"
        ;;
    *) echo "[ERROR] CLS_DATASET must be imagenet, flowers102, inat_plant, market1501 or line_seg (got: $CLS_DATASET)" >&2; exit 1 ;;
esac

case "$COCO_SUBSET" in
    train2017|minitrain|val2017) ;;
    *) echo "[ERROR] COCO_SUBSET must be train2017, minitrain, or val2017 (got: $COCO_SUBSET)" >&2; exit 1 ;;
esac

# Detection family dispatch (Part B). Both workers share the same CLI surface, so only the worker,
# geometry, num_classes and baseline stem differ. num_classes: effdet uses COCO-90 (no background
# row), torchvision SSDLite uses COCO-91 with background=0.
DET_MODEL="${DET_MODEL:-efficientdet_lite1}"
# Dispatch values come from families/<name>/family.yaml (int8_pruning.manifest). Capture first --
# see the same note in scripts/deliver.sh.
DET_ASSIGNMENTS="$(python -m int8_pruning.manifest det "$DET_MODEL")" || exit 1
eval "$DET_ASSIGNMENTS"

# Date stamp for the log directory (independent of filename "0521")
DATE_TAG="$(date +%Y%m%d_%H%M%S)"

echo "═══════════════════════════════════════════════════════════"
echo "  pruning.sh — production pruning launch"
echo "═══════════════════════════════════════════════════════════"
echo "  PROJECT_DIR  = $PROJECT_DIR"
echo "  DATASET_DIR  = $DATASET_DIR"
echo "  MODELS_DIR   = $MODELS_DIR"
echo "  PARTS        = $PARTS"
echo "  CLS_MODEL    = $CLS_MODEL  (Part A classification family)"
echo "  CLS_DATASET  = $CLS_DATASET  (baseline file=$CLS_BASELINE_FILE.pt, output prefix=$CLS_BASELINE_NAME, num_classes=$CLS_NUM_CLASSES)"
echo "  COCO_SUBSET  = $COCO_SUBSET  (Part B train split; val=val2017)"
echo "  DET_MODEL    = $DET_MODEL  (Part B worker=$DET_WORKER, input=$DET_IMAGE_SIZE, num_classes=$DET_NUM_CLASSES)"
CKPT_NOTE=""
if [[ "$CLS_DATASET" == "line_seg" ]]; then
    echo "  LANE_SEG     = upstream=$LANE_SEG_UPSTREAM epochs=${LANE_SEG_EPOCHS:-(30)} no_finetune=$LANE_SEG_NO_FINETUNE"
    CKPT_NOTE="  (CHANNEL ratios, not param ratios)"
fi
echo "  CHECKPOINTS  = $CHECKPOINTS$CKPT_NOTE"
echo "  IMPORTANCES  = $IMPORTANCES"
echo "  FINAL_EPOCHS = ${FINAL_EPOCHS:-(script default = 100)}"
echo "  DET_LR       = ${DET_LR:-(default = 1e-3)}"
echo "  DET_WD       = ${DET_WD:-(default = 4e-5)}"
echo "  DET_SCHED    = ${DET_SCHED:-(default = cosine)}"
echo "  DET_WARMUP   = ${DET_WARMUP:-(default = 1 ep)}"
echo "  DET_SCOPE    = ${DET_SCOPE:-(default = backbone)}"
[[ -n "$DET_MILESTONES" ]] && echo "  DET_MILESTONES = $DET_MILESTONES"
echo "  DATE_TAG     = $DATE_TAG"
echo "═══════════════════════════════════════════════════════════"
echo

cd "$PROJECT_DIR"

# Tee all output to a log file alongside the script
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/run_${DATE_TAG}.log"
exec > >(tee -a "$RUN_LOG") 2>&1
echo "  Logging to : $RUN_LOG"
echo

# Build optional --final_epochs flag (shared by both parts)
extra_args=()
if [[ -n "$FINAL_EPOCHS" ]]; then
    extra_args+=(--final_epochs "$FINAL_EPOCHS")
fi

# Part-B FT recipe overrides (Python falls back to RECIPE_FT defaults if unset)
det_extra_args=()
[[ -n "$DET_LR"        ]] && det_extra_args+=(--lr "$DET_LR")
[[ -n "$DET_WD"        ]] && det_extra_args+=(--weight_decay "$DET_WD")
[[ -n "$DET_SCHED"     ]] && det_extra_args+=(--scheduler "$DET_SCHED")
[[ -n "$DET_WARMUP"    ]] && det_extra_args+=(--warmup_epochs "$DET_WARMUP")
[[ -n "$DET_MILESTONES" ]] && det_extra_args+=(--lr_decay_epochs $DET_MILESTONES)
# --scope exists on the effdet worker only; the ssdlite worker would reject it.
[[ -n "$DET_SCOPE" ]] && det_extra_args+=(--scope "$DET_SCOPE")

# Part-A overrides. PRUNE_MODE=iterative does LTH-style iterative pruning (one trajectory ->
# all ratios); default (unset) = independent per-ratio. CLS_LR/CLS_WD/CLS_SCHED/
# CLS_MILESTONES tune the recovery FT, mirroring the Part-B DET_* knobs.
# --prune_mode is its own array, NOT part of cls_extra_args: that bundle also carries
# --scheduler/--lr_decay_epochs, which only the classification worker accepts. Passing it to
# a worker that does not implement the flag is a hard argparse error.
prune_mode_args=()
if [[ -n "$PRUNE_MODE" ]]; then
    # Refuse a mode the family has not implemented, for whichever part is running. Without this the flag
    # is dropped for the workers it is not spliced into and the run comes back independent -- a different
    # experiment under the name that was asked for. Part A keys on CLS_MODEL, Part B on DET_MODEL.
    if [[ "$PARTS" == *A* ]]; then
        python -m int8_pruning.manifest prune-mode "$CLS_MODEL" "$PRUNE_MODE" || exit 1
    fi
    if [[ "$PARTS" == *B* ]]; then
        python -m int8_pruning.manifest prune-mode "$DET_MODEL" "$PRUNE_MODE" || exit 1
    fi
    prune_mode_args+=(--prune_mode "$PRUNE_MODE")
fi

cls_extra_args=()
[[ -n "$CLS_LR"         ]] && cls_extra_args+=(--lr "$CLS_LR")
[[ -n "$CLS_WD"         ]] && cls_extra_args+=(--weight_decay "$CLS_WD")
[[ -n "$CLS_SCHED"      ]] && cls_extra_args+=(--scheduler "$CLS_SCHED")
[[ -n "$CLS_MILESTONES" ]] && cls_extra_args+=(--lr_decay_epochs $CLS_MILESTONES)


# Part A — classification family (CLS_MODEL) + dataset
if [[ "$PARTS" == *A* ]]; then
    echo "════════════════════════════════════════════════════════════"
    echo "  Part A — $CLS_MODEL $CLS_DATASET  (classification)"
    echo "════════════════════════════════════════════════════════════"
    baseline="$MODELS_DIR/${CLS_BASELINE_FILE}.pt"

    if [[ ! -f "$baseline" ]]; then
        echo "[ERREUR] $baseline missing." >&2
        case "$CLS_DATASET" in
            imagenet)
                echo "   Run first (on the frontend, with internet access):" >&2
                echo "     (on a cluster login node, your own env-setup script)" >&2
                echo "   or: CLS_DATASET=imagenet ./scripts/download_and_finetune_models.sh" >&2
                ;;
            flowers102|inat_plant)
                echo "   Run first: CLS_DATASET=$CLS_DATASET ./scripts/download_and_finetune_models.sh" >&2
                ;;
        esac
        exit 1
    fi

    cls_dataset_args=(--dataset "$CLS_DATASET")
    case "$CLS_DATASET" in
        imagenet)
            cls_dataset_args+=(--data_root "$DATASET_DIR/Imagenet_1k")
            ;;
        flowers102)
            cls_dataset_args+=(--data_root "$DATASET_DIR/flowers102")
            ;;
        inat_plant)
            inat="$DATASET_DIR/inat2017_plant"
            cls_dataset_args+=(
                --inat_train_json "$inat/train2017.json"
                --inat_val_json   "$inat/val2017.json"
                --inat_image_root "$inat"
            )
            ;;
    esac

    t_start_A=$(date +%s)
    if [[ "$CLS_DATASET" == "line_seg" ]]; then
        # Lane-following segmentation: separate worker. Output is a [1,1,H/2,W/2] mask-logit map, and
        # recovery FT runs the upstream mix-stage recipe on the upstream self-collected dataset, imported
        # and never copied: a second copy of the frames would be a truth source that silently drifts.
        lane_args=()
        [[ -n "$LANE_SEG_EPOCHS"  ]] && lane_args+=(--epochs "$LANE_SEG_EPOCHS")
        [[ -n "$LANE_SEG_SAMPLES" ]] && lane_args+=(--samples "$LANE_SEG_SAMPLES")
        [[ -n "$LANE_SEG_OUT_DIR" ]] && lane_args+=(--out_dir "$LANE_SEG_OUT_DIR")
        [[ "$LANE_SEG_NO_FINETUNE" == "1" ]] && lane_args+=(--no_finetune)
        python "$PROJECT_DIR/families/segmentation/line_seg/prune.py" \
            --variant "$CLS_MODEL" \
            --upstream "$LANE_SEG_UPSTREAM" \
            --baseline_path "$baseline" \
            --batch "${CLS_BATCH:-32}" \
            --checkpoints $CHECKPOINTS \
            --importance $IMPORTANCES \
            "${lane_args[@]}"
    elif [[ "$CLS_MODEL" == "reid_youtu_lite" ]]; then
        # Re-ID embedder: separate worker. Non-square 256x128 input, no labels -- recovery is
        # feature-distillation from a frozen copy of the four-source teacher on unlabeled crops
        # (Duke/MSMT17/CUHK03 are unobtainable, so the ORIGINAL label space is out of reach anyway).
        reid_args=()
        [[ -n "$REID_EXTRA_IMAGE_DIRS" ]] && reid_args+=(--extra_image_dirs $REID_EXTRA_IMAGE_DIRS)
        python "$PROJECT_DIR/families/re-identification/reid/prune.py" \
            --market_dir "$DATASET_DIR/market1501" \
            --baseline_path "$baseline" \
            --baseline_name "$CLS_BASELINE_NAME" \
            --embed_dim "$REID_EMBED_DIM" \
            --distill_loss "$REID_DISTILL_LOSS" \
            --global_pruning "$REID_GLOBAL_PRUNING" \
            --batch_size "${CLS_BATCH:-64}" \
            --checkpoints $CHECKPOINTS \
            --importance $IMPORTANCES \
            "${reid_args[@]}" \
            "${extra_args[@]}"
    elif [[ "$CLS_MODEL" == "clip_rn50" ]]; then
        # Vision-language tower: separate worker. No labels -- cosine feature-distillation from a frozen
        # teacher on unlabeled ImageNet train images. Zero-shot eval needs CLIP_TEXT_EMB + CLIP_VAL_DIR.
        clip_eval_args=()
        [[ -n "$CLIP_TEXT_EMB" ]] && clip_eval_args+=(--text_emb "$CLIP_TEXT_EMB")
        [[ -n "$CLIP_VAL_DIR"  ]] && clip_eval_args+=(--val_images_dir "$CLIP_VAL_DIR")
        python "$PROJECT_DIR/families/classification/clip_rn50/prune.py" \
            --data_root "$DATASET_DIR/Imagenet_1k" \
            --baseline_path "$baseline" \
            --baseline_name "$CLS_BASELINE_NAME" \
            --image_size 224 --batch_size "${CLS_BATCH:-64}" \
            --checkpoints $CHECKPOINTS \
            --importance $IMPORTANCES \
            "${clip_eval_args[@]}" \
            "${prune_mode_args[@]}" \
            "${extra_args[@]}"
    elif [[ "$CLS_MODEL" == "relu_clip" ]]; then
        # The relu-clip efflite4 student: also an embedding, also no labels, so also cosine distillation
        # from a frozen pre-pruning teacher. Differs from clip_rn50 in the allocation scope (per-layer here)
        # and in the transform, the published Resize(224, BICUBIC) + CenterCrop(224).
        relu_clip_args=()
        [[ -n "$RELU_CLIP_TEXT_EMB" ]] && relu_clip_args+=(--text_emb "$RELU_CLIP_TEXT_EMB")
        [[ -n "$RELU_CLIP_VAL_DIR"  ]] && relu_clip_args+=(--val_images_dir "$RELU_CLIP_VAL_DIR")
        python "$PROJECT_DIR/families/classification/relu_clip/prune.py" \
            --data_root "$DATASET_DIR/Imagenet_1k" \
            --baseline_path "$baseline" \
            --baseline_name "$CLS_BASELINE_NAME" \
            --image_size 224 --batch_size "${CLS_BATCH:-64}" \
            --global_pruning "$RELU_CLIP_GLOBAL_PRUNING" \
            --round_to "$RELU_CLIP_ROUND_TO" \
            --checkpoints $CHECKPOINTS \
            --importance $IMPORTANCES \
            "${relu_clip_args[@]}" \
            "${prune_mode_args[@]}" \
            "${extra_args[@]}"
    else
        python "$PROJECT_DIR/families/classification/imagenet_backbones/prune.py" \
            "${cls_dataset_args[@]}" \
            --num_classes "$CLS_NUM_CLASSES" \
            --baseline_path "$baseline" \
            --baseline_name "$CLS_BASELINE_NAME" \
            --image_size 224 --batch_size "${CLS_BATCH:-64}" \
            --checkpoints $CHECKPOINTS \
            --importance $IMPORTANCES \
            "${extra_args[@]}" \
            "${prune_mode_args[@]}" \
            "${cls_extra_args[@]}"
    fi

    # Stash baseline as the 0% data point (symlink, no extra disk). One 0% file per baseline:
    # the importance dimension is irrelevant at 0%. Skipped for the Re-ID e<dim> variants, whose
    # head is pruned 768 -> dim before the body, so a "0%-pruned" model of that family does not
    # exist and symlinking the 768-d baseline under an e256 name would ship the wrong dim.
    mkdir -p "$PT_PRUNED_DIR"
    zero_A="$PT_PRUNED_DIR/${CLS_BASELINE_NAME}_pruned0pct.pt"
    if [[ "$CLS_MODEL" == "reid_youtu_lite" && "$REID_EMBED_DIM" != "768" ]]; then
        echo "  [stash] skipped 0% data point (embed_dim=$REID_EMBED_DIM is not the baseline's 768)"
    elif [[ "$CLS_DATASET" == "line_seg" && ( "$LANE_SEG_NO_FINETUNE" == "1" || -n "$LANE_SEG_OUT_DIR" ) ]]; then
        # Measurement-only run: its models live outside PT_PRUNED_DIR, so do not seed that dir with a 0% row.
        echo "  [stash] skipped 0% data point (measurement-only line_seg run)"
    elif [[ ! -e "$zero_A" ]]; then
        ln -sf "$baseline" "$zero_A"
        echo "  [stash] 0% data point: $zero_A -> ${CLS_BASELINE_FILE}.pt"
    fi
    t_end_A=$(date +%s)
    printf "  Part A duration : %d min\n" $(( (t_end_A - t_start_A) / 60 ))
    echo
fi


# Part B — detection family (DET_MODEL) + COCO 2017 (backbone-only)
if [[ "$PARTS" == *B* ]]; then
    echo "════════════════════════════════════════════════════════════"
    echo "  Part B — $DET_MODEL  (detection, backbone-only)"
    echo "════════════════════════════════════════════════════════════"
    baseline="$MODELS_DIR/${DET_BASELINE_FILE}.pt"
    coco="$DATASET_DIR/coco"

    if [[ ! -e "$baseline" ]]; then
        echo "[ERREUR] $baseline missing." >&2
        echo "   Required step before Part B : run scripts/download_and_finetune_models.sh" >&2
        echo "   to download the $DET_MODEL COCO baseline into \$MODELS_DIR/." >&2
        exit 1
    fi

    # Pre-flight: bail early with a hint if COCO_SUBSET=minitrain but its annotation file is missing.
    if [[ "$COCO_SUBSET" == "minitrain" ]]; then
        mini_json="$coco/annotations/instances_minitrain2017.json"
        if [[ ! -f "$mini_json" ]]; then
            echo "[ERROR] COCO_SUBSET=minitrain but $mini_json is missing." >&2
            echo "        Run scripts/download_datasets.sh to fetch it (~5 MB)." >&2
            exit 1
        fi
    fi

    t_start_B=$(date +%s)
    python "$PROJECT_DIR/$DET_WORKER" \
        --coco_root "$coco" \
        --coco_subset "$COCO_SUBSET" \
        --baseline_name "$DET_BASELINE_NAME" \
        --num_classes "$DET_NUM_CLASSES" \
        --image_size "$DET_IMAGE_SIZE" --batch_size 16 \
        --checkpoints $CHECKPOINTS \
        --importance $IMPORTANCES \
        --eval_every 5 \
        "${extra_args[@]}" \
        "${det_extra_args[@]}"

    # Stash baseline as the 0% data point (backbone-only pruning, so 0% = the original baseline). Tag it
    # with COCO_SUBSET to match the non-0% convention (<baseline>_coco-<subset>_pruned<P>pct_<imp>.pt)
    # and so different subsets get distinct 0pct stashes, all symlinks to the same baseline.
    mkdir -p "$PT_PRUNED_DIR"
    zero_B="$PT_PRUNED_DIR/${DET_BASELINE_NAME}_coco-${COCO_SUBSET}_pruned0pct.pt"
    if [[ ! -e "$zero_B" ]]; then
        ln -sf "$baseline" "$zero_B"
        echo "  [stash] 0% data point: $zero_B -> ${DET_BASELINE_FILE}.pt"
    fi
    t_end_B=$(date +%s)
    printf "  Part B duration : %d min\n" $(( (t_end_B - t_start_B) / 60 ))
    echo
fi


echo "═══════════════════════════════════════════════════════════"
echo "  ✓ pruning.sh completed"
echo "═══════════════════════════════════════════════════════════"
echo "  Pruned models : $PT_PRUNED_DIR/"
echo "  Per-run logs  : $LOG_DIR/"
echo "  Launch log    : $RUN_LOG"
echo
echo "  Inventory check :"
# Glob matches both the 0pct stash and the per-run outputs:
#   cls: `<prefix>_pruned*.pt`           (prefix carries the dataset)
#   det: `<prefix>_coco-*_pruned*.pt`    (subset segment is always present, incl. 0pct)
for prefix in "$CLS_BASELINE_NAME" "$DET_BASELINE_NAME"; do
    # `|| true`: a prefix with no matches makes `ls` exit 2, which under `set -euo pipefail` aborts the
    # whole script *after* the run already succeeded (e.g. PARTS=A leaves the det prefix empty).
    count=$(ls "$PT_PRUNED_DIR/${prefix}"*_pruned*.pt 2>/dev/null | wc -l || true)
    echo "    ${prefix}*_pruned*.pt  →  $count files"
done
