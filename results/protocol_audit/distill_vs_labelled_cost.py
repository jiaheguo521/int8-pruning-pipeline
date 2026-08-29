#!/usr/bin/env python3
"""What the distillation arm costs per step, against a labelled arm on the same model.

Recovery has two arms in this repo (families/*/family.yaml, `recovery:`), and the
distillation one is often assumed to be the cheap one because it needs no annotated
set. In GPU time it is the expensive one, and this measures by how much: the student
forward and backward are the same in both arms, so the whole difference is the
teacher's forward pass -- which does NOT shrink when the student does, because the
teacher is the unpruned model. The harder you prune, the larger a share of the step
it becomes.

Method. One real Market-1501 batch is pinned on the GPU and reused, so the dataloader
is out of the comparison: it is identical for both arms and would only add noise.
Both arms run the same optimizer at the same batch size and resolution, on the same
student, timed after a warm-up.

  arm D  teacher(x) under no_grad -> student(x) -> 1 - cos -> backward   (what
         families/re-identification/reid/prune.py actually runs, via prune.recover.distill_finetune)
  arm L  student(x) -> 751-way linear head -> CrossEntropy -> backward

Arm L is a COST MODEL, not a runnable recovery for this family. The Youtu baseline is
four-source and its label space is unobtainable (Duke withdrawn, MSMT17 licensed),
which is the reason families/re-identification/reid/prune.py distils in the first place; the 751-way
head over Market identities stands in for what a labelled step would cost.

What this does NOT measure: convergence. A cheaper step is not a cheaper recovery if
it needs more epochs, and nothing here compares epochs-to-target. Read it as the cost
of one step, which is all it is.

    python3 results/protocol_audit/distill_vs_labelled_cost.py          # rewrite the JSON
    python3 results/protocol_audit/distill_vs_labelled_cost.py --check  # print, write nothing

Needs torch, a GPU, outputs/ and data/datasets/market1501 -- so it is not in any tier
of scripts/check.sh. The committed JSON is the record.
"""
import glob
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "families" / "re-identification" / "reid"))   # youreid_model, for the full-module pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image

from int8_pruning.prune.core import seed_everything, count_params
from int8_pruning.prune.recover import cosine_loss

OUT = REPO / "results" / "protocol_audit" / "distill_vs_labelled_cost.json"
DEV, BS, H, W = "cuda", 64, 256, 128
WARM, N_TIMED = 8, 30
N_IDENT = 751                      # Market-1501 training identities
TEACHER = REPO / "outputs" / "models" / "reid_youtu_lite.pt"
STUDENTS = [("dense (0%)", "reid_youtu_lite_market1501_pruned0pct.pt"),
            ("pruned 90%", "reid_youtu_lite_market1501_pruned90pct_magnitude_l2.pt")]


def main(check):
    seed_everything(42)
    tf = T.Compose([T.Resize((H, W)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    crops = sorted(glob.glob(str(REPO / "data/datasets/market1501/Market-1501-v15.09.15"
                                 / "bounding_box_train" / "*.jpg")))[:BS]
    if len(crops) < BS:
        sys.exit(f"[skip] need {BS} Market crops, found {len(crops)}")
    images = torch.stack([tf(Image.open(p).convert("RGB")) for p in crops]).to(DEV)
    # CrossEntropy's cost does not depend on which label is correct.
    labels = torch.randint(0, N_IDENT, (BS,), device=DEV)

    teacher = torch.load(TEACHER, map_location="cpu", weights_only=False).to(DEV).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    def timeit(step):
        for _ in range(WARM):
            step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_TIMED):
            step()
        torch.cuda.synchronize()
        return 1000 * (time.perf_counter() - t0) / N_TIMED

    rows = []
    for tag, ckpt in STUDENTS:
        student = torch.load(REPO / "outputs" / "pytorch_pruned" / ckpt,
                             map_location="cpu", weights_only=False).to(DEV)
        student.train()
        head = nn.Linear(student(images).shape[-1], N_IDENT).to(DEV)

        opt_d = optim.AdamW(student.parameters(), lr=3e-4, weight_decay=5e-4)
        def step_distill():
            with torch.no_grad():
                te = teacher(images).float()
            loss = cosine_loss(student(images).float(), te)
            opt_d.zero_grad(set_to_none=True); loss.backward(); opt_d.step()

        opt_l = optim.AdamW(list(student.parameters()) + list(head.parameters()),
                            lr=3e-4, weight_decay=5e-4)
        ce = nn.CrossEntropyLoss()
        def step_labelled():
            loss = ce(head(student(images).float()), labels)
            opt_l.zero_grad(set_to_none=True); loss.backward(); opt_l.step()

        def teacher_forward():                    # to attribute the difference
            with torch.no_grad():
                teacher(images)

        d, l, t = timeit(step_distill), timeit(step_labelled), timeit(teacher_forward)
        rows.append({"student": tag, "checkpoint": ckpt,
                     "student_params": count_params(student),
                     "ms_per_step_distill": round(d, 2),
                     "ms_per_step_labelled": round(l, 2),
                     "ms_teacher_forward_alone": round(t, 2),
                     "distill_over_labelled": round(d / l, 3),
                     "teacher_share_of_distill_step": round(t / d, 3)})
        print(f"  {tag:12s}  distill {d:7.2f} ms   labelled {l:7.2f} ms   "
              f"teacher fwd {t:6.2f} ms   ratio {d / l:.2f}x")
        del student, head, opt_d, opt_l
        torch.cuda.empty_cache()

    doc = {
        "produced_by": "results/protocol_audit/distill_vs_labelled_cost.py",
        "what": "Per-step GPU cost of the two recovery arms on the same student, same batch, "
                "same optimizer. The difference between them is the frozen teacher's forward "
                "pass, which does not shrink with the student.",
        "caveat": "Per-step cost only -- says nothing about epochs to converge. The labelled arm "
                  "is a cost model: this family's real label space is unobtainable, which is why "
                  "it distils. One GPU, one model.",
        "teacher": {"file": str(TEACHER.relative_to(REPO)), "params": count_params(teacher)},
        "batch": BS, "input_hw": [H, W], "warmup_steps": WARM, "timed_steps": N_TIMED,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "rows": rows,
    }
    if check:
        print(json.dumps(doc, indent=1))
    else:
        OUT.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main("--check" in sys.argv)
