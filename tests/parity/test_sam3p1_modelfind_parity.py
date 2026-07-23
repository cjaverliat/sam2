# SPDX-License-Identifier: LicenseRef-SAM
"""Model-find parity: detector mid-stream spawn (text tracking) vs the facebook golden.

Masks are matched id-AGNOSTICALLY (greedy best-IoU): our tracklet lifecycle labels a
re-entering object with a fresh id where upstream re-associates the original id, and
leave/enter transitions can differ by ~1 frame. Those are tracklet re-ID nuances
outside Feature 1b's scope; the SEGMENTATIONS are numerically equivalent, which is
what 1b (dynamic add) guarantees. So we assert per-frame matched-mask IoU and a
count within one object, not exact id/timing agreement.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from sam.prompts import ConceptPrompt

FIX = Path("tests/parity/fixtures/sam3p1")
CKPT = "checkpoints/sam3.1_multiplex.pt"
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.isfile(CKPT)
    or not (FIX / "modelfind_dance.npz").is_file(),
    reason="needs CUDA + sam3.1_multiplex.pt + captured golden",
)

IOU_MEAN = 0.95       # per-frame mean matched IoU (stable frames hit >=0.99)
IOU_FLOOR = 0.80      # per-object floor: catch gross mismatch, allow boundary dips
COUNT_SLACK = 1       # transition timing may lag/lead by one frame


def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else inter / union


def _match(ours, gold):
    """Greedy id-agnostic best-IoU matching; return the matched IoUs."""
    used, ious = set(), []
    for om in ours:
        best, bj = 0.0, -1
        for j, gm in enumerate(gold):
            if j in used:
                continue
            v = _iou(om, gm)
            if v > best:
                best, bj = v, j
        if bj >= 0:
            used.add(bj)
            ious.append(best)
    return ious


def test_modelfind_detector_midstream_parity():
    from sam.build_sam import build_sam3_multiplex_video_predictor
    from sam.models.sam3_predictor import Sam3VideoPredictorState

    g = np.load(FIX / "modelfind_dance.npz")
    scn = json.loads((FIX / "modelfind_scenario.json").read_text())
    n, (h, w) = scn["num_frames"], scn["hw"]
    frames = [
        np.asarray(Image.open(f"notebooks/videos/dance/{i:05d}.jpg").convert("RGB"))
        for i in range(n)
    ]

    pred = build_sam3_multiplex_video_predictor(
        config_file="configs/sam3/sam3.1.yaml", ckpt_path=CKPT, device="cuda")
    st = Sam3VideoPredictorState(video_hw=(h, w))
    pred.set_concept(st, ConceptPrompt(scn["phrase"]))

    our_ids, gold_ids = set(), set()
    for i, fr in enumerate(frames):
        out = pred.forward(st, i, fr)
        our_ids.update(out)
        gold_ids.update(int(o) for o in g[f"frame{i}_obj_ids"])
        ours = [(out[k].masks_logits[0, 0] > 0).cpu().numpy() for k in sorted(out)]
        gids = g[f"frame{i}_obj_ids"]
        gold = [g[f"frame{i}_obj{o}"].astype(bool) for o in gids]
        assert abs(len(ours) - len(gold)) <= COUNT_SLACK, (
            f"frame {i}: object count {len(ours)} vs golden {len(gold)}"
        )
        ious = _match(ours, gold)
        assert ious, f"frame {i}: no masks matched"
        assert min(ious) >= IOU_FLOOR, (
            f"frame {i}: matched IoU {min(ious):.3f} < floor {IOU_FLOOR} (gross mismatch)"
        )
        assert float(np.mean(ious)) >= IOU_MEAN, (
            f"frame {i}: mean matched IoU {np.mean(ious):.3f} < {IOU_MEAN}"
        )

    # Re-ID: a re-entering object reuses its id (no fresh spawn). Upstream tracks the
    # whole clip with a fixed id set; our port must not mint spurious extra ids.
    assert len(our_ids) == len(gold_ids), (
        f"distinct ids {sorted(our_ids)} vs golden {sorted(gold_ids)} "
        "(a re-entering object must reuse its id, not spawn a fresh one)"
    )
