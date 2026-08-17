# SPDX-License-Identifier: LicenseRef-SAM
"""Image box-prompt parity vs the facebook SAM 3.1 golden.

Detections are matched to the golden by box-IoU (id-agnostic). The box/point
geometry tokens lengthen the VL-encoder + 200-query decoder sequence, so under
bf16 the outputs drift more than the bit-exact text-only path (which passes at
2px/1e-2 in test_sam3_parity). We therefore assert a matched detection set with
close-but-not-bitexact tolerances: same count + presence, each matched box IoU
high, score and mask close. Tightening to bit-exact is tracked as follow-up.

Both box signs are covered (upstream ``add_geometric_prompt(box, label)``; the
label indexes ``geometry_encoder.label_embed``, an ``nn.Embedding(2, d)``):

- ``box_prompt`` (label 1, positive) -- the prompt carries no ``boxes_labels``,
  exercising the all-positive default.
- ``box_prompt_neg`` (label 0, negative) -- "not this one". Upstream drops the
  detection enclosed by the box (3 dets -> 2, presence 0.99999 -> 0.8606), so
  this case fails outright if ``boxes_labels`` is ignored.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from sam.prompts import ConceptPrompt, GeometryPrompt

FIX = Path("tests/parity/fixtures/sam3")
CKPT = "checkpoints/sam3.1_multiplex.pt"
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.isfile(CKPT)
    or not (FIX / "box_prompt.npz").is_file()
    or not (FIX / "box_prompt_neg.npz").is_file(),
    reason="needs CUDA + sam3.1_multiplex.pt + captured goldens",
)

BOX_IOU_MIN = 0.80
SCORE_ATOL = 0.06
PRESENCE_ATOL = 1e-2
MASK_IOU_MIN = 0.85


def _box_iou(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else inter / union


@pytest.mark.parametrize("stem", ["box_prompt", "box_prompt_neg"])
def test_image_box_prompt_parity(stem):
    from sam.build_sam import build_sam3_multiplex

    scn = json.loads((FIX / f"{stem}_scenario.json").read_text())
    g = np.load(FIX / f"{stem}.npz")
    g_boxes, g_scores = g["boxes"], g["scores"]
    g_presence, g_masks = float(g["presence"]), g["masks"]
    frame = np.asarray(Image.open(scn["frame"]).convert("RGB"))

    pred = build_sam3_multiplex(
        config_file="configs/sam3/sam3.1.yaml", ckpt_path=CKPT, device="cuda")
    # a scenario without "box_label" exercises the all-positive default
    labels = (
        None if "box_label" not in scn
        else torch.tensor([scn["box_label"]], dtype=torch.long)
    )
    box = GeometryPrompt(obj_id=1, boxes=torch.tensor([scn["box_xyxy"]], dtype=torch.float32),
                         boxes_labels=labels)
    det = pred.predict(frame, ConceptPrompt(scn["phrase"]),
                       confidence_threshold=scn["confidence_threshold"], geometry=box)

    assert det.num_detections == int(g["n"]), (
        f"{det.num_detections} detections vs golden {int(g['n'])}"
    )
    assert abs(det.presence - g_presence) <= PRESENCE_ATOL, (
        f"presence {det.presence:.4f} vs {g_presence:.4f}"
    )

    our_boxes = det.boxes.cpu().numpy()
    our_scores = det.scores.cpu().numpy()
    our_masks = (det.masks_logits > 0).cpu().numpy()
    used = set()
    for gi in range(int(g["n"])):
        best, bj = 0.0, -1
        for j in range(det.num_detections):
            if j in used:
                continue
            v = _box_iou(g_boxes[gi], our_boxes[j])
            if v > best:
                best, bj = v, j
        assert bj >= 0 and best >= BOX_IOU_MIN, (
            f"golden det {gi} box {g_boxes[gi].round(0)} unmatched (best IoU {best:.3f})"
        )
        used.add(bj)
        assert abs(our_scores[bj] - g_scores[gi]) <= SCORE_ATOL, (
            f"det {gi}: score {our_scores[bj]:.3f} vs {g_scores[gi]:.3f}"
        )
        assert _mask_iou(our_masks[bj], g_masks[gi].astype(bool)) >= MASK_IOU_MIN, (
            f"det {gi}: mask IoU below {MASK_IOU_MIN}"
        )
