# SPDX-License-Identifier: LicenseRef-SAM
"""Image box-prompt parity vs the facebook SAM 3.1 golden.

Detections are matched to the golden by box-IoU (id-agnostic), then gated at the
same tolerances as the text-only image parity in test_sam3_parity: boxes 2px,
scores 1e-2, presence 1e-2. Geometry tokens cost nothing in accuracy -- the
measured residual is 0.0 on every score of ``box_prompt`` and 3.8e-3 on one
``box_prompt_neg`` score, with the mask-derived boxes bit-identical.

Two conventions have to line up for that, and both bit the earlier (much looser)
version of this test:

* **Preprocessing.** The golden is captured in the IMAGE regime (upstream
  ``Sam3Processor``: uint8 -> GPU -> ``v2.Resize(1008)`` bilinear+antialias ->
  float32), which is what ``predict`` runs via ``preprocess_to_1008``. Capturing it
  through ``init_state`` instead -- the image-FOLDER video loader, PIL CPU resize ->
  float16, mirrored by ``preprocess_to_1008_video`` -- injects a systematic encoder
  input difference (``enc_feat`` median delta ~0.037) that reaches 5.5e-2 on a
  near-threshold score and drops a mask IoU to 0.93. That is the resize mismatch,
  not the geometry path.
* **Box convention.** ``predict`` returns ``masks_to_boxes`` of the output mask (the
  multiplex demo semantics), while the golden's ``boxes`` array is the raw DETR
  ``pred_boxes_xyxy``. The two differ by up to 15.7px on the same detection, so the
  golden box is re-derived here from the golden MASKS, putting both sides in the
  same convention.

Both box signs are covered (upstream ``add_geometric_prompt(box, label)``; the
label indexes ``geometry_encoder.label_embed``, an ``nn.Embedding(2, d)``):

- ``box_prompt`` (label 1, positive) -- driven with ``boxes_labels=None``, since 1 is
  what the all-positive default produces; that default is what this stem exercises.
- ``box_prompt_neg`` (label 0, negative) -- "not this one", passed explicitly. Upstream
  drops the detection enclosed by the box (3 dets -> 2, presence 0.99999 -> 0.8615), so
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

BOX_ATOL_PX = 2.0     # text-only image-parity bar; measured max delta 0.00px
SCORE_ATOL = 1e-2     # text-only image-parity bar; measured max delta 3.8e-3
PRESENCE_ATOL = 1e-2  # measured max delta 9.3e-4
MASK_IOU_MIN = 0.98   # measured min 0.9898 (the boxed, partly occluded person)


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
def test_image_box_prompt_parity(stem, determinism_no_det_algos):
    from sam.build_sam import build_sam3_multiplex

    from sam.models.sam3_predictor import _masks_to_boxes

    # the capture's regime: seed 0, cuDNN deterministic, TF32 off (it does NOT set
    # use_deterministic_algorithms, so neither does this)
    determinism_no_det_algos()

    scn = json.loads((FIX / f"{stem}_scenario.json").read_text())
    g = np.load(FIX / f"{stem}.npz")
    g_scores = g["scores"]
    g_presence, g_masks = float(g["presence"]), g["masks"]
    # the npz `boxes` are the raw DETR pred_boxes_xyxy; predict() returns
    # mask-derived boxes, so re-derive the golden's the same way (see module docstring)
    g_boxes = _masks_to_boxes(torch.from_numpy(g_masks.astype(bool))).numpy()
    frame = np.asarray(Image.open(scn["frame"]).convert("RGB"))

    pred = build_sam3_multiplex(
        config_file="configs/sam3/sam3.1.yaml", ckpt_path=CKPT, device="cuda")
    # label 1 IS the all-positive default, so drive that stem with boxes_labels=None
    # to exercise the default path; the negative stem passes its label explicitly
    labels = (
        None if scn["box_label"] == 1
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
        assert bj >= 0 and best > 0.0, (
            f"golden det {gi} box {g_boxes[gi].round(0)} unmatched (best IoU {best:.3f})"
        )
        used.add(bj)
        d_box = float(np.abs(our_boxes[bj] - g_boxes[gi]).max())
        assert d_box <= BOX_ATOL_PX, (
            f"det {gi}: box {our_boxes[bj].round(1)} vs {g_boxes[gi].round(1)} "
            f"max|delta|={d_box:.2f}px"
        )
        assert abs(our_scores[bj] - g_scores[gi]) <= SCORE_ATOL, (
            f"det {gi}: score {our_scores[bj]:.3f} vs {g_scores[gi]:.3f}"
        )
        assert _mask_iou(our_masks[bj], g_masks[gi].astype(bool)) >= MASK_IOU_MIN, (
            f"det {gi}: mask IoU below {MASK_IOU_MIN}"
        )
