# SPDX-License-Identifier: LicenseRef-SAM
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from sam.prompts import GeometryPrompt

FIX = Path("tests/parity/fixtures/sam3p1")
CKPT = "checkpoints/sam3.1_multiplex.pt"
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.isfile(CKPT)
    or not (FIX / "interactive_noconcept.npz").is_file(),
    reason="needs CUDA + sam3.1_multiplex.pt + captured golden",
)


def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else inter / union


def test_interactive_seed_click_parity():
    from sam.build_sam import build_sam3_multiplex_video_predictor
    from sam.models.sam3_predictor import Sam3VideoPredictorState

    scenario = json.loads((FIX / "interactive_scenario.json").read_text())
    golden = np.load(FIX / "interactive_noconcept.npz")["masklets"]  # (n,H,W) bool
    n, h, w = golden.shape
    frames = [
        np.asarray(Image.open(f"notebooks/videos/bedroom/{i:05d}.jpg").convert("RGB"))
        for i in range(n)
    ]

    pred = build_sam3_multiplex_video_predictor(
        config_file="configs/sam3/sam3.1.yaml", ckpt_path=CKPT, device="cuda")
    st = Sam3VideoPredictorState(video_hw=(h, w))
    x, y = scenario["frame0_click_xy"]
    click = GeometryPrompt(
        obj_id=scenario["obj_id"],
        points_coords=torch.tensor([[x, y]]),
        points_labels=torch.tensor([scenario["label"]]),
    )
    for i, fr in enumerate(frames):
        out = pred.forward(st, i, fr, geometry_prompts=[click] if i == 0 else [])
        oid = scenario["obj_id"]
        assert oid in out, f"frame {i}: object {oid} lost"
        pred_mask = (out[oid].masks_logits[0, 0] > 0).cpu().numpy()
        iou = _iou(pred_mask, golden[i])
        assert iou >= 0.99, f"frame {i}: IoU {iou:.3f} < 0.99"
