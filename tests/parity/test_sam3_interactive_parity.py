# SPDX-License-Identifier: LicenseRef-SAM
"""Interactive click parity vs the facebook base SAM 3 (``sam3.pt``) golden.

A single positive click on frame 0, NO text concept: the clicked object is seeded
through the tracker's prompt encoder and tracked forward for the whole clip.

The golden (``interactive_noconcept.npz``, 30 frames, captured by
``reference_sam3/capture_sam3_interactive_golden.py``) emits the clicked object on
EVERY frame -- upstream force-confirms it in ``add_tracker_new_points``
(``sam3_video_inference.py:1522-1531``: ``masklet_confirmation["status"] = 1`` and
``consecutive_det_num`` set to the threshold), and its hotstart purge only reaps ids
the DETECTOR registered (``_process_hotstart`` walks ``unmatched_frame_inds``, keyed
from ``new_det_obj_ids``). A click-seeded object is therefore never a purge candidate.

The clip length matters: with detection off, our lifecycle counts an unmatched frame
per step, so a tracklet under the base constants dies at frame 8
(``hotstart_unmatch_thresh=8`` inside ``hotstart_delay=15``). 30 frames puts that well
inside the gated range.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from sam.prompts import GeometryPrompt

FIX = Path("tests/parity/fixtures/sam3")
CKPT = "checkpoints/sam3.pt"
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.isfile(CKPT)
    or not (FIX / "interactive_noconcept.npz").is_file(),
    reason="needs CUDA + sam3.pt + captured golden",
)

MIN_IOU = 0.90   # per-frame floor across the clip
MEAN_IOU = 0.95  # clip mean


def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else inter / union


def test_interactive_click_tracks_whole_clip(determinism_no_det_algos):
    from sam.build_sam import build_sam3_video_predictor
    from sam.models.sam3_predictor import Sam3VideoPredictorState

    determinism_no_det_algos()
    scn = json.loads((FIX / "interactive_scenario.json").read_text())
    g = np.load(FIX / "interactive_noconcept.npz")
    golden = g["masklets"]
    n, (h, w) = scn["num_frames"], scn["hw"]
    frames = [
        np.asarray(Image.open(f"{scn['frames_dir']}/{i:05d}.jpg").convert("RGB"))
        for i in range(n)
    ]

    pred = build_sam3_video_predictor(
        config_file="configs/sam3/sam3.yaml", ckpt_path=CKPT, device="cuda")
    st = Sam3VideoPredictorState(video_hw=(h, w))
    x, y = scn["frame0_click_xy"]
    obj_id = scn["obj_id"]
    click = GeometryPrompt(
        obj_id=obj_id,
        points_coords=torch.tensor([[x, y]]),
        points_labels=torch.tensor([scn["label"]]),
    )

    ious = []
    for i, frame in enumerate(frames):
        out = pred.forward(st, i, frame, geometry_prompts=[click] if i == 0 else [])
        assert list(g[f"frame{i}_obj_ids"]) == [obj_id], (
            f"golden frame {i} is not the single-object scenario this test assumes"
        )
        assert obj_id in out, (
            f"frame {i}: clicked object lost; the golden emits it on every frame"
        )
        mask = (out[obj_id].masks_logits[0, 0] > 0).cpu().numpy()
        ious.append(_iou(mask, golden[i]))

    worst = int(np.argmin(ious))
    assert min(ious) >= MIN_IOU, (
        f"frame {worst}: IoU {ious[worst]:.4f} < {MIN_IOU} "
        f"(per-frame: {[round(v, 3) for v in ious]})"
    )
    assert float(np.mean(ious)) >= MEAN_IOU, (
        f"mean IoU {np.mean(ious):.4f} < {MEAN_IOU}"
    )
