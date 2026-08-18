# SPDX-License-Identifier: LicenseRef-SAM
"""Mask-only ``GeometryPrompt`` on the base SAM 3 video predictor.

A mask prompt takes the same tracker route as a click: it seeds one object, no
detection runs, and the object tracks to the end of the clip. The weights are the
tracker's own (``tracker.sam_prompt_encoder.mask_downscaling``, 10 tensors in
``sam3.pt``) -- the same ones the loop already uses when a detection seeds a tracklet.

Only the DETECTOR's geometry mask slot is unsupported (no ``mask_encoder`` weights in
either checkpoint), so pairing a mask with a box still raises.

The reference masks are the interactive-click golden, since seeding with that golden's
frame-0 mask should reproduce the object it was captured from.
"""
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from sam.prompts import BoxRoute, GeometryPrompt

FIX = Path("tests/parity/fixtures/sam3")
CKPT = "checkpoints/sam3.pt"
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.isfile(CKPT)
    or not (FIX / "interactive_noconcept.npz").is_file(),
    reason="needs CUDA + sam3.pt + the interactive golden (for its masks)",
)

N_FRAMES = 6
MIN_IOU = 0.90


def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else inter / union


@pytest.fixture(scope="module")
def predictor():
    from sam.build_sam import build_sam3_video_predictor

    return build_sam3_video_predictor(
        config_file="configs/sam3/sam3.yaml", ckpt_path=CKPT, device="cuda")


@pytest.fixture(scope="module")
def clip():
    golden = np.load(FIX / "interactive_noconcept.npz")["masklets"]
    frames = [
        np.asarray(Image.open(f"notebooks/videos/bedroom/{i:05d}.jpg").convert("RGB"))
        for i in range(N_FRAMES)
    ]
    return frames, golden


def _mask_logits(mask_bool):
    """The golden's boolean mask as the logits a caller would pass in."""
    return torch.from_numpy(mask_bool.astype(np.float32))[None] * 20.0 - 10.0


def test_mask_only_prompt_seeds_and_tracks(predictor, clip):
    from sam.models.sam3_predictor import Sam3VideoPredictorState

    frames, golden = clip
    h, w, _ = frames[0].shape
    state = Sam3VideoPredictorState(video_hw=(h, w))
    prompt = GeometryPrompt(obj_id=1, masks_logits=_mask_logits(golden[0]).cuda())

    ious = []
    for i, frame in enumerate(frames):
        out = predictor.forward(state, i, frame, prompts=[prompt] if i == 0 else [])
        assert 1 in out, f"frame {i}: mask-seeded object missing from the output"
        assert sorted(out) == [1], (
            f"frame {i}: expected only the seeded object, got {sorted(out)} "
            "(a mask prompt must not switch detection on)"
        )
        ious.append(_iou((out[1].masks_logits[0, 0] > 0).cpu().numpy(), golden[i]))

    worst = int(np.argmin(ious))
    assert min(ious) >= MIN_IOU, (
        f"frame {worst}: IoU {ious[worst]:.4f} < {MIN_IOU} "
        f"(per-frame: {[round(v, 3) for v in ious]})"
    )


def test_mask_with_detector_box_is_rejected(predictor, clip):
    """The detector's geometry mask slot has no weights -- that pairing must raise."""
    from sam.models.sam3_predictor import Sam3VideoPredictorState

    frames, golden = clip
    h, w, _ = frames[0].shape
    state = Sam3VideoPredictorState(video_hw=(h, w))
    prompt = GeometryPrompt(
        obj_id=1,
        masks_logits=_mask_logits(golden[0]).cuda(),
        boxes=torch.tensor([[285.0, 0.0, 535.0, 430.0]], device="cuda"),
        box_route=BoxRoute.DETECTOR,
    )
    with pytest.raises(NotImplementedError, match="mask"):
        predictor.forward(state, 0, frames[0], prompts=[prompt])
