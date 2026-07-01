# SPDX-License-Identifier: Apache-2.0
"""Characterization harness for the sam2->sam refactor.

Freezes a deterministic block output (CPU, fp32, fixed seed) for EfficientTAM-tiny
so the rename can be proven behavior-preserving. Capture once with CAPTURE_GOLDEN=1,
then this test compares against the committed fixture. Skips if the checkpoint is absent.

The .npy fixture is the invariant across refactor steps.
"""
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from sam.build_sam import build_sam2_video_predictor

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures"
CKPT = ROOT / "checkpoints" / "efficienttam_ti.pt"
CONFIG = "configs/efficienttam/efficienttam_ti.yaml"
ATOL = RTOL = 1e-4


@pytest.mark.skipif(not CKPT.is_file(), reason=f"checkpoint absent: {CKPT}")
def test_image_encode_parity():
    torch.manual_seed(0)
    model = build_sam2_video_predictor(
        CONFIG, str(CKPT), device="cpu", mode="eval", use_half=False
    )
    frame = torch.rand(3, model.image_size, model.image_size)
    with torch.inference_mode():
        emb, _pos = model.encode_image(frame)
    got = emb[-1].float().cpu().numpy()  # lowest-res feature level

    golden = FIXTURES / "etam_ti_image_emb.npy"
    if os.environ.get("CAPTURE_GOLDEN"):
        FIXTURES.mkdir(parents=True, exist_ok=True)
        np.save(golden, got)
        pytest.skip("captured golden fixture")
    np.testing.assert_allclose(got, np.load(golden), atol=ATOL, rtol=RTOL)
