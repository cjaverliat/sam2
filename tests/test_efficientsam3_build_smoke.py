# SPDX-License-Identifier: LicenseRef-SAM
"""CI-safe build-smoke tests for EfficientSAM3 (no checkpoint required).

Verifies that ``build_efficientsam3`` composes the full ``Sam3Predictor``
correctly (hydra config -> instantiate) with RANDOM weights (``ckpt_path=None``)
and that the vision encoder produces the expected feature pyramid shape for a
1008×1008 input (detector resolution 72×72, d_model 256).

No GPU or checkpoint is needed — the test runs on CPU and is suitable for CI.
"""
from __future__ import annotations

import torch
import pytest

from sam.build_sam import build_efficientsam3
from sam.models.sam3_predictor import Sam3Predictor
from sam.modeling.encoders.efficientsam3_trunk import EfficientSam3Trunk
from sam.modeling.text.mobileclip_text_encoder import MobileClipTextEncoder


def test_efficientsam3_meta_build_and_forward():
    """Build with random weights and run a dummy vision forward on CPU.

    Guards:
    * The hydra config composes without error (correct ``_target_`` paths, dims).
    * The returned model is a ``Sam3Predictor`` owning an EfficientSam3Trunk vision
      trunk and a MobileClipTextEncoder text tower.
    * A zero image (1, 3, 1008, 1008) produces 3 pyramid levels, the detector level
      (``feats[-1]``) at (1, 256, 72, 72) — matching the RepViT-M1.1 stride-14 grid
      for a 1008-px image with ``d_model=256``.
    """
    model = build_efficientsam3(ckpt_path=None, device="cpu", mode="eval")

    assert isinstance(model, Sam3Predictor), (
        f"Expected Sam3Predictor, got {type(model).__name__}"
    )
    assert isinstance(model.vision_encoder.vision_backbone.trunk, EfficientSam3Trunk), (
        "vision_backbone.trunk must be EfficientSam3Trunk (RepViT-M1.1)"
    )
    assert isinstance(model.text_encoder, MobileClipTextEncoder), (
        "text_encoder must be MobileClipTextEncoder (MobileCLIP-S0)"
    )

    with torch.inference_mode():
        feats, pos = model.vision_encoder(torch.zeros(1, 3, 1008, 1008))

    # 3 pyramid levels produced by the Simple-FPN neck (scale_factors 4/2/1)
    assert len(feats) == 3, f"Expected 3 pyramid levels, got {len(feats)}"
    # Detector-resolution level: 1008 / 14 = 72, d_model = 256
    assert feats[-1].shape[-2:] == (72, 72), (
        f"Detector level spatial size: expected (72, 72), got {feats[-1].shape[-2:]}"
    )
    assert feats[-1].shape[1] == 256, (
        f"Detector level channels: expected 256, got {feats[-1].shape[1]}"
    )
    # Position embeddings have the same spatial layout as the features
    assert pos[-1].shape[-2:] == (72, 72), (
        f"Pos-enc spatial size mismatch: {pos[-1].shape[-2:]}"
    )
