# SPDX-License-Identifier: LicenseRef-SAM
"""Strict-load build test for EfficientSAM3.1 distilled-RepViT multiplex-video (MobileCLIP-S0, ctx16).

``build_efficientsam3p1_video_predictor(
    config_file="configs/efficientsam3/efficientsam3p1_repvit_m_mobileclip_s0_ctx16.yaml",
    ckpt_path=..., backbone_type="repvit", model_name="m1_1")`` must compose a
``Sam3MultiplexVideoPredictor`` whose text encoder is ``MobileClipTextEncoder``, vision
trunk is ``EfficientSam3Trunk`` with ``channel_list==[1024]``, and tracker carries 457
keys.  The ``_load_sam3_multiplex_video_checkpoint`` loader then STRICT-loads all 1672
keys of ``efficient_sam3p1_repvit_m_mobileclip_s0_ctx16.pt`` (0 missing / 0 unexpected).
Skipped when the checkpoint is absent.
"""
import os

import pytest

# Try the flat checkpoints/ copy first (written by download_efficientsam3.py), then the
# validated-during-review copy kept under checkpoints/_esam3_validate/.
_CKPT_PRIMARY = "checkpoints/efficient_sam3p1_repvit_m_mobileclip_s0_ctx16.pt"
_CKPT_VALIDATE = "checkpoints/_esam3_validate/stage1_sam3p1/efficient_sam3p1_repvit_m_mobileclip_s0_ctx16.pt"
_CKPT = _CKPT_PRIMARY if os.path.exists(_CKPT_PRIMARY) else _CKPT_VALIDATE


@pytest.mark.skipif(not os.path.exists(_CKPT), reason="EfficientSAM3.1 RepViT-M s0/ctx16 ckpt absent")
def test_build_efficientsam3p1_repvit_m_s0_ctx16_strict_load():
    import torch

    from sam.build_sam import build_efficientsam3p1_video_predictor
    from sam.modeling.encoders.efficientsam3_trunk import EfficientSam3Trunk
    from sam.modeling.text.mobileclip_text_encoder import MobileClipTextEncoder
    from sam.models.sam3_predictor import Sam3MultiplexVideoPredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # A successful build IS the strict-load assertion: _load_sam3_multiplex_video_checkpoint
    # calls model.load_state_dict(..., strict=True) and raises on any key mismatch.
    model = build_efficientsam3p1_video_predictor(
        config_file="configs/efficientsam3/efficientsam3p1_repvit_m_mobileclip_s0_ctx16.yaml",
        ckpt_path=_CKPT,
        device=device,
        backbone_type="repvit",
        model_name="m1_1",
    )
    assert isinstance(model, Sam3MultiplexVideoPredictor)
    assert isinstance(model.text_encoder, MobileClipTextEncoder)
    # Vision trunk must be the distilled EfficientSam3Trunk (not PE ViT).
    assert isinstance(model.vision_encoder.vision_backbone.trunk, EfficientSam3Trunk)
    assert model.vision_encoder.vision_backbone.trunk.channel_list == [1024]
    # Tracker carries exactly 457 keys (multiplex tracker, unchanged from SAM 3.1).
    assert len(list(model.tracker.state_dict())) == 457
