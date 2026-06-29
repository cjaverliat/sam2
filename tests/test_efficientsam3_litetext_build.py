# SPDX-License-Identifier: LicenseRef-SAM
"""Strict-load build test for SAM3-LiteText base-video (MobileCLIP-S0, ctx16).

``build_sam3_video_predictor(config_file="configs/efficientsam3/sam3_litetext_s0_ctx16.yaml",
ckpt_path=...)`` must compose a base-lineage ``Sam3VideoPredictor`` whose vision encoder
is the PE ViT (add_sam2_neck=True), text encoder is ``MobileClipTextEncoder``, detector
includes the trained geometry encoder, and tracker is the base ``Sam3Tracker``.  The
``_load_sam3_video_checkpoint`` loader then STRICT-loads all 1281 keys of
``sam3_litetext_mobileclip_s0_ctx16.pt`` (0 missing / 0 unexpected).  Skipped when the
checkpoint is absent.
"""
import os

import pytest

# Try the flat checkpoints/ copy first (written by download_efficientsam3.py), then the
# validated-during-review copy kept under checkpoints/_esam3_validate/.
_CKPT_PRIMARY = "checkpoints/sam3_litetext_mobileclip_s0_ctx16.pt"
_CKPT_VALIDATE = "checkpoints/_esam3_validate/sam3_litetext/sam3_litetext_mobileclip_s0_ctx16.pt"
_CKPT = _CKPT_PRIMARY if os.path.exists(_CKPT_PRIMARY) else _CKPT_VALIDATE


@pytest.mark.skipif(not os.path.exists(_CKPT), reason="SAM3-LiteText s0/ctx16 ckpt absent")
def test_build_efficientsam3_litetext_s0_ctx16_strict_load():
    import torch

    from sam.build_sam import build_sam3_video_predictor
    from sam.modeling.text.mobileclip_text_encoder import MobileClipTextEncoder
    from sam.models.sam3_predictor import Sam3VideoPredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # A successful build IS the strict-load assertion: _load_sam3_video_checkpoint
    # calls model.load_state_dict(..., strict=True) and raises on any key mismatch.
    model = build_sam3_video_predictor(
        config_file="configs/efficientsam3/sam3_litetext_s0_ctx16.yaml",
        ckpt_path=_CKPT,
        device=device,
    )
    assert isinstance(model, Sam3VideoPredictor)
    assert isinstance(model.text_encoder, MobileClipTextEncoder)
