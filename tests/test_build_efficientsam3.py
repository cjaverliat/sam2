# SPDX-License-Identifier: LicenseRef-SAM
"""Strict-load build test for the EfficientSAM3 (RepViT-M1.1 + MobileCLIP-S0) image predictor.

``build_efficientsam3(ckpt_path=...)`` must compose a base-lineage ``Sam3Predictor`` whose
vision trunk is the ``EfficientSam3Trunk`` and whose text tower is the
``MobileClipTextEncoder``, then STRICT-load all 1107 keys of
``efficientsam3_ft/efficientsam3_repvit.pt`` (0 missing / 0 unexpected) -- i.e. the model has
exactly the checkpoint's parameters (no geometry encoder, no SAM 2 / interactive / tracker
params). Skipped when the gated checkpoint is absent.
"""
import os

import pytest

CKPT = "checkpoints/_esam3_validate/efficientsam3_ft/efficientsam3_repvit.pt"


@pytest.mark.skipif(not os.path.exists(CKPT), reason="EfficientSAM3 RepViT ckpt absent")
def test_build_efficientsam3_strict_load():
    from sam.build_sam import build_efficientsam3
    from sam.models.sam3_predictor import Sam3Predictor

    model = build_efficientsam3(ckpt_path=CKPT, device="cpu")
    assert isinstance(model, Sam3Predictor)
    # the swapped vision trunk + text encoder are the EfficientSAM3 ones
    assert model.vision_encoder.vision_backbone.trunk.channel_list == [1024]
    from sam.modeling.text.mobileclip_text_encoder import MobileClipTextEncoder

    assert isinstance(model.text_encoder, MobileClipTextEncoder)
