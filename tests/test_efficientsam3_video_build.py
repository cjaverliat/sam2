# SPDX-License-Identifier: LicenseRef-SAM
"""Strict-load build test for EfficientSAM3 base-lineage (NON-multiplex) VIDEO variants.

EfficientSAM3 ships distilled VIDEO checkpoints (``stage1_all_converted/efficient_sam3_<bb>_m_geo_ft.pt``)
that are the BASE SAM 3 lineage (per-object, 309-key ``Sam3Tracker`` — NOT the 457-key multiplex
tracker of EfficientSAM3.1). They pair a distilled ``EfficientSam3Trunk`` vision encoder with the
PE text tower (``Sam3TextEncoder``, ctx32) and carry a TRAINED geometry encoder.

``build_sam3_video_predictor(config_file="configs/efficientsam3/efficientsam3_<bb>_video.yaml",
ckpt_path=...)`` reuses the existing base-lineage builder + ``_load_sam3_full_checkpoint`` (the
3-group remap is vision-trunk- and text-agnostic), strict-loading all 1698 keys
(vision 697 = trunk 653 + convs 22 + sam2_convs 22, language 295, detector head 397, tracker 309).

Skipped per-variant when the checkpoint is absent (CI-safe).
"""
import os

import pytest

# variant -> (config, checkpoint basename, expected EfficientSam3Trunk backbone_type)
_VARIANTS = {
    "repvit": (
        "configs/efficientsam3/efficientsam3_repvit_video.yaml",
        "efficient_sam3_repvit_m_geo_ft.pt",
        "repvit",
    ),
    "tinyvit": (
        "configs/efficientsam3/efficientsam3_tinyvit_video.yaml",
        "efficient_sam3_tinyvit_m_geo_ft.pt",
        "tinyvit",
    ),
    "efficientvit": (
        "configs/efficientsam3/efficientsam3_efficientvit_video.yaml",
        "efficient_sam3_efficientvit_m_geo_ft.pt",
        "efficientvit",
    ),
}


def _resolve_ckpt(basename: str) -> str | None:
    """Flat checkpoints/ copy first (download tool), then the validated-review copy."""
    primary = os.path.join("checkpoints", basename)
    validate = os.path.join("checkpoints", "_esam3_validate", "stage1_all_converted", basename)
    if os.path.exists(primary):
        return primary
    if os.path.exists(validate):
        return validate
    return None


@pytest.mark.parametrize("variant", sorted(_VARIANTS))
def test_build_efficientsam3_video_strict_load(variant):
    config_file, basename, backbone_type = _VARIANTS[variant]
    ckpt = _resolve_ckpt(basename)
    if ckpt is None:
        pytest.skip(f"EfficientSAM3 {variant} geo_ft video checkpoint absent ({basename})")

    import torch

    from sam.build_sam import build_sam3_video_predictor
    from sam.modeling.encoders.efficientsam3_trunk import EfficientSam3Trunk
    from sam.modeling.text.text_encoder import Sam3TextEncoder
    from sam.models.sam3_predictor import Sam3VideoPredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # A successful build IS the strict-load assertion: _load_sam3_full_checkpoint calls
    # model.load_state_dict(..., strict=True) and raises on any key mismatch (1698 keys).
    model = build_sam3_video_predictor(config_file=config_file, ckpt_path=ckpt, device=device)

    assert isinstance(model, Sam3VideoPredictor)
    trunk = model.vision_encoder.vision_backbone.trunk
    assert isinstance(trunk, EfficientSam3Trunk)
    assert trunk.channel_list == [1024]
    # Base (non-multiplex) lineage: PE text tower + 309-key base tracker (not 457 multiplex).
    assert isinstance(model.text_encoder, Sam3TextEncoder)
    assert len(list(model.tracker.state_dict())) == 309
