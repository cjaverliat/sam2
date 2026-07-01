# SPDX-License-Identifier: LicenseRef-SAM
"""Strict-load build tests for the single-variant EfficientSAM3 / SAM3-LiteText / SAM3.1
predictors (parametrized).

A successful build IS the strict-load assertion: each builder's checkpoint loader calls
``load_state_dict(..., strict=True)`` and raises on any key mismatch. Each case additionally
checks the predictor class, the text-encoder class, and (where applicable) the vision-trunk
type / ``channel_list`` and the tracker key count. Skipped per-case when the checkpoint is
absent (CI-safe).

Consolidates the former one-test-each modules (identical skeletons, per-variant builder /
config / checkpoint / assertions preserved verbatim as parametrize cases):
  * test_build_efficientsam3.py                 -> efficientsam3_repvit (image, CPU)
  * test_efficientsam3_litetext_build.py        -> sam3_litetext_s0_ctx16 (base video)
  * test_efficientsam3p1_litetext_build.py      -> sam3p1_litetext_s0_ctx16 (multiplex video)
  * test_efficientsam3p1_repvit_build.py        -> efficientsam3p1_repvit_m_s0_ctx16 (multiplex video)

The 3-backbone RGB video-build matrix lives separately in test_efficientsam3_video_build.py.
"""
import os

import pytest


def _first_existing(*paths: str) -> str | None:
    """First existing path among candidates (primary flat copy, then validated-review copy)."""
    return next((p for p in paths if os.path.exists(p)), None)


# --- per-case checkpoint resolution (evaluated at import, like the former skipif marks) ------
_CKPT_ESAM3 = _first_existing(
    "checkpoints/_esam3_validate/efficientsam3_ft/efficientsam3_repvit.pt",
)
_CKPT_LITETEXT = _first_existing(
    "checkpoints/sam3_litetext_mobileclip_s0_ctx16.pt",
    "checkpoints/_esam3_validate/sam3_litetext/sam3_litetext_mobileclip_s0_ctx16.pt",
)
_CKPT_P1_LITETEXT = _first_existing(
    "checkpoints/efficient_sam3p1_litetext_mobileclip_s0_ctx16.pt",
    "checkpoints/_esam3_validate/sam3p1_litetext/efficient_sam3p1_litetext_mobileclip_s0_ctx16.pt",
)
_CKPT_P1_REPVIT = _first_existing(
    "checkpoints/efficient_sam3p1_repvit_m_mobileclip_s0_ctx16.pt",
    "checkpoints/_esam3_validate/stage1_sam3p1/efficient_sam3p1_repvit_m_mobileclip_s0_ctx16.pt",
)


# --- per-case builders (device policy preserved: efficientsam3 is CPU; others cuda-if-avail) --
def _build_efficientsam3(ckpt: str):
    from sam.build_sam import build_efficientsam3

    return build_efficientsam3(ckpt_path=ckpt, device="cpu")


def _build_litetext(ckpt: str):
    import torch

    from sam.build_sam import build_sam3_video_predictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return build_sam3_video_predictor(
        config_file="configs/efficientsam3/sam3_litetext_s0_ctx16.yaml",
        ckpt_path=ckpt,
        device=device,
    )


def _build_p1_litetext(ckpt: str):
    import torch

    from sam.build_sam import build_sam3_multiplex_video_predictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return build_sam3_multiplex_video_predictor(
        config_file="configs/efficientsam3/sam3p1_litetext_s0_ctx16.yaml",
        ckpt_path=ckpt,
        device=device,
    )


def _build_p1_repvit(ckpt: str):
    import torch

    from sam.build_sam import build_efficientsam3p1_video_predictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return build_efficientsam3p1_video_predictor(
        config_file="configs/efficientsam3/efficientsam3p1_repvit_m_mobileclip_s0_ctx16.yaml",
        ckpt_path=ckpt,
        device=device,
        backbone_type="repvit",
        model_name="m1_1",
    )


# --- per-case assertions (verbatim from the former modules) ----------------------------------
def _check_efficientsam3(model) -> None:
    from sam.models.sam3_predictor import Sam3Predictor
    from sam.modeling.text.mobileclip_text_encoder import MobileClipTextEncoder

    assert isinstance(model, Sam3Predictor)
    # the swapped vision trunk + text encoder are the EfficientSAM3 ones
    assert model.vision_encoder.vision_backbone.trunk.channel_list == [1024]
    assert isinstance(model.text_encoder, MobileClipTextEncoder)


def _check_litetext(model) -> None:
    from sam.models.sam3_predictor import Sam3VideoPredictor
    from sam.modeling.text.mobileclip_text_encoder import MobileClipTextEncoder

    assert isinstance(model, Sam3VideoPredictor)
    assert isinstance(model.text_encoder, MobileClipTextEncoder)


def _check_p1_litetext(model) -> None:
    from sam.models.sam3_predictor import Sam3MultiplexVideoPredictor
    from sam.modeling.text.mobileclip_text_encoder import MobileClipTextEncoder

    assert isinstance(model, Sam3MultiplexVideoPredictor)
    assert isinstance(model.text_encoder, MobileClipTextEncoder)
    assert len([k for k in model.tracker.state_dict()]) == 457


def _check_p1_repvit(model) -> None:
    from sam.models.sam3_predictor import Sam3MultiplexVideoPredictor
    from sam.modeling.encoders.efficientsam3_trunk import EfficientSam3Trunk
    from sam.modeling.text.mobileclip_text_encoder import MobileClipTextEncoder

    assert isinstance(model, Sam3MultiplexVideoPredictor)
    assert isinstance(model.text_encoder, MobileClipTextEncoder)
    # Vision trunk must be the distilled EfficientSam3Trunk (not PE ViT).
    assert isinstance(model.vision_encoder.vision_backbone.trunk, EfficientSam3Trunk)
    assert model.vision_encoder.vision_backbone.trunk.channel_list == [1024]
    # Tracker carries exactly 457 keys (multiplex tracker, unchanged from SAM 3.1).
    assert len(list(model.tracker.state_dict())) == 457


_CASES = [
    pytest.param(
        _build_efficientsam3, _CKPT_ESAM3, _check_efficientsam3,
        id="efficientsam3_repvit",
        marks=pytest.mark.skipif(_CKPT_ESAM3 is None, reason="EfficientSAM3 RepViT ckpt absent"),
    ),
    pytest.param(
        _build_litetext, _CKPT_LITETEXT, _check_litetext,
        id="sam3_litetext_s0_ctx16",
        marks=pytest.mark.skipif(_CKPT_LITETEXT is None, reason="SAM3-LiteText s0/ctx16 ckpt absent"),
    ),
    pytest.param(
        _build_p1_litetext, _CKPT_P1_LITETEXT, _check_p1_litetext,
        id="sam3p1_litetext_s0_ctx16",
        marks=pytest.mark.skipif(_CKPT_P1_LITETEXT is None, reason="SAM3.1-LiteText s0/ctx16 ckpt absent"),
    ),
    pytest.param(
        _build_p1_repvit, _CKPT_P1_REPVIT, _check_p1_repvit,
        id="efficientsam3p1_repvit_m_s0_ctx16",
        marks=pytest.mark.skipif(_CKPT_P1_REPVIT is None, reason="EfficientSAM3.1 RepViT-M s0/ctx16 ckpt absent"),
    ),
]


@pytest.mark.parametrize("build,ckpt,check", _CASES)
def test_strict_load_build(build, ckpt, check):
    model = build(ckpt)
    check(model)
