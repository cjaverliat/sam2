# SPDX-License-Identifier: Apache-2.0
"""Per-family build smoke test.

Each model-config family must actually instantiate through the builders (no
checkpoint, CPU) — this catches a broken ``_target_``, a wrong dim, or a moved
module that mere config composition (``test_config_compose``) would not surface.
No weights are loaded (``ckpt_path=None``), so this is fast and needs no GPU.
"""
import pytest

from sam.build_sam import build_sam2_predictor, build_sam2_video_predictor
from sam.models.sam2_predictor import Sam2Predictor, Sam2VideoPredictor

CONFIGS = [
    "configs/efficienttam/efficienttam_ti.yaml",
    "configs/sam2/sam2_hiera_t.yaml",
    "configs/sam2.1/sam2.1_hiera_t.yaml",
]


@pytest.mark.parametrize("cfg", CONFIGS)
def test_build_predictor_instantiates(cfg):
    model = build_sam2_predictor(cfg, ckpt_path=None, device="cpu", mode="eval", use_half=False)
    assert isinstance(model, Sam2Predictor)


@pytest.mark.parametrize("cfg", CONFIGS)
def test_build_video_predictor_instantiates(cfg):
    model = build_sam2_video_predictor(cfg, ckpt_path=None, device="cpu", mode="eval", use_half=False)
    assert isinstance(model, Sam2VideoPredictor)
