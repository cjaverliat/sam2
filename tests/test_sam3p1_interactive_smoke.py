# SPDX-License-Identifier: LicenseRef-SAM
import os

import numpy as np
import pytest
import torch
from PIL import Image

from sam.prompts import ConceptPrompt, GeometryPrompt

CKPT = "checkpoints/sam3.1_multiplex.pt"
CFG = "configs/sam3/sam3.1.yaml"
needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.isfile(CKPT),
    reason="mux interactive needs CUDA + sam3.1_multiplex.pt",
)


def _build():
    from sam.build_sam import build_sam3_multiplex_video_predictor
    return build_sam3_multiplex_video_predictor(config_file=CFG, ckpt_path=CKPT, device="cuda")


def _state(hw):
    from sam.models.sam3_predictor import Sam3VideoPredictorState
    return Sam3VideoPredictorState(video_hw=hw)


@needs_gpu
def test_box_prompt_raises_pointing_to_exemplars():
    pred = _build()
    st = _state((540, 960))
    box = GeometryPrompt(obj_id=1, boxes=torch.tensor([[10.0, 10.0, 50.0, 50.0]]))
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    with pytest.raises(NotImplementedError, match="exemplar"):
        pred.forward(st, 0, frame, geometry_prompts=[box])


@needs_gpu
def test_click_prompt_with_concept_raises_coseed():
    pred = _build()
    st = _state((540, 960))
    pred.set_concept(st, ConceptPrompt("person"))
    click = GeometryPrompt(
        obj_id=1, points_coords=torch.tensor([[385.0, 230.0]]),
        points_labels=torch.tensor([1]),
    )
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    with pytest.raises(NotImplementedError, match="co-seed|concept"):
        pred.forward(st, 0, frame, geometry_prompts=[click])


@needs_gpu
def test_seed_frame_click_spawns_and_tracks():
    pred = _build()
    frames = [
        np.asarray(Image.open(f"notebooks/videos/bedroom/{i:05d}.jpg").convert("RGB"))
        for i in range(3)
    ]
    h, w, _ = frames[0].shape
    st = _state((h, w))
    click = GeometryPrompt(
        obj_id=1, points_coords=torch.tensor([[385.0, 230.0]]),
        points_labels=torch.tensor([1]),
    )
    out0 = pred.forward(st, 0, frames[0], geometry_prompts=[click])
    assert set(out0) == {1}
    assert out0[1].masks_logits.shape[-2:] == (h, w)
    assert (out0[1].masks_logits > 0).sum() > 0  # non-empty seed mask
    # mid-stream add now blocked (1b territory)
    with pytest.raises(NotImplementedError, match="mid-stream"):
        pred.forward(st, 1, frames[1], geometry_prompts=[
            GeometryPrompt(obj_id=2, points_coords=torch.tensor([[500.0, 300.0]]),
                           points_labels=torch.tensor([1]))])
    out1 = pred.forward(st, 1, frames[1])           # plain propagation still works
    assert set(out1) == {1}
