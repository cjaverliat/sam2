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


def _bedroom(n):
    return [
        np.asarray(Image.open(f"notebooks/videos/bedroom/{i:05d}.jpg").convert("RGB"))
        for i in range(n)
    ]


def _click(obj_id, xy):
    return GeometryPrompt(
        obj_id=obj_id, points_coords=torch.tensor([list(xy)]),
        points_labels=torch.tensor([1]),
    )


@needs_gpu
def test_box_prompt_raises_pointing_to_exemplars():
    pred = _build()
    st = _state((540, 960))
    box = GeometryPrompt(obj_id=1, boxes=torch.tensor([[10.0, 10.0, 50.0, 50.0]]))
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    with pytest.raises(NotImplementedError, match="exemplar"):
        pred.forward(st, 0, frame, geometry_prompts=[box])


@needs_gpu
def test_seed_frame_click_spawns_and_tracks():
    pred = _build()
    frames = _bedroom(3)
    h, w, _ = frames[0].shape
    st = _state((h, w))
    out0 = pred.forward(st, 0, frames[0], geometry_prompts=[_click(1, (385.0, 230.0))])
    assert set(out0) == {1}
    assert out0[1].masks_logits.shape[-2:] == (h, w)
    assert (out0[1].masks_logits > 0).sum() > 0
    out1 = pred.forward(st, 1, frames[1])           # plain propagation
    assert set(out1) == {1}


@needs_gpu
def test_midstream_click_add_second_object():
    pred = _build()
    frames = _bedroom(4)
    h, w, _ = frames[0].shape
    st = _state((h, w))
    pred.forward(st, 0, frames[0], geometry_prompts=[_click(1, (385.0, 230.0))])
    pred.forward(st, 1, frames[1])
    # add a SECOND object mid-stream via a click
    out2 = pred.forward(st, 2, frames[2], geometry_prompts=[_click(2, (600.0, 300.0))])
    assert set(out2) == {1, 2}                       # both now tracked
    out3 = pred.forward(st, 3, frames[3])
    assert set(out3) == {1, 2}                       # both propagate forward


@needs_gpu
def test_coseed_concept_plus_click():
    pred = _build()
    frames = _bedroom(2)
    h, w, _ = frames[0].shape
    st = _state((h, w))
    pred.set_concept(st, ConceptPrompt("person"))
    # concept detects the people; the click adds one more object on the SAME frame
    out0 = pred.forward(st, 0, frames[0], geometry_prompts=[_click(99, (600.0, 300.0))])
    assert 99 in out0 and len(out0) >= 2             # detector persons + the clicked obj
