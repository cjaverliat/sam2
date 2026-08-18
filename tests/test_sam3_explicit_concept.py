# SPDX-License-Identifier: LicenseRef-SAM
"""A box prompt must not switch detection on behind the caller's back.

Upstream adopts a placeholder caption whenever a box arrives with no text
(``sam3_video_inference.py:868-876``), which silently turns a "track what I boxed"
call into "detect everything this caption matches, forever". We keep the behaviour but
require the caller to ask for it, so the two intents read differently at the call site:

    pred.set_concept(state, ConceptPrompt("person"))   # detect this concept
    pred.set_placeholder_concept(state)                # detect with the box-only caption

A box with neither raises.
"""
import os

import numpy as np
import pytest
import torch
from PIL import Image

from sam.prompts import BoxRoute, ConceptPrompt, GeometryPrompt

CKPT = "checkpoints/sam3.pt"
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.isfile(CKPT),
    reason="needs CUDA + sam3.pt",
)

BOX_XYXY = [285.0, 0.0, 535.0, 430.0]


@pytest.fixture(scope="module")
def predictor():
    from sam.build_sam import build_sam3_video_predictor

    return build_sam3_video_predictor(
        config_file="configs/sam3/sam3.yaml", ckpt_path=CKPT, device="cuda")


@pytest.fixture(scope="module")
def frame0():
    return np.asarray(Image.open("notebooks/videos/bedroom/00000.jpg").convert("RGB"))


def _box(route=BoxRoute.DETECTOR):
    return GeometryPrompt(obj_id=1, boxes=torch.tensor([BOX_XYXY], device="cuda"),
                          box_route=route)


def test_box_without_concept_raises(predictor, frame0):
    from sam.models.sam3_predictor import Sam3VideoPredictorState

    h, w, _ = frame0.shape
    state = Sam3VideoPredictorState(video_hw=(h, w))
    with pytest.raises(ValueError, match="set_placeholder_concept"):
        predictor.forward(state, 0, frame0, prompts=[_box()])


def test_placeholder_concept_is_opt_in(predictor, frame0):
    from sam.models.sam3_predictor import Sam3VideoPredictorState

    h, w, _ = frame0.shape
    state = Sam3VideoPredictorState(video_hw=(h, w))
    predictor.set_placeholder_concept(state)
    assert state.concept.prompt.text == predictor.BOX_ONLY_CAPTION

    out = predictor.forward(state, 0, frame0, prompts=[_box()])
    assert len(out) == 2, (
        f"expected the caption to detect both children, got {sorted(out)}"
    )


def test_explicit_concept_still_takes_a_box(predictor, frame0):
    from sam.models.sam3_predictor import Sam3VideoPredictorState

    h, w, _ = frame0.shape
    state = Sam3VideoPredictorState(video_hw=(h, w))
    predictor.set_concept(state, ConceptPrompt("person"))
    out = predictor.forward(state, 0, frame0, prompts=[_box()])
    assert len(out) == 2
    assert state.concept.prompt.text == "person", "the box must not overwrite the concept"


def test_default_box_is_interactive(predictor, frame0):
    """A plain `boxes=` prompt is SAM 2 semantics: corner points, one object, no detection."""
    from sam.models.sam3_predictor import Sam3VideoPredictorState

    h, w, _ = frame0.shape
    state = Sam3VideoPredictorState(video_hw=(h, w))
    out = predictor.forward(state, 0, frame0, prompts=[_box(BoxRoute.TRACKER)])
    assert sorted(out) == [1], f"expected only the boxed object, got {sorted(out)}"
    assert state.concept is None, "a TRACKER-route box must not adopt a caption"


def test_interactive_prompts_need_no_concept(predictor, frame0):
    """Points and masks take the tracker route, so they must stay concept-free."""
    from sam.models.sam3_predictor import Sam3VideoPredictorState

    h, w, _ = frame0.shape
    state = Sam3VideoPredictorState(video_hw=(h, w))
    click = GeometryPrompt(
        obj_id=1,
        points_coords=torch.tensor([[385.0, 230.0]], device="cuda"),
        points_labels=torch.tensor([1], device="cuda"),
    )
    out = predictor.forward(state, 0, frame0, prompts=[click])
    assert sorted(out) == [1]
    assert state.concept is None, "an interactive prompt must not adopt a caption"
