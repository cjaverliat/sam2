# SPDX-License-Identifier: LicenseRef-SAM
"""``start_session``: the session wrapper over ``forward`` + a state.

A session binds a predictor to one video: it owns the state, counts frames, and
declares the concept at birth (SAM 3). It adds NO model logic -- every test here
asserts a session run is identical to the equivalent explicit ``forward`` loop.
"""
import os

import numpy as np
import pytest
import torch
from PIL import Image

from sam.prompts import ConceptPrompt, GeometryPrompt

CKPT = "checkpoints/sam3.pt"
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.isfile(CKPT),
    reason="needs CUDA + sam3.pt",
)

N_FRAMES = 4


@pytest.fixture(scope="module")
def predictor():
    from sam.build_sam import build_sam3_video_predictor

    return build_sam3_video_predictor(
        config_file="configs/sam3/sam3.yaml", ckpt_path=CKPT, device="cuda")


@pytest.fixture(scope="module")
def frames():
    return [
        np.asarray(Image.open(f"notebooks/videos/bedroom/{i:05d}.jpg").convert("RGB"))
        for i in range(N_FRAMES)
    ]


def _masks(masklets):
    return {o: (r.best_mask_logits[0, 0] > 0).cpu().numpy() for o, r in masklets.items()}


def test_interactive_session_matches_forward_loop(predictor, frames):
    from sam.models.sam3_predictor import Sam3VideoPredictorState

    h, w, _ = frames[0].shape
    prompt = GeometryPrompt.box(1, (285, 0, 535, 430))

    state = Sam3VideoPredictorState(video_hw=(h, w))
    expected = []
    for i, frame in enumerate(frames):
        out = predictor.forward(state, i, frame,
                                prompts=[prompt.clone()] if i == 0 else [])
        expected.append(_masks(out))

    interactive_session = predictor.start_session()
    for i, frame in enumerate(frames):
        masklets = interactive_session.process(
            frame, prompts=[prompt.clone()] if i == 0 else [])
        got = _masks(masklets)
        assert sorted(got) == sorted(expected[i]), f"frame {i}: ids differ"
        for oid in got:
            assert (got[oid] == expected[i][oid]).all(), f"frame {i}: obj {oid} differs"


def test_concept_session_declares_concept_at_birth(predictor, frames):
    person_session = predictor.start_session(concept="person")
    masklets = person_session.process(frames[0])
    assert len(masklets) == 2  # both children on frame 0 (measured elsewhere)
    assert person_session.state.concept.prompt.text == "person"


def test_placeholder_session_runs_concept_box(predictor, frames):
    hint_session = predictor.start_session(concept=predictor.PLACEHOLDER)
    masklets = hint_session.process(
        frames[0], prompts=[GeometryPrompt.concept_box((285, 0, 535, 430))])
    assert len(masklets) == 2
    assert hint_session.state.concept.prompt.text == predictor.BOX_ONLY_CAPTION


def test_concept_box_in_conceptless_session_raises(predictor, frames):
    interactive_session = predictor.start_session()
    with pytest.raises(ValueError, match="concept"):
        interactive_session.process(
            frames[0], prompts=[GeometryPrompt.concept_box((285, 0, 535, 430))])


def test_frame_idx_override(predictor, frames):
    interactive_session = predictor.start_session()
    interactive_session.process(frames[0], prompts=[GeometryPrompt.click(1, (385, 230))])
    interactive_session.process(frames[1])
    # re-run frame 1, then continue: the counter must resume from the override
    interactive_session.process(frames[1], frame_idx=1)
    masklets = interactive_session.process(frames[2])
    assert 1 in masklets
    assert interactive_session.state.num_frames_processed == 4  # 4 forward calls


def test_two_sessions_share_one_model(predictor, frames):
    person_session = predictor.start_session(concept="person")
    interactive_session = predictor.start_session()
    for i, frame in enumerate(frames[:2]):
        people = person_session.process(frame)
        clicked = interactive_session.process(
            frame, prompts=[GeometryPrompt.click(7, (385, 230))] if i == 0 else [])
    assert len(people) == 2
    assert sorted(clicked) == [7]
    assert person_session.state is not interactive_session.state


def test_sam2_start_session_exists():
    """SAM 2 gets the same wrapper (no concept parameter)."""
    from sam.models.sam2_predictor import Sam2VideoPredictor

    predictor = Sam2VideoPredictor.__new__(Sam2VideoPredictor)  # no weights needed
    session = predictor.start_session(video_hw=(540, 960))
    assert session.state.video_hw == (540, 960)
    with pytest.raises(TypeError):
        predictor.start_session(concept="person")
