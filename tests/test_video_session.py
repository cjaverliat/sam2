# SPDX-License-Identifier: LicenseRef-SAM
"""Sessions: the wrapper over ``forward`` + a state.

A session binds a predictor to one video: it owns the state and counts frames. It
adds NO model logic -- every test here asserts a session run is identical to the
equivalent explicit ``forward`` loop.

Two entry points, deliberately distinct: ``start_session()`` is interactive (no
detection, caller-chosen obj_ids) and exists on SAM 2 and SAM 3 alike, while
``start_concept_session(concept)`` is SAM 3 only and declares the concept at birth.
"""
import os

import numpy as np
import pytest
import torch
from PIL import Image

from sam.prompts import ConceptPrompt, GeometryPrompt

CKPT = "checkpoints/sam3.pt"
needs_model = pytest.mark.skipif(
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


@needs_model
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


@needs_model
def test_concept_session_declares_concept_at_birth(predictor, frames):
    person_session = predictor.start_concept_session("person")
    masklets = person_session.process(frames[0])
    assert len(masklets) == 2  # both children on frame 0 (measured elsewhere)
    assert person_session.state.concept.prompt.text == "person"


@needs_model
def test_placeholder_session_runs_exemplar_box(predictor, frames):
    hint_session = predictor.start_concept_session(predictor.PLACEHOLDER)
    masklets = hint_session.process(
        frames[0], prompts=[GeometryPrompt.exemplar_box((285, 0, 535, 430))])
    assert len(masklets) == 2
    assert hint_session.state.concept.prompt.text == predictor.BOX_ONLY_CAPTION


@needs_model
def test_exemplar_box_in_conceptless_session_raises(predictor, frames):
    interactive_session = predictor.start_session()
    with pytest.raises(ValueError, match="concept"):
        interactive_session.process(
            frames[0], prompts=[GeometryPrompt.exemplar_box((285, 0, 535, 430))])


@needs_model
def test_frame_idx_override(predictor, frames):
    interactive_session = predictor.start_session()
    interactive_session.process(frames[0], prompts=[GeometryPrompt.click(1, (385, 230))])
    interactive_session.process(frames[1])
    # re-run frame 1, then continue: the counter must resume from the override
    interactive_session.process(frames[1], frame_idx=1)
    masklets = interactive_session.process(frames[2])
    assert 1 in masklets
    assert interactive_session.state.num_frames_processed == 4  # 4 forward calls


@needs_model
def test_two_sessions_share_one_model(predictor, frames):
    person_session = predictor.start_concept_session("person")
    interactive_session = predictor.start_session()
    for i, frame in enumerate(frames[:2]):
        people = person_session.process(frame)
        clicked = interactive_session.process(
            frame, prompts=[GeometryPrompt.click(7, (385, 230))] if i == 0 else [])
    assert len(people) == 2
    assert sorted(clicked) == [7]
    assert person_session.state is not interactive_session.state


def test_sam2_start_session_exists():
    """SAM 2 gets the same interactive wrapper, same (empty) signature."""
    import inspect

    from sam.models.sam2_predictor import Sam2VideoPredictor
    from sam.models.sam3_predictor import Sam3VideoPredictor

    predictor = Sam2VideoPredictor.__new__(Sam2VideoPredictor)  # no weights needed
    session = predictor.start_session()
    assert session.state is None  # the size arrives with the first frame

    sig = inspect.signature(Sam2VideoPredictor.start_session)
    assert sig == inspect.signature(Sam3VideoPredictor.start_session)
    assert list(sig.parameters) == ["self"]  # no video_hw: it is never passed in


def test_sam2_has_no_concept_session():
    """A concept is a SAM 3 capability: SAM 2 lacks the method entirely.

    The two entry points are split precisely so this misuse names the missing
    capability instead of surfacing as an unexpected keyword (or, worse, silently
    binding a phrase to ``video_hw``).
    """
    from sam.models.sam2_predictor import Sam2VideoPredictor

    assert not hasattr(Sam2VideoPredictor, "start_concept_session")


@pytest.mark.parametrize("bad", ["person", (540,), (540, 960, 3), (0, 960), (540.0, 960)])
def test_state_rejects_bad_video_hw(bad):
    """The explicit state form validates ``video_hw`` -- sessions never take one."""
    from sam.models.sam2_predictor import Sam2VideoPredictorState
    from sam.models.sam3_predictor import Sam3VideoPredictorState

    for cls in (Sam2VideoPredictorState, Sam3VideoPredictorState):
        with pytest.raises(ValueError, match="video_hw"):
            cls(video_hw=bad)


def test_frame_size_must_match_the_session():
    """SAM 3 never re-reads (H, W) from the frame, so a mismatch must raise.

    ``forward`` takes (H, W) from the state to normalize prompt coordinates and to
    resize every output mask, while the preprocessor squashes the frame to 1008x1008
    regardless -- so without this check a wrong-sized frame is silently mis-tracked.
    """
    from sam.models.sam3_predictor import Sam3VideoPredictor, Sam3VideoPredictorState

    predictor = Sam3VideoPredictor.__new__(Sam3VideoPredictor)
    state = Sam3VideoPredictorState(video_hw=(540, 960))

    predictor._check_frame_hw(state, np.zeros((540, 960, 3), dtype=np.uint8))  # ok
    with pytest.raises(ValueError, match="this session runs at"):
        predictor._check_frame_hw(state, np.zeros((270, 480, 3), dtype=np.uint8))


def test_concept_is_never_implicit():
    """Neither entry point guesses: ``concept`` is required, and never a keyword.

    ``concept=None`` is a request for an interactive session, so it points at
    ``start_session``; ``start_session(concept=...)`` no longer exists on SAM 3
    either, which keeps the interactive signature identical to SAM 2's.
    """
    from sam.models.sam3_predictor import Sam3VideoPredictor

    predictor = Sam3VideoPredictor.__new__(Sam3VideoPredictor)
    with pytest.raises(ValueError, match="start_session"):
        predictor.start_concept_session(None)
    with pytest.raises(TypeError):
        predictor.start_session(concept="person")
