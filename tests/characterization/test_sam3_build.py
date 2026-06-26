# SPDX-License-Identifier: LicenseRef-SAM
"""Guard unit tests for Sam3VideoPredictor concept state (CPU-only, no checkpoint).

Verifies:
  - set_concept returns concept id 0 on the first call.
  - A second set_concept raises RuntimeError (MAX_CONCEPTS=1 guard).
  - set_concept raises RuntimeError when called after the first frame is processed
    (state.started guard).

The "returns id 0" test reaches encode_text; a lightweight subclass stubs it out so
no checkpoint or GPU is needed anywhere in this file.
"""
from __future__ import annotations

import pytest
import torch

from sam.prompts import ConceptPrompt
from sam.models.sam3_predictor import (
    Sam3VideoPredictor,
    Sam3VideoPredictorState,
    MAX_CONCEPTS,
)


# ---------------------------------------------------------------------------
# CPU-only stub: overrides the two encode_* methods so no model is loaded.
# ---------------------------------------------------------------------------

class _FakeSam3VideoPredictor(Sam3VideoPredictor):
    """Minimal test double — encode_text / encode_exemplars return dummy tensors."""

    def encode_text(self, text: str) -> torch.Tensor:
        return torch.zeros(1, 256)

    def encode_exemplars(self, exemplars) -> torch.Tensor:
        return torch.zeros(1, 256)


def _predictor() -> _FakeSam3VideoPredictor:
    return _FakeSam3VideoPredictor()


def _state() -> Sam3VideoPredictorState:
    return Sam3VideoPredictorState(video_hw=(480, 640))


# ---------------------------------------------------------------------------
# Guard tests
# ---------------------------------------------------------------------------

def test_set_concept_returns_id_zero():
    """First set_concept must return concept id 0."""
    pred = _predictor()
    state = _state()
    cid = pred.set_concept(state, ConceptPrompt(text="cat"))
    assert cid == 0


def test_set_concept_raises_on_second_concept():
    """A second set_concept must raise when MAX_CONCEPTS=1."""
    pred = _predictor()
    state = _state()
    pred.set_concept(state, ConceptPrompt(text="cat"))
    with pytest.raises(RuntimeError, match="at most"):
        pred.set_concept(state, ConceptPrompt(text="dog"))


def test_set_concept_raises_after_started():
    """set_concept must raise when called after any frame has been processed."""
    pred = _predictor()
    state = _state()
    state.num_frames_processed = 1  # simulate a processed frame
    with pytest.raises(RuntimeError, match="before the first frame"):
        pred.set_concept(state, ConceptPrompt(text="cat"))


# ---------------------------------------------------------------------------
# Association + tracklet lifecycle tests (Task 7) — synthetic masks, CPU-only.
# No GPU, no checkpoint.  All assertions are real (no "pass" stubs).
# ---------------------------------------------------------------------------

from sam.modeling.association import associate_det_trk
from sam.modeling.association.tracklet import TrackletManager


def test_associate_identical_masks_are_matched():
    """Identical det & track masks must be matched (high IoU = 1.0).

    Expected: det index 0 is NOT in new_dets; track index 0 is NOT in
    unmatched_tracks; det2track maps det 0 → [track 0].
    """
    H, W = 32, 32
    mask = torch.zeros(1, H, W)
    mask[0, 10:20, 10:20] = 1.0  # non-empty patch

    new_dets, unmatched_trks, det2track, matched_scores = associate_det_trk(
        det_masks=mask,
        track_masks=mask.clone(),
        iou_threshold=0.5,
        iou_threshold_trk=0.5,
        det_scores=torch.tensor([0.9]),
        new_det_thresh=0.3,
    )
    assert 0 not in new_dets, "identical mask pair must be matched, not a new detection"
    assert len(unmatched_trks) == 0, "matched track must not appear in unmatched_tracks"
    assert 0 in det2track, "det2track must contain det index 0"
    assert 0 in det2track[0], "det 0 must map to track 0 in det2track"


def test_associate_novel_high_score_det_is_new():
    """A high-score det with no track overlap must appear in new_dets.

    Two non-overlapping patches: det at top-left, track at bottom-right.
    IoU = 0. Score 0.9 >= new_det_thresh 0.3 → should be a new detection.
    """
    H, W = 32, 32
    det_mask = torch.zeros(1, H, W)
    det_mask[0, 0:5, 0:5] = 1.0   # top-left corner
    trk_mask = torch.zeros(1, H, W)
    trk_mask[0, 20:30, 20:30] = 1.0  # bottom-right, non-overlapping

    new_dets, unmatched_trks, det2track, matched_scores = associate_det_trk(
        det_masks=det_mask,
        track_masks=trk_mask,
        iou_threshold=0.5,
        iou_threshold_trk=0.5,
        det_scores=torch.tensor([0.9]),
        new_det_thresh=0.3,
    )
    assert 0 in new_dets, "non-overlapping high-score det must appear in new_dets"


def test_tracklet_manager_kills_unmatched_track():
    """A track unmatched for kill_thresh=3 consecutive frames must become DEAD.

    Step 1, 2: not yet dead (unmatched count < kill_thresh).
    Step 3: unmatched_count == kill_thresh → DEAD.
    """
    mgr = TrackletManager(confirmation_thresh=3, kill_thresh=3)
    mgr.spawn(obj_id=0)

    mgr.step(matched_track_ids=set(), new_det_ids=set())
    assert not mgr.is_dead(0), "track must not be dead after 1 unmatched frame"

    mgr.step(matched_track_ids=set(), new_det_ids=set())
    assert not mgr.is_dead(0), "track must not be dead after 2 unmatched frames"

    mgr.step(matched_track_ids=set(), new_det_ids=set())
    assert mgr.is_dead(0), "track must be dead after 3 consecutive unmatched frames"
