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

from sam.prompts import ConceptPrompt, GeometryPrompt
from sam.models.sam3_predictor import (
    Sam3VideoPredictor,
    Sam3VideoPredictorState,
    MAX_CONCEPTS,
)


# ---------------------------------------------------------------------------
# CPU-only stub: overrides the two encode_* methods so no model is loaded.
# ---------------------------------------------------------------------------

class _FakeSam3VideoPredictor(Sam3VideoPredictor):
    """Minimal test double — encode_text / encode_exemplars return dummy tensors.

    ``encode_text`` takes the full ``ConceptPrompt`` (the Task 8 seam fix) rather than a
    bare ``text: str``, matching ``Sam3VideoPredictor.encode_text``'s updated signature.
    """

    def encode_text(self, concept) -> torch.Tensor:
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


def test_tracklet_manager_removes_within_hotstart_only():
    """A within-hotstart object unmatched for hotstart_unmatch_thresh is removed; an
    ESTABLISHED (past-hotstart) object is never removed, only suppressed."""
    mgr = TrackletManager(hotstart_delay=15, hotstart_unmatch_thresh=8)
    mgr.spawn(obj_id=0, frame_idx=0)
    for f in range(1, 9):                      # unmatched within the hotstart window
        mgr.step(set(), set(), frame_idx=f)
    assert 0 in mgr.removed_ids(), "within-hotstart object must be removed after 8 unmatched"

    mgr.spawn(obj_id=1, frame_idx=0)
    for f in range(1, 30):                     # establish it well past hotstart
        mgr.step({1}, set(), frame_idx=f)
    for f in range(30, 50):                    # long absence, but established
        mgr.step(set(), set(), frame_idx=f)
    assert 1 not in mgr.removed_ids(), "established object must never be removed"
    assert 1 in mgr.alive_ids() and 1 not in mgr.visible_ids()  # dormant, hidden


# ---------------------------------------------------------------------------
# Lifecycle tests (Task 9) — remove_object + allocator + kill path.
# CPU-only, no checkpoint, no GPU.
# ---------------------------------------------------------------------------

def test_remove_object_purges_everything():
    """remove_object must purge obj_id from known_obj_ids, both memory dicts, and tracklet_mgr.

    Constructs a Sam3VideoPredictorState, registers an object in BOTH the bank (known_obj_ids,
    conditional_memories, non_conditional_memories) AND the TrackletManager, then asserts that
    remove_object wipes all four stores atomically.
    """
    pred = _predictor()
    state = _state()
    obj_id = 42

    # Register the object in the bank's known set and both per-object memory dicts.
    state.bank.known_obj_ids.add(obj_id)
    state.bank.conditional_memories[obj_id] = ["dummy_cond_memory"]
    state.bank.non_conditional_memories[obj_id] = ["dummy_non_cond_memory"]
    # Register in tracklet manager.
    state.tracklet_mgr.spawn(obj_id, 0)

    # Sanity: object is present in all four stores before removal.
    assert obj_id in state.bank.known_obj_ids
    assert obj_id in state.bank.conditional_memories
    assert obj_id in state.bank.non_conditional_memories
    assert obj_id in state.tracklet_mgr._tracks

    pred.remove_object(state, obj_id)

    assert obj_id not in state.bank.known_obj_ids, "known_obj_ids must not contain removed id"
    assert obj_id not in state.bank.conditional_memories, "conditional_memories must drop removed id"
    assert obj_id not in state.bank.non_conditional_memories, "non_conditional_memories must drop removed id"
    assert obj_id not in state.tracklet_mgr._tracks, "tracklet_mgr must drop removed id"


def test_alloc_obj_id_monotonic_after_remove():
    """After removing an obj_id, allocating a new one must yield a FRESH id (monotonic, no reuse).

    Validates the 'remove-then-re-add yields a NEW id, never colliding with the removed
    object's stale memories' guarantee of _alloc_obj_id.
    """
    pred = _predictor()
    state = _state()

    # Allocate first id (must be 0).
    first_id = pred._alloc_obj_id(state)
    assert first_id == 0, "first allocation must be 0"

    # Register and then remove.
    state.bank.known_obj_ids.add(first_id)
    state.tracklet_mgr.spawn(first_id, 0)
    pred.remove_object(state, first_id)

    # Allocate again: must NOT reuse the removed id (no collision with stale memories).
    second_id = pred._alloc_obj_id(state)
    assert second_id != first_id, "re-allocation must not reuse the removed id (no collision)"
    assert second_id == 1, "allocator must be monotonic: second allocation must be 1"


def test_kill_path_removes_dead_tracklet():
    """The removed_ids -> remove_object kill path purges a within-hotstart failure.

    Drives a within-hotstart tracklet to removed (hotstart_unmatch_thresh unmatched
    frames), then exercises the same purge loop used in _associate_and_update.
    """
    pred = _predictor()
    state = _state()
    obj_id = 7

    # Register in bank and tracklet_mgr.
    state.bank.known_obj_ids.add(obj_id)
    state.bank.conditional_memories[obj_id] = ["dummy_cond"]
    state.bank.non_conditional_memories[obj_id] = ["dummy_non_cond"]
    mgr = state.tracklet_mgr
    mgr.spawn(obj_id, 0)

    # Drive to removed: hotstart_unmatch_thresh unmatched frames within the window.
    for f in range(1, mgr.hotstart_unmatch_thresh + 1):
        mgr.step(set(), set(), frame_idx=f)

    assert obj_id in mgr.removed_ids(), "within-hotstart tracklet must be removed"

    # Exercise the purge loop from _associate_and_update.
    for oid in mgr.removed_ids():
        pred.remove_object(state, oid)

    # All four stores must be purged after the kill path runs.
    assert obj_id not in state.bank.known_obj_ids, "dead tracklet must be purged from known_obj_ids"
    assert obj_id not in state.bank.conditional_memories, "dead tracklet must be purged from conditional_memories"
    assert obj_id not in state.bank.non_conditional_memories, "dead tracklet must be purged from non_conditional_memories"
    assert obj_id not in mgr._tracks, "dead tracklet must be removed from tracklet_mgr after kill path"


# ---------------------------------------------------------------------------
# Build smoke tests (Task 10): CPU, no checkpoint, instantiate from config.
# Mirrors tests/characterization/test_build_instantiate.py — catches a broken
# _target_, a wrong dim, or a moved module that test_config_compose.py would
# not surface.  No weights are loaded (ckpt_path=None), no GPU needed.
# ---------------------------------------------------------------------------

from sam.build_sam import (  # noqa: E402
    build_sam3,
    build_sam3_video_predictor,
    build_sam3_multiplex,
    build_sam3_multiplex_video_predictor,
)
from sam.models.sam3_predictor import (  # noqa: E402
    Sam3Predictor,
    Sam3MultiplexPredictor,
    Sam3MultiplexVideoPredictor,
)


@pytest.mark.parametrize(
    "builder,config,cls",
    [
        pytest.param(build_sam3, "configs/sam3/sam3.yaml", Sam3Predictor, id="sam3"),
        pytest.param(
            build_sam3_video_predictor, "configs/sam3/sam3.yaml", Sam3VideoPredictor,
            id="sam3_video",
        ),
        pytest.param(
            build_sam3_multiplex, "configs/sam3/sam3.1.yaml", Sam3MultiplexPredictor,
            id="sam3_multiplex",
        ),
        pytest.param(
            build_sam3_multiplex_video_predictor, "configs/sam3/sam3.1.yaml",
            Sam3MultiplexVideoPredictor, id="sam3_multiplex_video",
        ),
    ],
)
def test_build_predictor_instantiates(builder, config, cls):
    """Each SAM 3 / SAM 3.1 builder (ckpt_path=None, CPU) instantiates its full
    hydra-compose -> instantiate (or direct-construction) path without loading any weights,
    returning its predictor class. Catches a broken _target_, a wrong dim, or a moved module
    that test_config_compose.py would not surface. Mirrors test_build_instantiate.py.
    """
    model = builder(config, ckpt_path=None, device="cpu", mode="eval")
    assert isinstance(model, cls)


# ---------------------------------------------------------------------------
# Priority-1 guard: multiplex video predictor must reject geometry prompts.
# CPU-only, no checkpoint.  The check fires at the very top of forward(),
# before self.device, so a no-parameter stub is sufficient.
# ---------------------------------------------------------------------------

def test_sam3p1_video_rejects_mask_prompts():
    """Sam3MultiplexVideoPredictor.forward rejects MASK geometry prompts loudly.

    Point and box prompts are supported; mask geometry has no ``mask_encoder``
    weights in either checkpoint, so it raises. The check (``_check_mux_geometry``)
    precedes ``self.device`` / any tensor ops, so no checkpoint or GPU is required.
    """
    from sam.models.sam3_predictor import Sam3MultiplexVideoPredictor, Sam3VideoPredictorState

    # Instantiate with no weights (the check precedes self.device / any tensor ops).
    pred = Sam3MultiplexVideoPredictor()
    state = Sam3VideoPredictorState(video_hw=(288, 512))
    gp = GeometryPrompt(obj_id=0, masks_logits=torch.zeros(288, 512))

    with pytest.raises(NotImplementedError, match="mask"):
        pred.forward(state, frame_idx=0, frame=None, geometry_prompts=[gp])
