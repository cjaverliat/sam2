# SPDX-License-Identifier: LicenseRef-SAM
"""Predictor-side lifecycle of the live ``MultiplexState``.

Covers the two ways the multiplex state and the threaded bucket-space memory can
fall out of step: total tracklet loss (every seeded detection dies in hotstart)
and a grow at a frame whose joint decode was never stored.

CPU-only: both paths are bookkeeping around the tracker, so they run on an
uninitialised predictor.
"""
import pytest
import torch

from sam.modeling.multiplex import MultiplexController
from sam.models.sam3_predictor import (
    Sam3MultiplexVideoPredictor,
    Sam3VideoPredictorState,
)


def _predictor():
    return object.__new__(Sam3MultiplexVideoPredictor)


def _seeded_state(num_objects):
    """A state as ``_seed_mux_state`` leaves it: a live grid + a stored cond frame."""
    state = Sam3VideoPredictorState(video_hw=(8, 8))
    ctrl = MultiplexController(multiplex_count=16)
    state.mux_state = ctrl.get_state(
        num_objects, torch.device("cpu"), torch.float32, random=False
    )
    state.mux_obj_ids = list(range(num_objects))
    state.mux_output_dict["cond_frame_outputs"][0] = {
        "obj_ptr": torch.zeros(1, 16, 4), "obj_ids": list(state.mux_obj_ids),
    }
    for obj_id in state.mux_obj_ids:
        state.bank.known_obj_ids.add(obj_id)
    return state


def test_losing_the_last_object_drops_the_state_instead_of_emptying_it():
    state = _seeded_state(2)
    pred = _predictor()
    pred._shrink_mux_state(state, 0)
    assert state.mux_state is not None       # one object left: the grid stays
    pred._shrink_mux_state(state, 1)
    assert state.mux_state is None
    assert state.mux_obj_ids == []


def test_losing_the_last_object_drops_the_bucket_space_memory_too():
    state = _seeded_state(1)
    _predictor()._shrink_mux_state(state, 0)
    assert state.mux_output_dict == {
        "cond_frame_outputs": {}, "non_cond_frame_outputs": {}
    }


def test_growing_at_an_unstored_frame_says_so():
    state = _seeded_state(1)
    with pytest.raises(RuntimeError, match="no multiplex output stored for frame 3"):
        _predictor()._grow_mux_state(
            state, 3, torch.zeros(1, 1, 4, 4), True, [7], None, None
        )


def _stored_frame(obj_ids, num_buckets, capacity=16, channels=4):
    """A frame output as the tracker leaves it: data-space rows + bucket-space memory.

    Row / bucket ``i`` is filled with the value ``i`` so a re-slice is checkable.
    """
    n = len(obj_ids)
    rows = torch.arange(n, dtype=torch.float32)
    buckets = torch.arange(num_buckets, dtype=torch.float32)
    return {
        "pred_masks": rows.view(n, 1, 1, 1).expand(n, 1, 4, 4).clone(),
        "pred_masks_high_res": rows.view(n, 1, 1, 1).expand(n, 1, 8, 8).clone(),
        "object_score_logits": rows.view(n, 1).clone(),
        "obj_ptr": buckets.view(num_buckets, 1, 1).expand(
            num_buckets, capacity, channels).clone(),
        "maskmem_features": buckets.view(num_buckets, 1, 1, 1).expand(
            num_buckets, channels, 2, 2).clone(),
        "maskmem_pos_enc": [buckets.view(num_buckets, 1, 1, 1).expand(
            num_buckets, channels, 2, 2).clone()],
        "conditioning_objects": set(range(n)),
        "obj_ids": list(obj_ids),
    }


def _state_with_stored_frame(num_objects, num_buckets=1):
    state = _seeded_state(num_objects)
    state.mux_output_dict["cond_frame_outputs"].clear()
    state.mux_output_dict["non_cond_frame_outputs"][4] = _stored_frame(
        state.mux_obj_ids, num_buckets
    )
    return state


def test_removal_drops_the_object_row_from_a_stored_frame():
    state = _state_with_stored_frame(3)
    _predictor()._shrink_mux_state(state, 1)
    out = state.mux_output_dict["non_cond_frame_outputs"][4]
    assert out["obj_ids"] == [0, 2]
    assert out["pred_masks"].shape[0] == 2
    # the surviving rows keep their own values, compacted in place
    assert out["pred_masks"][:, 0, 0, 0].tolist() == [0.0, 2.0]
    assert out["object_score_logits"].flatten().tolist() == [0.0, 2.0]


def test_stored_rows_stay_in_step_with_the_live_state():
    state = _state_with_stored_frame(3)
    _predictor()._shrink_mux_state(state, 1)
    out = state.mux_output_dict["non_cond_frame_outputs"][4]
    assert out["pred_masks"].shape[0] == state.mux_state.total_valid_entries


def test_removal_reindexes_the_conditioning_objects():
    state = _state_with_stored_frame(3)
    state.mux_output_dict["non_cond_frame_outputs"][4]["conditioning_objects"] = {1, 2}
    _predictor()._shrink_mux_state(state, 1)
    out = state.mux_output_dict["non_cond_frame_outputs"][4]
    assert out["conditioning_objects"] == {1}   # object 2 moved from row 2 to row 1


def test_a_frame_predating_the_object_is_left_alone():
    state = _state_with_stored_frame(3)
    early = _stored_frame([0], num_buckets=1)   # stored before objects 1 and 2 existed
    state.mux_output_dict["non_cond_frame_outputs"][2] = early
    _predictor()._shrink_mux_state(state, 1)
    assert state.mux_output_dict["non_cond_frame_outputs"][2]["obj_ids"] == [0]
    assert state.mux_output_dict["non_cond_frame_outputs"][2]["pred_masks"].shape[0] == 1


def test_losing_a_whole_bucket_slices_the_bucket_space_memory():
    state = _state_with_stored_frame(17, num_buckets=2)
    _predictor()._shrink_mux_state(state, 16)   # the sole occupant of bucket 1
    out = state.mux_output_dict["non_cond_frame_outputs"][4]
    assert state.mux_state.num_buckets == 1
    assert out["maskmem_features"].shape[0] == 1
    assert out["maskmem_pos_enc"][0].shape[0] == 1
    assert out["obj_ptr"].shape[0] == 1
    assert out["maskmem_features"][0, 0, 0, 0] == 0.0   # bucket 0 survived
