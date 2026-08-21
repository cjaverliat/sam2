# SPDX-License-Identifier: LicenseRef-SAM
import os

import pytest
import torch


def test_multiplexstate_slot_fill_no_bucket_growth():
    from sam.modeling.multiplex import MultiplexController
    ctrl = MultiplexController(multiplex_count=16)
    st = ctrl.get_state(2, torch.device("cpu"), torch.float32, random=False)
    assert st.num_buckets == 1 and st.total_valid_entries == 2
    assert st.available_slots >= 1
    idx = st.find_next_batch_of_available_indices(1, allow_new_buckets=False)
    st.add_objects(idx, object_ids=None, allow_new_buckets=False)
    assert st.total_valid_entries == 3 and st.num_buckets == 1


def test_slot_fill_raises_when_bucket_full():
    from sam.modeling.multiplex import MultiplexController
    ctrl = MultiplexController(multiplex_count=16)
    st = ctrl.get_state(16, torch.device("cpu"), torch.float32, random=False)
    assert st.available_slots == 0
    with pytest.raises(AssertionError):
        st.find_next_batch_of_available_indices(1, allow_new_buckets=False)


CKPT = "checkpoints/sam3.1_multiplex.pt"


@pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.isfile(CKPT),
    reason="needs CUDA + sam3.1_multiplex.pt",
)
def test_tracker_has_dynamic_add():
    from sam.build_sam import build_sam3_multiplex_video_predictor
    pred = build_sam3_multiplex_video_predictor(
        config_file="configs/sam3/sam3.1.yaml", ckpt_path=CKPT, device="cuda")
    assert hasattr(pred.tracker, "add_new_masks_to_existing_state")


def _state(num_objects, capacity=16):
    from sam.modeling.multiplex import MultiplexController
    ctrl = MultiplexController(multiplex_count=capacity)
    return ctrl.get_state(num_objects, torch.device("cpu"), torch.float32, random=False)


def _grow(st, num_new=1):
    idx = st.find_next_batch_of_available_indices(num_new, allow_new_buckets=False)
    st.add_objects(idx, object_ids=None, allow_new_buckets=False)


def test_removed_slots_are_reclaimed_by_the_next_grow():
    st = _state(16)
    assert st.available_slots == 0
    st.remove_objects([0])
    assert st.available_slots == 1
    _grow(st)
    assert st.total_valid_entries == 16 and st.num_buckets == 1


def test_a_long_churn_never_exhausts_the_grid():
    """Two objects at a time, 30 arrivals: the free pool must not only shrink."""
    st = _state(2)
    for _ in range(30):
        st.remove_objects([0])          # the oldest leaves (indices are compacted)
        _grow(st)
        assert st.total_valid_entries == 2
    assert st.num_buckets == 1


def test_emptying_the_state_leaves_no_stale_bookkeeping():
    st = _state(2)
    st.remove_objects([0, 1])
    assert st.is_empty
    assert st.num_buckets == 0
    assert st.total_valid_entries == 0
    assert st.available_slots == 0
    assert st.get_all_valid_object_idx() == set()


def test_an_emptied_state_refuses_to_be_reused():
    st = _state(1)
    st.remove_objects([0])
    with pytest.raises(ValueError, match="emptied multiplex state"):
        st.add_objects([0], object_ids=None)
