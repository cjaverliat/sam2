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
