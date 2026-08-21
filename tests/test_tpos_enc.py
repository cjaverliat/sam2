# SPDX-License-Identifier: LicenseRef-SAM
"""Temporal pointer encoding on a one-frame history.

Our streaming loop passes ``num_frames = frame_idx + 1``, so a non-init-cond
``track_step`` on frame 0 -- a refining click on an object seeded on that same
frame -- reaches ``_get_tpos_enc`` with ``max_abs_pos == 1``. CPU-only: the
encoding is a pure function of the pointer distances, so it runs on a stub that
carries only the three attributes it reads.
"""
import torch

from sam.modeling.tracking.sam3_tracker import Sam3Tracker
from sam.modeling.utils import get_1d_sine_pe


class _Stub:
    """The three attributes ``_get_tpos_enc`` reads off the tracker."""

    def __init__(self):
        self.hidden_dim = 8
        self.mem_dim = 4
        self.obj_ptr_tpos_proj = torch.nn.Identity()


def _tpos(rel_pos_list, max_abs_pos):
    return Sam3Tracker._get_tpos_enc(
        _Stub(), rel_pos_list, torch.device("cpu"), max_abs_pos=max_abs_pos
    )


def test_single_frame_history_does_not_produce_nan():
    enc = _tpos([0], max_abs_pos=1)
    assert torch.isfinite(enc).all()


def test_single_frame_history_encodes_a_zero_distance():
    assert torch.equal(_tpos([0], max_abs_pos=1), _tpos([0], max_abs_pos=16))


def test_multi_frame_history_still_normalizes_by_max_abs_pos_minus_one():
    expected = get_1d_sine_pe(torch.tensor([0, 1, 2]) / 15, dim=_Stub().hidden_dim)
    assert torch.equal(_tpos([0, 1, 2], max_abs_pos=16), expected)
