# SPDX-License-Identifier: LicenseRef-SAM
"""Output-policy (``Emit``) selection over the tracklet lifecycle.

CPU-only: drives a real ``TrackletManager`` and the pure ``select_emitted``
filter, so the default (``Emit.CONFIRMED``) path is covered without a GPU or a
captured golden.
"""
import torch

from sam.modeling.association.tracklet import TrackletManager, TrackletState
from sam.models.sam3_predictor import select_emitted
from sam.results import Emit, MaskletResult


def _mgr():
    return TrackletManager(confirmation_thresh=3, hotstart_delay=15,
                           hotstart_unmatch_thresh=8)


def _result(logit: float = 1.0):
    """A MaskletResult whose mask is non-empty for ``logit > 0``."""
    return MaskletResult(
        masks_logits=torch.full((1, 1, 4, 4), logit),
        ious=torch.ones(1, 1),
        obj_ptrs=torch.zeros(1, 1),
        obj_scores_logits=torch.ones(1, 1),
    )


def _spawn_and_detect(mgr, obj_id, n_frames):
    """Spawn on frame 0, then match a detection on ``n_frames`` frames total."""
    mgr.spawn(obj_id, frame_idx=0)
    mgr.step(set(), {obj_id}, frame_idx=0)
    for f in range(1, n_frames):
        mgr.step({obj_id}, set(), frame_idx=f)


def test_pending_object_is_not_emitted_under_confirmed():
    m = _mgr()
    _spawn_and_detect(m, 1, n_frames=2)      # 2 consecutive dets, thresh is 3
    assert m.emitted_ids(Emit.CONFIRMED) == set()


def test_object_is_emitted_under_confirmed_on_third_consecutive_detection():
    m = _mgr()
    _spawn_and_detect(m, 1, n_frames=3)
    assert m.emitted_ids(Emit.CONFIRMED) == {1}


def test_pending_object_is_emitted_under_visible():
    m = _mgr()
    _spawn_and_detect(m, 1, n_frames=1)      # keep_alive 1 from its first match
    assert m.emitted_ids(Emit.VISIBLE) == {1}


def test_dormant_object_is_emitted_only_under_alive():
    m = _mgr()
    _spawn_and_detect(m, 1, n_frames=20)     # established, then leaves the scene
    for f in range(20, 40):
        m.step(set(), set(), frame_idx=f)
    assert m.emitted_ids(Emit.VISIBLE) == set()   # keep_alive decayed
    assert m.emitted_ids(Emit.ALIVE) == {1}       # still tracked, memory retained


def test_removed_object_is_emitted_under_no_mode():
    m = _mgr()
    m.spawn(1, frame_idx=100)
    for f in range(101, 109):                # unmatched through the hotstart gate
        m.step(set(), set(), frame_idx=f)
    assert 1 in m.removed_ids()
    for mode in Emit:
        assert m.emitted_ids(mode) == set(), mode


def test_select_emitted_drops_an_empty_mask():
    m = _mgr()
    _spawn_and_detect(m, 1, n_frames=3)
    kept = select_emitted({1: _result(logit=-1.0)}, m, Emit.CONFIRMED)
    assert kept == {}


def test_select_emitted_passes_unmanaged_ids():
    m = _mgr()                               # 7 was never spawned (click-seeded)
    kept = select_emitted({7: _result()}, m, Emit.CONFIRMED)
    assert set(kept) == {7}


def test_select_emitted_stamps_the_tracklet_state():
    m = _mgr()
    _spawn_and_detect(m, 1, n_frames=3)
    kept = select_emitted({1: _result()}, m, Emit.CONFIRMED)
    assert kept[1].tracklet_state is TrackletState.CONFIRMED


def test_select_emitted_stamps_none_for_an_unmanaged_id():
    m = _mgr()
    kept = select_emitted({7: _result()}, m, Emit.ALIVE)
    assert kept[7].tracklet_state is None


def test_select_emitted_hides_a_pending_object_under_confirmed():
    m = _mgr()
    _spawn_and_detect(m, 1, n_frames=1)
    assert select_emitted({1: _result()}, m, Emit.CONFIRMED) == {}
    assert set(select_emitted({1: _result()}, m, Emit.VISIBLE)) == {1}


def test_clicked_object_stays_visible_through_a_click_only_session():
    """A click-seeded object neither decays nor dies (upstream never registers it)."""
    m = _mgr()
    m.spawn(1, frame_idx=0, interactive=True)
    for f in range(1, 60):                   # no detection ever matches it
        m.step(set(), set(), frame_idx=f)
    assert m.emitted_ids(Emit.VISIBLE) == {1}
    assert m.emitted_ids(Emit.CONFIRMED) == {1}
    assert m.removed_ids() == set()


def test_clicking_an_existing_object_pins_it_visible():
    m = _mgr()
    _spawn_and_detect(m, 1, n_frames=20)     # established, then leaves the scene
    for f in range(20, 40):
        m.step(set(), set(), frame_idx=f)
    assert m.emitted_ids(Emit.VISIBLE) == set()
    m.force_confirm(1)                       # user clicks it again
    assert m.emitted_ids(Emit.VISIBLE) == {1}
    for f in range(40, 60):
        m.step(set(), set(), frame_idx=f)
    assert m.emitted_ids(Emit.VISIBLE) == {1}
