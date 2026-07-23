# SPDX-License-Identifier: LicenseRef-SAM
from sam.modeling.association.tracklet import TrackletManager


def _mgr():
    return TrackletManager(confirmation_thresh=3, hotstart_delay=15,
                           hotstart_unmatch_thresh=8)


def test_established_object_suppressed_not_removed_then_reid():
    m = _mgr()
    m.spawn(1, frame_idx=0)
    for f in range(1, 20):                 # matched every frame -> keep_alive saturates
        m.step({1}, set(), frame_idx=f)
    assert 1 in m.visible_ids()
    for f in range(20, 40):                # absent 20 frames, but past hotstart
        m.step(set(), set(), frame_idx=f)
    assert 1 not in m.removed_ids()        # NOT killed (established)
    assert 1 in m.alive_ids()              # stays alive (memory retained)
    assert 1 not in m.visible_ids()        # hidden while gone
    for f in range(40, 45):                # re-enters; keep_alive recovers from the floor
        m.step({1}, set(), frame_idx=f)
    assert 1 in m.visible_ids()            # same id, visible again (no fresh id)


def test_new_object_within_hotstart_is_removed():
    m = _mgr()
    m.spawn(5, frame_idx=100)              # spawned mid-clip
    for f in range(101, 101 + 8):         # unmatched for hotstart_unmatch_thresh
        m.step(set(), set(), frame_idx=f)
    assert 5 in m.removed_ids()


def test_keep_alive_clamps_and_flips_visibility():
    m = _mgr()
    m.spawn(1, frame_idx=0)
    for f in range(1, 30):
        m.step({1}, set(), frame_idx=f)
    assert m._tracks[1].keep_alive == 8    # max clamp
    for f in range(30, 50):
        m.step(set(), set(), frame_idx=f)
    assert m._tracks[1].keep_alive == -4   # min clamp
    assert 1 not in m.visible_ids()
