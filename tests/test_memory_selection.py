# SPDX-License-Identifier: LicenseRef-SAM
"""Capping conditional memories: ``max_cond_frames_in_attn`` on the SAM 2 tracker.

The cap is off by default (-1 keeps everything), which is why the selector below went
unexercised long enough to grow a crash. It is a documented knob, so it is tested.
"""
import pytest
import torch

from sam.modeling.memory.bank import ObjectMemory
from sam.modeling.memory.banks import _select_N_closest_conditional_memories


def _memory(frame_idx: int) -> ObjectMemory:
    return ObjectMemory(
        obj_id=1,
        frame_idx=frame_idx,
        memory_embeddings=torch.zeros(1, 1),
        memory_pos_embeddings=torch.zeros(1, 1),
        ptr=torch.zeros(1, 1),
        is_conditional=True,
    )


def _frames(memories):
    return [m.frame_idx for m in memories]


def test_no_cap_keeps_everything():
    memories = [_memory(i) for i in (0, 5, 9)]
    selected, unselected = _select_N_closest_conditional_memories(memories, -1, 7)
    assert selected is memories and unselected == []


def test_cap_keeps_the_frames_nearest_the_current_one():
    """Neighbours either side first, then the next-closest until the cap is reached."""
    memories = [_memory(i) for i in (0, 2, 6, 10, 20)]
    selected, unselected = _select_N_closest_conditional_memories(memories, 3, 7)

    assert len(selected) == 3
    assert set(_frames(selected)) == {2, 6, 10}, "6 and 10 bracket frame 7; 2 is next nearest"
    assert set(_frames(unselected)) == {0, 20}
    assert len(selected) + len(unselected) == len(memories), "every memory is accounted for"


@pytest.mark.parametrize("cap", [1, 2, 4, 5, 9])
def test_cap_never_exceeds_itself_or_loses_a_memory(cap):
    memories = [_memory(i) for i in (0, 3, 4, 8, 11)]
    selected, unselected = _select_N_closest_conditional_memories(memories, cap, 5)

    assert len(selected) <= max(cap, 2), "the two bracketing frames are always kept"
    assert len(selected) + len(unselected) == len(memories)
    assert not set(_frames(selected)) & set(_frames(unselected))
