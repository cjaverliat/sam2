from __future__ import annotations

import torch

from sam2.modeling.sam2_result import SAM2Result
from sam2.modeling.sam2_prompt import SAM2Prompt
from sam2.modeling.sam2_scored_periodic_memory import (
    SAM2ScoredPeriodicObjectMemoryBank,
)

OBJ_ID = 7


def _result(score: float) -> SAM2Result:
    """A single-object SAM2Result whose obj_score_logits max == `score`."""
    return SAM2Result(
        masks_logits=torch.zeros(1, 1, 4, 4),
        ious=torch.ones(1, 1),
        obj_ptrs=torch.zeros(1, 8),
        obj_scores_logits=torch.tensor([[score]]),
    )


def _mem_tensors():
    return torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4)


def _add(bank, frame_idx, score, prompt=False):
    emb, pos = _mem_tensors()
    prompts = (
        [SAM2Prompt(obj_id=OBJ_ID, boxes=torch.tensor([[0.0, 0.0, 1.0, 1.0]]))]
        if prompt
        else []
    )
    return bank.try_add_memories(
        frame_idx=frame_idx,
        obj_ids=[OBJ_ID],
        memory_embeddings=emb,
        memory_pos_embeddings=pos,
        results=_result(score),
        prompts=prompts,
    )[0]  # (added: bool, memory)


def test_period_and_score_gates():
    bank = SAM2ScoredPeriodicObjectMemoryBank(score_threshold=0.0, storage_period=3)

    # frame 0: first non-cond, period_ok (last is None), score ok -> stored
    added, _ = _add(bank, 0, score=1.0)
    assert added
    # frames 1,2: within period (elapsed < 3) -> skipped
    assert not _add(bank, 1, score=1.0)[0]
    assert not _add(bank, 2, score=1.0)[0]
    # frame 3: 3 frames elapsed, score ok -> stored
    assert _add(bank, 3, score=1.0)[0]
    assert bank.count_object_non_conditional_memories(OBJ_ID) == 2


def test_probe_each_frame_after_period():
    bank = SAM2ScoredPeriodicObjectMemoryBank(score_threshold=0.5, storage_period=3)

    assert _add(bank, 0, score=1.0)[0]           # stored, last=0
    assert not _add(bank, 3, score=0.2)[0]       # period ok but score < threshold -> skip
    assert not _add(bank, 4, score=0.1)[0]       # still low -> skip
    assert _add(bank, 5, score=0.9)[0]           # first frame >= threshold after period -> stored
    # last stored resets to 5, so next store needs frame >= 8
    assert not _add(bank, 6, score=1.0)[0]
    assert bank.count_object_non_conditional_memories(OBJ_ID) == 2


def test_conditional_always_stored():
    bank = SAM2ScoredPeriodicObjectMemoryBank(score_threshold=10.0, storage_period=100)
    # Score far below threshold and period huge, but prompt frames bypass both gates.
    assert _add(bank, 0, score=-5.0, prompt=True)[0]
    assert _add(bank, 1, score=-5.0, prompt=True)[0]
    assert bank.count_object_conditional_memories(OBJ_ID) == 2
    assert bank.count_object_non_conditional_memories(OBJ_ID) == 0


def test_recency_selection_over_sparse_storage():
    bank = SAM2ScoredPeriodicObjectMemoryBank(score_threshold=0.0, storage_period=5)
    stored_frames = []
    for f in range(0, 40, 5):
        if _add(bank, f, score=1.0)[0]:
            stored_frames.append(f)
    assert stored_frames == [0, 5, 10, 15, 20, 25, 30, 35]

    # Select 3 most-recent stored non-cond memories before frame 33.
    sel = bank.select_memories(
        obj_ids=[OBJ_ID],
        current_frame_idx=33,
        max_conditional_memories=-1,
        max_non_conditional_memories=3,
        max_ptr_memories=0,
    )[OBJ_ID]
    picked = [m.frame_idx for m in sel.non_conditional_memories]
    # closest-first (most-recent first), matching base selector ordering
    assert picked == [30, 25, 20]


if __name__ == "__main__":
    test_period_and_score_gates()
    test_probe_each_frame_after_period()
    test_conditional_always_stored()
    test_recency_selection_over_sparse_storage()
    print("all tests passed")
