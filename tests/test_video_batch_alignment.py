# SPDX-License-Identifier: LicenseRef-SAM
"""An object skipped for lack of memories must leave the batch entirely.

``Sam2VideoPredictor.forward_embeddings`` iterates every known object, but an
object with neither a prompt nor a memory is skipped without producing a result.
If the batch size and the id list still count it, the memory encoder is asked to
expand to one row too many and the returned masks are paired with the wrong ids.

CPU-only: the decode is stubbed out, since what is under test is the bookkeeping
around it, not the model.
"""
import torch

from sam.modeling.memory.bank import ObjectMemory
from sam.models.sam2_predictor import Sam2VideoPredictor, Sam2VideoPredictorState
from sam.results import MaskletResult


def _result(value: float) -> MaskletResult:
    """A one-object result whose mask logits identify the object that made it."""
    return MaskletResult(
        masks_logits=torch.full((1, 1, 4, 4), value),
        ious=torch.ones(1, 1),
        obj_ptrs=torch.zeros(1, 8),
        obj_scores_logits=torch.ones(1, 1),
    )


class _StubPredictor(Sam2VideoPredictor):
    """The predictor's bookkeeping with every model call replaced."""

    def __init__(self):
        self.max_cond_frames_in_attn = -1
        self.num_maskmem = 7
        self.max_obj_ptrs_in_encoder = 16
        self.only_obj_ptrs_in_the_past_for_eval = True
        self.encoded_batch_sizes: list[int] = []
        self._decoded = 0

    @property
    def device(self):
        return torch.device("cpu")

    def condition_image_embeddings_on_memories(self, **kwargs):
        return kwargs["img_embeddings"][-1]

    def generate_masks(self, **kwargs):
        self._decoded += 1
        return _result(float(self._decoded))

    def encode_memory(self, img_embeddings, masks_logits, obj_score_logits, is_prompt):
        n = img_embeddings[0].shape[0]
        self.encoded_batch_sizes.append(n)
        return torch.zeros(n, 4, 2, 2), torch.zeros(n, 4, 2, 2)


def _state_with_one_memory_of_two_known_objects():
    """Object 1 carries a memory; object 2 is known but has none (e.g. pruned)."""
    state = Sam2VideoPredictorState(video_hw=(4, 4))
    state.memory_bank.try_add_memories(
        frame_idx=0,
        obj_ids=[1],
        memory_embeddings=torch.zeros(1, 4, 2, 2),
        memory_pos_embeddings=torch.zeros(1, 4, 2, 2),
        results=_result(1.0),
        prompts=[],
    )
    state.memory_bank.known_obj_ids.add(2)
    return state


def _forward(predictor, state):
    return predictor.forward_embeddings(
        state=state,
        frame_idx=1,
        img_embeddings=[torch.zeros(1, 4, 2, 2)],
        img_pos_embeddings=[torch.zeros(1, 4, 2, 2)],
    )


def test_memoryless_object_is_absent_from_the_output():
    state = _state_with_one_memory_of_two_known_objects()
    out = _forward(_StubPredictor(), state)
    assert set(out) == {1}


def test_memoryless_object_does_not_inflate_the_memory_batch():
    predictor = _StubPredictor()
    _forward(predictor, _state_with_one_memory_of_two_known_objects())
    assert predictor.encoded_batch_sizes == [1]


def test_masks_stay_paired_with_the_object_that_produced_them():
    predictor = _StubPredictor()
    out = _forward(predictor, _state_with_one_memory_of_two_known_objects())
    assert out[1].masks_logits.flatten()[0] == 1.0   # the first (only) decode
