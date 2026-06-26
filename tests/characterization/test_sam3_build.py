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

from sam.prompts import ConceptPrompt
from sam.models.sam3_predictor import (
    Sam3VideoPredictor,
    Sam3VideoPredictorState,
    MAX_CONCEPTS,
)


# ---------------------------------------------------------------------------
# CPU-only stub: overrides the two encode_* methods so no model is loaded.
# ---------------------------------------------------------------------------

class _FakeSam3VideoPredictor(Sam3VideoPredictor):
    """Minimal test double — encode_text / encode_exemplars return dummy tensors."""

    def encode_text(self, text: str) -> torch.Tensor:
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
