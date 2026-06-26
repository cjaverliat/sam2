# SPDX-License-Identifier: LicenseRef-SAM
"""SAM 3 video predictor skeleton — concept state + guard (Task 6).

Provides:
  ConceptState              — per-concept encoded state stored in Sam3VideoPredictorState.
  Sam3VideoPredictorState   — streaming-session state (bank + concepts + frame counter).
  MAX_CONCEPTS              — current concept-count cap (D9: 1; raise to relax).
  Sam3VideoPredictor        — skeleton class; encode_text / encode_exemplars are wired
                              to real encoders in Task 8.

Guard contract (spec §9):
  set_concept checks ``state.started`` THEN ``MAX_CONCEPTS`` BEFORE calling encode_text,
  so the two error-path tests never reach encoding and stay CPU-only / checkpoint-free.
"""
from __future__ import annotations

import torch
from dataclasses import dataclass, field

from sam.modeling.association.tracklet import TrackletManager
from sam.modeling.memory.bank import ObjectMemoryBank
from sam.modeling.memory.forgetful import ForgetfulObjectMemoryBank
from sam.prompts import ConceptPrompt


@dataclass
class ConceptState:
    concept_id: int
    prompt: ConceptPrompt            # original (text, exemplars, negatives)
    text_emb: torch.Tensor           # encoded once
    exemplar_emb: torch.Tensor | None


@dataclass
class Sam3VideoPredictorState:
    video_hw: tuple[int, int]
    bank: ObjectMemoryBank = field(default_factory=ForgetfulObjectMemoryBank)
    concepts: list[ConceptState] = field(default_factory=list)  # 0..1 now; list keeps multi open
    num_frames_processed: int = 0
    _next_obj_id: int = 0
    # Tracklet lifecycle state machine (Task 7).  Holds per-obj-id
    # PENDING → CONFIRMED → DEAD transitions driven by det-match signal.
    tracklet_mgr: TrackletManager = field(default_factory=TrackletManager)

    @property
    def started(self) -> bool:
        return self.num_frames_processed > 0


MAX_CONCEPTS = 1   # relax (or remove) to enable multi-concept


class Sam3VideoPredictor:
    """Skeleton SAM 3 video predictor.

    Only ``set_concept`` (and its guard) is active in this task.
    ``encode_text`` / ``encode_exemplars`` are wired to the real text tower and
    geometry encoder in Task 8; they raise ``NotImplementedError`` here so that any
    accidental call surfaces clearly.

    Style mirrors ``Sam2Predictor`` (same package, no nn.Module base needed for the
    skeleton — Task 8 adds the model-loading ``__init__``).
    """

    # ------------------------------------------------------------------
    # Encoding stubs (Task 8 overrides / fills these in)
    # ------------------------------------------------------------------

    def encode_text(self, text: str) -> torch.Tensor:
        """Embed *text* (and its associated negatives) via the SAM 3 text tower.

        Must embed both positive phrases and ``negative_phrases`` so that
        both flow into ``detect()`` (negatives sharpen the presence head /
        suppress near-misses).  Implemented in Task 8.
        """
        raise NotImplementedError("encode_text is wired to Sam3TextEncoder in Task 8")

    def encode_exemplars(self, exemplars) -> torch.Tensor | None:
        """Embed optional reference geometry via the geometry encoder.

        Implemented in Task 8.
        """
        raise NotImplementedError("encode_exemplars is wired to the geometry encoder in Task 8")

    # ------------------------------------------------------------------
    # Concept management (spec §9, verbatim)
    # ------------------------------------------------------------------

    def set_concept(self, state: Sam3VideoPredictorState, concept: ConceptPrompt) -> int:
        if state.started:
            raise RuntimeError("concept must be set before the first frame is processed")
        if len(state.concepts) >= MAX_CONCEPTS:
            raise RuntimeError(f"at most {MAX_CONCEPTS} concept(s) supported")
        cid = len(state.concepts)
        text_emb = self.encode_text(concept.text)
        ex_emb = self.encode_exemplars(concept.exemplars) if concept.exemplars else None
        state.concepts.append(ConceptState(cid, concept, text_emb, ex_emb))
        return cid
