# SPDX-License-Identifier: LicenseRef-SAM
"""SAM 3 image concept predictor + video predictor skeleton.

Provides:
  Sam3Predictor             — IMAGE concept predictor (Task 8). OWNS the shared PE vision
                              encoder + text tower + DETR detector and exposes
                              ``predict(image, ConceptPrompt) -> Sam3DetectionResult``.
  ConceptState              — per-concept encoded state stored in Sam3VideoPredictorState.
  Sam3VideoPredictorState   — streaming-session state (bank + concepts + frame counter).
  MAX_CONCEPTS              — current concept-count cap (D9: 1; raise to relax).
  Sam3VideoPredictor        — streaming skeleton class (Task 6 guard); its encode_text /
                              encode_exemplars are wired to real encoders in the streaming
                              task. Its ``encode_text`` seam now takes the full
                              ``ConceptPrompt`` so positives AND negatives are embedded.

Guard contract (spec §9):
  set_concept checks ``state.started`` THEN ``MAX_CONCEPTS`` BEFORE calling encode_text,
  so the two error-path tests never reach encoding and stay CPU-only / checkpoint-free.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.nn as nn

from sam.modeling.association.tracklet import TrackletManager
from sam.modeling.memory.bank import ObjectMemoryBank
from sam.modeling.memory.forgetful import ForgetfulObjectMemoryBank
from sam.prompts import ConceptPrompt
from sam.utils.sam3_transforms import preprocess_to_1008

if TYPE_CHECKING:
    from sam.results import Sam3DetectionResult


class Sam3Predictor(nn.Module):
    """SAM 3 image concept predictor (base / per-object, text-only path).

    Composes (OWNS) the shared PE vision encoder, the text tower, and the DETR detector
    (spec §5/§7): the vision encoder runs ONCE per image and its features are injected into
    the detector. Built by :func:`sam.build_sam.build_sam3` (hydra-compose
    ``configs/sam3/sam3.yaml`` -> instantiate -> strict-load the ``detector.*`` subtree).

    ``predict(image, concept)`` returns a :class:`~sam.results.Sam3DetectionResult` with the
    per-instance masks / boxes / scores for ``concept`` in ``image``.
    """

    def __init__(self, vision_encoder: nn.Module, text_encoder: nn.Module, detector: nn.Module):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.detector = detector

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # ------------------------------------------------------------------
    # Block methods (spec §7) — encode_image once, encode_text per concept, detect.
    # ------------------------------------------------------------------

    def encode_image(self, image_tensor: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Run the owned PE encoder ONCE -> ``(features, pos)`` (the detector's pyramid view).

        The encoder is built with the SAM 2 neck (``add_sam2_neck=True``) so the full
        checkpoint loads strict, but the default ``return_sam2=False`` view returns the
        SAM 3 pyramid the detector consumes (the tracker selects ``return_sam2=True``).
        """
        return self.vision_encoder(image_tensor)

    def encode_text(self, concept: ConceptPrompt) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed the concept's positive text AND ``negative_phrases`` (spec §6/§9).

        The positive phrase and any ``concept.negative_phrases`` are run through the text
        tower together (negatives piggy-backed onto the same forward, mirroring upstream
        ``SAM3VLBackbone.forward_text``'s ``additional_text``). Returns the positive slice
        ``(text_emb, text_mask)`` the base per-object detector consumes:
        ``text_emb`` (seq, n_pos, d_model), ``text_mask`` (n_pos, seq) True-where-PAD. With
        no negatives this reduces to encoding the single positive — bitwise-identical to the
        captured golden. (Feeding the embedded negatives into the presence head is deferred:
        the base detect path is text-positive-only, like the Task 4 detector.)
        """
        positives = [concept.text]
        negatives = list(concept.negative_phrases or [])
        all_phrases = positives + negatives
        # forward returns (text_attention_mask (batch, seq) True-where-PAD,
        #                  text_memory_resized (seq, batch, d_model), inputs_embeds_T)
        text_attention_mask, text_memory_resized, _ = self.text_encoder(
            all_phrases, device=self.device
        )
        n_pos = len(positives)
        text_emb = text_memory_resized[:, :n_pos]   # (seq, n_pos, d_model)
        text_mask = text_attention_mask[:n_pos]     # (n_pos, seq) True where PAD
        return text_emb, text_mask

    def encode_exemplars(self, exemplars) -> Optional[torch.Tensor]:
        """Embed optional reference geometry (deferred — base text-only path).

        The base per-object detector runs the text-only geometry cls path (Task 4); full
        exemplar / geometry-prompt encoding is a later task, so this returns ``None``.
        """
        return None

    def detect(
        self,
        feats: List[torch.Tensor],
        pos: List[torch.Tensor],
        text_emb: torch.Tensor,
        text_mask: torch.Tensor,
        image_hw: Tuple[int, int],
        confidence_threshold: float = 0.5,
        exemplar_emb: Optional[torch.Tensor] = None,
    ) -> "Sam3DetectionResult":
        """Ground the encoded text into per-object detections via the owned detector."""
        return self.detector.detect(
            feats, pos, text_emb, text_mask, image_hw,
            confidence_threshold=confidence_threshold, exemplar_emb=exemplar_emb,
        )

    # ------------------------------------------------------------------
    # Image concept prediction (spec §10): encode_image -> encode_text -> detect.
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def predict(
        self, image, concept: ConceptPrompt, confidence_threshold: float = 0.5
    ) -> "Sam3DetectionResult":
        """Detect every instance of ``concept`` in ``image`` -> ``Sam3DetectionResult``.

        Args:
            image: ``(H, W, 3)`` uint8 RGB array. The predictor owns its preprocessing
                (GPU resize -> 1008 + normalise; CPU resize fails parity).
            concept: the :class:`~sam.prompts.ConceptPrompt` (text [+ negatives]).
            confidence_threshold: presence-weighted score threshold (default 0.5).

        Runs under bf16 autocast + inference_mode — the only supported SAM 3 regime (the PE
        MLP path hardcodes bf16). The shared encoder runs ONCE and its features are injected
        into the detector.
        """
        device = self.device
        image_hw = (int(image.shape[0]), int(image.shape[1]))
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            x = preprocess_to_1008(image, device=device)
            feats, pos = self.encode_image(x)
            text_emb, text_mask = self.encode_text(concept)
            exemplar_emb = (
                self.encode_exemplars(concept.exemplars) if concept.exemplars else None
            )
            return self.detect(
                feats, pos, text_emb, text_mask, image_hw,
                confidence_threshold=confidence_threshold, exemplar_emb=exemplar_emb,
            )


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

    def encode_text(self, concept: ConceptPrompt) -> torch.Tensor:
        """Embed the concept's positive text AND ``negative_phrases`` via the text tower.

        Takes the full ``ConceptPrompt`` (seam fixed in Task 8 — was ``text: str``) so
        positives AND ``negative_phrases`` are embedded; both flow into ``detect()``
        (negatives sharpen the presence head / suppress near-misses). See
        :meth:`Sam3Predictor.encode_text` for the concrete image-path implementation; the
        streaming predictor's tower is wired in the streaming task.
        """
        raise NotImplementedError(
            "encode_text is wired to Sam3TextEncoder in the streaming task"
        )

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
        text_emb = self.encode_text(concept)
        ex_emb = self.encode_exemplars(concept.exemplars) if concept.exemplars else None
        state.concepts.append(ConceptState(cid, concept, text_emb, ex_emb))
        return cid
