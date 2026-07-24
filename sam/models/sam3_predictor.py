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

import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam.modeling.association import associate_det_trk
from sam.modeling.association.tracklet import TrackletManager
from sam.modeling.memory.bank import ObjectMemoryBank
from sam.modeling.memory.forgetful import ForgetfulObjectMemoryBank
from sam.modeling.tracking.sam3_tracker_utils import fill_holes_in_mask_scores
from sam.prompts import ConceptPrompt, GeometryPrompt
from sam.results import MaskletResult
from sam.utils.sam3_transforms import preprocess_to_1008, preprocess_to_1008_video

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

        The base per-object detector runs the text-only geometry cls path; full
        exemplar / geometry-prompt encoding is deferred to the streaming task, so this
        returns ``None``.
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
        geo: Optional[dict] = None,
    ) -> "Sam3DetectionResult":
        """Ground the encoded text into per-object detections via the owned detector."""
        return self.detector.detect(
            feats, pos, text_emb, text_mask, image_hw,
            confidence_threshold=confidence_threshold, exemplar_emb=exemplar_emb, geo=geo,
        )

    # ------------------------------------------------------------------
    # Image concept prediction (spec §10): encode_image -> encode_text -> detect.
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def predict(
        self, image, concept: ConceptPrompt, confidence_threshold: float = 0.5,
        dtype: torch.dtype = torch.bfloat16, geometry=None,
    ) -> "Sam3DetectionResult":
        """Detect every instance of ``concept`` in ``image`` -> ``Sam3DetectionResult``.

        Args:
            image: ``(H, W, 3)`` uint8 RGB array. The predictor owns its preprocessing
                (GPU resize -> 1008 + normalise; CPU resize fails parity).
            concept: the :class:`~sam.prompts.ConceptPrompt` (text [+ negatives]).
            confidence_threshold: presence-weighted score threshold (default 0.5).
            dtype: autocast dtype (default ``torch.bfloat16``).  Pass
                ``torch.float32`` to disable mixed-precision, e.g. when comparing
                against a float32 golden (EfficientSAM3 parity).  The PE-ViTDet SAM 3
                model requires ``bfloat16`` (its PE MLP path hardcodes bf16); the
                EfficientSAM3 student works in either regime.

        Runs under autocast(*dtype*) + inference_mode.  When *dtype* is ``torch.float32``
        the autocast context is a no-op (float32 is the native dtype of all ops).
        """
        device = self.device
        image_hw = (int(image.shape[0]), int(image.shape[1]))
        # float32 → nullcontext (no mixed-precision); any other dtype → autocast.
        ac = (
            contextlib.nullcontext()
            if dtype == torch.float32
            else torch.autocast(device_type=device.type, dtype=dtype)
        )
        with ac:
            x = preprocess_to_1008(image, device=device)
            feats, pos = self.encode_image(x)
            text_emb, text_mask = self.encode_text(concept)
            exemplar_emb = (
                self.encode_exemplars(concept.exemplars) if concept.exemplars else None
            )
            geo = _pack_geometry(geometry, image_hw, device)
            return self.detect(
                feats, pos, text_emb, text_mask, image_hw,
                confidence_threshold=confidence_threshold, exemplar_emb=exemplar_emb,
                geo=geo,
            )


def _masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """Tight ``xyxy`` (pixel) bounding box of each binary mask -- ``torchvision.ops.
    masks_to_boxes`` semantics (xmax/ymax are inclusive max indices).

    The SAM 3.1 multiplex demo derives detection boxes from the OUTPUT masks
    (``Sam3MultiplexTracking._postprocess_output`` runs ``masks_to_boxes`` on
    ``out_binary_masks``), so the committed ``image_sam31.npz`` ``boxes`` are mask-derived,
    not the raw DETR ``pred_boxes``. Reproducing the box the same way keeps the box and mask
    consistent (the gate compares against the mask-derived golden).
    """
    n = masks.shape[0]
    boxes = masks.new_zeros((n, 4), dtype=torch.float32)
    for i in range(n):
        ys, xs = torch.where(masks[i])
        if ys.numel() == 0:
            continue
        boxes[i, 0] = xs.min()
        boxes[i, 1] = ys.min()
        boxes[i, 2] = xs.max()
        boxes[i, 3] = ys.max()
    return boxes


def _build_mux_point_inputs(prompts, video_hw, image_size, device):
    """Batch point-only ``GeometryPrompt``s into the tracker's ``point_inputs`` dict.

    Coords are scaled from video pixels (or ``[0,1]`` when ``is_normalized``) to the
    ``image_size`` prompt grid. Ragged point counts are right-padded with coord
    ``(0,0)`` and label ``-1`` (the SAM "no point" padding), so all objects batch
    into one interactive ``track_step``.

    Args:
        prompts: point-only prompts, one per object (each ``points_coords`` (P,2),
            ``points_labels`` (P,), no boxes/masks).
        video_hw: ``(H, W)`` of the source video.
        image_size: the tracker's square input resolution.
        device: target device for the built tensors.

    Returns:
        ``(point_inputs, obj_ids)`` where ``point_inputs`` has ``point_coords``
        ``(n, P, 2)`` float and ``point_labels`` ``(n, P)`` int32.
    """
    height, width = video_hw
    scale = torch.tensor([width, height], device=device, dtype=torch.float32)
    obj_ids, per_coords, per_labels = [], [], []
    for prompt in prompts:
        obj_ids.append(prompt.obj_id)
        coords = prompt.points_coords.to(device).float()
        coords = coords if prompt.is_normalized else coords / scale
        per_coords.append(coords * image_size)
        per_labels.append(prompt.points_labels.to(device).to(torch.int32))
    max_points = max(labels.shape[0] for labels in per_labels)
    n = len(prompts)
    coords = torch.zeros(n, max_points, 2, device=device)
    labels = torch.full((n, max_points), -1, device=device, dtype=torch.int32)
    for i, (obj_coords, obj_labels) in enumerate(zip(per_coords, per_labels)):
        coords[i, : obj_coords.shape[0]] = obj_coords
        labels[i, : obj_labels.shape[0]] = obj_labels
    return {"point_coords": coords, "point_labels": labels}, obj_ids


def _pack_geometry(prompt, image_hw, device):
    """Pack a point/box ``GeometryPrompt`` into the detector's geometry inputs.

    Points -> normalized xy ``(N,1,2)`` + labels ``(N,1)``; boxes xyxy (pixel, or
    normalized via ``is_normalized``) -> normalized cxcywh ``(N,1,4)`` + labels
    ``(N,1)``. Mask geometry has no weights in either checkpoint -> raises.
    """
    if prompt is None:
        return None
    if prompt.masks_logits is not None:
        raise NotImplementedError(
            "mask geometry prompts are unsupported (no mask_encoder weights); "
            "use box or point prompts"
        )
    h, w = image_hw
    geo = {}
    if prompt.points_coords is not None:
        c = prompt.points_coords.to(device).float()
        c = c if prompt.is_normalized else c / torch.tensor([w, h], device=device)
        geo["point_coords"] = c[:, None, :]
        geo["point_labels"] = prompt.points_labels.to(device)[:, None]
    if prompt.boxes is not None:
        b = prompt.boxes.to(device).float()
        b = b if prompt.is_normalized else b / torch.tensor([w, h, w, h], device=device)
        cx = (b[:, 0] + b[:, 2]) / 2
        cy = (b[:, 1] + b[:, 3]) / 2
        bw = (b[:, 2] - b[:, 0]).abs()
        bh = (b[:, 3] - b[:, 1]).abs()
        geo["box_coords"] = torch.stack([cx, cy, bw, bh], -1)[:, None, :]
        geo["box_labels"] = torch.ones(b.shape[0], 1, device=device)
    return geo or None


class Sam3MultiplexPredictor(Sam3Predictor):
    """SAM 3.1 (multiplex) image concept predictor (text-only path).

    Mirrors :class:`Sam3Predictor` -- OWNS the SAM 3.1 PE vision encoder (the DETECTION
    tri-neck head ``convs``, run ONCE per image), the SAM 3.1 text tower, and the SAM 3.1
    DETR detector -- but the detector is built with ``supervise_joint_box_scores=True`` (the
    SAM 3.1 difference). It reuses the base ``encode_image`` / ``encode_text`` / ``device``
    (the text path feeds the detector exactly the text-id-0 slice == encoding the positive
    phrase alone, which equals ``image_sam31.npz``'s ``text_emb[:, 0]``).

    ``predict`` overrides the base post-processing to the SAM 3.1 demo semantics:
      * the score is the joint ``sigmoid(pred_logits)`` (presence is already folded into
        ``pred_logits`` by ``supervise_joint_box_scores``; NO extra presence multiply), and
      * the box is ``masks_to_boxes`` of the output mask (the multiplex demo derives boxes
        from masks), not the raw DETR ``pred_boxes``.
    Built by :func:`sam.build_sam.build_sam3_multiplex` (compose ``configs/sam3/sam3.1.yaml``
    -> instantiate -> strict-load the relevant ``detector.*`` subtree of
    ``sam3.1_multiplex.pt``).
    """

    @torch.inference_mode()
    def predict(
        self, image, concept: ConceptPrompt, confidence_threshold: float = 0.5,
        geometry=None,
    ) -> "Sam3DetectionResult":
        """Detect every instance of ``concept`` in ``image`` -> ``Sam3DetectionResult``.

        Args mirror :meth:`Sam3Predictor.predict`. Runs under bf16 autocast + inference_mode
        (the only supported SAM 3 regime); the shared encoder runs ONCE.
        """
        from sam.results import Sam3DetectionResult

        device = self.device
        image_hw = (int(image.shape[0]), int(image.shape[1]))
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            x = preprocess_to_1008(image, device=device)
            feats, pos = self.encode_image(x)
            text_emb, text_mask = self.encode_text(concept)
            geo = _pack_geometry(geometry, image_hw, device)
            out = self.detector.forward_grounding(feats, pos, text_emb, text_mask, geo=geo)

        pred_logits = out["pred_logits"]              # (P, nq, 1) -- JOINT (presence folded)
        pred_masks = out["pred_masks"]                # (P, nq, h, w) logits
        presence_logit = out["presence_logit_dec"]    # (P, 1)

        out_probs = pred_logits.sigmoid().squeeze(-1)  # (P, nq) -- joint score, no extra mult
        keep = out_probs > confidence_threshold
        kept_probs = out_probs[keep]
        kept_masks = pred_masks[keep]                  # (N, h, w)

        img_h, img_w = image_hw
        masks_logits = F.interpolate(
            kept_masks.unsqueeze(1).float(), (img_h, img_w),
            mode="bilinear", align_corners=False,
        ).squeeze(1)  # (N, H, W) logits (binarise at 0 == prob > 0.5)
        boxes = _masks_to_boxes(masks_logits > 0.0)    # (N, 4) xyxy px (mask-derived)

        presence = float(presence_logit.float().sigmoid().reshape(-1)[0])
        instance_ids = torch.arange(boxes.shape[0], device=boxes.device)
        return Sam3DetectionResult(
            masks_logits=masks_logits,
            boxes=boxes,
            scores=kept_probs,
            presence=presence,
            instance_ids=instance_ids,
        )


@dataclass
class ConceptState:
    concept_id: int
    prompt: ConceptPrompt            # original (text, exemplars, negatives)
    text_emb: torch.Tensor           # encoded once (positive slice, (seq, n_pos, d))
    exemplar_emb: torch.Tensor | None
    text_mask: torch.Tensor | None = None  # (n_pos, seq) True-where-PAD, for the detector


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
    # --- SAM 3.1 multiplex only (M3); untouched by the base per-object path. ---
    # The multiplex tracker's SPATIAL memory is BUCKET-space (``num_buckets, C, H, W`` -- a JOINT
    # K-object encoding, NOT per-object-separable), so it is threaded as the tracker's native
    # ``output_dict`` (the M1-proven format) rather than the per-object bank. The bank +
    # tracklet_mgr + allocator still drive the per-object LIFECYCLE (``known_obj_ids`` ->
    # active set, spawn/confirm/kill); the loop's OUTPUT masks are demuxed per-object.
    mux_output_dict: dict = field(
        default_factory=lambda: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    )
    mux_state: object = None        # the MultiplexState for the (stable) tracked set
    mux_obj_ids: list = field(default_factory=list)  # obj index i (mux_state) -> obj_id

    @property
    def started(self) -> bool:
        return self.num_frames_processed > 0


MAX_CONCEPTS = 1   # relax (or remove) to enable multi-concept


class Sam3VideoPredictor(nn.Module):
    """SAM 3 streaming video concept predictor (spec §10).

    COMPOSES the shared PE vision encoder (built ``add_sam2_neck=True``, run ONCE per frame —
    its ``return_sam2=False`` view feeds the detector, its ``return_sam2=True`` view feeds the
    tracker), the text tower, the DETR detector, and the per-object :class:`Sam3Tracker` (via
    ``self.tracker``; delegation, not inheritance — see spec §5 layering). ``set_concept`` +
    ``forward(state, frame_idx, frame)`` reproduce the official ``sam3_video_predictor_example``
    streaming flow: per frame, encode once -> (gated) detect new instances -> propagate existing
    tracklets -> associate det<->track -> spawn/confirm/kill via the ``TrackletManager`` ->
    update the forgetful bank (which bounds per-object memory -> constant VRAM).

    The bank owns temporal memory **selection** (its abstraction), so the tracker runs with
    ``use_memory_selection=False`` and is conditioned on exactly the frames the bank returns.

    ``__init__`` args are optional so the CPU guard tests can subclass with stubbed encoders.
    """

    def __init__(
        self,
        vision_encoder: nn.Module | None = None,
        text_encoder: nn.Module | None = None,
        detector: nn.Module | None = None,
        tracker: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.detector = detector
        self.tracker = tracker
        # Output-mask hole-fill / sprinkle-removal area (upstream build_sam3_video_model uses 16);
        # applied to the low-res OUTPUT masks only (not to seeding / memory, matching upstream).
        self.fill_hole_area = 16
        # New-tracklet spawn threshold (upstream Sam3VideoInferenceWithInstanceInteractivity
        # new_det_thresh=0.7): on frames AFTER the initial detection frame, an unmatched detection
        # must clear this score to spawn a new tracklet. The initial (prompt) frame spawns the
        # detection set at the detection threshold (0.5, applied in _detect) -- so borderline
        # later detections (0.5-0.7) are suppressed, matching upstream (which additionally hides
        # newly-spawned objects via the hotstart-delay buffer).
        self.new_det_thresh = 0.7

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # ------------------------------------------------------------------
    # Block methods (encode once; the encoder is shared by detector + tracker)
    # ------------------------------------------------------------------

    def encode_image(self, x: torch.Tensor):
        """One trunk pass -> BOTH the detector (SAM 3) and tracker (SAM 2) pyramid views.

        Returns ``(det_feats, det_pos, sam2_feats, sam2_pos)`` with ``scalp`` applied to each
        (so the heaviest ViT trunk runs once, not once per view).
        """
        sam3_f, sam3_p, sam2_f, sam2_p = self.vision_encoder.vision_backbone(x)
        s = self.vision_encoder.scalp
        if s > 0:
            sam3_f, sam3_p = sam3_f[:-s], sam3_p[:-s]
            sam2_f, sam2_p = sam2_f[:-s], sam2_p[:-s]
        return sam3_f, sam3_p, sam2_f, sam2_p

    def encode_text(self, concept: ConceptPrompt):
        """Embed the concept's positive text AND ``negative_phrases`` -> ``(text_emb, text_mask)``.

        Mirrors :meth:`Sam3Predictor.encode_text` (negatives piggy-backed on the same forward;
        the positive slice is returned for the base per-object detector).
        """
        positives = [concept.text]
        negatives = list(concept.negative_phrases or [])
        all_phrases = positives + negatives
        text_attention_mask, text_memory_resized, _ = self.text_encoder(
            all_phrases, device=self.device
        )
        n_pos = len(positives)
        return text_memory_resized[:, :n_pos], text_attention_mask[:n_pos]

    def encode_exemplars(self, exemplars) -> torch.Tensor | None:
        """Embed optional reference geometry (deferred — base text-only path; returns None)."""
        return None

    # ------------------------------------------------------------------
    # Concept management (spec §9)
    # ------------------------------------------------------------------

    def set_concept(self, state: Sam3VideoPredictorState, concept: ConceptPrompt) -> int:
        if state.started:
            raise RuntimeError("concept must be set before the first frame is processed")
        if len(state.concepts) >= MAX_CONCEPTS:
            raise RuntimeError(f"at most {MAX_CONCEPTS} concept(s) supported")
        cid = len(state.concepts)
        encoded = self.encode_text(concept)
        # The real tower returns (text_emb, text_mask); the CPU guard-test stub returns a bare
        # tensor — handle both so those tests stay checkpoint-free.
        if isinstance(encoded, tuple):
            text_emb, text_mask = encoded
        else:
            text_emb, text_mask = encoded, None
        ex_emb = self.encode_exemplars(concept.exemplars) if concept.exemplars else None
        state.concepts.append(ConceptState(cid, concept, text_emb, ex_emb, text_mask))
        return cid

    # ------------------------------------------------------------------
    # Object-id allocator + removal (spec §10 "session API")
    # ------------------------------------------------------------------

    def _alloc_obj_id(self, state: Sam3VideoPredictorState) -> int:
        """Issue a fresh obj id for BOTH detector-spawned and user-prompted objects.

        Monotonic -> never reuses an id, so a remove-then-re-add yields a NEW id (no collision
        with the removed object's stale memories).
        """
        oid = state._next_obj_id
        state._next_obj_id += 1
        return oid

    def remove_object(self, state: Sam3VideoPredictorState, obj_id: int) -> None:
        """Kill a tracklet + purge its bank memories (spec §10)."""
        state.bank.known_obj_ids.discard(obj_id)
        # ForgetfulObjectMemoryBank / Sam2ObjectMemoryBank store per-object memory dicts.
        for store in ("conditional_memories", "non_conditional_memories"):
            d = getattr(state.bank, store, None)
            if isinstance(d, dict):
                d.pop(obj_id, None)
        state.tracklet_mgr.remove(obj_id)

    @staticmethod
    def _filter_visible(state, results: dict) -> dict:
        """Keep only visible tracklet objects; non-managed ids (e.g. click-seeded)
        always pass. Suppressed (dormant) objects propagate but are hidden."""
        managed = state.tracklet_mgr.managed_ids()
        visible = state.tracklet_mgr.visible_ids()
        return {
            oid: r for oid, r in results.items()
            if oid not in managed or oid in visible
        }

    # ------------------------------------------------------------------
    # Streaming forward (spec §10 data-flow)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def forward(
        self,
        state: Sam3VideoPredictorState,
        frame_idx: int,
        frame,
        geometry_prompts: list[GeometryPrompt] = [],
    ) -> dict[int, MaskletResult]:
        device = self.device
        H, W = state.video_hw
        state.num_frames_processed += 1
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            # encode the frame ONCE (shared by detector + tracker)
            x = preprocess_to_1008(frame, device=device)
            det_feats, det_pos, sam2_feats, sam2_pos = self.encode_image(x)
            vis, vpos, feat_sizes = self._prepare_tracker_feats(sam2_feats, sam2_pos)
            num_frames = frame_idx + 1  # frames seen so far (forward streaming)

            concept = state.concepts[0] if state.concepts else None

            # 1) propagate existing tracklets (memory-conditioned, per object)
            active_ids = sorted(state.bank.known_obj_ids)
            trk_low_masks: dict[int, torch.Tensor] = {}
            trk_results: dict[int, dict] = {}
            for obj_id in active_ids:
                out = self._propagate_object(
                    state, frame_idx, obj_id, vis, vpos, feat_sizes, num_frames
                )
                trk_results[obj_id] = out
                trk_low_masks[obj_id] = out["pred_masks"][0, 0].float()

            # 2) concept-driven detection (GATED: only when a concept is set)
            det = self._detect(det_feats, det_pos, concept) if concept is not None else None

            # 3) associate det<->trk, then spawn / confirm / kill via the TrackletManager
            new_objects: list[tuple[int, int]] = []  # (obj_id, det_idx)
            if det is not None:
                new_objects = self._associate_and_update(
                    state, det, active_ids, trk_low_masks, trk_results
                )

            # 4) seed new detector instances into the tracker (soft-mask cond-frame memory)
            for obj_id, det_idx in new_objects:
                self._seed_object(
                    state, frame_idx, obj_id, vis, vpos, feat_sizes,
                    det.masks_logits[det_idx], num_frames,
                )

            # 5) route geometry prompts to the tracker (new obj_id -> spawn, existing -> refine)
            geom_results: dict[int, MaskletResult] = {}
            for prompt in (geometry_prompts or []):
                oid, out = self._apply_geometry_prompt(
                    state, frame_idx, prompt, vis, vpos, feat_sizes, num_frames
                )
                geom_results[oid] = self._masklet_from_lowres(
                    out["pred_masks"][0, 0].float(), out, H, W
                )

            # 6) build outputs: existing tracklets -> tracker masks; new dets -> detector masks
            results: dict[int, MaskletResult] = {}
            for obj_id in active_ids:
                if obj_id not in state.bank.known_obj_ids:
                    continue  # killed this frame
                results[obj_id] = self._masklet_from_lowres(
                    trk_low_masks[obj_id], trk_results[obj_id], H, W
                )
            for obj_id, det_idx in new_objects:
                results[obj_id] = self._masklet_from_lowres(
                    det.masks_logits[det_idx].float(), None, H, W
                )
            results.update(geom_results)
            return self._filter_visible(state, results)

    # ------------------------------------------------------------------
    # Streaming helpers
    # ------------------------------------------------------------------

    def _prepare_tracker_feats(self, sam2_feats, sam2_pos):
        """Project the two hi-res SAM 2 levels (conv_s0/conv_s1) and flatten to ``(HW, B, C)``.

        Run ONCE per frame (batch=1); every object's ``track_step`` shares these features
        (the base per-object tracker has no cross-object attention).
        """
        fpn = list(sam2_feats)
        fpn[0] = self.tracker.sam_mask_decoder.conv_s0(fpn[0])
        fpn[1] = self.tracker.sam_mask_decoder.conv_s1(fpn[1])
        feat_sizes = [(f.shape[-2], f.shape[-1]) for f in fpn]
        vis = [f.flatten(2).permute(2, 0, 1) for f in fpn]
        vpos = [p.flatten(2).permute(2, 0, 1) for p in sam2_pos]
        return vis, vpos, feat_sizes

    def _detect(self, det_feats, det_pos, concept: ConceptState) -> "Sam3DetectionResult":
        """Run the DETR detector at the tracker's low-res mask grid (squashed space).

        ``image_hw`` is set to the tracker's ``low_res_mask_size`` so the returned masks share
        the tracker's mask grid (association is mask-IoU; seeding + output resize from here).
        """
        m = self.tracker.low_res_mask_size
        return self.detector.detect(
            det_feats, det_pos, concept.text_emb, concept.text_mask,
            image_hw=(m, m), confidence_threshold=0.5,
        )

    def _associate_and_update(self, state, det, active_ids, trk_low_masks, trk_results):
        """Match detections to tracks, spawn new tracklets, advance lifecycle, kill dead ones.

        Returns ``[(obj_id, det_idx), ...]`` for the newly-spawned detections.
        """
        det_masks = det.masks_logits  # (N, Hm, Wm) logits
        if len(active_ids) > 0:
            trk_stack = torch.stack([trk_low_masks[o] for o in active_ids], dim=0)
        else:
            trk_stack = det_masks.new_zeros((0, *det_masks.shape[-2:]))

        # The prompt frame (first forward) spawns the initial detection set at the detection
        # threshold; subsequent frames gate new (unmatched) detections at new_det_thresh (0.7).
        spawn_thresh = 0.0 if state.num_frames_processed == 1 else self.new_det_thresh
        new_dets, unmatched_tracks, det2track, _scores = associate_det_trk(
            det_masks=det_masks,
            track_masks=trk_stack,
            iou_threshold=0.5,
            iou_threshold_trk=0.5,
            det_scores=det.scores,
            new_det_thresh=spawn_thresh,
        )

        # det2track values are TRACK INDICES -> map to obj_ids via the active_ids ordering.
        matched_track_ids: set[int] = set()
        for track_indices in det2track.values():
            for t in track_indices:
                matched_track_ids.add(active_ids[t])
        # Tracks the tracker itself reports ABSENT (object_score_logits <= 0) must be counted as
        # unmatched (the 5th upstream `empty_trk_obj_ids` return is omitted from associate_det_trk;
        # derive it here so the kill counter increments).
        for o in active_ids:
            if float(trk_results[o]["object_score_logits"].reshape(-1)[0]) <= 0.0:
                matched_track_ids.discard(o)

        frame_idx = state.num_frames_processed - 1

        # spawn new tracklets (allocator-issued ids) for the new detections
        new_objects: list[tuple[int, int]] = []
        for det_idx in new_dets:
            oid = self._alloc_obj_id(state)
            new_objects.append((oid, int(det_idx)))
            state.tracklet_mgr.spawn(oid, frame_idx)
        new_ids = {oid for oid, _ in new_objects}

        # make sure every live track is registered before stepping the lifecycle
        managed = state.tracklet_mgr.managed_ids()
        for o in active_ids:
            if o not in managed:
                state.tracklet_mgr.spawn(o, frame_idx)
        state.tracklet_mgr.step(matched_track_ids, new_ids, frame_idx)

        # purge ONLY removed tracklets (within-hotstart failures). Absent established
        # objects are suppressed, not removed -> memory retained for re-ID.
        for oid in state.tracklet_mgr.removed_ids():
            if oid in state.bank.known_obj_ids:
                self._purge_removed(state, oid)
        return new_objects

    def _purge_removed(self, state, obj_id: int) -> None:
        """Fully drop a removed tracklet. Subclasses that own extra per-object state
        (e.g. a multiplex slot) override to release it too."""
        self.remove_object(state, obj_id)

    def _propagate_object(self, state, frame_idx, obj_id, vis, vpos, feat_sizes, num_frames):
        """Propagate one tracklet a single frame, conditioned on its bank memories."""
        sel = state.bank.select_memories(
            obj_ids=[obj_id],
            current_frame_idx=frame_idx,
            # -1 = return all conditional memories (the forgetful bank keeps them indefinitely);
            # the tracker caps them to ``max_cond_frames_in_attn`` via ``select_closest_cond_frames``.
            max_conditional_memories=-1,
            max_non_conditional_memories=self.tracker.num_maskmem - 1,
            max_ptr_memories=self.tracker.max_obj_ptrs_in_encoder,
            only_include_pointers_in_past=True,
            reverse_tracking=False,
        )[obj_id]
        output_dict = self._selection_to_output_dict(sel)
        out = self.tracker.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=False,
            current_vision_feats=vis,
            current_vision_pos_embeds=vpos,
            feat_sizes=feat_sizes,
            image=None,
            point_inputs=None,
            mask_inputs=None,
            output_dict=output_dict,
            num_frames=num_frames,
        )
        self._store_memory(state, frame_idx, obj_id, out, conditional=False)
        return out

    def _seed_object(self, state, frame_idx, obj_id, vis, vpos, feat_sizes, det_mask, num_frames):
        """Seed a new tracklet's cond-frame memory from the detector's mask (spec §10).

        Mirrors upstream ``_tracker_add_new_objects``: resize the detector mask to the tracker's
        input-mask size and BINARIZE, then run a mask-prompted init-cond ``track_step``.
        """
        ims = self.tracker.input_mask_size
        m = F.interpolate(
            det_mask[None, None].float(), size=(ims, ims), mode="bilinear", align_corners=False
        )
        m = (m > 0.0).float()
        out = self.tracker.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=True,
            current_vision_feats=vis,
            current_vision_pos_embeds=vpos,
            feat_sizes=feat_sizes,
            image=None,
            point_inputs=None,
            mask_inputs=m,
            output_dict={"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
            num_frames=num_frames,
        )
        self._store_memory(state, frame_idx, obj_id, out, conditional=True)
        return out

    def _apply_geometry_prompt(self, state, frame_idx, prompt, vis, vpos, feat_sizes, num_frames):
        """Route a :class:`GeometryPrompt` to the tracker (new obj_id -> spawn, existing -> refine).

        Reuses the SAM 2 tracker prompt path: points (scaled to the 1008 input grid) and/or a
        mask drive a cond-frame ``track_step``; a new ``obj_id`` spawns a tracklet, an existing
        one is re-conditioned (refined) at this frame. Not exercised by the parity gate (the
        golden uses text only) but completes the spec §10 session API.
        """
        device = self.device
        H, W = state.video_hw
        obj_id = prompt.obj_id
        is_new = obj_id not in state.bank.known_obj_ids
        prompt = prompt.to(device)

        point_inputs = None
        coords_list, labels_list = [], []
        if prompt.points_coords is not None:
            c = prompt.points_coords.float()
            c = c if prompt.is_normalized else c / torch.tensor([W, H], device=device)
            coords_list.append(c * self.tracker.image_size)
            labels_list.append(prompt.points_labels.to(device))
        if prompt.boxes is not None:
            # encode each box as its two corners (labels 2 / 3), like the SAM prompt encoder
            b = prompt.boxes.float()
            b = b if prompt.is_normalized else b / torch.tensor([W, H, W, H], device=device)
            corners = b.reshape(-1, 2, 2) * self.tracker.image_size
            coords_list.append(corners.reshape(-1, 2))
            labels_list.append(
                torch.tensor([2, 3] * b.shape[0], device=device, dtype=torch.int32)
            )
        if coords_list:
            point_inputs = {
                "point_coords": torch.cat(coords_list, dim=0)[None],
                "point_labels": torch.cat(labels_list, dim=0)[None],
            }

        mask_inputs = None
        if prompt.masks_logits is not None:
            ims = self.tracker.input_mask_size
            mm = F.interpolate(
                prompt.masks_logits[None].float(), size=(ims, ims),
                mode="bilinear", align_corners=False,
            )
            mask_inputs = (mm > 0.0).float() if prompt.masks_logits.dtype == torch.bool else mm

        if is_new:
            output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
            state.tracklet_mgr.spawn(obj_id, frame_idx)
        else:
            sel = state.bank.select_memories(
                obj_ids=[obj_id], current_frame_idx=frame_idx,
                max_conditional_memories=-1,
                max_non_conditional_memories=self.tracker.num_maskmem - 1,
                max_ptr_memories=self.tracker.max_obj_ptrs_in_encoder,
                only_include_pointers_in_past=True,
            )[obj_id]
            output_dict = self._selection_to_output_dict(sel)

        out = self.tracker.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_new,
            current_vision_feats=vis,
            current_vision_pos_embeds=vpos,
            feat_sizes=feat_sizes,
            image=None,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=num_frames,
        )
        self._store_memory(state, frame_idx, obj_id, out, conditional=True)
        return obj_id, out

    # ------------------------------------------------------------------
    # Bank <-> tracker output-dict bridge
    # ------------------------------------------------------------------

    def _selection_to_output_dict(self, sel) -> dict:
        """Reconstruct the tracker's ``output_dict`` from a bank ``ObjectMemorySelection``.

        The bank already performed temporal selection (conditional kept indefinitely,
        non-conditional bounded by the forgetful window) -> the tracker conditions on exactly
        these frames (``use_memory_selection=False``). Only ``maskmem_features`` /
        ``maskmem_pos_enc`` / ``obj_ptr`` are consumed by the memory-conditioning path.
        """
        device = self.device
        output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        for memory in sel.conditional_memories:
            output_dict["cond_frame_outputs"][memory.frame_idx] = self._memory_to_out(memory, device)
        for memory in sel.non_conditional_memories:
            output_dict["non_cond_frame_outputs"][memory.frame_idx] = self._memory_to_out(memory, device)
        return output_dict

    @staticmethod
    def _memory_to_out(memory, device) -> dict:
        return {
            "maskmem_features": memory.memory_embeddings.to(device, non_blocking=True),
            "maskmem_pos_enc": [memory.memory_pos_embeddings.to(device, non_blocking=True)],
            "obj_ptr": memory.ptr.to(device, non_blocking=True),
        }

    def _store_memory(self, state, frame_idx, obj_id, out, conditional: bool) -> None:
        """Push one frame's tracker memory into the (forgetful) bank, then prune.

        ``conditional`` (seed / prompt frame) is marked by passing a prompt so the bank keeps
        it indefinitely; propagation frames are non-conditional (windowed away -> constant VRAM).
        """
        result = MaskletResult(
            masks_logits=out["pred_masks"],
            ious=out["object_score_logits"],          # placeholder (unused by the bank)
            obj_ptrs=out["obj_ptr"],
            obj_scores_logits=out["object_score_logits"],
        )
        prompts = (
            [GeometryPrompt(obj_id=obj_id, masks_logits=out["pred_masks"][0])]
            if conditional
            else []
        )
        state.bank.try_add_memories(
            frame_idx=frame_idx,
            obj_ids=[obj_id],
            memory_embeddings=out["maskmem_features"],
            memory_pos_embeddings=out["maskmem_pos_enc"][-1],
            results=result,
            prompts=prompts,
        )
        state.bank.prune_memories(obj_ids=[obj_id], current_frame_idx=frame_idx)

    def _masklet_from_lowres(self, low_mask, out, H: int, W: int) -> MaskletResult:
        """Un-squash a low-res mask (squashed 1008 space) to video res -> a ``MaskletResult``.

        Hole-fill + sprinkle-removal at the low-res grid (upstream ``build_outputs`` /
        ``_propogate_tracker_one_frame_local_gpu`` apply ``fill_holes_in_mask_scores`` before the
        resize). This cleans the OUTPUT mask only; the tracker's internal memory is untouched.
        """
        low = fill_holes_in_mask_scores(
            low_mask[None, None].float(), max_area=self.fill_hole_area
        )
        masks_logits = F.interpolate(
            low, size=(H, W), mode="bilinear", align_corners=False
        )
        if out is not None:
            ious = out["object_score_logits"]
            obj_ptrs = out["obj_ptr"]
            obj_scores = out["object_score_logits"]
        else:  # detector-spawned object on its first frame (no tracker output yet)
            ious = masks_logits.new_ones(1, 1)
            obj_ptrs = masks_logits.new_zeros(1, self.tracker.hidden_dim)
            obj_scores = masks_logits.new_zeros(1, 1)
        return MaskletResult(
            masks_logits=masks_logits,
            ious=ious,
            obj_ptrs=obj_ptrs,
            obj_scores_logits=obj_scores,
        )


class Sam3MultiplexVideoPredictor(Sam3VideoPredictor):
    """SAM 3.1 (multiplex) streaming video concept predictor (spec §8 + §10).

    SUBCLASSES :class:`Sam3VideoPredictor` to REUSE the spec §10 streaming machinery verbatim --
    the obj-id allocator (``_alloc_obj_id``), ``remove_object``, the Task-7 association +
    ``TrackletManager`` lifecycle (``_associate_and_update``), the forgetful bank, concept
    management (``set_concept`` / ``encode_text``), and output un-squashing
    (``_masklet_from_lowres``) -- but swaps in the SAM 3.1 components: the M1 multiplex vision
    encoder (a tri-neck: detection / interactive / propagation, run ONCE per frame), the SAM 3.1
    text tower (base ``Sam3TextEncoder``), the M2 SAM 3.1 detector
    (``supervise_joint_box_scores`` -> joint score), and the M1 :class:`Sam3MultiplexTracker`.

    Multiplex mux/demux is INTERNAL to the tracker (spec §8): the loop + forgetful bank only ever
    see PER-OBJECT tensors. Each frame the active objects are packed into one
    :class:`~sam.modeling.multiplex.MultiplexState` (``num_buckets = ceil(N/K)``, K=16); the SAM
    head + decoupled memory attention run JOINTLY at ``batch = num_buckets``; the K-slot decode is
    DEMUXED back to per-object BEFORE it is stored in the (per-object) bank, and the per-object
    object-pointers are re-MUXED on read (so memory conditioning is bucket-space again). The
    inherited data-space aliases (``encode_memory`` / ``condition_on_memories`` / ``decode``) are
    NOT used -- the loop calls the multiplex ``track_step`` directly (they would need a
    ``multiplex_state`` and raise ``TypeError`` otherwise).

    Scope (the committed parity scenario, like the golden): objects are seeded together on the
    cond frame and co-tracked thereafter (no mid-stream spawn/despawn mixed into one decode), so a
    single ``MultiplexState`` per frame + the per-object bank reassembly are exact. Mid-stream
    spawn-alongside-existing (the upstream dynamic add-object plumbing M1 stripped) is out of
    scope for this gate.
    """

    def encode_image(self, x: torch.Tensor):
        """One trunk pass -> the THREE sam3.1 pyramids (tri-neck), scalp applied to each.

        Returns ``(det_feats, det_pos, prop_feats, prop_pos, int_feats, int_pos)``:
        ``det_*`` (detection ``convs``) feed the detector; ``prop_*`` (``sam2_convs`` /
        propagation) feed the tracker's per-frame propagation; ``int_*`` (``interactive_convs``)
        feed the tracker's cond-frame interactive object-pointer head.
        """
        nb = self.vision_encoder.vision_backbone
        det_f, det_p, prop_f, prop_p, int_f, int_p = nb.forward_all(x)
        s = self.vision_encoder.scalp
        if s > 0:
            det_f, det_p = det_f[:-s], det_p[:-s]
            prop_f, prop_p = prop_f[:-s], prop_p[:-s]
            int_f, int_p = int_f[:-s], int_p[:-s]
        return det_f, det_p, prop_f, prop_p, int_f, int_p

    def _placeholder_concept(self):
        """A cached placeholder ConceptState for box-only prompts (no text set).

        Upstream keeps the TEXT slot at the literal ``"<text placeholder>"`` for a
        box-only frame (the box drives detection via the geometric slot).
        """
        if getattr(self, "_geo_concept", None) is None:
            text = "<text placeholder>"
            emb, mask = self.encode_text(ConceptPrompt(text))
            self._geo_concept = ConceptState(0, ConceptPrompt(text), emb, None, mask)
        return self._geo_concept

    def _detect(self, det_feats, det_pos, concept: ConceptState, geo=None) -> "Sam3DetectionResult":
        """Run the SAM 3.1 detector at the tracker's low-res mask grid, joint-score post-proc.

        The SAM 3.1 detector folds presence into ``pred_logits`` (``supervise_joint_box_scores``),
        so the score is ``sigmoid(pred_logits)`` directly -- the base ``Sam3DetrDetector.detect``
        would multiply by presence a SECOND time. Mirrors M2's ``Sam3MultiplexPredictor.predict``
        post-processing but emits masks at ``(low_res_mask_size, low_res_mask_size)`` (the squashed
        tracker grid the loop seeds/associates/outputs at). Boxes are unused by the loop
        (association is mask-IoU) so they are returned as zeros.
        """
        from sam.results import Sam3DetectionResult

        m = self.tracker.low_res_mask_size
        out = self.detector.forward_grounding(
            det_feats, det_pos, concept.text_emb, concept.text_mask, geo=geo,
        )
        pred_logits = out["pred_logits"]            # (P, nq, 1) JOINT (presence folded)
        pred_masks = out["pred_masks"]              # (P, nq, h, w) logits
        presence_logit = out["presence_logit_dec"]  # (P, 1)
        out_probs = pred_logits.sigmoid().squeeze(-1)  # (P, nq) joint, no extra presence mult
        keep = out_probs > 0.5
        kept_probs = out_probs[keep]
        kept_masks = pred_masks[keep]               # (N, h, w)
        masks_logits = F.interpolate(
            kept_masks.unsqueeze(1).float(), (m, m), mode="bilinear", align_corners=False,
        ).squeeze(1)  # (N, m, m) logits (binarise at 0)
        n = masks_logits.shape[0]
        presence = float(presence_logit.float().sigmoid().reshape(-1)[0])
        return Sam3DetectionResult(
            masks_logits=masks_logits,
            boxes=masks_logits.new_zeros((n, 4)),
            scores=kept_probs,
            presence=presence,
            instance_ids=torch.arange(n, device=masks_logits.device),
        )

    # ------------------------------------------------------------------
    # Streaming forward (spec §10 data-flow; multiplex track_step internal)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def forward(
        self,
        state: Sam3VideoPredictorState,
        frame_idx: int,
        frame,
        geometry_prompts: list[GeometryPrompt] | None = None,
    ) -> dict[int, MaskletResult]:
        geometry_prompts = geometry_prompts or []
        if geometry_prompts:
            self._check_mux_geometry(geometry_prompts)
        device = self.device
        H, W = state.video_hw
        state.num_frames_processed += 1
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            # encode the frame ONCE (the sam3.1 video regime: PIL TF.resize + float16 loader)
            x = preprocess_to_1008_video(frame, device=device)
            det_f, det_p, prop_f, prop_p, int_f, int_p = self.encode_image(x)
            bf_prop = self._mux_backbone_features(prop_f, prop_p, self.tracker.sam_mask_decoder)
            bf_int = self._mux_backbone_features(
                int_f, int_p, self.tracker.interactive_sam_mask_decoder
            )
            num_frames = frame_idx + 1
            concept = state.concepts[0] if state.concepts else None
            box_geo, point_prompts = self._split_and_pack_geometry(
                geometry_prompts, (H, W), device
            )

            # 1) joint-propagate existing tracklets (multiplex track_step at batch=num_buckets)
            active_ids = sorted(state.bank.known_obj_ids)
            trk_low_masks: dict[int, torch.Tensor] = {}
            trk_results: dict[int, dict] = {}
            if active_ids:
                per_obj = self._propagate_multiplex(state, frame_idx, bf_prop, num_frames)
                for oid in state.mux_obj_ids:
                    trk_results[oid] = per_obj[oid]
                    trk_low_masks[oid] = per_obj[oid]["pred_masks"][0, 0].float()

            # 2) detection (GATED) — concept text and/or a box in the geometric slot; a
            # box-only prompt uses the '<geometric>' placeholder concept.
            det_concept = concept or (self._placeholder_concept() if box_geo else None)
            det = (
                self._detect(det_f, det_p, det_concept, geo=box_geo)
                if det_concept is not None else None
            )

            # 3) associate det<->trk + spawn / confirm / kill (Task 7, per-object, unchanged)
            new_objects: list[tuple[int, int]] = []
            if det is not None:
                new_objects = self._associate_and_update(
                    state, det, active_ids, trk_low_masks, trk_results
                )

            # 4) seed (first frame) or grow (mid-stream) the new detector instances
            if new_objects:
                self._detector_add(
                    state, frame_idx, det, new_objects, bf_int, bf_prop, num_frames
                )

            # 5) build outputs: existing -> tracker masks; new dets -> detector masks
            results: dict[int, MaskletResult] = {}
            for oid in active_ids:
                if oid not in state.bank.known_obj_ids:
                    continue  # killed this frame
                results[oid] = self._masklet_from_lowres(
                    trk_low_masks[oid], trk_results[oid], H, W
                )
            for oid, det_idx in new_objects:
                results[oid] = self._masklet_from_lowres(
                    det.masks_logits[det_idx].float(), None, H, W
                )

            # 6) interactive point clicks: seed (first frame) or grow (mid-stream /
            # co-seed after a detector seed on this frame)
            if point_prompts:
                results.update(self._clicks_add(
                    state, frame_idx, point_prompts, bf_int, bf_prop, num_frames
                ))
            return self._filter_visible(state, results)

    def _split_and_pack_geometry(self, geometry_prompts, hw, device):
        """Split prompts into a packed box-geo dict (biases detection via the
        GEOMETRIC slot) and the list of point prompts (interactive click path)."""
        box_prompts = [p for p in geometry_prompts if p.boxes is not None]
        point_prompts = [p for p in geometry_prompts if p.points_coords is not None]
        box_geo = None
        if box_prompts:
            packed = [_pack_geometry(p, hw, device) for p in box_prompts]
            box_geo = {
                "box_coords": torch.cat([g["box_coords"] for g in packed], 0),
                "box_labels": torch.cat([g["box_labels"] for g in packed], 0),
            }
        return box_geo, point_prompts

    def _detector_add(self, state, frame_idx, det, new_objects, bf_int, bf_prop, num_frames):
        """Seed (first frame) or grow (mid-stream) the mux state with detector
        instances. Output masks are built by the caller from the detector logits."""
        if state.mux_state is None:
            self._seed_multiplex(
                state, frame_idx, new_objects, det, bf_int, bf_prop, num_frames
            )
        else:
            new_masks, new_ids = self._det_masks_for_seed(det, new_objects)
            self._grow_mux_state(
                state, frame_idx, new_masks, False, new_ids, bf_int, bf_prop
            )

    def _clicks_add(self, state, frame_idx, point_prompts, bf_int, bf_prop, num_frames) -> dict:
        """Seed (first frame) or grow (mid-stream) the mux state from point clicks;
        return per-object masklets for the clicked objects."""
        if state.mux_state is None:
            return self._seed_points_multiplex(
                state, frame_idx, point_prompts, bf_int, bf_prop, num_frames
            )
        click_masks, click_ids = self._click_masks_multiplex(
            point_prompts, state.video_hw, bf_int
        )
        return self._grow_mux_state(
            state, frame_idx, click_masks, True, click_ids, bf_int, bf_prop
        )

    def _purge_removed(self, state, obj_id: int) -> None:
        """Free the object's live mux slot (if seeded) before the base bank purge."""
        if state.mux_state is not None and obj_id in (state.mux_obj_ids or []):
            self._shrink_mux_state(state, obj_id)
        super()._purge_removed(state, obj_id)

    def _shrink_mux_state(self, state, obj_id: int) -> None:
        """Drop a removed object from the live mux state (frees its bucket slot).

        Marks the object's slot removed in the ``MultiplexState`` (``demux`` then
        excludes it) and drops it from the obj-id map. The threaded bucket-space
        memory keeps the (now-removed) slot; ``demux`` skips it, so per-object views
        stay aligned to ``mux_obj_ids``.
        """
        idx = state.mux_obj_ids.index(obj_id)
        state.mux_state.remove_objects([idx])
        state.mux_obj_ids = [o for o in state.mux_obj_ids if o != obj_id]

    # ------------------------------------------------------------------
    # Multiplex helpers (mux/demux internal; bank sees per-object)
    # ------------------------------------------------------------------

    def _mux_backbone_features(self, feats, pos, decoder) -> dict:
        """Project the two hi-res pyramid levels (``conv_s0``/``conv_s1`` of ``decoder``) and
        flatten to ``(HW, 1, C)`` -- the ``backbone_features_*`` dict the multiplex ``track_step``
        consumes (batch=1; the multiplex tracker expands to ``num_buckets`` internally)."""
        fpn = list(feats)
        fpn[0] = decoder.conv_s0(fpn[0])
        fpn[1] = decoder.conv_s1(fpn[1])
        feat_sizes = [(f.shape[-2], f.shape[-1]) for f in fpn]
        vis = [f.flatten(2).permute(2, 0, 1) for f in fpn]
        vpos = [p.flatten(2).permute(2, 0, 1) for p in pos]
        return {
            "vision_feats": vis,
            "vision_masks": [None] * len(vis),
            "vision_pos_embeds": vpos,
            "feat_sizes": feat_sizes,
        }

    def _propagate_multiplex(self, state, frame_idx, bf_prop, num_frames) -> dict:
        """Joint-propagate ALL tracked objects one frame; return per-object output views.

        Reuses the persistent ``MultiplexState`` (``state.mux_state``) + the threaded
        ``output_dict`` (``state.mux_output_dict``, the multiplex tracker's native BUCKET-space
        memory). The K-slot decode is demuxed to per-object outputs (masks / score / obj_ptr); the
        full frame output (bucket-space maskmem + image features) is appended to ``output_dict``
        for the next frame, then the non-conditional window is pruned (bounded memory).
        """
        out = self.tracker.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=False,
            backbone_features_interactive=None,
            backbone_features_propagation=bf_prop,
            point_inputs=None,
            mask_inputs=None,
            output_dict=state.mux_output_dict,
            num_frames=num_frames,
            multiplex_state=state.mux_state,
        )
        state.mux_output_dict["non_cond_frame_outputs"][frame_idx] = out
        self._prune_mux_memory(state, frame_idx)
        return self._demux_outputs(out, state.mux_state, state.mux_obj_ids)

    @staticmethod
    def _check_mux_geometry(geometry_prompts) -> None:
        """Validate geometry prompts: points and boxes are supported; masks are not
        (no ``mask_encoder`` weights in either checkpoint)."""
        for prompt in geometry_prompts:
            if prompt.masks_logits is not None:
                raise NotImplementedError(
                    "mask geometry prompts are unsupported (no mask_encoder weights); "
                    "use box or point prompts"
                )
            if prompt.boxes is None and prompt.points_coords is None:
                raise ValueError("geometry prompt has neither points nor boxes")

    def _click_masks_multiplex(self, prompts, video_hw, bf_int):
        """Build point inputs from clicks and decode them into binarised masks.

        The interactive-head decode is owned by the tracker
        (:meth:`Sam3MultiplexTracker.masks_from_points`); the predictor only maps the
        prompts into the tracker's normalized ``point_inputs``.
        """
        point_inputs, new_ids = _build_mux_point_inputs(
            prompts, video_hw, self.tracker.image_size, self.device
        )
        return self.tracker.masks_from_points(point_inputs, bf_int), new_ids

    def _masklets_from_demux(self, out, mux_state, demux_ids, height, width, return_ids=None):
        """Demux a joint ``track_step`` output into per-object ``MaskletResult``s.

        ``demux_ids`` is the full object order of ``mux_state`` (needed for correct
        slicing); ``return_ids`` (default: all) selects which to return.
        """
        per_obj = self._demux_outputs(out, mux_state, demux_ids)
        ids = demux_ids if return_ids is None else return_ids
        return {
            oid: self._masklet_from_lowres(
                per_obj[oid]["pred_masks"][0, 0].float(), per_obj[oid], height, width
            )
            for oid in ids
        }

    def _seed_mux_state(self, state, frame_idx, new_ids, *, point_inputs=None,
                        mask_inputs=None, bf_int, bf_prop, num_frames):
        """Build the persistent ``MultiplexState`` from ONE init-cond ``track_step``.

        Shared by the detector-mask seed (:meth:`_seed_multiplex`) and the click seed
        (:meth:`_seed_points_multiplex`): allocate the state, run the joint cond-frame
        track_step (point OR mask inputs), record the obj-id map + cond-frame output,
        and register the ids on the bank. Returns the raw ``out``.
        """
        mux_state = self.tracker.multiplex_controller.get_state(
            len(new_ids), self.device, torch.float32, random=False
        )
        out = self.tracker.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=True,
            backbone_features_interactive=bf_int,
            backbone_features_propagation=bf_prop,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict={"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
            num_frames=num_frames,
            multiplex_state=mux_state,
        )
        state.mux_state = mux_state
        state.mux_obj_ids = list(new_ids)
        state.mux_output_dict["cond_frame_outputs"][frame_idx] = out
        for obj_id in new_ids:
            state.bank.known_obj_ids.add(obj_id)
        return out

    def _det_masks_for_seed(self, det, new_objects):
        """Binarised ``(n, 1, ims, ims)`` masks + ids for detector-spawned objects."""
        ims = self.tracker.input_mask_size
        masks = []
        for _oid, det_idx in new_objects:
            m = F.interpolate(
                det.masks_logits[det_idx][None, None].float(), size=(ims, ims),
                mode="bilinear", align_corners=False,
            )
            masks.append((m > 0.0).float())
        return torch.cat(masks, dim=0), [oid for oid, _ in new_objects]

    def _seed_multiplex(self, state, frame_idx, new_objects, det, bf_int, bf_prop, num_frames):
        """Seed new detector instances jointly (multiplex ``mask_as_output`` cond frame)."""
        mask_inputs, new_ids = self._det_masks_for_seed(det, new_objects)
        self._seed_mux_state(
            state, frame_idx, new_ids, mask_inputs=mask_inputs,
            bf_int=bf_int, bf_prop=bf_prop, num_frames=num_frames,
        )

    def _seed_points_multiplex(
        self, state, frame_idx, prompts, bf_int, bf_prop, num_frames
    ) -> dict:
        """Seed click-prompted objects on the seed frame (interactive VOS, no text)."""
        height, width = state.video_hw
        point_inputs, new_ids = _build_mux_point_inputs(
            prompts, (height, width), self.tracker.image_size, self.device
        )
        out = self._seed_mux_state(
            state, frame_idx, new_ids, point_inputs=point_inputs,
            bf_int=bf_int, bf_prop=bf_prop, num_frames=num_frames,
        )
        return self._masklets_from_demux(out, state.mux_state, new_ids, height, width)

    def _grow_mux_state(
        self, state, frame_idx, new_masks, is_mask_from_pts, new_ids, bf_int, bf_prop
    ) -> dict:
        """Add new objects to the live mux state at ``frame_idx`` (forward-only).

        The current frame's output must already be stored in
        ``mux_output_dict["non_cond_frame_outputs"][frame_idx]`` (propagation) or,
        for a seed-frame co-seed, in ``cond_frame_outputs[frame_idx]``. After growth
        the frame is a conditioning frame (re-keyed so ``_prune_mux_memory`` keeps
        it). Returns per-object masklets for ``new_ids``.
        """
        height, width = state.video_hw
        prev = state.mux_output_dict["non_cond_frame_outputs"].get(frame_idx)
        was_cond = prev is None
        if was_cond:  # seed-frame co-seed: grow the cond-frame output in place
            prev = state.mux_output_dict["cond_frame_outputs"][frame_idx]
        out, _ = self.tracker.add_new_masks_to_existing_state(
            prev, new_masks, bf_int, bf_prop, state.mux_state, is_mask_from_pts
        )
        if not was_cond:
            state.mux_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
            state.mux_output_dict["cond_frame_outputs"][frame_idx] = out
        state.mux_obj_ids = state.mux_obj_ids + list(new_ids)
        for obj_id in new_ids:
            state.bank.known_obj_ids.add(obj_id)
        return self._masklets_from_demux(
            out, state.mux_state, state.mux_obj_ids, height, width, return_ids=new_ids
        )

    def _demux_outputs(self, out, mux_state, obj_ids) -> dict:
        """Slice the joint track_step output into per-object dicts. ``pred_masks`` /
        ``object_score_logits`` are already demuxed; ``obj_ptr`` is muxed in ``out`` so it is
        demuxed here. Indexed in the ``MultiplexState`` object order == ``obj_ids`` order."""
        per_obj_ptr = mux_state.demux(out["obj_ptr"])  # (N, C)
        per_obj = {}
        for i, oid in enumerate(obj_ids):
            per_obj[oid] = {
                "pred_masks": out["pred_masks"][i : i + 1],
                "pred_masks_high_res": out["pred_masks_high_res"][i : i + 1],
                "object_score_logits": out["object_score_logits"][i : i + 1],
                "obj_ptr": per_obj_ptr[i : i + 1],
            }
        return per_obj

    @staticmethod
    def _prune_mux_memory(state, frame_idx: int) -> None:
        """Drop non-conditional frame memories outside the forgetful window (cond kept
        indefinitely) so the threaded bucket-space ``output_dict`` stays bounded vs clip length."""
        window = getattr(state.bank, "memory_window_size", 7)
        nc = state.mux_output_dict["non_cond_frame_outputs"]
        for t in list(nc):
            if t < frame_idx - window:
                del nc[t]
