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
import torch.nn.functional as F

from sam.modeling.association import associate_det_trk
from sam.modeling.association.tracklet import TrackletManager
from sam.modeling.memory.bank import ObjectMemoryBank
from sam.modeling.memory.forgetful import ForgetfulObjectMemoryBank
from sam.modeling.tracking.sam3_tracker_utils import fill_holes_in_mask_scores
from sam.prompts import ConceptPrompt, GeometryPrompt
from sam.results import MaskletResult
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
            return results

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

        new_dets, unmatched_tracks, det2track, _scores = associate_det_trk(
            det_masks=det_masks,
            track_masks=trk_stack,
            iou_threshold=0.5,
            iou_threshold_trk=0.5,
            det_scores=det.scores,
            new_det_thresh=0.0,
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

        # spawn new tracklets (allocator-issued ids) for the new detections
        new_objects: list[tuple[int, int]] = []
        for det_idx in new_dets:
            oid = self._alloc_obj_id(state)
            new_objects.append((oid, int(det_idx)))
            state.tracklet_mgr.spawn(oid)
        new_ids = {oid for oid, _ in new_objects}

        # make sure every live track is registered before stepping the lifecycle
        for o in active_ids:
            if o not in state.tracklet_mgr._tracks:
                state.tracklet_mgr.spawn(o)
        state.tracklet_mgr.step(matched_track_ids, new_ids)

        # kill dead tracklets: purge bank + manager so VRAM stays bounded
        for oid in list(state.tracklet_mgr._tracks):
            if state.tracklet_mgr.is_dead(oid):
                self.remove_object(state, oid)
        return new_objects

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
            state.tracklet_mgr.spawn(obj_id)
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
