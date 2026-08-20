# SPDX-License-Identifier: LicenseRef-SAM
"""SAM 3 image concept predictor + video predictor skeleton.

Provides:
  Sam3Predictor             — IMAGE concept predictor (Task 8). OWNS the shared PE vision
                              encoder + text tower + DETR detector and exposes
                              ``process(image, concept=...) -> Sam3DetectionResult``.
  ConceptState              — the session's encoded concept, stored in Sam3VideoPredictorState.
  Sam3VideoPredictorState   — streaming-session state (bank + concept + frame counter).
  Sam3VideoPredictor        — streaming skeleton class (Task 6 guard); its ``encode_text``
                              is wired to the real text tower in the streaming task, and
                              takes the full ``ConceptPrompt``.

Guard contract (spec §9):
  set_concept checks ``state.started`` THEN "already set" BEFORE calling encode_text,
  so the two error-path tests never reach encoding and stay CPU-only / checkpoint-free.
"""
from __future__ import annotations

import contextlib
import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam.modeling.association import associate_det_trk
from sam.modeling.association.tracklet import TrackletManager
from sam.modeling.memory.bank import ObjectMemoryBank
from sam.modeling.memory.forgetful import ForgetfulObjectMemoryBank
from sam.models.video_session import VideoSession, validate_video_hw
from sam.modeling.tracking.sam3_tracker_utils import fill_holes_in_mask_scores
from sam.prompts import ConceptPrompt, GeometryPrompt, PromptRoute
from sam.results import Emit, MaskletResult
from sam.utils.sam3_transforms import preprocess_to_1008, preprocess_to_1008_video

if TYPE_CHECKING:
    from sam.results import Sam3DetectionResult


# The caption upstream encodes when a prompt carries geometry but no text. Lineage
# specific: the base path selects TEXT_ID_FOR_VISUAL, i.e. the literal "visual"
# (sam3_video_inference.py:868-876), while the multiplex path has no such else-branch
# (sam3_multiplex_tracking.py:1698-1705) and leaves text_ids at 0, i.e. slot 0.
@dataclasses.dataclass
class EncodedImage:
    """One image encoded once: the detector's pyramid and the tracker's SAM 2 view.

    Returned by :meth:`Sam3Predictor.encode` and accepted by
    :meth:`Sam3Predictor.process` in place of the array, so repeated prompting of one
    image pays for the vision encoder once.
    """

    det_feats: list
    det_pos: list
    sam2_feats: list
    sam2_pos: list
    image_hw: tuple
    dtype: "torch.dtype"


def _split_image_geometry(geometry):
    """Split ``geometry`` into ``(exemplars, objects)`` by each prompt's own route.

    Args:
        geometry: one :class:`GeometryPrompt`, a list of them, or None.

    Returns:
        ``(exemplars, objects)``: a single merged DETECTOR-route prompt (or None) and
        the list of TRACKER-route prompts.

    Raises:
        ValueError: if two exemplars are passed -- one prompt can carry several points
            and boxes, so merge them there.
    """
    if geometry is None:
        return None, []
    prompts = [geometry] if isinstance(geometry, GeometryPrompt) else list(geometry)
    exemplars = [p for p in prompts if p.to_detector]
    objects = [p for p in prompts if not p.to_detector]
    if len(exemplars) > 1:
        raise ValueError(
            "pass one exemplar prompt carrying every point/box you want to bias with, "
            "not several GeometryPrompts"
        )
    return (exemplars[0] if exemplars else None), objects


BOX_ONLY_CAPTION_BASE = "visual"
BOX_ONLY_CAPTION_MUX = "<text placeholder>"


class Sam3Predictor(nn.Module):
    """SAM 3 image concept predictor (base / per-object, text-only path).

    Composes (OWNS) the shared PE vision encoder, the text tower, and the DETR detector
    (spec §5/§7): the vision encoder runs ONCE per image and its features are injected into
    the detector. Built by :func:`sam.build_sam.build_sam3` (hydra-compose
    ``configs/sam3/sam3.yaml`` -> instantiate -> strict-load the ``detector.*`` subtree).

    ``process(image, concept=...)`` returns a :class:`~sam.results.Sam3DetectionResult`
    with the per-instance masks / boxes / scores for that concept, and
    ``process(image, geometry=...)`` runs the owned tracker to return exactly the objects
    you marked (SAM 2's contract). Predictors built without a tracker -- EfficientSAM3,
    the multiplex -- answer the first only.
    """

    def __init__(self, vision_encoder: nn.Module, text_encoder: nn.Module,
                 detector: nn.Module, tracker: nn.Module | None = None):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.detector = detector
        self.tracker = tracker
        self.fill_hole_area = 16

    # Sentinel for predict/process: "detect under the box-only caption".
    PLACEHOLDER = object()
    BOX_ONLY_CAPTION = BOX_ONLY_CAPTION_BASE

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _resolve_concept(self, concept) -> ConceptPrompt:
        """A phrase, a :class:`ConceptPrompt`, or :attr:`PLACEHOLDER` -> what to encode.

        :attr:`PLACEHOLDER` is this lineage's box-only caption (:attr:`BOX_ONLY_CAPTION`),
        which is what upstream encodes when geometry arrives with no text. It has to be
        asked for by name: adopting it silently would turn "find what I marked" into
        "search for this caption", and the caption differs per lineage -- passing the
        wrong one returns nothing at all.
        """
        if concept is self.PLACEHOLDER:
            return ConceptPrompt(self.BOX_ONLY_CAPTION)
        if isinstance(concept, str):
            return ConceptPrompt(concept)
        if concept is None:
            raise ValueError(
                "predict needs a concept: pass a phrase, a ConceptPrompt, or "
                f"{type(self).__name__}.PLACEHOLDER for the box-only caption"
            )
        return concept

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
        """Embed the concept's text (spec §6/§9) -> ``(text_emb, text_mask)``.

        Returns what the base per-object detector consumes: ``text_emb``
        (seq, 1, d_model) and ``text_mask`` (1, seq) True-where-PAD.

        No negative phrases: upstream SAM 3 encodes an optional ``additional_text``
        alongside the captions (``SAM3VLBackbone.forward_text``) but no caller ever
        passes it and nothing reads the resulting ``additional_text_features`` — the
        positives-only slice is what reaches the detector. Verified against
        facebookresearch/sam3 @ ``8f0b7f4`` (2026-08-13). No head takes a
        negative-caption input, so honouring negatives would mean inventing untrained
        behaviour.
        """
        # forward returns (text_attention_mask (batch, seq) True-where-PAD,
        #                  text_memory_resized (seq, batch, d_model), inputs_embeds_T)
        text_attention_mask, text_memory_resized, _ = self.text_encoder(
            [self._resolve_concept(concept).text], device=self.device
        )
        return text_memory_resized, text_attention_mask

    def detect(
        self,
        feats: List[torch.Tensor],
        pos: List[torch.Tensor],
        text_emb: torch.Tensor,
        text_mask: torch.Tensor,
        image_hw: Tuple[int, int],
        confidence_threshold: float = 0.5,
        geo: Optional[dict] = None,
    ) -> "Sam3DetectionResult":
        """Ground the encoded text into per-object detections via the owned detector."""
        return self.detector.detect(
            feats, pos, text_emb, text_mask, image_hw,
            confidence_threshold=confidence_threshold, geo=geo,
        )

    # ------------------------------------------------------------------
    # Image prediction (spec §10): one verb, two paths.
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def process(
        self, image, concept=None, geometry=None, confidence_threshold: float = 0.5,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "Sam3DetectionResult":
        """Segment ``image``, either by concept or by pointing at objects.

        The two paths are the two halves of the model, and which one runs is decided by
        what you pass -- never guessed:

        * ``concept=`` runs detection: every instance the phrase matches comes back,
          with ids minted by the detector. ``geometry=`` may carry
          :meth:`~sam.prompts.GeometryPrompt.exemplar_box` /
          :meth:`~sam.prompts.GeometryPrompt.exemplar_point` to bias that search.
        * ``geometry=`` alone, holding ``click`` / ``box`` / ``mask`` prompts, runs the
          tracker: exactly the objects you marked come back, under the ``obj_id`` you
          chose. This is SAM 2's contract on SAM 3's weights.

        Args:
            image: ``(H, W, 3)`` uint8 RGB array, or an :class:`EncodedImage` from
                :meth:`encode` when you are prompting the same image repeatedly.
            concept: a phrase, a :class:`~sam.prompts.ConceptPrompt`, or
                :attr:`PLACEHOLDER` for this lineage's box-only caption.
            geometry: one :class:`~sam.prompts.GeometryPrompt` or a list of them.
            confidence_threshold: detection threshold (concept path only).
            dtype: autocast dtype. ``torch.float32`` disables mixed precision (the
                EfficientSAM3 student only; the PE-ViTDet SAM 3 model requires bf16).

        Returns:
            A :class:`~sam.results.Sam3DetectionResult` either way. On the object path
            ``scores`` are the tracker's object-score probabilities, ``instance_ids``
            are the ``obj_id`` values you passed, and ``presence`` is None -- there is no
            concept to be present.

        Raises:
            ValueError: if neither ``concept`` nor tracker geometry is given, or if both
                are (their ids would collide in one result -- make two calls), or if
                exemplar geometry arrives without a concept.
            NotImplementedError: for tracker prompts on a predictor built without a
                tracker.
        """
        enc = image if isinstance(image, EncodedImage) else self.encode(image, dtype)
        exemplars, objects = _split_image_geometry(geometry)

        if concept is None and exemplars is not None:
            raise ValueError(
                "exemplar_box / exemplar_point bias a concept search, so they need a "
                f"concept: pass a phrase or {type(self).__name__}.PLACEHOLDER. To select "
                "the object you marked instead, use GeometryPrompt.box / .click"
            )
        if concept is None and not objects:
            raise ValueError(
                "process needs something to do: pass concept=<phrase> (or PLACEHOLDER) "
                "to detect, and/or geometry=GeometryPrompt.click/box/mask to select "
                "objects"
            )
        if concept is not None and objects:
            raise ValueError(
                "concept detection and object selection return different ids -- the "
                "detector mints 0..N-1 while click/box/mask carry the obj_id you chose, "
                "and one result cannot hold both. Call process twice"
            )
        if objects:
            return self._select_objects(enc, objects)
        return self._detect_encoded(
            enc.det_feats, enc.det_pos, enc.image_hw, concept,
            confidence_threshold=confidence_threshold, geometry=exemplars, dtype=enc.dtype,
        )

    @torch.inference_mode()
    def encode(self, image, dtype: torch.dtype = torch.bfloat16) -> "EncodedImage":
        """Encode ``image`` once, for repeated :meth:`process` calls.

        The vision encoder is most of a ``process`` call and depends on nothing in the
        prompt, so sweeping a threshold or comparing exemplars should pay for it once.
        Hand the result back to :meth:`process` in place of the array.
        """
        with self._autocast(dtype):
            x = preprocess_to_1008(image, device=self.device)
            det_f, det_p, sam2_f, sam2_p = self._encode_views(x)
        return EncodedImage(
            det_feats=det_f, det_pos=det_p, sam2_feats=sam2_f, sam2_pos=sam2_p,
            image_hw=(int(image.shape[0]), int(image.shape[1])), dtype=dtype,
        )

    def _encode_views(self, x: torch.Tensor):
        """One trunk pass -> the detector pyramid AND the tracker's SAM 2 view.

        The encoder is built with ``add_sam2_neck=True``, so both views come out of the
        same forward -- which is what lets one predictor answer both paths without
        running the heavy ViT twice.
        """
        sam3_f, sam3_p, sam2_f, sam2_p = self.vision_encoder.vision_backbone(x)
        s = self.vision_encoder.scalp
        if s > 0:
            sam3_f, sam3_p = sam3_f[:-s], sam3_p[:-s]
            sam2_f, sam2_p = sam2_f[:-s], sam2_p[:-s]
        return sam3_f, sam3_p, sam2_f, sam2_p

    def _autocast(self, dtype: torch.dtype):
        """Autocast under *dtype*; float32 means no mixed precision at all."""
        # float32 → nullcontext (no mixed-precision); any other dtype → autocast.
        if dtype == torch.float32:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    def _detect_encoded(
        self, feats, pos, image_hw, concept, confidence_threshold: float = 0.5,
        geometry=None, dtype: torch.dtype = torch.bfloat16,
    ) -> "Sam3DetectionResult":
        """Ground ``concept`` in already-encoded features (spec §10, second half)."""
        with self._autocast(dtype):
            text_emb, text_mask = self.encode_text(concept)
            geo = _pack_geometry(geometry, image_hw, self.device)
            return self.detect(
                feats, pos, text_emb, text_mask, image_hw,
                confidence_threshold=confidence_threshold, geo=geo,
            )

    def _select_objects(self, enc: "EncodedImage", prompts) -> "Sam3DetectionResult":
        """Run the tracker on one image: one mask per prompt, under the caller's ids.

        A still image is a one-frame stream, so every prompt is an initial conditioning
        frame and there is no memory to carry: the same ``track_step`` the video
        predictor runs on frame 0, minus the bank.
        """
        from sam.results import Sam3DetectionResult

        if self.tracker is None:
            raise NotImplementedError(
                "this predictor was built without a tracker, so it can only detect "
                "concepts; build_sam3 loads one, build_sam3_multiplex does not"
            )
        H, W = enc.image_hw
        masks, scores, ids = [], [], []
        with self._autocast(enc.dtype):
            vis, vpos, feat_sizes = self._prepare_tracker_feats(enc.sam2_feats, enc.sam2_pos)
            for prompt in prompts:
                out = self._track_one(prompt, vis, vpos, feat_sizes, (H, W))
                low = out["pred_masks"][0, 0].float()
                low = fill_holes_in_mask_scores(low[None, None], max_area=self.fill_hole_area)
                masks.append(
                    F.interpolate(low, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
                )
                scores.append(out["object_score_logits"].float().reshape(-1)[0].sigmoid())
                ids.append(prompt.obj_id)

        masks_logits = torch.stack(masks)
        return Sam3DetectionResult(
            masks_logits=masks_logits,
            boxes=_masks_to_boxes(masks_logits > 0.0),
            scores=torch.stack(scores),
            presence=None,  # no concept was asked for, so nothing to be present
            instance_ids=torch.as_tensor(ids, device=masks_logits.device),
        )

    def _track_one(self, prompt, vis, vpos, feat_sizes, image_hw):
        """One prompt -> one ``track_step`` on a fresh conditioning frame."""
        device = self.device
        prompt = prompt.to(device)

        point_inputs = None
        points = prompt.tracker_points()
        if points is not None:
            raw_coords, labels = points
            coords = _normalized_points(
                prompt, image_hw, device, coords=raw_coords
            ) * self.tracker.image_size
            point_inputs = {"point_coords": coords[None], "point_labels": labels.to(device)[None]}

        mask_inputs = None
        if prompt.masks_logits is not None:
            ims = self.tracker.input_mask_size
            mm = F.interpolate(
                prompt.masks_logits[None].float(), size=(ims, ims),
                mode="bilinear", align_corners=False,
            )
            mask_inputs = (mm > 0.0).float() if prompt.masks_logits.dtype == torch.bool else mm

        return self.tracker.track_step(
            frame_idx=0,
            is_init_cond_frame=True,
            current_vision_feats=vis,
            current_vision_pos_embeds=vpos,
            feat_sizes=feat_sizes,
            image=None,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict={"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
            num_frames=1,
        )

    def _prepare_tracker_feats(self, sam2_feats, sam2_pos):
        """Project the two hi-res SAM 2 levels and flatten to ``(HW, B, C)``.

        The same projection :class:`Sam3VideoPredictor` runs per frame; shared by every
        prompt in one call, since the base tracker has no cross-object attention.
        """
        fpn = list(sam2_feats)
        fpn[0] = self.tracker.sam_mask_decoder.conv_s0(fpn[0])
        fpn[1] = self.tracker.sam_mask_decoder.conv_s1(fpn[1])
        feat_sizes = [(f.shape[-2], f.shape[-1]) for f in fpn]
        vis = [f.flatten(2).permute(2, 0, 1) for f in fpn]
        vpos = [p.flatten(2).permute(2, 0, 1) for p in sam2_pos]
        return vis, vpos, feat_sizes


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
    obj_ids, per_coords, per_labels = [], [], []
    for prompt in prompts:
        obj_ids.append(prompt.obj_id)
        per_coords.append(_normalized_points(prompt, video_hw, device) * image_size)
        per_labels.append(prompt.points_labels.to(device).to(torch.int32))
    max_points = max(labels.shape[0] for labels in per_labels)
    n = len(prompts)
    coords = torch.zeros(n, max_points, 2, device=device)
    labels = torch.full((n, max_points), -1, device=device, dtype=torch.int32)
    for i, (obj_coords, obj_labels) in enumerate(zip(per_coords, per_labels)):
        coords[i, : obj_coords.shape[0]] = obj_coords
        labels[i, : obj_labels.shape[0]] = obj_labels
    return {"point_coords": coords, "point_labels": labels}, obj_ids


def _normalized_points(prompt, image_hw, device, coords=None):
    """Tracker-bound point coords as ``(N, 2)`` in ``[0, 1]``, honouring ``is_normalized``.

    Args:
        prompt: the :class:`GeometryPrompt` the coords came from (for ``is_normalized``).
        image_hw: ``(height, width)`` of the video.
        device: device to normalize on.
        coords: coords to normalize; defaults to ``prompt.points_coords``. Pass the
            output of :meth:`GeometryPrompt.tracker_points` to include box corners.
    """
    coords = (prompt.points_coords if coords is None else coords).to(device).float()
    if prompt.is_normalized:
        return coords
    height, width = image_hw
    return coords / torch.tensor([width, height], device=device)


def _pack_geometry(prompt, image_hw, device):
    """Pack a DETECTOR-route ``GeometryPrompt`` into the detector's geometry inputs.

    TRACKER-route prompts are rejected rather than reinterpreted: an image predictor
    owns no tracker, so silently treating ``GeometryPrompt.box`` as a detector
    exemplar answers a question the caller did not ask.

    Points -> normalized xy ``(N,1,2)`` + labels ``(N,1)``; boxes xyxy (pixel, or
    normalized via ``is_normalized``) -> normalized cxcywh ``(N,1,4)`` + labels
    ``(N,1)``, defaulting to all-positive when the prompt carries no
    ``boxes_labels``. Mask geometry has no weights in either checkpoint -> raises.
    """
    if prompt is None:
        return None
    if prompt.masks_logits is not None:
        raise NotImplementedError(
            "mask geometry prompts are unsupported (no mask_encoder weights); "
            "use GeometryPrompt.exemplar_box / exemplar_point instead"
        )
    if prompt.route is not PromptRoute.DETECTOR:
        what = "box" if prompt.boxes is not None else "click"
        instead = "exemplar_box(xyxy)" if prompt.boxes is not None else "exemplar_point((x, y))"
        raise ValueError(
            f"GeometryPrompt.{what} marks ONE object for the tracker, which this "
            f"predictor does not have. To bias the concept search use "
            f"GeometryPrompt.{instead}; to select one object, open an interactive "
            f"session on the video predictor (start_session)"
        )
    h, w = image_hw
    geo = {}
    if prompt.points_coords is not None:
        geo["point_coords"] = _normalized_points(prompt, image_hw, device)[:, None, :]
        geo["point_labels"] = prompt.points_labels.to(device)[:, None]
    if prompt.boxes is not None:
        b = prompt.boxes.to(device).float()
        b = b if prompt.is_normalized else b / torch.tensor([w, h, w, h], device=device)
        cx = (b[:, 0] + b[:, 2]) / 2
        cy = (b[:, 1] + b[:, 3]) / 2
        bw = (b[:, 2] - b[:, 0]).abs()
        bh = (b[:, 3] - b[:, 1]).abs()
        geo["box_coords"] = torch.stack([cx, cy, bw, bh], -1)[:, None, :]
        geo["box_labels"] = (
            torch.ones(b.shape[0], 1, device=device)
            if prompt.boxes_labels is None
            else prompt.boxes_labels.to(device)[:, None]
        )
    return geo or None


class Sam3MultiplexPredictor(Sam3Predictor):
    """SAM 3.1 (multiplex) image concept predictor (text-only path).

    Mirrors :class:`Sam3Predictor` -- OWNS the SAM 3.1 PE vision encoder (the DETECTION
    tri-neck head ``convs``, run ONCE per image), the SAM 3.1 text tower, and the SAM 3.1
    DETR detector -- but the detector is built with ``supervise_joint_box_scores=True`` (the
    SAM 3.1 difference). It reuses the base ``encode_image`` / ``encode_text`` / ``device``
    (the text path feeds the detector exactly the text-id-0 slice == encoding the positive
    phrase alone, which equals ``image_sam31.npz``'s ``text_emb[:, 0]``).

    ``_detect_encoded`` overrides the base post-processing to the SAM 3.1 demo semantics
    (``process`` itself is inherited unchanged):
      * the score is the joint ``sigmoid(pred_logits)`` (presence is already folded into
        ``pred_logits`` by ``supervise_joint_box_scores``; NO extra presence multiply), and
      * the box is ``masks_to_boxes`` of the output mask (the multiplex demo derives boxes
        from masks), not the raw DETR ``pred_boxes``.
    Built by :func:`sam.build_sam.build_sam3_multiplex` (compose ``configs/sam3/sam3.1.yaml``
    -> instantiate -> strict-load the relevant ``detector.*`` subtree of
    ``sam3.1_multiplex.pt``).
    """

    BOX_ONLY_CAPTION = BOX_ONLY_CAPTION_MUX

    def _detect_encoded(
        self, feats, pos, image_hw, concept: ConceptPrompt,
        confidence_threshold: float = 0.5, geometry=None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "Sam3DetectionResult":
        """SAM 3.1 grounding + demo post-processing on already-encoded features.

        The grounding runs under autocast and the post-processing outside it, which is
        the split :meth:`predict` has always had -- keep it, the goldens are captured
        against these numerics.
        """
        from sam.results import Sam3DetectionResult

        device = self.device
        with self._autocast(dtype):
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
    prompt: ConceptPrompt            # original (text)
    text_emb: torch.Tensor           # encoded once (positive slice, (seq, n_pos, d))
    text_mask: torch.Tensor | None = None  # (n_pos, seq) True-where-PAD, for the detector


@dataclass
class Sam3VideoPredictorState:
    video_hw: tuple[int, int]
    bank: ObjectMemoryBank = field(default_factory=ForgetfulObjectMemoryBank)
    concept: ConceptState | None = None  # set once, before the first frame (see set_concept)
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

    def __post_init__(self):
        self.video_hw = validate_video_hw(self.video_hw)

    @property
    def started(self) -> bool:
        return self.num_frames_processed > 0


def select_emitted(
    results: dict[int, MaskletResult],
    mgr: TrackletManager,
    emit: Emit,
) -> dict[int, MaskletResult]:
    """Apply the output policy to one frame's masklets and stamp their state.

    Ids the manager does not track (click-seeded objects) always pass: upstream runs
    a click-only session through SAM 2 partial propagation, which never subjects it to
    detection-driven confirmation or hotstart. Empty masks are dropped in every mode,
    mirroring the unconditional ``mask.any()`` in upstream ``_postprocess_output``.

    Args:
        results: this frame's masklets, keyed by obj_id.
        mgr: the session's tracklet lifecycle state machine.
        emit: which objects to keep.

    Returns:
        The kept subset, each result carrying its ``tracklet_state``.
    """
    emitted = mgr.emitted_ids(emit)
    managed = mgr.managed_ids()
    kept = {}
    for obj_id, result in results.items():
        if obj_id in managed and obj_id not in emitted:
            continue
        if not bool((result.masks_logits > 0.0).any()):
            continue
        result.tracklet_state = mgr.state_of(obj_id)
        kept[obj_id] = result
    return kept


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

    # Upstream base-lineage tracklet lifecycle (model_builder.build_sam3_video_model ->
    # Sam3VideoInferenceWithInstanceInteractivity, ~746-762): no masklet confirmation and a
    # keep-alive that starts saturated, so an object stays visible through long absences.
    # The multiplex lineage overrides this -- its constants are different (see the subclass).
    LIFECYCLE = {
        "confirmation_enable": False,
        "confirmation_thresh": 3,
        "hotstart_delay": 15,
        "hotstart_unmatch_thresh": 8,
        "init_keep_alive": 30,
        "max_keep_alive": 30,
        "min_keep_alive": -1,
    }
    # caption encoded for a box-only (no text) prompt -- see _placeholder_concept
    BOX_ONLY_CAPTION = BOX_ONLY_CAPTION_BASE

    def __init__(
        self,
        vision_encoder: nn.Module | None = None,
        text_encoder: nn.Module | None = None,
        detector: nn.Module | None = None,
        tracker: nn.Module | None = None,
        emit: Emit = Emit.CONFIRMED,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.detector = detector
        self.tracker = tracker
        # Output policy (see Emit). Constant for a session in practice, so it lives on
        # the predictor rather than on every forward() call; reassign it to switch.
        self.emit = emit
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
        """Embed the concept's text -> ``(text_emb, text_mask)``.

        Mirrors :meth:`Sam3Predictor.encode_text` (including why there are no
        negative phrases).
        """
        text_attention_mask, text_memory_resized, _ = self.text_encoder(
            [concept.text], device=self.device
        )
        return text_memory_resized, text_attention_mask

    # ------------------------------------------------------------------
    # Concept management (spec §9)
    # ------------------------------------------------------------------

    def set_concept(self, state: Sam3VideoPredictorState, concept: ConceptPrompt) -> int:
        """Encode ``concept`` into ``state`` — once, before the first frame.

        A session tracks ONE concept. Upstream SAM 3 is the same: its only
        multi-concept path (``Sam3MultiplexTracking.forward``, benchmark eval)
        re-runs the whole video once per phrase and calls ``reset_state`` in
        between, offsetting the object ids to merge the runs. There is no
        cross-concept interaction anywhere — no shared association, no dedup,
        separate id spaces — so N concepts means N independent sessions:

            for phrase in phrases:
                state = Sam3VideoPredictorState(video_hw=hw)
                pred.set_concept(state, ConceptPrompt(phrase))
                for i, frame in enumerate(frames):
                    results[phrase][i] = pred(state, i, frame)

        Sharing one session across concepts would NOT be upstream-equivalent:
        ``associate_det_trk`` matches every detection against every tracklet (so
        one concept's detection could capture another's tracklet), and the
        multiplex bucket memory is a joint K-object encoding, so co-bucketing
        two concepts' objects changes their masks.

        Returns:
            The concept id, always 0 (one concept per session).

        Raises:
            RuntimeError: if a frame has already been processed, or a concept is
                already set.
        """
        if state.started:
            raise RuntimeError("concept must be set before the first frame is processed")
        if state.concept is not None:
            raise RuntimeError(
                "a concept is already set; one concept per session — run one session "
                "per concept (see set_concept's docstring)"
            )
        encoded = self.encode_text(concept)
        # The real tower returns (text_emb, text_mask); the CPU guard-test stub returns a bare
        # tensor — handle both so those tests stay checkpoint-free.
        if isinstance(encoded, tuple):
            text_emb, text_mask = encoded
        else:
            text_emb, text_mask = encoded, None
        state.concept = ConceptState(concept, text_emb, text_mask)
        return 0

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
        prompts: list[GeometryPrompt] = [],
    ) -> dict[int, MaskletResult]:
        device = self.device
        self._check_frame_hw(state, frame)
        H, W = state.video_hw
        if not state.started:
            # the caller builds the state, so it cannot know which lineage drives it
            state.tracklet_mgr.configure(**self.LIFECYCLE)
        state.num_frames_processed += 1
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            # encode the frame ONCE (shared by detector + tracker), in the VIDEO regime:
            # upstream's video path always loads frames through the image-folder loader
            # (PIL TF.resize + float16), never through the image processor
            x = preprocess_to_1008_video(frame, device=device)
            det_feats, det_pos, sam2_feats, sam2_pos = self.encode_image(x)
            vis, vpos, feat_sizes = self._prepare_tracker_feats(sam2_feats, sam2_pos)
            num_frames = frame_idx + 1  # frames seen so far (forward streaming)

            geo, tracker_prompts = self._split_and_pack_geometry(
                prompts or [], (H, W), device
            )
            concept = self._concept_for_detection(state, geo)

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

            # 2) detection (GATED: needs a concept — text, or the box-only placeholder)
            det = (
                self._detect(det_feats, det_pos, concept, geo=geo)
                if concept is not None else None
            )

            # 3) associate det<->trk, then spawn / confirm / kill via the TrackletManager
            new_objects: list[tuple[int, int]] = []  # (obj_id, det_idx)
            if det is not None:
                new_objects = self._associate_and_update(
                    state, det, active_ids, trk_low_masks, trk_results
                )
            else:
                self._advance_lifecycle(state, set(), set(), frame_idx)

            # 4) seed new detector instances into the tracker (soft-mask cond-frame memory)
            for obj_id, det_idx in new_objects:
                self._seed_object(
                    state, frame_idx, obj_id, vis, vpos, feat_sizes,
                    det.masks_logits[det_idx], num_frames,
                )

            # 5) route clicks and masks to the tracker (new obj_id -> spawn, existing ->
            # refine); boxes already went through the detector's geometric slot in step 2
            geom_results: dict[int, MaskletResult] = {}
            for prompt in tracker_prompts:
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
            return select_emitted(results, state.tracklet_mgr, self.emit)

    # ------------------------------------------------------------------
    # Streaming helpers
    # ------------------------------------------------------------------

    def _check_frame_hw(self, state: Sam3VideoPredictorState, frame) -> None:
        """Reject a frame whose size disagrees with the session's ``video_hw``.

        ``forward`` reads ``(H, W)`` from the state -- to normalize prompt coordinates
        and to resize every output mask -- and never re-reads it from the frame, which
        the preprocessor squashes to 1008x1008 regardless. So a mismatch has no other
        symptom: prompts land on the wrong pixels and masks come back at the wrong
        resolution. :class:`~sam.models.sam2_predictor.Sam2VideoPredictor` asserts the
        same invariant on its own ``(C, H, W)`` frames.

        Raises:
            ValueError: if the frame's ``(H, W)`` differs from ``state.video_hw``.
        """
        shape = getattr(frame, "shape", None)
        if shape is None or len(shape) != 3:
            return  # not an (H, W, C) array: preprocess_to_1008_video will reject it
        if tuple(shape[:2]) != tuple(state.video_hw):
            raise ValueError(
                f"frame is {tuple(shape[:2])} but this session runs at "
                f"{tuple(state.video_hw)}; SAM 3 takes (H, W) from the state to "
                "normalize prompts and resize masks, and never re-reads it from the "
                "frame. Open one session per resolution, or resize the frame."
            )

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

    # Sentinel for start_concept_session: "detect under the box-only caption".
    PLACEHOLDER = object()

    def start_session(self):
        """An INTERACTIVE :class:`~sam.models.video_session.VideoSession` (no concept).

        Detection stays off: nothing appears that you did not prompt for, and the
        obj_ids are the ones you pass on your :class:`~sam.prompts.GeometryPrompt` list.
        This is the SAM 2 interface unchanged, and the same call means the same thing
        on a :class:`~sam.models.sam2_predictor.Sam2VideoPredictor`.

        For concept-driven tracking (detection on every frame, ids minted by the
        detector) use :meth:`start_concept_session` instead.

        Returns:
            An independent session; hold several over one model and interleave. The
            video size comes from the first frame, so ``state`` is None until then.
        """
        return self._make_session(None)

    def start_concept_session(self, concept):
        """A CONCEPT :class:`~sam.models.video_session.VideoSession` (detection on).

        The concept is declared here, at birth, because it is session-scoped and
        immutable (it must be set before the first frame and never changes). Detection
        then runs on EVERY frame: every instance the concept matches is spawned and
        tracked, under an obj_id the detector mints.

        Because of that, tracker prompts (click / box / mask) in a concept session are
        for REFINING ids the session already returned -- seeding a new object with them
        is an id minefield, since detection spawns its own ids in the same call, before
        tracker prompts apply. Seed new objects in a :meth:`start_session` session.

        Args:
            concept: what detection should find -- a phrase (str or
                :class:`ConceptPrompt`), or :attr:`PLACEHOLDER` for this lineage's
                box-only caption (see :meth:`set_placeholder_concept`).

        Returns:
            An independent session; hold several over one model and interleave. The
            video size comes from the first frame, so ``state`` is None until then.

        Raises:
            ValueError: if ``concept`` is None -- that is a request for an interactive
                session, which is :meth:`start_session`.
        """
        if concept is None:
            raise ValueError(
                "start_concept_session needs a concept: pass a phrase, a ConceptPrompt, "
                f"or {type(self).__name__}.PLACEHOLDER for the box-only caption. For an "
                "interactive session with detection off, call start_session()."
            )
        if concept is self.PLACEHOLDER:
            on_state = self.set_placeholder_concept
        else:
            if isinstance(concept, str):
                concept = ConceptPrompt(concept)
            on_state = lambda st: self.set_concept(st, concept)
        return self._make_session(on_state)

    def _make_session(self, on_state):
        """Build a :class:`VideoSession` over this predictor's state class."""
        return VideoSession(
            self, lambda hw: Sam3VideoPredictorState(video_hw=hw), on_state=on_state,
        )

    def set_placeholder_concept(self, state: Sam3VideoPredictorState) -> None:
        """Run detection under this lineage's box-only caption (:attr:`BOX_ONLY_CAPTION`).

        The explicit form of what upstream does implicitly for a box-with-no-text session:
        ``"visual"`` on the base lineage, ``"<text placeholder>"`` on the multiplex. Every
        instance that caption matches gets tracked, on every frame -- if you want only the
        object you boxed, use a TRACKER-route box instead and set no concept at all.

        Args:
            state: the session to configure, before its first frame.
        """
        state.concept = self._placeholder_concept()

    def _placeholder_concept(self):
        """A cached placeholder ConceptState for box-only prompts (no text set).

        The caption is lineage-specific (:attr:`BOX_ONLY_CAPTION`): a box-only
        ``add_prompt`` selects ``TEXT_ID_FOR_VISUAL`` on the base video path
        (``sam3_video_inference.py:868-876`` -- the encoded caption is
        ``find_text_batch[1]``, the literal ``"visual"``), whereas the multiplex path
        has no such else-branch (``sam3_multiplex_tracking.py:1698-1705``) and leaves
        ``text_ids`` at 0, i.e. ``"<text placeholder>"``. The box itself drives
        detection through the geometric slot either way.
        """
        if getattr(self, "_geo_concept", None) is None:
            text = self.BOX_ONLY_CAPTION
            emb, mask = self.encode_text(ConceptPrompt(text))
            self._geo_concept = ConceptState(ConceptPrompt(text), emb, mask)
        return self._geo_concept

    def _concept_for_detection(self, state, geo):
        """The concept driving this frame's detection.

        Detection runs only for a session that asked for it, via :meth:`set_concept` or
        :meth:`set_placeholder_concept`. Upstream instead adopts the placeholder caption
        the moment a box arrives with no text (``sam3_video_inference.py:868-877``), which
        silently turns "track what I boxed" into "detect everything this caption matches,
        on every frame". We keep that behaviour but make the caller ask for it.

        Returns:
            The session's :class:`ConceptState`, or None when no concept was set (so
            detection stays gated off and only tracker prompts produce objects).

        Raises:
            ValueError: if DETECTOR-route geometry arrives with no concept set.
        """
        if geo is not None and state.concept is None:
            raise ValueError(
                "exemplar_box / exemplar_point drive detection, which needs a concept: "
                "open the session with start_concept_session(<phrase>) for your own "
                "phrase, or start_concept_session(PLACEHOLDER) for upstream's box-only "
                f"caption ({self.BOX_ONLY_CAPTION!r}) -- on an explicit state, "
                "set_concept(state, ConceptPrompt(...)) / set_placeholder_concept(state)."
                " To track only the object you marked instead, use GeometryPrompt.box / "
                ".click, which go to the tracker."
            )
        return state.concept

    def _split_and_pack_geometry(self, prompts, hw, device):
        """Split prompts by the route they take through the model.

        DETECTOR-route geometry -- ``exemplar_box`` and ``exemplar_point`` -- biases
        detection through the DETR geometric slot; TRACKER-route points, box corners and
        masks seed or refine one object through the tracker's prompt encoder. The route
        is the prompt's, so the same frame can carry both.

        Detector POINTS are ours, not upstream's: the weights are there and trained
        (``geometry_encoder.points_*`` ships in both checkpoints, and upstream samples
        point prompts during training), but upstream's video inference hardcodes
        ``point_embeddings=None`` (``sam3_video_inference.py:208-210``), so this path
        has no golden to pin it. Boxes remain the parity-tested route.

        Args:
            prompts: this frame's :class:`GeometryPrompt` list.
            hw: ``(height, width)`` of the video, for normalizing box/point coords.
            device: device to build the packed tensors on.

        Returns:
            ``(geo, tracker_prompts)``: the packed geometric-slot dict (or None), which
            may carry boxes, points or both, and the prompts to route to the tracker.

        Raises:
            NotImplementedError: if a DETECTOR-route prompt also carries a mask.
            ValueError: if a TRACKER-route box carries ``boxes_labels`` (a detector
                notion: the tracker's prompt encoder has no sign for a box).
        """
        mislabelled = [p for p in prompts if p.boxes_labels is not None and not p.to_detector]
        if mislabelled:
            raise ValueError(
                "boxes_labels signs a box for the detector; build it with "
                "GeometryPrompt.exemplar_box(xyxy, label=...), or drop the labels to "
                "seed the boxed object through the tracker with GeometryPrompt.box"
            )
        detector_prompts = [p for p in prompts if p.to_detector]
        tracker_prompts = [
            p for p in prompts
            if p.tracker_points() is not None or p.masks_logits is not None
        ]
        geo = None
        if detector_prompts:
            packed = [_pack_geometry(p, hw, device) for p in detector_prompts]
            geo = {
                key: torch.cat([g[key] for g in packed if key in g], 0)
                for key in ("box_coords", "box_labels", "point_coords", "point_labels")
                if any(key in g for g in packed)
            }
        return geo, tracker_prompts

    def _detect(self, det_feats, det_pos, concept: ConceptState, geo=None) -> "Sam3DetectionResult":
        """Run the DETR detector at the tracker's low-res mask grid (squashed space).

        ``image_hw`` is set to the tracker's ``low_res_mask_size`` so the returned masks share
        the tracker's mask grid (association is mask-IoU; seeding + output resize from here).
        ``geo`` carries the packed box/point geometry prompt for the GEOMETRIC slot.
        """
        m = self.tracker.low_res_mask_size
        return self.detector.detect(
            det_feats, det_pos, concept.text_emb, concept.text_mask,
            image_hw=(m, m), confidence_threshold=0.5, geo=geo,
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
        self._advance_lifecycle(state, matched_track_ids, new_ids, frame_idx)
        return new_objects

    def _advance_lifecycle(self, state, matched_track_ids, new_ids, frame_idx):
        """Step every managed tracklet one frame, then purge the hotstart-killed ones.

        Runs on EVERY frame, including frames with no detection pass -- upstream
        advances its hotstart bookkeeping unconditionally, and "no detections" simply
        means every track is unmatched. Skipping it froze the counters of a
        box-seeded (concept-less) session, so its object was never suppressed or
        killed.

        Only *managed* tracklets step. Click-seeded objects are deliberately left
        unmanaged: upstream routes a click-only session through SAM 2 partial
        propagation, which never runs detection/hotstart for those objects, so they
        neither decay nor die.
        """
        state.tracklet_mgr.step(matched_track_ids, new_ids, frame_idx)

        # purge ONLY removed tracklets (within-hotstart failures). Absent established
        # objects are suppressed, not removed -> memory retained for re-ID.
        for oid in state.tracklet_mgr.removed_ids():
            if oid in state.bank.known_obj_ids:
                self._purge_removed(state, oid)

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
        """Route a point/mask :class:`GeometryPrompt` to the tracker (new obj_id -> spawn,
        existing -> refine).

        Reuses the SAM 2 tracker prompt path: points (scaled to the 1008 input grid) and/or a
        mask drive a cond-frame ``track_step``; a new ``obj_id`` spawns a tracklet, an existing
        one is re-conditioned (refined) at this frame.

        BOXES do not come here: upstream routes a box to the detector's geometry encoder
        (``sam3_video_inference._get_visual_prompt`` stores it as the frame's GEOMETRIC
        prompt), never to the tracker as corner points, so ``forward`` handles boxes in its
        detection step.
        """
        device = self.device
        H, W = state.video_hw
        obj_id = prompt.obj_id
        is_new = obj_id not in state.bank.known_obj_ids
        prompt = prompt.to(device)

        point_inputs = None
        points = prompt.tracker_points()
        if points is not None:
            raw_coords, labels = points
            coords = _normalized_points(
                prompt, (H, W), device, coords=raw_coords
            ) * self.tracker.image_size
            point_inputs = {
                "point_coords": coords[None],
                "point_labels": labels.to(device)[None],
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
            state.tracklet_mgr.spawn(obj_id, frame_idx, interactive=True)
        else:
            state.tracklet_mgr.force_confirm(obj_id)
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

    # Upstream multiplex lifecycle: the demo builder turns masklet confirmation ON
    # (model_builder.py ~1184, thresh 3) and leaves the keep-alive bounds at the
    # Sam3MultiplexBase class defaults (~228-230), which are much tighter than the base
    # lineage's -- a multiplex object is hidden after a few unmatched frames.
    LIFECYCLE = {
        "confirmation_enable": True,
        "confirmation_thresh": 3,
        "hotstart_delay": 15,
        "hotstart_unmatch_thresh": 8,
        "init_keep_alive": 0,
        "max_keep_alive": 8,
        "min_keep_alive": -4,
    }
    BOX_ONLY_CAPTION = BOX_ONLY_CAPTION_MUX

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
        prompts: list[GeometryPrompt] | None = None,
    ) -> dict[int, MaskletResult]:
        prompts = prompts or []
        if prompts:
            self._check_mux_geometry(prompts)
        device = self.device
        self._check_frame_hw(state, frame)
        H, W = state.video_hw
        if not state.started:
            # the caller builds the state, so it cannot know which lineage drives it
            state.tracklet_mgr.configure(**self.LIFECYCLE)
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
            geo, point_prompts = self._split_and_pack_geometry(
                prompts, (H, W), device
            )
            concept = self._concept_for_detection(state, geo)

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
            det = (
                self._detect(det_f, det_p, concept, geo=geo)
                if concept is not None else None
            )

            # 3) associate det<->trk + spawn / confirm / kill (Task 7, per-object, unchanged)
            new_objects: list[tuple[int, int]] = []
            if det is not None:
                new_objects = self._associate_and_update(
                    state, det, active_ids, trk_low_masks, trk_results
                )
            else:
                self._advance_lifecycle(state, set(), set(), frame_idx)

            # 4) seed (first frame) or grow (mid-stream) the new detector instances
            if new_objects:
                self._detector_add(
                    state, frame_idx, det, new_objects, bf_int, bf_prop, num_frames
                )

            # 5) build outputs: existing -> tracker masks; new dets -> detector masks.
            # Skip the full-res masklet build for tracklets the emit policy will drop --
            # they still propagate + retain memory, but their output would go straight
            # out of select_emitted, so upsampling it is wasted work.
            managed = state.tracklet_mgr.managed_ids()
            emitted = state.tracklet_mgr.emitted_ids(self.emit)
            results: dict[int, MaskletResult] = {}
            for oid in active_ids:
                if oid not in state.bank.known_obj_ids:
                    continue  # killed this frame
                if oid in managed and oid not in emitted:
                    continue  # policy hides it -> nothing to build
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
            return select_emitted(results, state.tracklet_mgr, self.emit)

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
    def _check_mux_geometry(prompts) -> None:
        """Validate this lineage's prompts: points and boxes are supported, masks are not.

        The multiplex seeds objects through ``_seed_mux_state`` / ``_grow_mux_state``,
        which take point inputs from a user prompt and masks only from the detector, so
        there is no route for a caller's mask here (the base lineage does support one).
        """
        for prompt in prompts:
            if prompt.masks_logits is not None:
                raise NotImplementedError(
                    "mask prompts are unsupported on the multiplex lineage; use the base "
                    "video predictor for a mask prompt, or prompt with points/boxes"
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
