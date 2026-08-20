# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import enum

import torch


class PromptRoute(enum.Enum):
    """Which half of SAM 3 a geometry prompt is talking to.

    A prompt is one thing or the other, never both: you are either pointing at ONE
    object you want back (TRACKER, SAM 2 semantics) or describing what the concept
    search should look for (DETECTOR, SAM 3 only). The named constructors set this
    for you -- ``click`` / ``box`` / ``mask`` are TRACKER, ``concept_point`` /
    ``concept_box`` are DETECTOR -- so callers rarely name the route itself.

    TRACKER:
        The default. Points, box corners (labels 2 and 3) and masks go to the
        tracker's prompt encoder and seed or refine ONE object under the ``obj_id``
        you chose. Detection does not run, so nothing else can appear. Video only:
        the image predictor owns no tracker.
    DETECTOR:
        Points and boxes go to the detector's geometric slot and bias that frame's
        concept search; every instance the concept matches is returned, not only the
        one you marked, and the ids are the detector's. Needs a concept -- a phrase,
        or the box-only ``PLACEHOLDER`` caption. Masks have no detector slot: neither
        checkpoint ships ``mask_encoder`` weights.
    """

    TRACKER = "tracker"
    DETECTOR = "detector"


# The name this enum shipped under when only boxes could be routed.
BoxRoute = PromptRoute


class GeometryPrompt:
    def __init__(
        self,
        obj_id: int,
        points_coords: torch.Tensor | None = None,
        points_labels: torch.Tensor | None = None,
        boxes: torch.Tensor | None = None,
        boxes_labels: torch.Tensor | None = None,
        masks_logits: torch.Tensor | None = None,
        is_normalized: bool = False,
        route: PromptRoute = PromptRoute.TRACKER,
    ):
        if (
            points_coords is None
            and points_labels is None
            and boxes is None
            and masks_logits is None
        ):
            raise ValueError(
                "At least one of points_coords, points_labels, boxes, or masks_logits must be provided"
            )

        if points_coords is not None and points_labels is None:
            raise ValueError(
                "points_labels must be provided if points_coords is provided"
            )
        
        if points_coords is not None and (points_coords.ndim != 2 or points_coords.shape[1] != 2):
            raise ValueError(f"Expected points_coords to be of shape (N, 2), got {points_coords.shape}")
        
        if points_labels is not None and (points_labels.ndim != 1 or points_labels.shape[0] != points_coords.shape[0]):
            raise ValueError(f"Expected points_labels to be of shape (N,), got {points_labels.shape}")
        
        if boxes is not None and (boxes.ndim != 2 or boxes.shape[1] != 4):
            raise ValueError(f"Expected boxes to be of shape (N, 4), got {boxes.shape}")

        if boxes_labels is not None and boxes is None:
            raise ValueError("boxes must be provided if boxes_labels is provided")

        if boxes_labels is not None and (boxes_labels.ndim != 1 or boxes_labels.shape[0] != boxes.shape[0]):
            raise ValueError(f"Expected boxes_labels to be of shape (N,), got {boxes_labels.shape}")

        if masks_logits is not None and route is PromptRoute.DETECTOR:
            raise NotImplementedError(
                "the detector has no mask slot: neither SAM 3 checkpoint ships "
                "mask_encoder weights. Use concept_box / concept_point to bias the "
                "search, or GeometryPrompt.mask(obj_id, mask) to prompt the tracker"
            )

        if masks_logits is not None:
            mask_res = masks_logits.shape[-2:]
            masks_logits = masks_logits.reshape(1, *mask_res) # Reshape to (1, H, W)

        self.obj_id = obj_id
        self.points_coords = points_coords
        self.points_labels = points_labels
        self.boxes = boxes
        self.boxes_labels = boxes_labels
        self.masks_logits = masks_logits
        self.is_normalized = is_normalized
        self.route = route

    @classmethod
    def click(cls, obj_id: int, xy, label: int = 1) -> GeometryPrompt:
        """A click on ONE object at pixel ``xy``; ``label`` 1 positive, 0 negative.

        The SAM 2 gesture: this seeds or refines the object you name with ``obj_id``
        through the tracker, and nothing else comes back. To bias a concept search
        instead, use :meth:`concept_point`.
        """
        coords = torch.as_tensor(xy, dtype=torch.float32).reshape(-1)
        if coords.numel() != 2:
            raise ValueError(f"click expects an (x, y) pair, got {tuple(coords.shape)}")
        return cls(
            obj_id=obj_id,
            points_coords=coords.reshape(1, 2),
            points_labels=torch.tensor([label], dtype=torch.int32),
        )

    @classmethod
    def box(cls, obj_id: int, xyxy) -> GeometryPrompt:
        """A box around ONE object (interactive VOS): seeds it from pixel ``xyxy``.

        The SAM 2 gesture, encoded as the box's two corners. Detection does not run,
        so only this object is tracked. To bias a concept search instead, use
        :meth:`concept_box`.
        """
        coords = torch.as_tensor(xyxy, dtype=torch.float32).reshape(-1)
        if coords.numel() != 4:
            raise ValueError(f"box expects (xmin, ymin, xmax, ymax), got {tuple(coords.shape)}")
        return cls(obj_id=obj_id, boxes=coords.reshape(1, 4))

    @classmethod
    def concept_box(cls, xyxy, label: int = 1) -> GeometryPrompt:
        """A detector box (SAM 3): biases the concept search on this frame.

        Needs a concept on the session (``start_concept_session`` -- or ``set_concept``
        / ``set_placeholder_concept`` on an explicit state). ``label`` 1 keeps the boxed
        instance positive;
        0 means "everything matching the concept EXCEPT this one".

        Takes no ``obj_id``: a detector box only biases detection, and the spawned
        instances get their ids from the session's own counter.
        """
        coords = torch.as_tensor(xyxy, dtype=torch.float32).reshape(-1)
        if coords.numel() != 4:
            raise ValueError(f"concept_box expects (xmin, ymin, xmax, ymax), got {tuple(coords.shape)}")
        return cls(
            obj_id=-1,  # unused: detection mints the ids
            boxes=coords.reshape(1, 4),
            boxes_labels=None if label == 1 else torch.tensor([label]),
            route=PromptRoute.DETECTOR,
        )

    @classmethod
    def concept_point(cls, xy, label: int = 1) -> GeometryPrompt:
        """A detector point (SAM 3): biases the concept search toward pixel ``xy``.

        The point form of :meth:`concept_box`, and the same contract: the concept
        still decides WHAT comes back, this only says where to look harder. ``label``
        1 marks the point as an example, 0 as a counter-example ("everything matching
        the concept EXCEPT this one").

        Needs a concept -- a phrase, or the predictor's ``PLACEHOLDER`` for the
        box-only caption. Takes no ``obj_id``: a detector point selects nothing, so
        the ids come from detection.

        Contrast :meth:`click`, which is the SAM 2 gesture: that one picks out ONE
        object and returns it alone.
        """
        coords = torch.as_tensor(xy, dtype=torch.float32).reshape(-1)
        if coords.numel() != 2:
            raise ValueError(
                f"concept_point expects an (x, y) pair, got {tuple(coords.shape)}")
        if label not in (0, 1):
            raise ValueError(f"concept_point label must be 1 or 0, got {label!r}")
        return cls(
            obj_id=-1,  # unused: detection mints the ids
            points_coords=coords.reshape(1, 2),
            points_labels=torch.tensor([label], dtype=torch.int32),
            route=PromptRoute.DETECTOR,
        )

    @classmethod
    def mask(cls, obj_id: int, mask) -> GeometryPrompt:
        """A mask over ONE object, from an ``(H, W)`` boolean mask or float logits.

        Tracker-only: the detector has no mask slot in either SAM 3 checkpoint, so
        there is no ``concept_mask`` counterpart.
        """
        m = torch.as_tensor(mask)
        if m.dtype == torch.bool:
            m = m.float() * 20.0 - 10.0  # binarising at 0 recovers the input
        return cls(obj_id=obj_id, masks_logits=m.float())

    @property
    def to_detector(self) -> bool:
        """Whether this prompt carries a box bound for the detector's geometric slot."""
        return self.route is PromptRoute.DETECTOR

    def tracker_points(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """The points this prompt feeds to the tracker's prompt encoder.

        Clicks keep their own labels (1 foreground / 0 background); a TRACKER-route box
        contributes its corners as labels 2 (top-left) and 3 (bottom-right), the same
        encoding :class:`~sam.models.sam2_predictor.Sam2VideoPredictor` uses. Coordinates
        stay in whatever space the prompt was built in (pixel unless ``is_normalized``).

        Returns:
            ``(coords, labels)`` of shapes ``(N, 2)`` / ``(N,)``, or None when the prompt
            has nothing for the tracker (a DETECTOR-route box, or a mask alone).
        """
        if self.route is PromptRoute.DETECTOR:
            return None  # every DETECTOR geometry belongs to the geometric slot
        parts = []
        if self.points_coords is not None:
            parts.append((self.points_coords, self.points_labels))
        if self.boxes is not None and self.route is PromptRoute.TRACKER:
            corners = self.boxes.reshape(-1, 2)
            labels = torch.tensor(
                [2, 3], dtype=torch.int32, device=corners.device
            ).repeat(self.boxes.shape[0])
            parts.append((corners, labels))
        if not parts:
            return None
        coords = torch.cat([c for c, _ in parts], dim=0)
        labels = torch.cat([lb.to(coords.device) for _, lb in parts], dim=0)
        return coords, labels

    def to(self, device: torch.device) -> GeometryPrompt:
        points_coords = (
            self.points_coords.to(device) if self.points_coords is not None else None
        )
        points_labels = (
            self.points_labels.to(device) if self.points_labels is not None else None
        )
        boxes = self.boxes.to(device) if self.boxes is not None else None
        boxes_labels = (
            self.boxes_labels.to(device) if self.boxes_labels is not None else None
        )
        masks_logits = (
            self.masks_logits.to(device) if self.masks_logits is not None else None
        )
        return GeometryPrompt(
            obj_id=self.obj_id,
            points_coords=points_coords,
            points_labels=points_labels,
            boxes=boxes,
            boxes_labels=boxes_labels,
            masks_logits=masks_logits,
            is_normalized=self.is_normalized,
            route=self.route,
        )

    def clone(self) -> GeometryPrompt:
        return GeometryPrompt(
            obj_id=self.obj_id,
            points_coords=self.points_coords.clone() if self.points_coords is not None else None,
            points_labels=self.points_labels.clone() if self.points_labels is not None else None,
            boxes=self.boxes.clone() if self.boxes is not None else None,
            boxes_labels=self.boxes_labels.clone() if self.boxes_labels is not None else None,
            masks_logits=self.masks_logits.clone() if self.masks_logits is not None else None,
            is_normalized=self.is_normalized,
            route=self.route,
        )


# SPDX-License-Identifier: LicenseRef-SAM
# SAM 3 concept prompt — carries the concept text.  Encoding (text → embeddings)
# is the predictor's responsibility; this type is a pure data carrier.


class ConceptPrompt:
    """Per-concept SAM 3 prompt.

    Text is the only field. Upstream SAM 3 has no inference-time semantics for
    a negative phrase (see :meth:`sam.models.sam3_predictor.Sam3Predictor.encode_text`)
    nor for a VISUAL-slot exemplar: ``visual_prompt_embed`` reaches a live
    consumer but no released code path — training included — ever builds one.
    Reference geometry is expressed as a :class:`GeometryPrompt` box/point, which
    is what upstream's own "image exemplar" resolves to.

    Args:
        text: Positive description of the concept to segment (e.g. "cat").
    """

    def __init__(self, text: str):
        self.text = text
