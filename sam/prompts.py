# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import enum

import torch


class BoxRoute(enum.Enum):
    """Which part of the model a VIDEO box prompt is meant for.

    The image predictor has no tracker, so its ``geometry=`` box is always a detector
    prompt and this route is ignored there. ``boxes_labels`` (positive/negative) is a
    detector notion likewise: on video it requires ``DETECTOR``.

    TRACKER:
        SAM 2 semantics, and the default: the box is encoded as its two corner points
        (labels 2 and 3) and seeds ONE object through the tracker's prompt encoder. No
        detection runs, so nothing else can appear.
    DETECTOR:
        SAM 3 only: the box goes to the detector's geometric slot and biases that
        frame's detection. Detection then runs on every frame, so every instance the
        concept matches is tracked, not only the boxed one -- which is why this route
        must be asked for, and why the session needs a concept (``set_concept`` or
        ``set_placeholder_concept``).
    """

    TRACKER = "tracker"
    DETECTOR = "detector"


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
        box_route: BoxRoute = BoxRoute.TRACKER,
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
        self.box_route = box_route

    @classmethod
    def click(cls, obj_id: int, xy, label: int = 1) -> GeometryPrompt:
        """A single point click at pixel ``xy = (x, y)``; ``label`` 1 positive, 0 negative."""
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
        """A tracker box (interactive VOS): seeds ONE object from pixel ``xyxy``."""
        coords = torch.as_tensor(xyxy, dtype=torch.float32).reshape(-1)
        if coords.numel() != 4:
            raise ValueError(f"box expects (xmin, ymin, xmax, ymax), got {tuple(coords.shape)}")
        return cls(obj_id=obj_id, boxes=coords.reshape(1, 4))

    @classmethod
    def concept_box(cls, obj_id: int, xyxy, label: int = 1) -> GeometryPrompt:
        """A detector box (SAM 3): biases the concept search on this frame.

        Needs a concept on the session (``set_concept`` / ``set_placeholder_concept`` /
        ``start_session(concept=...)``). ``label`` 1 keeps the boxed instance positive;
        0 means "everything matching the concept EXCEPT this one".
        """
        coords = torch.as_tensor(xyxy, dtype=torch.float32).reshape(-1)
        if coords.numel() != 4:
            raise ValueError(f"concept_box expects (xmin, ymin, xmax, ymax), got {tuple(coords.shape)}")
        return cls(
            obj_id=obj_id,
            boxes=coords.reshape(1, 4),
            boxes_labels=None if label == 1 else torch.tensor([label]),
            box_route=BoxRoute.DETECTOR,
        )

    @classmethod
    def mask(cls, obj_id: int, mask) -> GeometryPrompt:
        """A mask prompt from an ``(H, W)`` boolean mask or float logits."""
        m = torch.as_tensor(mask)
        if m.dtype == torch.bool:
            m = m.float() * 20.0 - 10.0  # binarising at 0 recovers the input
        return cls(obj_id=obj_id, masks_logits=m.float())

    @property
    def to_detector(self) -> bool:
        """Whether this prompt carries a box bound for the detector's geometric slot."""
        return self.boxes is not None and self.box_route is BoxRoute.DETECTOR

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
        parts = []
        if self.points_coords is not None:
            parts.append((self.points_coords, self.points_labels))
        if self.boxes is not None and self.box_route is BoxRoute.TRACKER:
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
            box_route=self.box_route,
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
            box_route=self.box_route,
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
