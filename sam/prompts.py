# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch


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
