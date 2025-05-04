from __future__ import annotations

import torch


class SAM2Prompt:
    def __init__(
        self,
        obj_id: int,
        orig_img_size_hw: tuple[int, int],
        points_coords: torch.Tensor | None = None,
        points_labels: torch.Tensor | None = None,
        boxes: torch.Tensor | None = None,
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

        self.obj_id = obj_id
        self.orig_img_size_hw = orig_img_size_hw
        self.points_coords = points_coords
        self.points_labels = points_labels
        self.boxes = boxes
        self.masks_logits = masks_logits
        self.is_normalized = is_normalized

    def to(self, device: torch.device) -> SAM2Prompt:
        points_coords = (
            self.points_coords.to(device) if self.points_coords is not None else None
        )
        points_labels = (
            self.points_labels.to(device) if self.points_labels is not None else None
        )
        boxes = self.boxes.to(device) if self.boxes is not None else None
        masks_logits = (
            self.masks_logits.to(device) if self.masks_logits is not None else None
        )
        return SAM2Prompt(
            obj_id=self.obj_id,
            orig_img_size_hw=self.orig_img_size_hw,
            points_coords=points_coords,
            points_labels=points_labels,
            boxes=boxes,
            masks_logits=masks_logits,
            is_normalized=self.is_normalized,
        )

    def normalize(self, model_input_size_hw: tuple[int, int]) -> SAM2Prompt:

        if self.is_normalized:
            return self

        orig_img_h, orig_img_w = self.orig_img_size_hw
        model_input_h, model_input_w = model_input_size_hw

        points_coords = None
        points_labels = None
        boxes = None
        masks_logits = None

        if self.points_coords is not None:
            points_coords = self.points_coords.clone()
            points_coords[..., 0] = points_coords[..., 0] / orig_img_w
            points_coords[..., 1] = points_coords[..., 1] / orig_img_h
            points_coords[..., 0] = points_coords[..., 0] * model_input_w
            points_coords[..., 1] = points_coords[..., 1] * model_input_h

        if self.boxes is not None:
            boxes = self.boxes.clone()
            boxes[..., 0] = boxes[..., 0] / orig_img_w
            boxes[..., 1] = boxes[..., 1] / orig_img_h
            boxes[..., 2] = boxes[..., 2] / orig_img_w
            boxes[..., 3] = boxes[..., 3] / orig_img_h
            boxes[..., 0] = boxes[..., 0] * model_input_w
            boxes[..., 1] = boxes[..., 1] * model_input_h
            boxes[..., 2] = boxes[..., 2] * model_input_w
            boxes[..., 3] = boxes[..., 3] * model_input_h

        if self.masks_logits is not None:
            masks_logits = self.masks_logits.clone()

            # Downsample the masks if needed
            if masks_logits.shape[-2:] != model_input_size_hw:
                masks_logits = torch.nn.functional.interpolate(
                    masks_logits,
                    size=model_input_size_hw,
                    mode="bilinear",
                    antialias=True,
                )

        return SAM2Prompt(
            obj_id=self.obj_id,
            orig_img_size_hw=self.orig_img_size_hw,
            points_coords=points_coords,
            points_labels=points_labels,
            boxes=boxes,
            masks_logits=masks_logits,
            is_normalized=True,
        )

    def unnormalize(self, model_input_size_hw: tuple[int, int]) -> SAM2Prompt:

        if not self.is_normalized:
            return self

        orig_img_h, orig_img_w = self.orig_img_size_hw
        model_input_h, model_input_w = model_input_size_hw

        points_coords = None
        points_labels = None
        boxes = None
        masks_logits = None

        if self.points_coords is not None:
            points_coords = self.points_coords.clone()
            points_coords[..., 0] = points_coords[..., 0] / model_input_w
            points_coords[..., 1] = points_coords[..., 1] / model_input_h
            points_coords[..., 0] = points_coords[..., 0] * orig_img_w
            points_coords[..., 1] = points_coords[..., 1] * orig_img_h

        if self.boxes is not None:
            boxes = self.boxes.clone()
            boxes[..., 0] = boxes[..., 0] / model_input_w
            boxes[..., 1] = boxes[..., 1] / model_input_h
            boxes[..., 2] = boxes[..., 2] / model_input_w
            boxes[..., 3] = boxes[..., 3] / model_input_h
            boxes[..., 0] = boxes[..., 0] * orig_img_w
            boxes[..., 1] = boxes[..., 1] * orig_img_h
            boxes[..., 2] = boxes[..., 2] * orig_img_w
            boxes[..., 3] = boxes[..., 3] * orig_img_h

        if self.masks_logits is not None:
            masks_logits = self.masks_logits.clone()

            # Upsample the masks if needed
            if masks_logits.shape[-2:] != self.orig_img_size_hw:
                masks_logits = torch.nn.functional.interpolate(
                    masks_logits,
                    size=self.orig_img_size_hw,
                    mode="bilinear",
                    antialias=True,
                )

        return SAM2Prompt(
            obj_id=self.obj_id,
            orig_img_size_hw=self.orig_img_size_hw,
            points_coords=points_coords,
            points_labels=points_labels,
            boxes=boxes,
            masks_logits=masks_logits,
            is_normalized=False,
        )
