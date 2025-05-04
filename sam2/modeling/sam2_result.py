from __future__ import annotations

import torch


class SAM2Result:
    def __init__(
        self,
        masks_logits: torch.Tensor,
        ious: torch.Tensor,
        obj_ptrs: torch.Tensor,
        obj_score_logits: torch.Tensor,
    ):
        self.masks_logits = masks_logits
        self.ious = ious
        self.obj_ptrs = obj_ptrs
        self.obj_score_logits = obj_score_logits

    def to(self, device: torch.device) -> SAM2Result:
        return SAM2Result(
            masks_logits=self.masks_logits.to(device),
            ious=self.ious.to(device),
            obj_ptrs=self.obj_ptrs.to(device),
            obj_score_logits=self.obj_score_logits.to(device),
        )
