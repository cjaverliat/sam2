# SPDX-License-Identifier: LicenseRef-SAM
# Vendored/adapted from facebookresearch/sam3 @ 5dd401d
# (sam3/model/vl_combiner.py::SAM3VLBackbone): the IMAGE path only. The text tower
# (``language_backbone`` / ``forward_text``) is Task 3 and is intentionally omitted here --
# ``Sam3VisionEncoder.forward`` is exactly ``SAM3VLBackbone._forward_image_no_act_ckpt``
# (minus the SAM 2 dual-neck branch the image model does not build). The attribute name
# ``vision_backbone`` is preserved so the checkpoint subtree
# ``detector.backbone.vision_backbone.*`` loads strictly.
"""SAM 3 Perception-Encoder vision encoder (shared by the detector and the tracker)."""

from typing import List, Tuple

import torch
import torch.nn as nn

from sam.modeling.encoders.necks import Sam3DualViTDetNeck


class Sam3VisionEncoder(nn.Module):
    """The PE vision backbone: a ViTDet Simple-FPN neck (PE ViT trunk + FPN) producing a
    multi-level feature pyramid.

    ``forward(image)`` consumes the preprocessed (B, 3, 1008, 1008) tensor and returns
    ``(features, pos)`` -- two parallel lists ordered coarse-stride-first; ``features[-1]``
    is the principal stride-14 (72x72, 256-ch) level the DETR detector consumes
    (``num_feature_levels=1``). ``scalp`` drops the ``scalp`` lowest-resolution levels from
    the neck output (the SAM 3 image config uses ``scalp=1``, dropping the 36x36 level so
    the pyramid is [288, 144, 72]).
    """

    def __init__(self, vision_backbone: Sam3DualViTDetNeck, scalp: int = 1):
        super().__init__()
        self.vision_backbone = vision_backbone
        self.scalp = scalp

    def forward(
        self, image: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        sam3_features, sam3_pos, _, _ = self.vision_backbone(image)
        if self.scalp > 0:
            sam3_features = sam3_features[: -self.scalp]
            sam3_pos = sam3_pos[: -self.scalp]
        return sam3_features, sam3_pos
