# SPDX-License-Identifier: Apache-2.0
# Vendored from SimonZeng7108/efficientsam3 @ d063e00 (sam3/backbones/efficientvit/efficientvit/); intra-package imports rewritten.
# Upstream source: MIT-HAN-Lab/efficientvit (Apache-2.0)
# NOTE: task heads (cls, dc_ae, sam, seg) are intentionally NOT re-exported here;
# only the backbone factories needed by _EfficientViTTrunk are included.
from .backbone import (
    EfficientViTBackbone,
    EfficientViTLargeBackbone,
    efficientvit_backbone_b0,
    efficientvit_backbone_b1,
    efficientvit_backbone_b2,
    efficientvit_backbone_b3,
    efficientvit_backbone_l0,
    efficientvit_backbone_l1,
    efficientvit_backbone_l2,
    efficientvit_backbone_l3,
)

__all__ = [
    "EfficientViTBackbone",
    "efficientvit_backbone_b0",
    "efficientvit_backbone_b1",
    "efficientvit_backbone_b2",
    "efficientvit_backbone_b3",
    "EfficientViTLargeBackbone",
    "efficientvit_backbone_l0",
    "efficientvit_backbone_l1",
    "efficientvit_backbone_l2",
    "efficientvit_backbone_l3",
]
