# SPDX-License-Identifier: LicenseRef-SAM
# Vendored from facebookresearch/sam3 @ 5dd401d (sam3/model/necks.py): the
# ``Sam3DualViTDetNeck`` Simple-FPN that sits between the PE ViT trunk and the rest of the
# detector. Trimmed to the image path (the plain-tensor branch -- no NestedTensor masking)
# and to the SAM 3 neck only; the optional SAM 2 ("dual") neck is preserved behind
# ``add_sam2_neck`` but defaults off (the image vision encoder does not use it). Attribute
# names (``trunk`` / ``position_encoding`` / ``convs``) are preserved verbatim so the
# checkpoint subtree ``detector.backbone.vision_backbone.{trunk,convs}.*`` loads strictly.
"""Necks are the interface between a vision backbone and the rest of the detection model."""

from copy import deepcopy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class Sam3DualViTDetNeck(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
        position_encoding: nn.Module,
        d_model: int,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        add_sam2_neck: bool = False,
        add_interactive_neck: bool = False,
    ):
        """SimpleFPN neck a la ViTDet (from detectron2, very lightly adapted).

        Optionally supports a "dual neck" setting (``add_sam2_neck``): two identical necks
        (for SAM 3 and SAM 2) with different weights. ``add_interactive_neck`` adds a THIRD
        identical neck (``interactive_convs``) -- the SAM 3.1 multiplex *video* path is a
        tri-neck (detection ``convs`` + ``interactive_convs`` + ``propagation_convs``), all fed
        by ONE trunk pass; the third neck is exposed only via :meth:`forward_all` so the base
        (dual/single) :meth:`forward` arity is byte-unchanged.

        :param trunk: the backbone
        :param position_encoding: the positional encoding to use
        :param d_model: the dimension of the model
        """
        super().__init__()
        self.trunk = trunk
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()

        self.scale_factors = scale_factors
        use_bias = True
        dim: int = self.trunk.channel_list[-1]

        for _, scale in enumerate(scale_factors):
            current = nn.Sequential()

            if scale == 4.0:
                current.add_module(
                    "dconv_2x2_0",
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                )
                current.add_module("gelu", nn.GELU())
                current.add_module(
                    "dconv_2x2_1",
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                )
                out_dim = dim // 4
            elif scale == 2.0:
                current.add_module(
                    "dconv_2x2",
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                )
                out_dim = dim // 2
            elif scale == 1.0:
                out_dim = dim
            elif scale == 0.5:
                current.add_module("maxpool_2x2", nn.MaxPool2d(kernel_size=2, stride=2))
                out_dim = dim
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            current.add_module(
                "conv_1x1",
                nn.Conv2d(
                    in_channels=out_dim, out_channels=d_model, kernel_size=1, bias=use_bias
                ),
            )
            current.add_module(
                "conv_3x3",
                nn.Conv2d(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                ),
            )
            self.convs.append(current)

        self.sam2_convs = None
        if add_sam2_neck:
            # Assumes the sam2 neck is just a clone of the original neck.
            self.sam2_convs = deepcopy(self.convs)

        self.interactive_convs = None
        if add_interactive_neck:
            # The SAM 3.1 multiplex tracker's interactive (cond-frame object-pointer) neck --
            # a third clone, fed by the same trunk output (see forward_all).
            self.interactive_convs = deepcopy(self.convs)

    def forward(
        self, tensor_list: List[torch.Tensor]
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        Optional[List[torch.Tensor]],
        Optional[List[torch.Tensor]],
    ]:
        # ``forward_all`` is a superset: its interactive branch is a no-op (stays ``None``)
        # unless the neck was built with ``add_interactive_neck``, so the base (detection +
        # optional SAM 2) 4-tuple is exactly ``forward_all(...)[:4]``.
        return self.forward_all(tensor_list)[:4]

    def forward_all(self, tensor_list: List[torch.Tensor]):
        """Tri-neck forward: ONE trunk pass -> ``(sam3_out, sam3_pos, sam2_out, sam2_pos,
        interactive_out, interactive_pos)``.

        ``interactive_*`` is ``None`` when the encoder was built without
        ``add_interactive_neck``. Additive to :meth:`forward` (which keeps its 4-tuple arity so
        every base caller is byte-unchanged); used only by the SAM 3.1 multiplex *video*
        predictor, whose ``encode_image`` needs all three pyramids (detection ``convs`` ->
        detector, ``sam2_convs`` -> tracker propagation, ``interactive_convs`` -> the tracker's
        cond-frame interactive object-pointer head) from a single heavy ViT trunk pass.
        """
        xs = self.trunk(tensor_list)
        x = xs[-1]  # simpleFPN
        sam3_out, sam3_pos = [], []
        sam2_out, sam2_pos = ([], []) if self.sam2_convs is not None else (None, None)
        int_out, int_pos = ([], []) if self.interactive_convs is not None else (None, None)
        for i in range(len(self.convs)):
            a = self.convs[i](x)
            sam3_out.append(a)
            sam3_pos.append(self.position_encoding(a).to(a.dtype))
            if self.sam2_convs is not None:
                b = self.sam2_convs[i](x)
                sam2_out.append(b)
                sam2_pos.append(self.position_encoding(b).to(b.dtype))
            if self.interactive_convs is not None:
                c = self.interactive_convs[i](x)
                int_out.append(c)
                int_pos.append(self.position_encoding(c).to(c.dtype))
        return sam3_out, sam3_pos, sam2_out, sam2_pos, int_out, int_pos
