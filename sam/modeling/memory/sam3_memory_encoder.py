# SPDX-License-Identifier: LicenseRef-SAM
# Vendored from facebookresearch/sam3 @ 5dd401d (sam3/model/memory.py): the SAM 3 tracker's
# memory ENCODER. ``SimpleMaskDownSampler`` genuinely diverges from the shared (SAM 2, Apache)
# ``memory/encoder.py`` ``MaskDownSampler`` (adds the SAM 3 ``interpol_size``=1152
# pre-interpolation and a different channel progression / multiplex knobs) and is defined here.
# ``SimpleFuser`` / ``SimpleMaskEncoder`` had implementations byte-identical to the shared Apache
# ``Fuser`` / ``MemoryEncoder``, so they are re-exported (aliased) from ``memory/encoder.py``
# instead of re-defined. ``CXBlock`` is kept local: its module is identical to the shared
# ``CXBlock`` but the docstring/comment text differs. Stripped: the timm ``DropPath`` import
# (reuse the Apache ``sam.modeling.utils.DropPath``; inert at inference, drop_path=0).
# ``LayerNorm2d`` is reused from the shared Apache ``sam.modeling.utils``.
"""SAM 3 tracker memory encoder (mask + pixel-feature -> spatial memory)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam.modeling.memory.encoder import (
    Fuser as SimpleFuser,
    MemoryEncoder as SimpleMaskEncoder,
)
from sam.modeling.utils import DropPath, LayerNorm2d

# ``SimpleFuser`` / ``SimpleMaskEncoder`` are re-exported from ``memory/encoder.py`` (identical
# implementations); listed in ``__all__`` so callers keep importing them from this module (e.g.
# ``sam.build_sam``).
__all__ = ["SimpleMaskDownSampler", "SimpleMaskEncoder", "SimpleFuser", "CXBlock"]


class SimpleMaskDownSampler(nn.Module):
    """Progressively downsample a mask by ``total_stride`` (each step by ``stride``), then
    linearly project to ``embed_dim`` channels. Optionally interpolates the input mask to
    ``interpol_size`` first (the SAM 3 tracker uses 1152x1152)."""

    def __init__(
        self,
        embed_dim=256,
        kernel_size=4,
        stride=4,
        padding=0,
        total_stride=16,
        activation=nn.GELU,
        interpol_size=None,
        multiplex_count: int = 1,
        starting_out_chan: int = 1,
        input_channel_multiplier: int = 1,
    ):
        super().__init__()
        num_layers = int(math.log2(total_stride) // math.log2(stride))
        multiplex_count = multiplex_count * input_channel_multiplier
        assert stride**num_layers == total_stride
        self.encoder = nn.Sequential()
        mask_in_chans, mask_out_chans = multiplex_count, starting_out_chan
        for _ in range(num_layers):
            mask_out_chans = mask_out_chans * (stride**2)
            self.encoder.append(
                nn.Conv2d(
                    mask_in_chans,
                    mask_out_chans,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            self.encoder.append(LayerNorm2d(mask_out_chans))
            self.encoder.append(activation())
            mask_in_chans = mask_out_chans

        self.encoder.append(nn.Conv2d(mask_out_chans, embed_dim, kernel_size=1))
        self.multiplex_count = multiplex_count
        self.interpol_size = interpol_size
        if self.interpol_size is not None:
            assert isinstance(self.interpol_size, (list, tuple)), (
                f"Unsupported type {type(self.interpol_size)}. Should be a list or tuple."
            )
            self.interpol_size = list(interpol_size)
            assert len(self.interpol_size) == 2

    def forward(self, x: torch.Tensor):
        if self.interpol_size is not None and self.interpol_size != list(x.shape[-2:]):
            x = F.interpolate(
                x.float(),
                size=self.interpol_size,
                align_corners=False,
                mode="bilinear",
                antialias=True,
            )
        return self.encoder(x)


# Lightly adapted from ConvNext (https://github.com/facebookresearch/ConvNeXt)
class CXBlock(nn.Module):
    r"""ConvNeXt Block (DwConv -> LayerNorm -> 1x1 Conv -> GELU -> 1x1 Conv, channels-last)."""

    def __init__(
        self,
        dim,
        kernel_size=7,
        padding=3,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        use_dwconv=True,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim if use_dwconv else 1,
        )  # depthwise conv
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # 1x1 convs as linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
