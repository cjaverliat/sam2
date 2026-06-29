# SPDX-License-Identifier: Apache-2.0
"""Small layer utilities vendored to avoid a `timm` dependency (see spec E9).
`SqueezeExcite` mirrors `timm.layers.SqueezeExcite` (Apache-2.0); `to_2tuple`
mirrors `timm.layers.to_2tuple`. Pair with the local `DropPath` in pe_vitdet.py
and `torch.nn.init.trunc_normal_`."""
from collections.abc import Iterable
import torch
import torch.nn as nn


def to_2tuple(x):
    if isinstance(x, Iterable) and not isinstance(x, str):
        return tuple(x)
    return (x, x)


def _make_divisible(v, divisor=8):
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation block (timm-compatible: gate after fc2, hard-sigmoid)."""
    def __init__(self, channels: int, rd_ratio: float = 0.25):
        super().__init__()
        rd = _make_divisible(channels * rd_ratio)
        self.fc1 = nn.Conv2d(channels, rd, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(rd, channels, kernel_size=1, bias=True)
        self.gate = nn.Hardsigmoid()

    def forward(self, x):
        s = x.mean((2, 3), keepdim=True)
        s = self.fc2(self.act(self.fc1(s)))
        return x * self.gate(s)
