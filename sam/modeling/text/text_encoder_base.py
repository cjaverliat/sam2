# SPDX-License-Identifier: LicenseRef-SAM
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
"""Structural contract shared by Sam3TextEncoder and MobileClipTextEncoder (spec §6).
No forced inheritance — both classes satisfy it via duck typing."""
from typing import Optional, Protocol, runtime_checkable

import torch


@runtime_checkable
class TextEncoder(Protocol):
    def forward(
        self,
        text,
        input_boxes: Optional[list] = None,
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def encode(self, phrases: list[str]) -> torch.Tensor: ...
