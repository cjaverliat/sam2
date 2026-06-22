# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Lightweight frame IO / warmup helpers shared by the example scripts.

Kept dependency-light (only cv2 + torch) so scripts that don't render figures,
e.g. examples/benchmark_onnx.py, can reuse them without pulling in matplotlib.
"""

import cv2
import torch


def read_frame(cap, device) -> torch.Tensor:
    ret, frame = cap.read()
    if not ret:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = torch.as_tensor(frame).permute(2, 0, 1).to(device)
    frame = frame / 255.0
    return frame


def warmup(predictor, video_state, device):
    """Pay one-time CUDA/cuDNN init cost up front instead of on the first real
    frame. Empty prompts return early without writing memory, so state stays
    clean."""
    dummy = torch.zeros(3, *video_state.video_hw, device=device)
    predictor.forward(
        state=video_state, frame=dummy, frame_idx=0, prompts=[], create_memory=False
    )
