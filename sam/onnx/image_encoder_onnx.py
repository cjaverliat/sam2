# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import torch

from sam.onnx.ort_block import OrtBlock
from sam.onnx.trt_options import TensorRTOptions

# Graph outputs: vision_features, then L backbone_fpn levels, then L pos levels.
FILENAME = "image_encoder.onnx"
INPUT_NAMES = ["image"]


def output_names(num_levels: int) -> list[str]:
    fpn = [f"fpn_{i}" for i in range(num_levels)]
    pos = [f"pos_{i}" for i in range(num_levels)]
    return ["vision_features", *fpn, *pos]


class ImageEncoderOnnx(torch.nn.Module):
    """Drop-in for ImageEncoder: ``forward(sample) -> dict`` with the same
    ``vision_features`` / ``backbone_fpn`` / ``vision_pos_enc`` keys the orchestration
    reads.

    An ``nn.Module`` (holding no parameters — just the ORT session) so it can be
    assigned over the torch block with a plain ``setattr`` and so ``.to()`` / ``.eval()``
    recurse harmlessly."""

    def __init__(
        self,
        onnx_dir: str | Path,
        num_levels: int,
        device: torch.device,
        use_trt: bool = True,
        trt_opts: TensorRTOptions | None = None,
    ):
        super().__init__()
        self.num_levels = num_levels
        self._all_output_names = output_names(num_levels)
        self.block = OrtBlock(
            Path(onnx_dir) / FILENAME,
            INPUT_NAMES,
            self._all_output_names,
            device,
            use_trt,
            trt_opts,
        )
        # The pos levels (vision_pos_enc) are sinusoidal position encodings over the
        # feature grid: they depend only on the input resolution, not the frame
        # content, so they are identical for every frame at a given size (the export
        # even constant-folds them). Returning them from the graph each frame is pure
        # waste — those 3 tensors total ~87 MiB and cost ~22 ms/frame to re-serve,
        # more than the encoder compute itself. Compute them once, cache, and skip
        # binding them on later frames (keyed by H,W so a resolution change rebuilds).
        self._pos_cache: list[torch.Tensor] | None = None
        self._pos_cache_hw: tuple[int, int] | None = None

    def forward(self, sample: torch.Tensor) -> dict:
        n = self.num_levels
        hw = (int(sample.shape[-2]), int(sample.shape[-1]))
        if self._pos_cache is None or self._pos_cache_hw != hw:
            # First frame at this resolution: request all outputs and cache the pos
            # levels (cloned off the reused ORT output buffers).
            self.block.output_names = self._all_output_names
            outs = self.block.run({"image": sample})
            self._pos_cache = [p.clone() for p in outs[1 + n :]]
            self._pos_cache_hw = hw
            # Later frames skip the constant pos outputs entirely.
            self.block.output_names = self._all_output_names[: 1 + n]
        else:
            outs = self.block.run({"image": sample})
        return {
            "vision_features": outs[0],
            "backbone_fpn": list(outs[1 : 1 + n]),
            "vision_pos_enc": list(self._pos_cache),
        }
