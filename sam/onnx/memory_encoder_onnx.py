from pathlib import Path

import torch

from sam.onnx.ort_block import OrtBlock
from sam.onnx.trt_options import TensorRTOptions

FILENAME = "memory_encoder.onnx"
INPUT_NAMES = ["pix_feat", "masks"]
OUTPUT_NAMES = ["vision_features", "vision_pos_enc"]


class MemoryEncoderOnnx(torch.nn.Module):
    """Drop-in for MemoryEncoder. ``skip_mask_sigmoid=True`` is baked into the graph
    (the only value the orchestration uses); returns the same dict shape.

    An ``nn.Module`` (no parameters) so it swaps in with a plain ``setattr``."""

    def __init__(
        self,
        onnx_dir: str | Path,
        device: torch.device,
        use_trt: bool = True,
        trt_opts: TensorRTOptions | None = None,
    ):
        super().__init__()
        self.block = OrtBlock(
            Path(onnx_dir) / FILENAME,
            INPUT_NAMES,
            OUTPUT_NAMES,
            device,
            use_trt,
            trt_opts,
        )

    def forward(self, pix_feat, masks, skip_mask_sigmoid: bool = True) -> dict:
        assert skip_mask_sigmoid, "exported memory encoder bakes skip_mask_sigmoid=True"
        feats, pos = self.block.run({"pix_feat": pix_feat, "masks": masks})
        return {"vision_features": feats, "vision_pos_enc": [pos]}
