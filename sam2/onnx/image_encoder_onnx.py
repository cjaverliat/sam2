from pathlib import Path

import torch

from sam2.onnx.ort_block import OrtBlock
from sam2.onnx.trt_options import TensorRTOptions

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
        self.block = OrtBlock(
            Path(onnx_dir) / FILENAME,
            INPUT_NAMES,
            output_names(num_levels),
            device,
            use_trt,
            trt_opts,
        )

    def forward(self, sample: torch.Tensor) -> dict:
        outs = self.block.run({"image": sample})
        vision_features = outs[0]
        fpn = outs[1 : 1 + self.num_levels]
        pos = outs[1 + self.num_levels :]
        return {
            "vision_features": vision_features,
            "backbone_fpn": list(fpn),
            "vision_pos_enc": list(pos),
        }
