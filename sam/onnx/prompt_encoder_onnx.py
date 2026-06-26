# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import numpy as np
import torch

from sam.onnx.ort_block import OrtBlock
from sam.onnx.trt_options import TensorRTOptions

POINTS_FILE = "prompt_encoder_points.onnx"
MASK_FILE = "prompt_encoder_mask.onnx"
WEIGHTS_FILE = "weights.npz"


class PromptEncoderOnnx(torch.nn.Module):
    """Drop-in for PromptEncoder, fully checkpoint-free.

    The point and mask embeddings (the only data-dependent compute) run as two ONNX
    graphs. The branches the orchestration relies on that used to need torch weights
    — ``get_dense_pe()`` and the no-mask (``no_mask_embed``) dense embedding — are
    loaded as constant tensors from the ``weights.npz`` dict dumped by
    tools/export_onnx.py (keys ``prompt.dense_pe`` / ``prompt.no_mask_embed``).
    Scalar shapes come from ``manifest.json``. Boxes are not used on the SAM2Generic
    path (always ``boxes=None``).

    An ``nn.Module`` (the baked tensors are plain attributes, not parameters) so it
    swaps in with a plain ``setattr``."""

    def __init__(
        self,
        onnx_dir: str | Path,
        device: torch.device,
        use_trt: bool = True,
        trt_opts: TensorRTOptions | None = None,
    ):
        super().__init__()
        onnx_dir = Path(onnx_dir)
        manifest = json.loads((onnx_dir / "manifest.json").read_text())
        self.device = device
        self.embed_dim = manifest["hidden_dim"]
        g = manifest["sam_image_embedding_size"]
        self.image_embedding_size = (g, g)
        self.mask_input_size = tuple(manifest["mask_input_size"])

        # Fixed (weight-derived) tensors baked at export — no checkpoint at runtime.
        w = np.load(onnx_dir / WEIGHTS_FILE)
        self._dense_pe = torch.from_numpy(w["prompt.dense_pe"]).to(device)
        self._no_mask_embed = torch.from_numpy(w["prompt.no_mask_embed"]).to(device)

        self.points_block = OrtBlock(
            onnx_dir / POINTS_FILE, ["coords", "labels"], ["sparse"],
            device, use_trt, trt_opts,
        )
        self.mask_block = OrtBlock(
            onnx_dir / MASK_FILE, ["mask"], ["dense"],
            device, use_trt, trt_opts,
        )

    def get_dense_pe(self) -> torch.Tensor:
        return self._dense_pe

    def forward(self, points=None, boxes=None, masks=None):
        assert boxes is None, "boxes are not supported by the ONNX prompt encoder"

        if points is not None:
            coords, labels = points
            sparse = self.points_block.run(
                {"coords": coords, "labels": labels.to(torch.int64)}
            )[0]
        else:
            bs = masks.shape[0] if masks is not None else 1
            sparse = torch.empty((bs, 0, self.embed_dim), device=self.device)

        if masks is not None:
            dense = self.mask_block.run({"mask": masks})[0]
        else:
            bs = sparse.shape[0]
            dense = self._no_mask_embed.expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )
        return sparse, dense
