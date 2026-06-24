from pathlib import Path

import torch

from sam2.onnx.ort_block import OrtBlock
from sam2.onnx.trt_options import TensorRTOptions


def filename(multimask: bool) -> str:
    return f"mask_decoder_mm{int(multimask)}.onnx"


BASE_INPUTS = ["image_embeddings", "image_pe", "sparse", "dense"]
HR_INPUTS = ["hr0", "hr1"]
OUTPUT_NAMES = ["masks", "ious", "sam_tokens", "obj_scores"]


class MaskDecoderOnnx(torch.nn.Module):
    """Drop-in for MaskDecoder. ``multimask_output`` selects one of two exported
    graphs; ``repeat_image=False`` is baked. The high-res projection convs
    (``conv_s0`` / ``conv_s1``) are folded into the image-encoder ONNX graph at
    export, so the copies here are identities — the orchestration still calls
    ``conv_s0`` / ``conv_s1`` in ``encode_image`` but they are no-ops.

    An ``nn.Module`` (no parameters of its own — the ORT sessions live in a plain
    dict) so it swaps in with a plain ``setattr``."""

    def __init__(
        self,
        onnx_dir: str | Path,
        use_high_res: bool,
        device: torch.device,
        use_trt: bool = True,
        trt_opts: TensorRTOptions | None = None,
    ):
        super().__init__()
        self.use_high_res = use_high_res
        # Projection is baked into the image-encoder graph; identities keep the
        # orchestration's conv_s0/conv_s1 calls as harmless no-ops.
        self.conv_s0 = torch.nn.Identity()
        self.conv_s1 = torch.nn.Identity()
        inputs = BASE_INPUTS + (HR_INPUTS if use_high_res else [])
        self.blocks = {
            mm: OrtBlock(
                Path(onnx_dir) / filename(mm),
                inputs,
                OUTPUT_NAMES,
                device,
                use_trt,
                trt_opts,
            )
            for mm in (False, True)
        }

    def forward(
        self,
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
        multimask_output,
        repeat_image=False,
        high_res_features=None,
    ):
        assert not repeat_image, "exported mask decoder bakes repeat_image=False"
        feeds = {
            "image_embeddings": image_embeddings,
            "image_pe": image_pe,
            "sparse": sparse_prompt_embeddings,
            "dense": dense_prompt_embeddings,
        }
        if self.use_high_res:
            assert high_res_features is not None and len(high_res_features) == 2
            feeds["hr0"], feeds["hr1"] = high_res_features
        masks, ious, sam_tokens, obj_scores = self.blocks[bool(multimask_output)].run(
            feeds
        )
        return masks, ious, sam_tokens, obj_scores
