import json
from pathlib import Path

import torch

from sam.onnx.ort_block import OrtBlock
from sam.onnx.trt_options import TensorRTOptions

FILENAME = "memory_attention.onnx"
# Memory is fed pre-split into the rotary (spatial) part and the no-rotary
# (object-pointer) part. The no-rotary length is a dynamic dim, which is how
# num_obj_ptr_tokens stays dynamic without an integer graph input.
INPUT_NAMES = [
    "curr",
    "curr_pos",
    "memory_rope",
    "memory_norope",
    "mpos_rope",
    "mpos_norope",
]
OUTPUT_NAMES = ["out"]


class MemoryAttentionOnnx(torch.nn.Module):
    """Drop-in for MemoryAttention. Matches the call signature used by the
    orchestration: ``forward(curr, memory, curr_pos, memory_pos, num_obj_ptr_tokens)``;
    splits the concatenated memory into rope / no-rope parts for the graph.

    An ``nn.Module`` (no parameters) so it swaps in with a plain ``setattr``."""

    def __init__(
        self,
        onnx_dir: str | Path,
        device: torch.device,
        use_trt: bool = True,
        trt_opts: TensorRTOptions | None = None,
    ):
        super().__init__()
        onnx_dir = Path(onnx_dir)
        min_shapes, opt_shapes, max_shapes = self._trt_profiles(onnx_dir)
        self.block = OrtBlock(
            onnx_dir / FILENAME,
            INPUT_NAMES,
            OUTPUT_NAMES,
            device,
            use_trt,
            trt_opts,
            min_shapes=min_shapes,
            opt_shapes=opt_shapes,
            max_shapes=max_shapes,
        )

    @staticmethod
    def _trt_profiles(onnx_dir: Path):
        """Build TensorRT min/opt/max shape profiles for the dynamic memory inputs so
        the EP builds one engine spanning the whole bounded range, instead of a fresh
        (static) engine per memory length — the per-frame rebuilds otherwise hit a
        TensorRT myelin bug on the dynamic-shape RoPE path. Needs the bounds the export
        baked into manifest.json (finite ``max_cond_frames_in_attn``); returns
        ``(None, None, None)`` for an unbounded/old export so the EP stays dynamic."""
        mani = json.loads((onnx_dir / "manifest.json").read_text())
        rope_max = mani.get("memory_rope_max_tokens")
        ptr_max = mani.get("memory_norope_max_tokens")
        if not rope_max or not ptr_max:
            return None, None, None
        c = mani["hidden_dim"]
        mem_dim = mani["mem_dim"]
        hw = mani["sam_image_embedding_size"] ** 2
        # opt = a typical length: the full mask-memory window for spatial, all pointers.
        rope_opt = min(mani.get("num_maskmem", 7) * hw, rope_max)
        # curr/curr_pos are static (one frame of image tokens); the memory inputs vary
        # along dim 0 (token count). Profiles cover [1 frame .. rope_max] spatial and
        # [1 .. ptr_max] object-pointer tokens.
        def shapes(rope_n, ptr_n):
            return {
                "curr": (hw, 1, c),
                "curr_pos": (hw, 1, c),
                "memory_rope": (rope_n, 1, mem_dim),
                "memory_norope": (ptr_n, 1, mem_dim),
                "mpos_rope": (rope_n, 1, mem_dim),
                "mpos_norope": (ptr_n, 1, mem_dim),
            }
        return shapes(hw, 1), shapes(rope_opt, ptr_max), shapes(rope_max, ptr_max)

    def forward(self, curr, memory, curr_pos=None, memory_pos=None, num_obj_ptr_tokens=0):
        m = memory.shape[0] - num_obj_ptr_tokens
        feeds = {
            "curr": curr,
            "curr_pos": curr_pos,
            "memory_rope": memory[:m],
            "memory_norope": memory[m:],
            "mpos_rope": memory_pos[:m],
            "mpos_norope": memory_pos[m:],
        }
        return self.block.run(feeds)[0]
