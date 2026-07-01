# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from pathlib import Path

# Blocks (matched by .onnx file stem) for which fp16_enable applies; all others stay
# fp32. Does NOT gate bf16 (safe everywhere).
FP16_SAFE_BLOCKS: frozenset[str] = frozenset({"image_encoder", "memory_attention"})


@dataclass
class TensorRTOptions:
    """Knobs for the ONNX Runtime TensorRT execution provider.

    Feed TensorRT the fp32 opset-18 export. Both precision flags off (default) is fp32;
    set bf16_enable on Ampere+ for full-speed (fp32 range, safe on all blocks). These
    build a weakly-typed network and let TensorRT auto-tune precision per layer.

    bf16_enable is safe everywhere. fp16_enable is applied ONLY to ``FP16_SAFE_BLOCKS``
    (the 2 heavy blocks); other blocks stay fp32 because fp16 error compounds through the
    recurrent memory loop and destroys the masklet (mask IoU ~0). OrtBlock logs, per
    block, whether fp16 was applied. Don't set both flags.
    """

    fp16_enable: bool = False
    bf16_enable: bool = False
    # Capture each TensorRT engine's execution as a CUDA graph and replay it, removing
    # per-kernel launch overhead. Needs stable input shapes + buffer addresses across
    # runs; ORT re-captures when they change, so it helps the static blocks (image
    # encoder) more than the dynamic-shape memory attention.
    cuda_graph_enable: bool = False
    cache_dir: str | Path | None = None
    # 0-5; higher = more tactic search => faster engine, slower build. Default 3 keeps
    # iteration fast; use 5 for a max-performance deployment engine (built once, cached).
    builder_optimization_level: int = 3
    max_workspace_size: int = 4 << 30  # 4 GiB

    def __post_init__(self) -> None:
        # Both precision flags build a weakly-typed network; setting both is ambiguous
        # (TRT would pick one) and conflates two distinct paths. Force the caller to
        # choose: bf16 (safe for all blocks) or fp16 (safe blocks only).
        if self.fp16_enable and self.bf16_enable:
            raise ValueError(
                "TensorRTOptions: set fp16_enable OR bf16_enable, not both. "
                "bf16_enable is safe for all blocks; fp16_enable applies only to "
                "FP16_SAFE_BLOCKS."
            )
