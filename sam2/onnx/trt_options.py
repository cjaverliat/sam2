from dataclasses import dataclass
from pathlib import Path


@dataclass
class TensorRTOptions:
    """Knobs for the ONNX Runtime TensorRT execution provider.

    No precision flag: enabling ``trt_fp16_enable`` makes the ORT TensorRT EP build a
    weakly-typed network, which rejects the native ONNX Attention/RoPE ops in the
    opset-23 exports (they require a strongly-typed network). The EP only builds a
    strongly-typed network when no precision flag is set, so precision is owned by the
    graph instead — use a mixed-fp16 export (``tools/export_onnx.py --fp16``) for fp16.

    Engine cache on, reasonable workspace. ``cache_dir=None`` puts the engine cache
    next to the .onnx file.

    The timing cache is separate from the engine cache: it stores per-tactic kernel
    timings and is reused across builds (even of different engines), cutting the cost
    of the *first* build of a new engine. ``timing_cache_dir=None`` shares the engine
    cache directory.
    """

    # Let TensorRT pick precision (weakly-typed network, per-layer auto-tuning) instead
    # of taking it from the graph — the trtexec --fp16 / --bf16 path. Keep the ONNX fp32
    # and TensorRT converts + tunes, holding numerically-sensitive layers at higher
    # precision automatically (no manual fp16 block-list). REQUIRES a decomposed graph
    # (opset 18; the native opset-23 Attention/RoPE ops need a strongly-typed network,
    # which any precision flag disables). Leave BOTH off (default) to take precision from
    # the graph (i.e. a mixed-fp16 export, which is strongly-typed). bf16 needs Ampere+
    # (sm_80); fp16 since Volta. Don't set both.
    #
    # WARNING: fp16 on ALL blocks BREAKS SAM2's masks — the fp16 error compounds across
    # blocks + the recurrent memory loop and the masklet is lost (mask IoU ~0). Each
    # block is individually fp16-safe, but not all together. So fp16_enable is applied
    # ONLY to the blocks in ``fp16_safe_blocks`` (the 2 heavy blocks the validated
    # mixed-fp16 export targets); the recurrent-loop / mask-sensitive blocks stay fp32.
    # OrtBlock logs, per block, whether fp16 was applied or held at fp32. bf16_enable is
    # safe for all blocks (fp32 range) — it is the recommended full-speed Ampere path.
    fp16_enable: bool = False
    bf16_enable: bool = False
    # Blocks (matched by .onnx file stem) for which fp16_enable actually applies. The
    # default is the two heavy blocks the mixed-fp16 export validates as fp16-safe;
    # every other block is held at fp32 even when fp16_enable is set, because fp16 error
    # compounds through the recurrent memory loop. Does NOT gate bf16 (safe everywhere).
    fp16_safe_blocks: frozenset[str] = frozenset({"image_encoder", "memory_attention"})
    # Capture each TensorRT engine's execution as a CUDA graph and replay it, removing
    # per-kernel launch overhead. Needs stable input shapes + buffer addresses across
    # runs; ORT re-captures when they change, so it helps the static blocks (image
    # encoder) more than the dynamic-shape memory attention.
    cuda_graph_enable: bool = False
    cache_enable: bool = True
    cache_dir: str | Path | None = None
    timing_cache_enable: bool = True
    # None -> share the engine cache directory.
    timing_cache_dir: str | Path | None = None
    # Reuse the timing cache even if the device/CUDA/TRT version differs (faster
    # builds at the risk of slightly sub-optimal tactic choices on a mismatch).
    force_timing_cache: bool = False
    # 0-5; higher = more tactic search => faster engine, slower build. Default 3 keeps
    # iteration fast; use 5 for a max-performance Ampere+ deployment engine (built once,
    # cached). On Ampere+ the strongly-typed fp16 graph lets TensorRT fuse a flash/MHA
    # attention kernel; on Turing (no fused-MHA) the CUDA EP is the faster ONNX path.
    builder_optimization_level: int = 3
    max_workspace_size: int = 4 << 30  # 4 GiB
    force_sequential_build: bool = False
    max_partition_iterations: int = 1000
    min_subgraph_size: int = 1
    # Optional path for the EP-context (pre-built engine) ONNX file.
    ep_context_file_path: str | Path | None = None

    def __post_init__(self) -> None:
        # Both precision flags build a weakly-typed network; setting both is ambiguous
        # (TRT would pick one) and conflates two distinct paths. Force the caller to
        # choose: bf16 (safe for all blocks) or fp16 (safe blocks only).
        if self.fp16_enable and self.bf16_enable:
            raise ValueError(
                "TensorRTOptions: set fp16_enable OR bf16_enable, not both. "
                "bf16_enable is safe for all blocks; fp16_enable applies only to "
                "fp16_safe_blocks."
            )
