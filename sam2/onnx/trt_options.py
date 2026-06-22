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

    cache_enable: bool = True
    cache_dir: str | Path | None = None
    timing_cache_enable: bool = True
    # None -> share the engine cache directory.
    timing_cache_dir: str | Path | None = None
    # Reuse the timing cache even if the device/CUDA/TRT version differs (faster
    # builds at the risk of slightly sub-optimal tactic choices on a mismatch).
    force_timing_cache: bool = False
    builder_optimization_level: int = 3
    max_workspace_size: int = 4 << 30  # 4 GiB
    force_sequential_build: bool = False
    max_partition_iterations: int = 1000
    min_subgraph_size: int = 1
    # Optional path for the EP-context (pre-built engine) ONNX file.
    ep_context_file_path: str | Path | None = None
