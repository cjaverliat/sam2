import logging
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from sam2.onnx.trt_options import TensorRTOptions

logger = logging.getLogger(__name__)

_TRT_DLLS_PRELOADED = False


def _preload_trt_dlls() -> None:
    """Make ORT's TensorRT EP loadable under the Windows pip layout.

    ORT resolves the TensorRT EP's DLLs only from its own ``onnxruntime/capi`` dir
    (its load ignores PATH and ``add_dll_directory``), but the wheels scatter them:
    nvinfer in ``tensorrt_libs``, cuDNN/cuBLAS/cudart in ``torch/lib``. Preload the
    whole chain by full path here so the EP's by-name imports bind to the
    already-loaded modules. cuDNN/cuBLAS first (nvinfer depends on them), then
    nvinfer. Best-effort, Windows-only, runs once."""
    global _TRT_DLLS_PRELOADED
    if _TRT_DLLS_PRELOADED or sys.platform != "win32":
        return
    _TRT_DLLS_PRELOADED = True
    import ctypes

    # cuDNN / cuBLAS / CUDA runtime (nvinfer's dependencies) via ORT's own helper.
    try:
        ort.preload_dlls(cuda=True, cudnn=True)
    except Exception as e:  # pragma: no cover - environment-dependent
        logger.debug("ort.preload_dlls failed: %s", e)
    # nvinfer + plugin + onnx parser from the tensorrt wheel.
    try:
        import tensorrt_libs

        libs = Path(tensorrt_libs.__file__).parent
        for dll in sorted(libs.glob("nvinfer*.dll")) + sorted(libs.glob("nvonnxparser*.dll")):
            try:
                ctypes.WinDLL(str(dll))
            except OSError as e:  # pragma: no cover - environment-dependent
                logger.debug("preload %s failed: %s", dll.name, e)
    except Exception as e:  # pragma: no cover - environment-dependent
        logger.debug("tensorrt_libs preload skipped: %s", e)

_TORCH_TO_NP = {
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.int64: np.int64,
    torch.int32: np.int32,
    torch.bool: np.bool_,
}


def _shape_str(profiles: dict[str, tuple[int, ...]]) -> str:
    """ "name:1x3x512x512,other:1x256" for a TRT profile dict."""
    return ",".join(f"{n}:{'x'.join(str(d) for d in s)}" for n, s in profiles.items())


class OrtBlock:
    """Runs one exported ONNX graph through ONNX Runtime, preferring the TensorRT
    execution provider (then CUDA, then CPU). Inputs/outputs are bound to GPU memory
    via IO binding so tensors stay resident on the device between blocks.

    Subclasses wrap this with a ``forward(...)`` matching the torch module they
    replace. They construct the block with the input/output names of the graph and,
    for TRT, the min/opt/max shape profiles per dynamic input.
    """

    def __init__(
        self,
        onnx_path: str | Path,
        input_names: list[str],
        output_names: list[str],
        device: torch.device,
        use_trt: bool = True,
        trt_opts: TensorRTOptions | None = None,
        min_shapes: dict[str, tuple[int, ...]] | None = None,
        opt_shapes: dict[str, tuple[int, ...]] | None = None,
        max_shapes: dict[str, tuple[int, ...]] | None = None,
    ):
        onnx_path = Path(onnx_path)
        if not onnx_path.is_file():
            raise FileNotFoundError(f"ONNX model not found at {onnx_path}")

        self.onnx_path = onnx_path
        self.input_names = input_names
        self.output_names = output_names
        self.device = device
        trt_opts = trt_opts or TensorRTOptions()

        self.session = self._init_session(
            onnx_path, device, use_trt, trt_opts, min_shapes, opt_shapes, max_shapes
        )
        active = self.session.get_providers()[0]
        self._bind_device = (
            device if active != "CPUExecutionProvider" else torch.device("cpu")
        )
        logger.info("%s loaded on %s (%s)", onnx_path.name, self._bind_device, active)

    @staticmethod
    def _init_session(
        onnx_path, device, use_trt, trt_opts, min_shapes, opt_shapes, max_shapes
    ) -> ort.InferenceSession:
        providers: list = ["CPUExecutionProvider"]
        if device.type == "cuda":
            # Preload the CUDA/cuDNN (+ TRT) DLLs so both the CUDA and TensorRT EPs load
            # under the Windows pip layout (their loader ignores PATH/add_dll_directory).
            _preload_trt_dlls()
            device_id = device.index if device.index is not None else 0
            stream = str(torch.cuda.current_stream(device).cuda_stream)
            providers.insert(
                0,
                (
                    "CUDAExecutionProvider",
                    {"device_id": device_id, "user_compute_stream": stream},
                ),
            )
            if use_trt:
                cache_dir = (
                    Path(trt_opts.cache_dir)
                    if trt_opts.cache_dir
                    else onnx_path.parent / "trt_cache"
                )
                if trt_opts.cache_enable:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                timing_dir = (
                    Path(trt_opts.timing_cache_dir)
                    if trt_opts.timing_cache_dir
                    else cache_dir
                )
                if trt_opts.timing_cache_enable:
                    timing_dir.mkdir(parents=True, exist_ok=True)
                trt = {
                    "device_id": device_id,
                    "user_compute_stream": stream,
                    # No trt_fp16_enable: it forces a weakly-typed network, which
                    # rejects the native opset-23 Attention/RoPE ops. Leaving it off
                    # makes the EP build a strongly-typed network (precision comes from
                    # the graph; use a mixed-fp16 export for fp16).
                    "trt_engine_cache_enable": trt_opts.cache_enable,
                    "trt_engine_cache_path": cache_dir.as_posix(),
                    "trt_engine_cache_prefix": onnx_path.stem,
                    # Timing cache: per-tactic kernel timings reused across builds,
                    # cutting the cost of the first build of each engine.
                    "trt_timing_cache_enable": trt_opts.timing_cache_enable,
                    "trt_timing_cache_path": timing_dir.as_posix(),
                    "trt_force_timing_cache": trt_opts.force_timing_cache,
                    "trt_builder_optimization_level": trt_opts.builder_optimization_level,
                    "trt_max_workspace_size": trt_opts.max_workspace_size,
                    "trt_force_sequential_engine_build": trt_opts.force_sequential_build,
                    "trt_max_partition_iterations": trt_opts.max_partition_iterations,
                    "trt_min_subgraph_size": trt_opts.min_subgraph_size,
                }
                if min_shapes and opt_shapes and max_shapes:
                    trt["trt_profile_min_shapes"] = _shape_str(min_shapes)
                    trt["trt_profile_opt_shapes"] = _shape_str(opt_shapes)
                    trt["trt_profile_max_shapes"] = _shape_str(max_shapes)
                if trt_opts.ep_context_file_path is not None:
                    trt["trt_ep_context_file_path"] = Path(
                        trt_opts.ep_context_file_path
                    ).as_posix()
                providers.insert(0, ("TensorrtExecutionProvider", trt))

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        session = ort.InferenceSession(
            onnx_path.as_posix(), providers=providers, sess_options=so
        )
        if use_trt and device.type == "cuda":
            if "TensorrtExecutionProvider" not in session.get_providers():
                logger.warning(
                    "TensorRT EP unavailable for %s; falling back to %s",
                    onnx_path.name,
                    session.get_providers()[0],
                )
        return session

    def _to_torch(self, ov: "ort.OrtValue") -> torch.Tensor:
        try:
            return torch.from_dlpack(ov).to(self.device)
        except Exception:
            return torch.from_numpy(ov.numpy()).to(self.device)

    def run(self, inputs: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """Bind ``inputs`` (by name) and run, returning outputs as torch tensors on
        ``self.device`` in ``output_names`` order."""
        io = self.session.io_binding()
        dev_type = self._bind_device.type
        dev_id = self._bind_device.index if self._bind_device.index is not None else 0

        held = []  # keep input tensors alive until run completes
        for name in self.input_names:
            t = inputs[name].to(self._bind_device).contiguous()
            held.append(t)
            io.bind_input(
                name=name,
                device_type=dev_type,
                device_id=dev_id,
                element_type=_TORCH_TO_NP[t.dtype],
                shape=tuple(t.shape),
                buffer_ptr=t.data_ptr(),
            )
        for name in self.output_names:
            io.bind_output(name=name, device_type=dev_type, device_id=dev_id)

        self.session.run_with_iobinding(io)
        return [self._to_torch(o) for o in io.get_outputs()]
