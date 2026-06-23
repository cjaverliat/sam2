# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Benchmark per-frame speed of the ONNX/TensorRT SAM 2 pipeline.

Times each ``predictor.forward(...)`` over a propagated masklet (one point prompt on
frame 0) and reports latency + throughput. No checkpoint needed at runtime — all
weights are baked into the export artifacts from tools/export_onnx.py.

Run from the repo root:
    pixi run python tools/export_onnx.py \
        --config configs/sam2.1/sam2.1_hiera_b+.yaml \
        --ckpt checkpoints/sam2.1_hiera_base_plus.pt --out-dir onnx_sam2
    pixi run python examples/benchmark_onnx.py --onnx-dir onnx_sam2 \
        --model-cfg configs/sam2.1/sam2.1_hiera_b+.yaml

Precision: point --onnx-dir at a mixed-fp16 export for fp16 (CUDA EP or strongly-typed
TRT). For the weakly-typed TRT auto-tune path, point it at the fp32 export and pass
--trt-fp16 / --trt-bf16 (decomposed opset-18 export only).
"""

import argparse
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

from bench_utils import run_video_benchmark
from sam2.build_sam import build_sam2_generic_video_predictor_onnx
from sam2.onnx.trt_options import TensorRTOptions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", default="notebooks/videos/bedroom.mp4")
    p.add_argument("--onnx-dir", default="outputs/onnx/sam2.1_hiera_base_plus/opset18-fp16")
    p.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    p.add_argument("--no-trt", action="store_true", help="CUDA EP instead of TensorRT")
    p.add_argument("--warmup-frames", type=int, default=5,
                   help="propagation frames discarded before timing (engine/cache settle)")
    p.add_argument("--max-frames", type=int, default=200, help="stop after N frames (0=all)")
    p.add_argument("--builder-opt-level", type=int, default=None,
                   help="TRT builder optimization level 0-5 (lower=faster build)")
    p.add_argument("--trt-cache-dir", default=None,
                   help="override TRT engine/timing cache dir (isolate distinct builds)")
    p.add_argument("--max-workspace-gib", type=float, default=None,
                   help="TRT max workspace (GiB); lower on small-VRAM GPUs to avoid OOM")
    p.add_argument("--trt-fp16", action="store_true",
                   help="weakly-typed TRT fp16 auto-tune (fp32 opset-18 export)")
    p.add_argument("--trt-bf16", action="store_true",
                   help="weakly-typed TRT bf16 auto-tune (Ampere+; fp32 opset-18 export)")
    p.add_argument("--trt-cuda-graph", action="store_true",
                   help="capture/replay engines as CUDA graphs (cut launch overhead)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    torch.no_grad().__enter__()

    trt_kwargs = {}
    if args.builder_opt_level is not None:
        trt_kwargs["builder_optimization_level"] = args.builder_opt_level
    if args.trt_cache_dir:
        trt_kwargs["cache_dir"] = args.trt_cache_dir
    if args.max_workspace_gib is not None:
        trt_kwargs["max_workspace_size"] = int(args.max_workspace_gib * (1 << 30))
    trt_kwargs["fp16_enable"] = args.trt_fp16
    trt_kwargs["bf16_enable"] = args.trt_bf16
    trt_kwargs["cuda_graph_enable"] = args.trt_cuda_graph

    predictor = build_sam2_generic_video_predictor_onnx(
        args.model_cfg, args.onnx_dir, device=device,
        use_trt=not args.no_trt, trt_opts=TensorRTOptions(**trt_kwargs),
        apply_postprocessing=True,
    )
    run_video_benchmark(predictor, args.video, device, args.warmup_frames,
                        args.max_frames, desc="ONNX")


if __name__ == "__main__":
    main()
