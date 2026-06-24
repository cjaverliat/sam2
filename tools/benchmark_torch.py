# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Benchmark per-frame speed of the plain PyTorch SAM 2 pipeline (no ONNX/TensorRT).

Identical pipeline, timing, memory bank and CLI to tools/benchmark_onnx.py — the
only difference is the predictor: this builds the torch model, that swaps in the
ONNX/TensorRT blocks. ``use_half`` (default on) lets the model's forward decorators
apply autocast + inference_mode: bf16 on Ampere+ (sm_80), fp16 on older GPUs.

Run from the repo root:
    pixi run python tools/benchmark_torch.py \
        --checkpoint checkpoints/sam2.1_hiera_base_plus.pt \
        --model-cfg configs/sam2.1/sam2.1_hiera_b+.yaml
"""

import argparse
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

from bench_utils import build_memory_bank, run_video_benchmark
from sam2.build_sam import build_sam2_generic_video_predictor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", default="notebooks/videos/bedroom.mp4")
    p.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    p.add_argument("--warmup-frames", type=int, default=5,
                   help="propagation frames discarded before timing (engine/cache settle)")
    p.add_argument("--max-frames", type=int, default=200, help="stop after N frames (0=all)")
    p.add_argument("--memory-window", type=int, default=7,
                   help="forgetful memory window in frames (0=infinite bank)")
    p.add_argument("--bank-device", default="cuda", choices=["cuda", "cpu"],
                   help="device the memory bank stores memories on")
    # torch predictor build
    p.add_argument("--checkpoint", default="checkpoints/sam2.1_hiera_base_plus.pt")
    p.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
                   help="auto=bf16 on Ampere+/fp16 otherwise; fp32 disables autocast")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the image encoder (vos_optimized path)")
    p.add_argument("--cudnn-benchmark", action="store_true",
                   help="autotune cuDNN conv algos for the fixed input shape")
    p.add_argument("--channels-last", action="store_true",
                   help="run conv stacks in channels-last (NHWC) memory format")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    torch.no_grad().__enter__()
    # tf32 for fp32 matmuls/convs (Ampere sm_80+; no-op on older GPUs like Turing).
    if device.type == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.cudnn_benchmark and device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    predictor = build_sam2_generic_video_predictor(
        args.model_cfg, args.checkpoint, device=device, use_half=args.precision != "fp32",
        vos_optimized=args.compile, apply_postprocessing=True,
    )
    if args.precision in ("bf16", "fp16"):
        predictor._half_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

    if args.channels_last:
        predictor.image_encoder = predictor.image_encoder.to(memory_format=torch.channels_last)
        _prepare = predictor._prepare_images

        def _prepare_channels_last(*a, **k):
            return _prepare(*a, **k).to(memory_format=torch.channels_last)

        predictor._prepare_images = _prepare_channels_last

    memory_bank = build_memory_bank(args.memory_window, args.bank_device)

    run_video_benchmark(predictor, args.video, device, args.warmup_frames,
                        args.max_frames, desc="torch", memory_bank=memory_bank)


if __name__ == "__main__":
    main()
