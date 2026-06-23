# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Benchmark per-frame speed of the plain PyTorch SAM 2 pipeline (no ONNX/TensorRT).

Same pipeline and timing as examples/benchmark_onnx.py — use it as the torch baseline.
``use_half`` (default on) lets the model's forward decorators apply autocast +
inference_mode: bf16 on Ampere+ (sm_80), fp16 on older GPUs like Turing.

Run from the repo root:
    pixi run python examples/benchmark_torch.py \
        --checkpoint checkpoints/sam2.1_hiera_base_plus.pt \
        --model-cfg configs/sam2.1/sam2.1_hiera_b+.yaml
"""

import argparse
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

from bench_utils import run_video_benchmark
from sam2.build_sam import build_sam2_generic_video_predictor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", default="notebooks/videos/bedroom.mp4")
    p.add_argument("--checkpoint", default="checkpoints/sam2.1_hiera_base_plus.pt")
    p.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    p.add_argument("--fp32", action="store_true", help="disable half precision (use_half=False)")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the image encoder (vos_optimized path)")
    p.add_argument("--warmup-frames", type=int, default=5,
                   help="propagation frames discarded before timing")
    p.add_argument("--max-frames", type=int, default=200, help="stop after N frames (0=all)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}  half: {not args.fp32}")
    # tf32 for fp32 matmuls/convs (Ampere sm_80+; no-op on older GPUs like Turing).
    if device.type == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # use_half: the model's forward decorators apply autocast (bf16 on sm_80+, fp16
    # otherwise) + inference_mode per call — no manual precision context needed.
    predictor = build_sam2_generic_video_predictor(
        args.model_cfg, args.checkpoint, device=device, use_half=not args.fp32,
        vos_optimized=args.compile, apply_postprocessing=True,
    )
    run_video_benchmark(predictor, args.video, device, args.warmup_frames,
                        args.max_frames, desc="torch")


if __name__ == "__main__":
    main()
