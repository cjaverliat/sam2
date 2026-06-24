# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Benchmark per-frame speed of the plain PyTorch SAM 2 pipeline (no ONNX/TensorRT).

Same pipeline and timing as examples/benchmark_onnx.py — use it as the torch baseline.
``use_half`` (default on) lets the model's forward decorators apply autocast +
inference_mode: bf16 on Ampere+ (sm_80), fp16 on older GPUs like Turing.

Throughput knobs that do NOT compile the image encoder:
    --cudnn-benchmark  autotune cuDNN convs for the fixed input shape
    --channels-last    run the conv stacks in channels-last (NHWC) layout
    --forgetful        bounded memory bank (keep only a window of recent memories)

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
from sam2.modeling.sam2_forgetful_memory import SAM2ForgetfulObjectMemoryBank


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", default="notebooks/videos/bedroom.mp4")
    p.add_argument("--checkpoint", default="checkpoints/sam2.1_hiera_base_plus.pt")
    p.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    p.add_argument("--fp32", action="store_true", help="disable half precision (use_half=False)")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the image encoder (vos_optimized path)")
    p.add_argument("--cudnn-benchmark", action="store_true",
                   help="autotune cuDNN conv algos for the fixed input shape")
    p.add_argument("--channels-last", action="store_true",
                   help="run conv stacks in channels-last (NHWC) memory format")
    p.add_argument("--forgetful", action="store_true",
                   help="use a bounded (forgetful) memory bank instead of the infinite one")
    p.add_argument("--memory-window", type=int, default=7,
                   help="window size for the forgetful memory bank (frames kept around current)")
    p.add_argument("--warmup-frames", type=int, default=5,
                   help="propagation frames discarded before timing")
    p.add_argument("--max-frames", type=int, default=200, help="stop after N frames (0=all)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}  half: {not args.fp32}  "
          f"cudnn_benchmark: {args.cudnn_benchmark}  channels_last: {args.channels_last}  "
          f"forgetful: {args.forgetful}")
    # tf32 for fp32 matmuls/convs (Ampere sm_80+; no-op on older GPUs like Turing).
    if device.type == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # Autotune cuDNN conv algos: the per-frame input shape is fixed, so the one-time
    # search cost is amortised over the whole video.
    if args.cudnn_benchmark and device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # use_half: the model's forward decorators apply autocast (bf16 on sm_80+, fp16
    # otherwise) + inference_mode per call — no manual precision context needed.
    predictor = build_sam2_generic_video_predictor(
        args.model_cfg, args.checkpoint, device=device, use_half=not args.fp32,
        vos_optimized=args.compile, apply_postprocessing=True,
    )

    # channels-last: NHWC lets the conv stacks (image-encoder patch embed + FPN neck,
    # mask-decoder / memory-encoder convs) hit faster tensor-core kernels. Put the
    # model in NHWC and feed the encoder NHWC input; the 3D attention tensors are
    # untouched. Does not compile anything.
    if args.channels_last:
        predictor.image_encoder = predictor.image_encoder.to(memory_format=torch.channels_last)
        _prepare = predictor._prepare_images

        def _prepare_channels_last(*a, **k):
            return _prepare(*a, **k).to(memory_format=torch.channels_last)

        predictor._prepare_images = _prepare_channels_last

    memory_bank = None
    if args.forgetful:
        memory_bank = SAM2ForgetfulObjectMemoryBank(memory_window_size=args.memory_window)

    run_video_benchmark(predictor, args.video, device, args.warmup_frames,
                        args.max_frames, desc="torch", memory_bank=memory_bank)


if __name__ == "__main__":
    main()
