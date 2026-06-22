# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Benchmark per-frame processing speed of the plain PyTorch SAM 2 pipeline.

Same pipeline and timing as examples/benchmark_onnx.py (single point prompt on frame
0, then propagate the masklet), but the model runs as the normal torch
SAM2GenericVideoPredictor — no ONNX, no TensorRT. Use it to compare against
benchmark_onnx.py.

``use_half`` (default on) lets the model's forward decorators apply autocast +
inference_mode themselves: bf16 on Ampere+ (sm_80), fp16 on older GPUs like Turing.

Defaults target the hiera base-plus (b+) model. Run from the repo root:
    pixi run python examples/benchmark_torch.py \
        --checkpoint checkpoints/sam2.1_hiera_base_plus.pt \
        --model-cfg configs/sam2.1/sam2.1_hiera_b+.yaml
"""

import argparse
import os
import statistics
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import torch
from tqdm import tqdm

from sam2.build_sam import build_sam2_generic_video_predictor
from sam2.modeling.sam2_prompt import SAM2Prompt
from sam2.sam2_generic_video_predictor import SAM2GenericVideoPredictorState

# Reuse the IO / warmup helpers (cv2/torch only, no matplotlib).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frame_utils import read_frame, warmup  # noqa: E402


def sync(device):
    """Block until queued GPU work finishes so timings are wall-accurate."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def summarize(times_ms):
    """Print latency stats (ms) and throughput (FPS) for a list of per-frame times."""
    n = len(times_ms)
    mean = statistics.mean(times_ms)
    median = statistics.median(times_ms)
    p90 = sorted(times_ms)[min(n - 1, int(0.90 * n))]
    p99 = sorted(times_ms)[min(n - 1, int(0.99 * n))]
    print("\n--- Benchmark results ---")
    print(f"frames timed : {n}")
    print(f"mean         : {mean:7.2f} ms")
    print(f"median       : {median:7.2f} ms")
    print(f"min / max    : {min(times_ms):7.2f} / {max(times_ms):7.2f} ms")
    print(f"p90 / p99    : {p90:7.2f} / {p99:7.2f} ms")
    print(f"throughput   : {1000.0 / mean:7.2f} FPS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="notebooks/videos/bedroom.mp4")
    parser.add_argument("--checkpoint", default="checkpoints/sam2.1_hiera_base_plus.pt")
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument("--fp32", action="store_true",
                        help="disable half precision (use_half=False); default runs half")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the image encoder (vos_optimized path)")
    parser.add_argument("--warmup-frames", type=int, default=5,
                        help="propagation frames to discard before timing")
    parser.add_argument("--max-frames", type=int, default=200,
                        help="stop after N propagation frames (0 = all)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}  half: {not args.fp32}")
    # tf32 for fp32 matmuls/convs (Ampere sm_80+; no-op on older GPUs like Turing).
    if device.type == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # use_half: the model's forward decorators apply autocast (bf16 on sm_80+, fp16
    # otherwise) + inference_mode per call — no manual precision context needed here.
    predictor = build_sam2_generic_video_predictor(
        args.model_cfg, args.checkpoint, device=device, use_half=not args.fp32,
        vos_optimized=args.compile, apply_postprocessing=True,
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_hw = (height, width)

    video_state = SAM2GenericVideoPredictorState.create(orig_hw)
    warmup(predictor, video_state, device)

    # Single point prompt on frame 0 (same as the ONNX benchmark).
    ann_frame_idx = 0
    ann_obj_id = 1
    points = torch.tensor([[210, 350]], dtype=torch.float32, device=device)
    labels = torch.tensor([1], device=device)
    prompt = SAM2Prompt(obj_id=ann_obj_id, points_coords=points, points_labels=labels)

    initial_frame = read_frame(cap, device)
    sync(device)
    t0 = time.perf_counter()
    predictor.forward(
        state=video_state, frame=initial_frame, frame_idx=ann_frame_idx, prompts=[prompt],
    )
    sync(device)
    print(f"prompt frame : {(time.perf_counter() - t0) * 1000.0:7.2f} ms (not in stats)")

    last = n_frames if args.max_frames <= 0 else min(n_frames, 1 + args.max_frames)
    times_ms = []
    pbar = tqdm(range(1, last), desc="Benchmarking (torch)")
    for f in pbar:
        frame = read_frame(cap, device)
        if frame is None:
            break
        sync(device)
        t0 = time.perf_counter()
        predictor.forward(state=video_state, frame=frame, frame_idx=f)
        sync(device)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if f > args.warmup_frames:
            times_ms.append(dt_ms)
        pbar.set_postfix({"ms": f"{dt_ms:.1f}"})

    pbar.close()
    cap.release()

    if not times_ms:
        print("No frames timed. Increase --max-frames or lower --warmup-frames.")
        return
    summarize(times_ms)


if __name__ == "__main__":
    main()
