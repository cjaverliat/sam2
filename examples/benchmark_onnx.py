# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Benchmark per-frame processing speed of the ONNX/TensorRT SAM 2 pipeline.

Same pipeline as examples/video_predictor_example_onnx.py: a single point prompt
is added on frame 0, then the masklet is propagated across the video. Instead of
saving figures, this script times each `predictor.forward(...)` call and reports
latency statistics and throughput (FPS).

Defaults target the hiera base-plus (b+) model.

Run from the repo root:
    pixi run python tools/export_onnx.py \
        --config configs/sam2.1/sam2.1_hiera_b+.yaml \
        --ckpt checkpoints/sam2.1_hiera_base_plus.pt --out-dir onnx_sam2

    pixi run python examples/benchmark_onnx.py \
        --onnx-dir onnx_sam2 \
        --model-cfg configs/sam2.1/sam2.1_hiera_b+.yaml

No checkpoint needed at runtime: all weights are baked into the export artifacts
produced by tools/export_onnx.py.
"""

import argparse
import os
import statistics
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import torch
from tqdm import tqdm

from frame_utils import read_frame, warmup
from sam2.build_sam import build_sam2_generic_video_predictor_onnx
from sam2.modeling.sam2_prompt import SAM2Prompt
from sam2.onnx.trt_options import TensorRTOptions
from sam2.sam2_generic_video_predictor import SAM2GenericVideoPredictorState


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
    fps = 1000.0 / mean
    print("\n--- Benchmark results ---")
    print(f"frames timed : {n}")
    print(f"mean         : {mean:7.2f} ms")
    print(f"median       : {median:7.2f} ms")
    print(f"min / max    : {min(times_ms):7.2f} / {max(times_ms):7.2f} ms")
    print(f"p90 / p99    : {p90:7.2f} / {p99:7.2f} ms")
    print(f"throughput   : {fps:7.2f} FPS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="notebooks/videos/bedroom.mp4")
    parser.add_argument("--onnx-dir", default="outputs/onnx/sam2.1_hiera_base_plus/opset18-fp16")
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument("--no-trt", action="store_true", help="CUDA EP instead of TensorRT")
    parser.add_argument(
        "--warmup-frames", type=int, default=5,
        help="propagation frames to discard before timing (engine/cache settle)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=200,
        help="stop after N propagation frames (0 = all)",
    )
    parser.add_argument(
        "--builder-opt-level", type=int, default=None,
        help="TRT builder optimization level (0-5). Lower = faster build, possibly "
             "slower engine. Default uses TensorRTOptions' value.",
    )
    parser.add_argument(
        "--trt-cache-dir", default=None,
        help="override the TRT engine/timing cache dir. Use a distinct dir to isolate "
             "builds with different settings (else the cached engine is reused).",
    )
    parser.add_argument(
        "--max-workspace-gib", type=float, default=None,
        help="TRT max workspace size (GiB). Lower it on small-VRAM GPUs: each engine "
             "partition reserves up to this much at build time, so the default can OOM.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    torch.no_grad().__enter__()

    # No fp16 flag: strongly-typed TRT network takes precision from the graph (use a
    # mixed-fp16 export for fp16).
    trt_kwargs = {}
    if args.builder_opt_level is not None:
        trt_kwargs["builder_optimization_level"] = args.builder_opt_level
    if args.trt_cache_dir:
        trt_kwargs["cache_dir"] = args.trt_cache_dir
    if args.max_workspace_gib is not None:
        trt_kwargs["max_workspace_size"] = int(args.max_workspace_gib * (1 << 30))
    trt_opts = TensorRTOptions(**trt_kwargs)
    predictor = build_sam2_generic_video_predictor_onnx(
        args.model_cfg, args.onnx_dir, device=device,
        use_trt=not args.no_trt, trt_opts=trt_opts, apply_postprocessing=True,
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

    # Single point prompt on frame 0 (same as the ONNX example).
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
    prompt_ms = (time.perf_counter() - t0) * 1000.0
    print(f"prompt frame : {prompt_ms:7.2f} ms (encode + prompt + memory, not in stats)")

    last = n_frames if args.max_frames <= 0 else min(n_frames, 1 + args.max_frames)
    times_ms = []
    pbar = tqdm(range(1, last), desc="Benchmarking (ONNX)")
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
