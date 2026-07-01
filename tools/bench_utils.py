# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Shared benchmark loop for the ONNX and torch SAM 2 predictors.

The pipeline is identical for both: one point prompt on frame 0, then propagate the
masklet across the video, timing each ``predictor.forward(...)``. Only the predictor
construction differs, so the timing loop lives here.
"""

import statistics
import time

import cv2
import torch
from tqdm import tqdm

from sam.modeling.memory.forgetful import ForgetfulObjectMemoryBank
from sam.modeling.memory.banks import Sam2ObjectMemoryBank
from sam.prompts import GeometryPrompt
from sam.models.sam2_predictor import Sam2VideoPredictorState


def add_common_bench_args(parser):
    """Add the CLI args shared by the ONNX and torch video benchmarks."""
    parser.add_argument("--video", default="notebooks/videos/bedroom.mp4")
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument("--warmup-frames", type=int, default=5,
                        help="propagation frames discarded before timing (engine/cache settle)")
    parser.add_argument("--max-frames", type=int, default=200, help="stop after N frames (0=all)")
    parser.add_argument("--memory-window", type=int, default=7,
                        help="forgetful memory window in frames (0=infinite bank)")
    parser.add_argument("--bank-device", default="cuda", choices=["cuda", "cpu"],
                        help="device the memory bank stores memories on")


def read_frame(cap, device):
    ret, frame = cap.read()
    if not ret:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = torch.as_tensor(frame).permute(2, 0, 1).to(device)
    return frame / 255.0


def warmup(predictor, video_state, device):
    """Pay one-time CUDA/cuDNN init cost up front. Empty prompts return early
    without writing memory, so state stays clean."""
    dummy = torch.zeros(3, *video_state.video_hw, device=device)
    predictor.forward(
        state=video_state, frame=dummy, frame_idx=0, prompts=[], create_memory=False
    )


def build_memory_bank(memory_window, bank_device):
    device = torch.device(bank_device)
    if memory_window > 0:
        return ForgetfulObjectMemoryBank(
            memory_window_size=memory_window, storage_device=device)
    return Sam2ObjectMemoryBank(storage_device=device)


def sync(device):
    """Block until queued GPU work finishes so timings are wall-accurate."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def summarize(times_ms):
    """Print latency stats (ms) and throughput (FPS) for the per-frame times."""
    n = len(times_ms)
    s = sorted(times_ms)
    mean = statistics.mean(times_ms)
    print("\n--- Benchmark results ---")
    print(f"frames timed : {n}")
    print(f"mean         : {mean:7.2f} ms")
    print(f"median       : {statistics.median(times_ms):7.2f} ms")
    print(f"min / max    : {s[0]:7.2f} / {s[-1]:7.2f} ms")
    print(f"p90 / p99    : {s[min(n - 1, int(0.90 * n))]:7.2f} / "
          f"{s[min(n - 1, int(0.99 * n))]:7.2f} ms")
    print(f"throughput   : {1000.0 / mean:7.2f} FPS")


def run_video_benchmark(predictor, video, device, warmup_frames=5, max_frames=200,
                        desc="bench", memory_bank=None):
    """Prompt frame 0 with a single point, propagate, time each frame, print stats.

    The first ``warmup_frames`` propagation frames are discarded (engine/cache settle).
    ``max_frames=0`` runs the whole video. ``memory_bank`` overrides the default
    (infinite) bank — pass a ``ForgetfulObjectMemoryBank`` for bounded memory.
    """
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_hw = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
               int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

    state = Sam2VideoPredictorState.create(orig_hw, memory_bank)
    warmup(predictor, state, device)

    prompt = GeometryPrompt(
        obj_id=1,
        points_coords=torch.tensor([[210, 350]], dtype=torch.float32, device=device),
        points_labels=torch.tensor([1], device=device),
    )
    frame = read_frame(cap, device)
    sync(device)
    t0 = time.perf_counter()
    predictor.forward(state=state, frame=frame, frame_idx=0, prompts=[prompt])
    sync(device)
    print(f"prompt frame : {(time.perf_counter() - t0) * 1000.0:7.2f} ms (not in stats)")

    last = n_frames if max_frames <= 0 else min(n_frames, 1 + max_frames)
    times_ms = []
    pbar = tqdm(range(1, last), desc=f"Benchmarking ({desc})")
    for f in pbar:
        frame = read_frame(cap, device)
        if frame is None:
            break
        sync(device)
        t0 = time.perf_counter()
        predictor.forward(state=state, frame=frame, frame_idx=f)
        sync(device)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if f > warmup_frames:
            times_ms.append(dt_ms)
        pbar.set_postfix({"ms": f"{dt_ms:.1f}"})
    pbar.close()
    cap.release()

    if not times_ms:
        print("No frames timed. Increase --max-frames or lower --warmup-frames.")
        return
    summarize(times_ms)
