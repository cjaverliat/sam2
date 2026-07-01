# SPDX-License-Identifier: LicenseRef-SAM
"""Video-prediction latency benchmark: EfficientTAM vs SAM 2.1 vs SAM 3.1 on the bedroom clip.

Each model runs its NATURAL video-predict workload over the same source frames:
  - EfficientTAM / SAM 2.1: a single geometry prompt (one click on frame 0) -> track 1 object.
  - SAM 3.1 (multiplex):     a text concept ("person") -> detect + track ALL instances every frame.
So this measures real per-model video-predict cost, not an apples-to-apples single-object workload.

Models are built/run/torn-down sequentially (the 3.5 GB SAM 3.1 weights won't co-reside).
Timing: torch.cuda.synchronize() around each frame; the first `--warmup` frames (incl. the
heavier prompt/detect-seed frame 0) are reported separately and excluded from steady-state stats.

Usage: pixi run python tools/benchmark_video.py [--n 30] [--warmup 3]
"""
import argparse
import gc
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BEDROOM = Path("notebooks/videos/bedroom")
CLICK_XY = (385.0, 230.0)  # a kid in the native 960x540 frame (x, y), for the sam2/etam prompt
SAM3_PHRASE = "person"

# (label, kind, config, checkpoint)
MODELS = [
    ("EfficientTAM-s", "sam2", "configs/efficienttam/efficienttam_s.yaml", "checkpoints/efficienttam_s.pt"),
    ("SAM2.1-small", "sam2", "configs/sam2.1/sam2.1_hiera_s.yaml", "checkpoints/sam2.1_hiera_small.pt"),
    ("SAM2.1-large", "sam2", "configs/sam2.1/sam2.1_hiera_l.yaml", "checkpoints/sam2.1_hiera_large.pt"),
    ("SAM3.1-multiplex", "sam3p1", "configs/sam3/sam3.1.yaml", "checkpoints/sam3.1_multiplex.pt"),
]


def load_frames(n):
    """First `n` bedroom frames as (H, W, 3) uint8 at native resolution."""
    fps = sorted(BEDROOM.glob("*.jpg"))[:n]
    assert fps, f"no frames in {BEDROOM}"
    frames = [np.asarray(Image.open(fp).convert("RGB")) for fp in fps]
    return frames


def _stats(latencies):
    """ms summary from a list of per-frame seconds."""
    ms = [x * 1e3 for x in latencies]
    ms_sorted = sorted(ms)
    p90 = ms_sorted[min(len(ms_sorted) - 1, int(round(0.9 * (len(ms_sorted) - 1))))]
    mean = statistics.fmean(ms)
    return {
        "mean_ms": mean,
        "median_ms": statistics.median(ms),
        "p90_ms": p90,
        "min_ms": min(ms),
        "max_ms": max(ms),
        "fps": 1000.0 / mean,
    }


def bench_sam2(cfg, ckpt, frames, warmup):
    from sam.build_sam import build_sam2_video_predictor
    from sam.models.sam2_predictor import Sam2VideoPredictorState
    from sam.prompts import GeometryPrompt

    h, w, _ = frames[0].shape
    pred = build_sam2_video_predictor(cfg, ckpt, device="cuda", mode="eval", use_half=True)
    nparams = sum(p.numel() for p in pred.parameters()) / 1e6

    pt = torch.tensor([list(CLICK_XY)], device="cuda")
    lbl = torch.tensor([1], device="cuda")
    state = Sam2VideoPredictorState.create((h, w))

    torch.cuda.reset_peak_memory_stats()
    per_frame, nobj = [], 0
    for i, fr in enumerate(frames):
        t = torch.as_tensor(fr, device="cuda").permute(2, 0, 1).float()
        prompts = [GeometryPrompt(obj_id=1, points_coords=pt, points_labels=lbl)] if i == 0 else []
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = pred.forward(state=state, frame_idx=i, frame=t, prompts=prompts)
        torch.cuda.synchronize()
        per_frame.append(time.perf_counter() - t0)
        nobj = len(out)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    del pred, state
    return per_frame, nobj, nparams, peak_gb, "fp16"


def bench_sam3p1(cfg, ckpt, frames, warmup):
    from sam.build_sam import build_sam3_multiplex_video_predictor
    from sam.models.sam3_predictor import Sam3VideoPredictorState
    from sam.prompts import ConceptPrompt

    h, w, _ = frames[0].shape
    pred = build_sam3_multiplex_video_predictor(config_file=cfg, ckpt_path=ckpt, device="cuda")
    nparams = sum(p.numel() for p in pred.parameters()) / 1e6

    state = Sam3VideoPredictorState(video_hw=(h, w))
    pred.set_concept(state, ConceptPrompt(SAM3_PHRASE))

    torch.cuda.reset_peak_memory_stats()
    per_frame, nobj = [], 0
    for i, fr in enumerate(frames):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = pred.forward(state, i, fr)
        torch.cuda.synchronize()
        per_frame.append(time.perf_counter() - t0)
        nobj = len(out)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    del pred, state
    return per_frame, nobj, nparams, peak_gb, "bf16-autocast"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="number of bedroom frames")
    ap.add_argument("--warmup", type=int, default=3, help="frames excluded from steady-state stats")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "benchmark requires CUDA"
    gpu = torch.cuda.get_device_name(0)
    frames = load_frames(args.n)
    h, w, _ = frames[0].shape
    print(f"GPU: {gpu} | frames: {len(frames)} @ {w}x{h} | warmup(excl): {args.warmup}\n")

    rows = []
    for label, kind, cfg, ckpt in MODELS:
        if not Path(ckpt).is_file():
            print(f"[skip] {label}: checkpoint absent {ckpt}")
            continue
        print(f"[run ] {label} ({cfg})...", flush=True)
        try:
            fn = bench_sam2 if kind == "sam2" else bench_sam3p1
            per_frame, nobj, nparams, peak_gb, regime = fn(cfg, ckpt, frames, args.warmup)
            init_ms = per_frame[0] * 1e3
            steady = per_frame[args.warmup:] if len(per_frame) > args.warmup else per_frame
            s = _stats(steady)
            rows.append((label, nparams, regime, nobj, init_ms, s, peak_gb))
            print(
                f"       params={nparams:.0f}M regime={regime} objs={nobj} "
                f"init(f0)={init_ms:.0f}ms steady={s['mean_ms']:.0f}ms ({s['fps']:.1f} FPS) "
                f"peakVRAM={peak_gb:.2f}GB"
            )
        except Exception as e:  # one model failing must not kill the others
            print(f"[FAIL] {label}: {type(e).__name__}: {e}")
        finally:
            gc.collect()
            torch.cuda.empty_cache()
        print()

    if not rows:
        print("no results")
        return
    print("\n==================== SUMMARY (steady-state propagation, warmup excluded) ====================")
    print(f"{'model':<18}{'params':>8}{'regime':>16}{'objs':>6}{'init_ms':>9}"
          f"{'mean_ms':>9}{'median':>8}{'p90':>7}{'FPS':>7}{'VRAM_GB':>9}")
    for label, nparams, regime, nobj, init_ms, s, peak_gb in rows:
        print(f"{label:<18}{nparams:>7.0f}M{regime:>16}{nobj:>6}{init_ms:>9.0f}"
              f"{s['mean_ms']:>9.0f}{s['median_ms']:>8.0f}{s['p90_ms']:>7.0f}{s['fps']:>7.1f}{peak_gb:>9.2f}")
    print("\nNote: sam2/EfficientTAM track 1 click-prompted object; SAM 3.1 detects+tracks ALL "
          "'person' instances every frame (heavier, multi-object) — each is the model's real video-predict mode.")


if __name__ == "__main__":
    main()
