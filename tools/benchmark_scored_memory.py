# SPDX-License-Identifier: LicenseRef-SAM
"""E2E check: default SAM2ObjectMemoryBank vs SAM2ScoredPeriodicObjectMemoryBank.

Tracks one click-prompted object across the bedroom clip with SAM 2.1-small on the
generic video-predict path, once per memory bank, and reports mask sanity, stored
non-conditional memory count, and steady-state latency.

Usage: pixi run python tools/benchmark_scored_memory.py [--n 60] [--period 5] [--threshold 0.5]
"""
import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam2.build_sam import build_sam2_generic_video_predictor
from sam2.sam2_generic_video_predictor import SAM2GenericVideoPredictorState
from sam2.modeling.sam2_prompt import SAM2Prompt
from sam2.modeling.sam2_memory import SAM2ObjectMemoryBank
from sam2.modeling.sam2_scored_periodic_memory import (
    SAM2ScoredPeriodicObjectMemoryBank,
)

BEDROOM = Path("notebooks/videos/bedroom")
CLICK_XY = (385.0, 230.0)
CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
CKPT = "checkpoints/sam2.1_hiera_small.pt"


def load_frames(n):
    fps = sorted(BEDROOM.glob("*.jpg"))[:n]
    assert fps, f"no frames in {BEDROOM}"
    return [np.asarray(Image.open(fp).convert("RGB")) for fp in fps]


def run(pred, frames, make_bank, warmup):
    h, w, _ = frames[0].shape
    state = SAM2GenericVideoPredictorState.create((h, w), memory_bank=make_bank())
    pt = torch.tensor([list(CLICK_XY)], device="cuda")
    lbl = torch.tensor([1], device="cuda")

    per_frame, mask_areas = [], []
    for i, fr in enumerate(frames):
        # Pass uint8 (C, H, W): the generic pipeline scales /255 + ImageNet-normalizes
        # internally, but ONLY when the input dtype is uint8 (sam2_generic._prepare_images).
        t = torch.as_tensor(fr.copy(), device="cuda").permute(2, 0, 1)
        prompts = (
            [SAM2Prompt(obj_id=1, points_coords=pt, points_labels=lbl)] if i == 0 else []
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = pred.forward(state=state, frame_idx=i, frame=t, prompts=prompts)
        torch.cuda.synchronize()
        per_frame.append(time.perf_counter() - t0)
        # fraction of the frame covered by the predicted mask for object 1
        area = (out[1].best_mask_logits > 0).float().mean().item()
        mask_areas.append(area)

    steady = per_frame[warmup:] if len(per_frame) > warmup else per_frame
    return {
        "mean_ms": statistics.fmean(x * 1e3 for x in steady),
        "n_noncond": state.memory_bank.count_non_conditional_memories(),
        "n_cond": state.memory_bank.count_conditional_memories(),
        "mask_first": mask_areas[0],
        "mask_mean": statistics.fmean(mask_areas),
        "mask_empty_frames": sum(1 for a in mask_areas if a == 0.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--period", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=None,
                    help="memory_window_size for forgetting; omit to keep all")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "benchmark requires CUDA"
    frames = load_frames(args.n)
    h, w, _ = frames[0].shape
    print(f"GPU: {torch.cuda.get_device_name(0)} | frames: {len(frames)} @ {w}x{h}")
    print(f"period={args.period} threshold={args.threshold} window={args.window}\n")

    pred = build_sam2_generic_video_predictor(CONFIG, CKPT, device="cuda", mode="eval", use_half=True)

    banks = {
        "default": lambda: SAM2ObjectMemoryBank(),
        "scored-periodic": lambda: SAM2ScoredPeriodicObjectMemoryBank(
            score_threshold=args.threshold,
            storage_period=args.period,
            memory_window_size=args.window,
        ),
    }

    rows = {}
    for name, make_bank in banks.items():
        print(f"[run ] {name}...", flush=True)
        rows[name] = run(pred, frames, make_bank, args.warmup)

    print(f"\n{'bank':<18}{'mean_ms':>9}{'non_cond':>10}{'cond':>6}"
          f"{'mask_f0':>9}{'mask_avg':>9}{'empty':>7}")
    for name, r in rows.items():
        print(f"{name:<18}{r['mean_ms']:>9.1f}{r['n_noncond']:>10}{r['n_cond']:>6}"
              f"{r['mask_first']:>9.3f}{r['mask_mean']:>9.3f}{r['mask_empty_frames']:>7}")

    d, s = rows["default"], rows["scored-periodic"]
    print(f"\nnon_cond memory: {d['n_noncond']} -> {s['n_noncond']} "
          f"({d['n_noncond'] / max(1, s['n_noncond']):.1f}x fewer)")


if __name__ == "__main__":
    main()
