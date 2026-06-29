# SPDX-License-Identifier: LicenseRef-SAM
"""EfficientSAM3 image-inference latency benchmark.

Mirrors ``efficientsam3_reference/_bench.py`` but uses OUR ``build_efficientsam3`` +
``Sam3Predictor`` API instead of the upstream processor.

Breakdown:
  * vision  — preprocess (GPU resize + normalize) + trunk + neck (``encode_image``)
  * prompt  — text-encode + DETR-detect + seg-head (``encode_text`` + ``detect``)
  * e2e     — vision + prompt in one call (``predict``)

Reports ``fp32`` (TF32 on, matching the upstream reference settings) and
``autocast`` (bfloat16, the standard deployment path).

Reference (RTX 3080 Ti, upstream _bench.py):
  fp32    vision ~31.5 ms   e2e ~136.6 ms
  autocast vision ~12.7 ms  e2e  ~43.3 ms

Usage:
    pixi run python tools/benchmark_efficientsam3.py --ckpt checkpoints/efficientsam3_repvit.pt
    pixi run python tools/benchmark_efficientsam3.py --ckpt ... --image ... --prompt person --iters 30
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _bench(fn, warmup: int, iters: int) -> dict:
    for _ in range(warmup):
        fn()
    _sync()
    ts = []
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    n = len(ts)
    return {
        "median_ms": statistics.median(ts) * 1e3,
        "mean_ms": statistics.fmean(ts) * 1e3,
        "std_ms": (statistics.pstdev(ts) * 1e3) if n > 1 else 0.0,
        "min_ms": ts[0] * 1e3,
        "p90_ms": ts[min(n - 1, int(0.9 * n))] * 1e3,
        "fps_median": 1.0 / statistics.median(ts),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="EfficientSAM3 image-inference latency benchmark."
    )
    ap.add_argument("--ckpt", required=True, help="Path to efficientsam3_repvit.pt")
    ap.add_argument(
        "--image",
        default=None,
        help="Path to a JPEG/PNG image (default: dog_person.jpeg from the reference repo or a "
             "dummy 1365x2048 image when the reference repo is absent).",
    )
    ap.add_argument("--prompt", default="dog", help="Text concept prompt (default: dog)")
    ap.add_argument("--threshold", type=float, default=0.1, help="Confidence threshold")
    ap.add_argument("--warmup", type=int, default=10, help="Warmup iterations (default: 10)")
    ap.add_argument("--iters", type=int, default=50, help="Timed iterations (default: 50)")
    ap.add_argument("--out", default=None, help="Optional JSON output path")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TF32 on -> matches the upstream reference settings for fp32 mode
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True  # fixed-size workload -> autotune

    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"device={device}  gpu={gpu}  torch={torch.__version__}")

    # ------------------------------------------------------------------ model
    from sam.build_sam import build_efficientsam3

    model = build_efficientsam3(ckpt_path=args.ckpt, device=str(device), mode="eval")
    print(f"Model built from {args.ckpt}")

    # ------------------------------------------------------------------ image
    if args.image is not None:
        image_path = Path(args.image)
    else:
        # Try the reference repo first; fall back to a dummy numpy array
        ref_img = Path(__file__).parent.parent.parent / "efficientsam3_reference/sam3/assets/dog_person.jpeg"
        if ref_img.is_file():
            image_path = ref_img
        else:
            image_path = None

    if image_path is not None and image_path.is_file():
        image_np = np.array(Image.open(image_path).convert("RGB"))
        image_label = image_path.name
    else:
        # Dummy 1365x2048 uint8 RGB (matches the dog_person.jpeg size)
        image_np = np.zeros((1365, 2048, 3), dtype=np.uint8)
        image_label = "dummy-1365x2048"

    print(
        f"image={image_label} shape={image_np.shape} "
        f"prompt={args.prompt!r} threshold={args.threshold} "
        f"warmup={args.warmup} iters={args.iters}"
    )

    # ------------------------------------------------------------------ bench
    from sam.prompts import ConceptPrompt
    from sam.utils.sam3_transforms import preprocess_to_1008

    concept = ConceptPrompt(text=args.prompt)
    image_hw = (int(image_np.shape[0]), int(image_np.shape[1]))

    report = {
        "gpu": gpu,
        "torch": torch.__version__,
        "ckpt": str(args.ckpt),
        "image": image_label,
        "image_shape": list(image_np.shape[:2]),
        "prompt": args.prompt,
        "threshold": args.threshold,
        "warmup": args.warmup,
        "iters": args.iters,
        "modes": {},
    }

    for mode in ("fp32", "autocast"):
        ctx: contextlib.AbstractContextManager
        if mode == "autocast":
            ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        else:
            ctx = contextlib.nullcontext()

        with torch.inference_mode(), ctx:
            # pre-process once so timing excludes CPU-Python overhead not in the model
            x = preprocess_to_1008(image_np, device=device)
            feats, pos = model.encode_image(x)
            text_emb, text_mask = model.encode_text(concept)

            def vision():
                _x = preprocess_to_1008(image_np, device=device)
                model.encode_image(_x)

            def prompt_only():
                _t, _m = model.encode_text(concept)
                model.detect(feats, pos, _t, _m, image_hw,
                             confidence_threshold=args.threshold)

            def e2e():
                if mode == "fp32":
                    model.predict(image_np, concept,
                                  confidence_threshold=args.threshold,
                                  dtype=torch.float32)
                else:
                    model.predict(image_np, concept,
                                  confidence_threshold=args.threshold,
                                  dtype=torch.bfloat16)

            r = {
                "vision_set_image": _bench(vision, args.warmup, args.iters),
                "prompt_text_detect_seg": _bench(prompt_only, args.warmup, args.iters),
                "end_to_end": _bench(e2e, args.warmup, args.iters),
            }

        report["modes"][mode] = r
        dt = torch.bfloat16 if mode == "autocast" else torch.float32
        print(f"\n=== mode={mode} (dtype~{dt}) ===")
        for phase, s in r.items():
            print(
                f"  {phase:28s}  median={s['median_ms']:8.2f} ms  "
                f"fps={s['fps_median']:7.2f}  "
                f"(mean={s['mean_ms']:.2f}  std={s['std_ms']:.2f}  min={s['min_ms']:.2f})"
            )

    # ------------------------------------------------------------------ output
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote {out_path}")
    else:
        print()


if __name__ == "__main__":
    main()
