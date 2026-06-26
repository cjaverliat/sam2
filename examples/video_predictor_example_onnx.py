# Copyright (c) Meta Platforms, Inc. and affiliates.
"""ONNX / TensorRT variant of video_predictor_example.py.

Identical pipeline, but the 5 neural-network blocks run as ONNX Runtime (TensorRT EP)
sessions instead of torch. Export the blocks first with tools/export_onnx.py, then point
this script at the resulting directory.

Run from the repo root:
    pixi run python tools/export_onnx.py \
        --config configs/sam2.1/sam2.1_hiera_b+.yaml \
        --ckpt checkpoints/sam2.1_hiera_base_plus.pt --out-dir onnx_sam2

    pixi run python examples/video_predictor_example_onnx.py \
        --onnx-dir onnx_sam2 \
        --model-cfg configs/sam2.1/sam2.1_hiera_b+.yaml \
        --output-dir examples/output_onnx

No checkpoint needed at runtime: all weights are baked into the export artifacts
(the 5 graphs + weights.npz, which holds dense_pe / no_mask_embed and the memory-path
residual params) produced by tools/export_onnx.py.
"""

import argparse
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from sam.build_sam import build_sam2_generic_video_predictor_onnx
from sam.prompts import SAM2Prompt
from sam.onnx.trt_options import TensorRTOptions
from sam.models.sam2_predictor import SAM2GenericVideoPredictorState

# Reuse the plotting / IO helpers from the torch example (same directory).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frame_utils import read_frame, warmup  # noqa: E402
from video_predictor_example import show_mask, show_points  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="notebooks/videos/bedroom.mp4")
    parser.add_argument("--onnx-dir", required=True, help="dir from tools/export_onnx.py")
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument("--output-dir", default="examples/output_onnx")
    parser.add_argument("--no-trt", action="store_true", help="CUDA EP instead of TensorRT")
    parser.add_argument("--vis-frame-stride", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    torch.no_grad().__enter__()

    # Build the ONNX/TRT-backed video predictor in one call. No checkpoint needed:
    # all weights are baked into the export artifacts under --onnx-dir.
    # Default fp32 TRT; set bf16_enable for full-speed on Ampere+.
    trt_opts = TensorRTOptions()
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

    ann_frame_idx = 0
    ann_obj_id = 1
    points = torch.tensor([[210, 350], [250, 220]], dtype=torch.float32, device=device)
    labels = torch.tensor([1, 1], device=device)
    prompt = SAM2Prompt(obj_id=ann_obj_id, points_coords=points, points_labels=labels)

    initial_frame = read_frame(cap, device)
    results = predictor.forward(
        state=video_state, frame=initial_frame, frame_idx=ann_frame_idx, prompts=[prompt],
    )

    plt.figure(figsize=(9, 6))
    plt.title("Initial Frame (ONNX)")
    plt.imshow(initial_frame.permute(1, 2, 0).cpu().numpy())
    show_mask((results[ann_obj_id].best_mask_logits > 0), plt.gca(), obj_id=ann_obj_id)
    show_points(points, labels, plt.gca())
    plt.savefig(os.path.join(args.output_dir, "frame_0000_mask.png"), bbox_inches="tight")
    plt.close()

    last = n_frames if args.max_frames <= 0 else min(n_frames, 1 + args.max_frames)
    pbar = tqdm(range(1, last), desc="Propagating in video (ONNX)")
    for f in pbar:
        frame = read_frame(cap, device)
        if frame is None:
            break
        results = predictor.forward(state=video_state, frame=frame, frame_idx=f)
        if f % args.vis_frame_stride == 0:
            plt.figure(figsize=(6, 4))
            plt.title(f"Frame {f} (ONNX)")
            plt.axis("off")
            plt.imshow(frame.permute(1, 2, 0).cpu().numpy())
            for obj_id, result in results.items():
                show_mask((result.best_mask_logits > 0), plt.gca(), obj_id=obj_id)
            plt.savefig(
                os.path.join(args.output_dir, f"frame_{f:04d}_mask.png"),
                bbox_inches="tight",
            )
            plt.close()

    pbar.close()
    cap.release()
    print(f"Done. Figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
