# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Video segmentation with SAM 2 using the generic video predictor.

Script equivalent of notebooks/video_predictor_example.ipynb. It:
- adds clicks on a frame to get and refine a mask,
- propagates the prompts to get a masklet across the whole video.

Figures are saved to an output directory instead of being shown interactively.

Run from the repo root:
    python examples/video_predictor_example.py \
        --video notebooks/videos/bedroom.mp4 \
        --checkpoint checkpoints/sam2.1_hiera_base_plus.pt \
        --model-cfg configs/sam2.1/sam2.1_hiera_b+.yaml \
        --output-dir examples/output
"""

import argparse
import os

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import matplotlib

matplotlib.use("Agg")  # non-interactive backend; save figures to files
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from frame_utils import read_frame, warmup
from sam.build_sam import build_sam2_generic_video_predictor
from sam.prompts import SAM2Prompt
from sam.models.sam2_predictor import SAM2GenericVideoPredictorState


def select_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            print("bfloat16 is not supported on this GPU")
        # use bfloat16 for the entire script
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs
        # (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    torch.no_grad().__enter__()
    return device


def show_mask(mask, ax, obj_id=None, random_color=False):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape((h, w, 1)) * color.reshape((1, 1, -1))
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0], pos_points[:, 1], color="green", marker="*",
        s=marker_size, edgecolor="white", linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0], neg_points[:, 1], color="red", marker="*",
        s=marker_size, edgecolor="white", linewidth=1.25,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="notebooks/videos/bedroom.mp4")
    parser.add_argument(
        "--checkpoint", default="checkpoints/sam2.1_hiera_base_plus.pt"
    )
    parser.add_argument(
        "--model-cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml"
    )
    parser.add_argument("--output-dir", default="examples/output")
    parser.add_argument(
        "--vis-frame-stride", type=int, default=30,
        help="save a propagation figure every N frames",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = select_device()

    predictor = build_sam2_generic_video_predictor(
        args.model_cfg, args.checkpoint, device=device,
        vos_optimized=False, apply_postprocessing=True,
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_hw = (height, width)

    video_state = SAM2GenericVideoPredictorState.create(orig_hw)

    # Move the one-time GPU init off the first prompted frame.
    warmup(predictor, video_state, device)

    # --- Step 1 & 2: add and refine clicks on frame 0 ---
    ann_frame_idx = 0
    ann_obj_id = 1

    # Two positive clicks to segment the child on the left.
    points = torch.tensor([[210, 350], [250, 220]], dtype=torch.float32, device=device)
    labels = torch.tensor([1, 1], device=device)

    prompt = SAM2Prompt(
        obj_id=ann_obj_id, points_coords=points, points_labels=labels
    )

    initial_frame = read_frame(cap, device)
    results = predictor.forward(
        state=video_state, frame=initial_frame,
        frame_idx=ann_frame_idx, prompts=[prompt],
    )

    plt.figure(figsize=(9, 6))
    plt.title("Initial Frame")
    plt.imshow(initial_frame.permute(1, 2, 0).cpu().numpy())
    show_mask((results[ann_obj_id].best_mask_logits > 0), plt.gca(), obj_id=ann_obj_id)
    show_points(points, labels, plt.gca())
    plt.savefig(os.path.join(args.output_dir, "frame_0000_mask.png"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.title("Raw logits")
    plt.imshow(
        results[ann_obj_id].best_mask_logits.squeeze().cpu().numpy(),
        cmap="seismic", vmin=-10, vmax=10,
    )
    plt.savefig(os.path.join(args.output_dir, "frame_0000_logits.png"), bbox_inches="tight")
    plt.close()

    # --- Step 3: propagate prompts across the video ---
    pbar = tqdm(range(1, n_frames), desc="Propagating in video")
    for f in pbar:
        frame = read_frame(cap, device)
        if frame is None:
            break

        results = predictor.forward(state=video_state, frame=frame, frame_idx=f)

        pbar.set_postfix(
            {
                "stored_cond_mem": video_state.memory_bank.count_conditional_memories(),
                "stored_non_cond_mem": video_state.memory_bank.count_non_conditional_memories(),
            }
        )

        if f % args.vis_frame_stride == 0:
            plt.figure(figsize=(6, 4))
            plt.title(f"Frame {f}")
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
    cv2.destroyAllWindows()
    print(f"Done. Figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
