# SPDX-License-Identifier: Apache-2.0
"""Replicates the notebook pipelines (image-predict + streaming video) on a fixed
input, for the pre/post-Phase-0 parity check.

The old (`sam2`) and new (`sam`) packages differ only in import paths / class names;
the method calls are identical. ``--api old`` is run from the pre-Phase-0 worktree
(via PYTHONPATH) to capture golden outputs; ``--api new`` is run against the current
package by the pytest. Deterministic: CUDA, fp32, fixed seeds, cuDNN deterministic,
TF32 off — so a pure refactor must reproduce the outputs.

Usage:
    python tests/parity/run_pipelines.py --api {old,new} --out <file.npz>
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
CKPT = "checkpoints/sam2.1_hiera_tiny.pt"
HW = (384, 512)  # downscaled input -> small masks/fixtures, same code paths
TRUCK = "notebooks/images/truck.jpg"
BEDROOM_DIR = "notebooks/videos/bedroom"


def _imports(api):
    if api == "old":
        from sam2.build_sam import build_sam2_generic as build_pred
        from sam2.build_sam import build_sam2_generic_video_predictor as build_vid
        from sam2.sam2_generic_video_predictor import SAM2GenericVideoPredictorState as VState
        from sam2.modeling.sam2_prompt import SAM2Prompt as Prompt
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator as AMG
    else:
        from sam.build_sam import build_sam2_predictor as build_pred
        from sam.build_sam import build_sam2_video_predictor as build_vid
        from sam.models.sam2_predictor import Sam2VideoPredictorState as VState
        from sam.prompts import GeometryPrompt as Prompt
        from sam.automatic_mask_generator import Sam2AutomaticMaskGenerator as AMG
    return build_pred, build_vid, VState, Prompt, AMG


def _determinism():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _load_rgb(path, hw):
    img = Image.open(path).convert("RGB").resize((hw[1], hw[0]))  # PIL is (W, H)
    return np.asarray(img)


def _run_image(build_pred, out):
    model = build_pred(CFG, CKPT, device="cuda", mode="eval", apply_postprocessing=True, use_half=False)
    image = _load_rgb(TRUCK, HW)
    img = torch.as_tensor(image, device="cuda").permute(2, 0, 1)
    img_embeddings, _ = model.encode_image(img)

    def predict(points=None, labels=None, box=None, multimask=False):
        pc = pl = bx = None
        if points is not None:
            pc = torch.as_tensor(points, dtype=torch.float32, device="cuda").reshape(1, -1, 2)
            pl = torch.as_tensor(labels, dtype=torch.int, device="cuda").reshape(1, -1)
        if box is not None:
            bx = torch.as_tensor(box, dtype=torch.float32, device="cuda").reshape(1, -1, 4)
        pe = model.encode_prompts(orig_hw=HW, batch_size=1, points_coords=pc, points_labels=pl, boxes=bx)
        r = model.generate_masks(orig_hw=HW, img_embeddings=img_embeddings, prompt_embeddings=pe, multimask_output=multimask)
        masks = (r.masks_logits[0] > model.mask_threshold).cpu().numpy().astype(np.uint8)
        scores = r.ious[0].float().cpu().numpy()
        return masks, scores

    m, s = predict(points=[[300, 200]], labels=[1], multimask=True)
    out["img_pt_masks"], out["img_pt_scores"] = m, s
    m, s = predict(box=[100, 100, 400, 300], multimask=False)
    out["img_box_masks"], out["img_box_scores"] = m, s
    m, s = predict(points=[[150, 250]], labels=[0], box=[100, 100, 400, 300], multimask=False)
    out["img_ptbox_masks"], out["img_ptbox_scores"] = m, s


def _run_video(build_vid, VState, Prompt, out):
    frames = sorted(Path(BEDROOM_DIR).glob("*.jpg"))[:3]
    assert len(frames) == 3, f"need 3 bedroom frames, found {len(frames)} in {BEDROOM_DIR}"
    predictor = build_vid(CFG, CKPT, device="cuda", mode="eval", use_half=False)
    state = VState.create(HW)
    prompt = Prompt(
        obj_id=1,
        points_coords=torch.tensor([[256.0, 192.0]], device="cuda"),
        points_labels=torch.tensor([1], device="cuda"),
    )
    for f_idx, fp in enumerate(frames):
        frame = torch.as_tensor(_load_rgb(str(fp), HW), device="cuda").permute(2, 0, 1).float()
        kw = {"prompts": [prompt]} if f_idx == 0 else {}
        results = predictor.forward(state=state, frame=frame, frame_idx=f_idx, **kw)
        mask = (results[1].best_mask_logits.squeeze() > 0).cpu().numpy().astype(np.uint8)
        out[f"vid_f{f_idx}_mask"] = mask


def _run_amg(build_pred, AMG, out):
    model = build_pred(CFG, CKPT, device="cuda", mode="eval", apply_postprocessing=False, use_half=False)
    gen = AMG(model, points_per_side=8)
    masks = gen.generate(_load_rgb(TRUCK, HW))
    out["amg_count"] = np.array([len(masks)], dtype=np.int64)
    out["amg_areas"] = np.array(sorted(int(m["area"]) for m in masks), dtype=np.int64)


def run_all(api):
    _determinism()
    build_pred, build_vid, VState, Prompt, AMG = _imports(api)
    out = {}
    with torch.inference_mode():
        _run_image(build_pred, out)
        _run_video(build_vid, VState, Prompt, out)
    _run_amg(build_pred, AMG, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", choices=["old", "new"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = run_all(args.api)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print("wrote", args.out, "keys:", sorted(out))


if __name__ == "__main__":
    main()
