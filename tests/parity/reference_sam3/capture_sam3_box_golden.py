# SPDX-License-Identifier: LicenseRef-SAM
"""Capture the IMAGE grounding detection golden (text + ONE box geometry prompt)
from the OFFICIAL SAM 3.1 multiplex image detector.

Run ONCE in the isolated reference env (NOT this repo's pixi env):

    cd C:/Users/javerlia/PycharmProjects/sam3_reference
    ./.venv/Scripts/python.exe \
        ../sam2/tests/parity/reference_sam3/capture_sam3_box_golden.py \
        --frame ../sam2/notebooks/videos/bedroom/00000.jpg \
        --ckpt ../sam2/checkpoints/sam3.1_multiplex.pt \
        --out ../sam2/tests/parity/fixtures/sam3 \
        --patches

Scenario (single-image PHRASE-GROUNDED detection with a geometric BOX prompt):
  Frame notebooks/videos/bedroom/00000.jpg (960x540), text phrase "person", and
  ONE geometric box prompt at native-pixel xyxy [300, 150, 470, 420],
  confidence_threshold 0.5.

  ``--label`` selects the box sign (upstream ``add_geometric_prompt(box, label)``;
  the label indexes ``geometry_encoder.label_embed``, an ``nn.Embedding(2, d)``):
    1 (default) -> positive box, written to ``box_prompt{,_scenario}.*``
    0           -> negative box, written to ``box_prompt_neg{,_scenario}.*``
  The default run reproduces the original detections; the scenario json now also
  records ``box_label`` explicitly.

API path: MODEL API (Sam3MultiplexTrackingWithInteractivity.detector, a
  Sam3MultiplexDetector -> Sam3MultiplexImageBase -> Sam3Image). We build the full
  multiplex predictor (for correct checkpoint remap + detector assembly), then
  drive the image detector directly:

    predictor = build_sam3_multiplex_video_predictor(checkpoint_path=..., ...)
    model = predictor.model
    detector = model.detector

    # init_state does the exact image preprocessing (resize to 1008, mean/std 0.5)
    # and builds a BatchedDatapoint (input_batch) with a per-frame FindStage.
    state = model.init_state(resource_path=<dir with the single frame>)
    input_batch = state["input_batch"]
    input_batch.find_text_batch[0] = "person"          # text_ids=[0] selects this
    find_input = input_batch.find_inputs[0]             # img_ids=[0], text_ids=[0]

    text_outputs = detector.backbone.forward_text(input_batch.find_text_batch, dev)
    backbone_out = {"img_batch_all_stages": input_batch.img_batch, **text_outputs}

    # geometric prompt = ONE box, NORMALIZED cxcywh, seq-first (N, B, 4), label 1.
    geom = Prompt(box_embeddings=torch.zeros(0, 1, 4, ...),
                  box_mask=torch.zeros(1, 0, bool, ...))
    geom.append_boxes(boxes=box_cxcywh.view(-1, 1, 4), labels=label.view(-1, 1))

    out = detector.forward_grounding(backbone_out, find_input, find_target=None,
                                     geometric_prompt=geom)

  forward_grounding fills out["pred_logits"] (B, Q, 1) JOINT logits (joint with the
  presence token because supervise_joint_box_scores=True), out["pred_boxes_xyxy"]
  (B, Q, 4) NORMALIZED xyxy, out["pred_masks"] (B, Q, h, w) low-res logits, and
  out["presence_logit_dec"] (B, 1, 1) the presence logit.

Post-processing (matches the model's own detection recipe):
  scores   = sigmoid(pred_logits)[..., 0]                 # joint, per query
  presence = sigmoid(presence_logit_dec).flatten()[0]     # scalar
  keep     = scores > confidence_threshold
  boxes    = pred_boxes_xyxy[keep] * [W, H, W, H]         # native pixels
  masks    = interpolate(pred_masks[keep], (H, W)) > 0    # binarized

Determinism: seed 0, cuDNN deterministic, TF32 off; inference_mode + bf16 autocast.
use_fa3=False, use_rope_real=False (triton/FA3 kernels absent in this venv).

Env concessions (kernel/dep dispatch ONLY -- no weights/logic change), REQUIRED
because triton/FA3 kernels are absent: edt scipy stub + all-backend SDPA + CPU
NMS/CC. Pass --patches to enable (used for the golden).
"""
import argparse
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PHRASE = "person"
BOX_XYXY = [300.0, 150.0, 470.0, 420.0]  # native 960x540 pixel
CONF_THRESH = 0.5


def _install_edt_stub():
    """Stub sam3.model.edt.edt_triton with a scipy CPU EDT (triton absent)."""
    from scipy.ndimage import distance_transform_edt

    def edt_triton(data: torch.Tensor) -> torch.Tensor:
        assert data.dim() == 3, "edt_triton expects (B, H, W)"
        arr = data.detach().to("cpu").numpy()
        out = np.empty(arr.shape, dtype=np.float32)
        for b in range(arr.shape[0]):
            out[b] = distance_transform_edt(arr[b].astype(bool)).astype(np.float32)
        return torch.from_numpy(out).to(data.device)

    stub = types.ModuleType("sam3.model.edt")
    stub.edt_triton = edt_triton
    sys.modules["sam3.model.edt"] = stub


def _patch_sdpa_all_backends():
    """Force all-backend SDPA in the multiplex decoder (FA kernels may be absent)."""
    from torch.nn.attention import SDPBackend
    from torch.nn.attention import sdpa_kernel as _sk
    from sam3.model import decoder as _dec

    _all = [
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.MATH,
    ]
    _dec.sdpa_kernel = lambda *a, **k: _sk(_all)


def _patch_nms_cpu():
    import sam3.perflib.nms as _nms_mod

    _nms_mod.generic_nms = _nms_mod.generic_nms_cpu


def _patch_connected_components_cpu():
    import sam3.perflib.connected_components as _cc_mod
    from sam3.perflib.connected_components import connected_components_cpu as _cc_cpu

    def _cc_cpu_fallback(input_tensor):
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(1)
        assert input_tensor.dim() == 4 and input_tensor.shape[1] == 1
        if input_tensor.shape[0] == 0:
            return (
                torch.zeros_like(input_tensor, dtype=torch.int64),
                torch.zeros_like(input_tensor, dtype=torch.int64),
            )
        return _cc_cpu(input_tensor)

    _cc_mod.connected_components = _cc_cpu_fallback


def _determinism():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", type=int, choices=(0, 1), default=1,
                    help="box sign: 1 positive (default), 0 negative")
    ap.add_argument("--patches", action="store_true")
    args = ap.parse_args()
    stem = "box_prompt" if args.label == 1 else "box_prompt_neg"

    assert torch.cuda.is_available(), "CUDA required"
    _determinism()

    if args.patches:
        _install_edt_stub()
        _patch_sdpa_all_backends()
        _patch_nms_cpu()
        _patch_connected_components_cpu()
        print("[capture] env concessions applied: edt + SDPA(all) + NMS(cpu) + CC(cpu)")

    from sam3.model_builder import build_sam3_multiplex_video_predictor
    from sam3.model.geometry_encoders import Prompt

    frame_path = Path(args.frame)
    w0, h0 = Image.open(frame_path).size
    print(f"[capture] native frame size (w, h) = ({w0}, {h0})")

    ckpt_path = str(Path(args.ckpt).resolve())
    print(f"[capture] Building SAM 3.1 multiplex predictor (ckpt={ckpt_path})")
    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path=ckpt_path,
        use_fa3=False,
        use_rope_real=False,
        compile=False,
        warm_up=False,
        async_loading_frames=False,
    )
    model = predictor.model
    model.eval()
    detector = model.detector
    _determinism()

    # native-pixel xyxy -> NORMALIZED cxcywh
    x0, y0, x1, y1 = BOX_XYXY
    cx = (x0 + x1) / 2.0 / w0
    cy = (y0 + y1) / 2.0 / h0
    bw = (x1 - x0) / w0
    bh = (y1 - y0) / h0
    print(f"[capture] box native xyxy={BOX_XYXY} -> norm cxcywh="
          f"({cx:.5f}, {cy:.5f}, {bw:.5f}, {bh:.5f}) label={args.label}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="sam3_box_ref_"))
    try:
        shutil.copyfile(frame_path, tmp_dir / "00000.jpg")

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                state = model.init_state(
                    resource_path=str(tmp_dir), async_loading_frames=False
                )
                assert state["num_frames"] == 1
                assert (state["orig_height"], state["orig_width"]) == (h0, w0)
                device = state["device"]

                input_batch = state["input_batch"]
                input_batch.find_text_batch[0] = PHRASE
                find_input = input_batch.find_inputs[0]

                text_outputs = detector.backbone.forward_text(
                    input_batch.find_text_batch, device=device
                )
                backbone_out = {
                    "img_batch_all_stages": input_batch.img_batch,
                    **text_outputs,
                }

                geom = Prompt(
                    box_embeddings=torch.zeros(0, 1, 4, device=device),
                    box_mask=torch.zeros(1, 0, device=device, dtype=torch.bool),
                )
                box_cxcywh = torch.tensor(
                    [cx, cy, bw, bh], dtype=torch.float32, device=device
                )
                labels = torch.tensor([args.label], dtype=torch.long, device=device)
                geom.append_boxes(
                    boxes=box_cxcywh.view(-1, 1, 4),
                    labels=labels.view(-1, 1),
                )

                out = detector.forward_grounding(
                    backbone_out=backbone_out,
                    find_input=find_input,
                    find_target=None,
                    geometric_prompt=geom,
                )

        print("[capture] out keys:", sorted(out.keys()))
        for k in ("pred_logits", "pred_boxes", "pred_boxes_xyxy", "pred_masks",
                  "presence_logit_dec"):
            if k in out:
                print(f"  {k}: shape={tuple(out[k].shape)} dtype={out[k].dtype}")

        pred_logits = out["pred_logits"].float()          # (B, Q, 1)
        pred_boxes_xyxy = out["pred_boxes_xyxy"].float()   # (B, Q, 4) norm
        pred_masks = out["pred_masks"].float()             # (B, Q, h, w) low-res
        presence_logit = out["presence_logit_dec"].float()

        scores_all = pred_logits.sigmoid()[0, :, 0]        # (Q,)
        presence = float(presence_logit.sigmoid().flatten()[0].item())

        keep = (scores_all > CONF_THRESH).nonzero(as_tuple=True)[0]
        n = int(keep.numel())
        print(f"[capture] presence={presence:.6f}  "
              f"n_over_{CONF_THRESH}={n}  max_score={float(scores_all.max()):.4f}")

        scores = scores_all[keep].cpu().numpy().astype(np.float32)
        scale = torch.tensor([w0, h0, w0, h0], dtype=torch.float32, device=device)
        boxes = (pred_boxes_xyxy[0, keep] * scale).cpu().numpy().astype(np.float32)

        masks_low = pred_masks[0, keep]                    # (n, h, w)
        if n > 0:
            masks_up = F.interpolate(
                masks_low.unsqueeze(1), size=(h0, w0),
                mode="bilinear", align_corners=False,
            )[:, 0]
            masks = (masks_up > 0).cpu().numpy().astype(np.uint8)
        else:
            masks = np.zeros((0, h0, w0), dtype=np.uint8)

        # Sort by score descending for stable, readable output.
        order = np.argsort(-scores)
        boxes, scores, masks = boxes[order], scores[order], masks[order]

        for i in range(n):
            b = boxes[i]
            print(f"  det[{i}] score={scores[i]:.4f} "
                  f"xyxy=[{b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f}] "
                  f"mask_px={int(masks[i].sum())}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{stem}.npz",
        boxes=boxes,
        scores=scores,
        presence=np.float32(presence),
        masks=masks,
        n=np.int64(n),
    )
    (out_dir / f"{stem}_scenario.json").write_text(json.dumps({
        "phrase": PHRASE,
        "box_xyxy": [int(v) for v in BOX_XYXY],
        "box_label": args.label,
        "confidence_threshold": CONF_THRESH,
        "hw": [h0, w0],
        "frame": "notebooks/videos/bedroom/00000.jpg",
    }, indent=2))
    print(f"\n[capture] saved {stem}.npz (n={n}) + {stem}_scenario.json -> {out_dir}")


if __name__ == "__main__":
    main()
