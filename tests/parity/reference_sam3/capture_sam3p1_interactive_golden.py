# SPDX-License-Identifier: LicenseRef-SAM
"""Capture the interactive seed-frame click golden from the OFFICIAL SAM 3.1 multiplex.

Run ONCE in the isolated reference env (NOT this repo's pixi env):

    cd C:/Users/javerlia/PycharmProjects/sam3_reference
    ./.venv/Scripts/python.exe \
        ../sam2/tests/parity/reference_sam3/capture_sam3p1_interactive_golden.py \
        --frames ../sam2/notebooks/videos/bedroom \
        --ckpt ../sam2/checkpoints/sam3.1_multiplex.pt \
        --n 8 \
        --out ../sam2/tests/parity/fixtures/sam3p1 \
        --patches

Scenario (INTERACTIVE point click, NO text concept):
  On frame 0, add a POSITIVE point click (label 1) at native pixel (x, y) =
  (385.0, 230.0), obj_id=2, no text concept -> propagate forward N frames and
  collect obj_id=2's binary mask per frame.

API path: MODEL API (Sam3MultiplexTrackingWithInteractivity, i.e. predictor.model).
  We build the predictor (for correct checkpoint remap + detector assembly), then
  drive the underlying model directly because the predictor's start_session
  hard-passes offload_state_to_cpu, which the multiplex init_state rejects in this
  reference rev.

  model = predictor.model
  state = model.init_state(resource_path=..., async_loading_frames=False)
  f0, out0 = model.add_prompt(state, frame_idx=0, points=[[x_rel, y_rel]],
                              point_labels=[1], obj_id=2)  # -> add_sam2_new_points

  add_prompt and each propagate step return (frame_idx, out) where ``out`` is a
  dict of PARALLEL arrays (NOT a {obj_id: mask} map):
      out["out_obj_ids"]      int64  (K,)
      out["out_binary_masks"] bool   (K, H, W)  at native video (orig) resolution
      out["out_probs"]        float32(K,)
  obj_id=2's mask is out_binary_masks[i] where out_obj_ids[i] == 2.

TWO-PASS cache seeding (why): the multiplex demo is concept-first. A newly clicked
object is tracked by the SAM2 tracker (its memory DOES propagate), but the demo's
``_postprocess_output`` -> ``_build_sam2_output`` only surfaces an object on frames
already present in ``inference_state["cached_frame_outputs"]``; that cache is
normally seeded by a prior text/concept propagation. With NO concept the cache is
empty, so a single propagate surfaces obj_id=2 ONLY on the clicked (seed) frame.
  Pass 1: add click on frame 0, then ``propagate_in_video(start_frame_idx=0)``.
          This captures the seed-frame mask AND seeds cached_frame_outputs for all
          frames (as empty entries).
  Pass 2: re-issue the identical click on frame 0 (clear_old_points=True, a no-op
          refine that re-marks obj_id=2 for partial propagation), then
          ``propagate_in_video(start_frame_idx=1)``. Now every frame is in the
          cache, so the tracker-propagated obj_id=2 mask surfaces on frames 1..N-1.
  Both passes use ONLY the public add_prompt / propagate_in_video API -- no model
  code or inference_state internals are mutated. The masklet for obj_id=2 is the
  genuine SAM2-tracked segmentation of the frame-0 click.

Points are converted to the relative [0, 1] coords the upstream add_prompt expects
(rel_coordinates defaults True).

Determinism: seed 0, cuDNN deterministic, TF32 off; inference_mode + bf16 autocast.
use_fa3=False, use_rope_real=False (triton/FA3 kernels absent in this venv).

Env concessions (kernel/dep dispatch ONLY -- no weights/logic change), REQUIRED here
because triton/FA3 kernels are absent (SDPA raises "No available kernel"): edt scipy
stub + all-backend SDPA + CPU NMS/CC. Pass --patches to enable (used for the golden).
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
from PIL import Image

CLICK_XY = (385.0, 230.0)  # native 960x540 pixel
OBJ_ID = 2
LABEL = 1


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
    ap.add_argument("--frames", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--patches",
        action="store_true",
        help="Apply env concessions (edt/SDPA/NMS/CC) if kernels are absent.",
    )
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    _determinism()

    if args.patches:
        _install_edt_stub()
        _patch_sdpa_all_backends()
        _patch_nms_cpu()
        _patch_connected_components_cpu()
        print("[capture] env concessions applied: edt + SDPA(all) + NMS(cpu) + CC(cpu)")

    from sam3.model_builder import build_sam3_multiplex_video_predictor

    ckpt_path = str(Path(args.ckpt).resolve())
    print(f"[capture] Building SAM 3.1 multiplex video predictor (ckpt={ckpt_path})")
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
    _determinism()

    # Copy the first N frames into an isolated dir so the session loads exactly N.
    src_dir = Path(args.frames)
    src_paths = sorted(src_dir.glob("*.jpg"))[: args.n]
    assert len(src_paths) == args.n, f"need {args.n} frames, found {len(src_paths)}"
    w0, h0 = Image.open(src_paths[0]).size
    print(f"[capture] native frame size (w, h) = ({w0}, {h0})")

    tmp_dir = Path(tempfile.mkdtemp(prefix="sam3p1_interactive_ref_"))
    try:
        for i, p in enumerate(src_paths):
            shutil.copyfile(p, tmp_dir / f"{i:05d}.jpg")

        masklets = np.zeros((args.n, h0, w0), dtype=bool)

        def _extract(frame_index, out):
            """Copy obj_id=2's mask (if present) into masklets[frame_index]."""
            if not isinstance(out, dict) or "out_obj_ids" not in out:
                print(f"  frame {frame_index}: <no output dict>")
                return 0
            obj_ids = np.asarray(out["out_obj_ids"], dtype=np.int64)
            masks = np.asarray(out["out_binary_masks"])
            probs = np.asarray(out.get("out_probs", []), dtype=np.float32)
            hit = np.where(obj_ids == OBJ_ID)[0]
            score = None
            npix = 0
            if hit.size:
                j = int(hit[0])
                m = masks[j].astype(bool)
                if 0 <= frame_index < args.n:
                    masklets[frame_index] = m
                npix = int(m.sum())
                if j < probs.size:
                    score = float(probs[j])
            print(f"  frame {frame_index}: obj_ids={obj_ids.tolist()} "
                  f"obj{OBJ_ID}_pix={npix} obj{OBJ_ID}_score={score}")
            return npix

        x_rel, y_rel = CLICK_XY[0] / w0, CLICK_XY[1] / h0
        print(f"[capture] click native=({CLICK_XY[0]}, {CLICK_XY[1]}) "
              f"rel=({x_rel:.4f}, {y_rel:.4f}) obj_id={OBJ_ID} label={LABEL}")

        def _click():
            return (
                torch.tensor([[x_rel, y_rel]], dtype=torch.float32),
                torch.tensor([LABEL], dtype=torch.int32),
            )

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                state = model.init_state(
                    resource_path=str(tmp_dir), async_loading_frames=False
                )
                assert state["num_frames"] == args.n, (
                    f"loaded {state['num_frames']} frames, expected {args.n}"
                )
                assert (state["orig_height"], state["orig_width"]) == (h0, w0)

                # Pass 1: click on frame 0, propagate forward FROM 0. Captures the
                # seed-frame mask and seeds cached_frame_outputs for all frames.
                pts, lbls = _click()
                model.add_prompt(state, frame_idx=0, points=pts,
                                 point_labels=lbls, obj_id=OBJ_ID)
                print("[capture] pass 1: propagate_in_video(start_frame_idx=0):")
                for f_idx, out in model.propagate_in_video(
                    state, start_frame_idx=0, max_frame_num_to_track=None,
                    reverse=False,
                ):
                    if f_idx >= args.n:
                        break
                    if f_idx == 0:  # seed frame: authoritative click segmentation
                        _extract(f_idx, out)

                # Pass 2: re-issue the identical click (no-op refine that re-marks
                # obj_id=2 for partial propagation), propagate forward FROM 1. Every
                # frame is now cached, so obj_id=2's tracked mask surfaces on 1..N-1.
                pts, lbls = _click()
                model.add_prompt(state, frame_idx=0, points=pts,
                                 point_labels=lbls, obj_id=OBJ_ID,
                                 clear_old_points=True)
                print("[capture] pass 2: propagate_in_video(start_frame_idx=1):")
                for f_idx, out in model.propagate_in_video(
                    state, start_frame_idx=1, max_frame_num_to_track=None,
                    reverse=False,
                ):
                    if f_idx >= args.n:
                        break
                    _extract(f_idx, out)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "interactive_noconcept.npz", masklets=masklets)
    (out_dir / "interactive_scenario.json").write_text(json.dumps({
        "frame0_click_xy": list(CLICK_XY),
        "label": LABEL,
        "obj_id": OBJ_ID,
        "num_frames": args.n,
        "hw": [h0, w0],
    }, indent=2))

    per_frame_pix = [int(masklets[i].sum()) for i in range(args.n)]
    print(f"\n[capture] per-frame obj{OBJ_ID} pixel counts: {per_frame_pix}")
    print(f"[capture] saved masklets {masklets.shape} bool -> {out_dir}")


if __name__ == "__main__":
    main()
