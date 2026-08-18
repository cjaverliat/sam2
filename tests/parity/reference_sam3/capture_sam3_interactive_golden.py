# SPDX-License-Identifier: LicenseRef-SAM
"""Capture the interactive CLICK golden from the OFFICIAL base SAM 3 (``sam3.pt``).

The base counterpart of ``capture_sam3p1_interactive_golden.py`` (multiplex). Run ONCE
in the isolated reference env (NOT this repo's pixi env):

    cd C:/Users/javerlia/PycharmProjects/sam3_reference
    ./.venv/Scripts/python.exe \
        ../sam2/tests/parity/reference_sam3/capture_sam3_interactive_golden.py \
        --frames ../sam2/notebooks/videos/bedroom \
        --ckpt ../sam2/checkpoints/sam3.pt \
        --n 30 \
        --out ../sam2/tests/parity/fixtures/sam3

Scenario (INTERACTIVE point click, NO text concept):
  On frame 0, add a POSITIVE point click (label 1) at native pixel (385.0, 230.0),
  ``obj_id=2``, no text concept -> propagate forward N frames and record, per frame,
  WHICH obj_ids upstream emits and obj 2's binary mask.

Why N=30 rather than the 8 of the sam3p1 capture: this capture exists to settle
whether upstream's base lineage PURGES a click-seeded object once it goes unmatched.
Our streaming predictor registers a clicked tracklet with the same lifecycle as a
detected one (``_apply_geometry_prompt`` -> ``TrackletManager.spawn``); with no
concept, detection never runs, so nothing re-matches the tracklet and it is removed
at frame 8 (``hotstart_unmatch_thresh=8`` inside ``hotstart_delay=15``). The existing
8-frame golden stops one frame short of that, so it cannot adjudicate. The multiplex
lineage never registers clicked objects at all and keeps them for the whole clip.

API path: MODEL API (``build_sam3_video_model`` -> ``Sam3VideoInferenceWithInstance-
Interactivity``), driven directly:

    model = build_sam3_video_model(checkpoint_path=..., load_from_HF=False,
                                   bpe_path=..., device="cuda",
                                   apply_temporal_disambiguation=True)
    state = model.init_state(resource_path=<dir of N frames>)
    model.add_prompt(state, frame_idx=0, points=<(1,2) rel>, point_labels=<(1,)>,
                     obj_id=2)                    # -> add_tracker_new_points

  ``add_prompt`` routes points to the tracker (``sam3_video_inference.py:1372-1389``):
  text and boxes must be None, and ``obj_id`` is required. Points are relative [0, 1]
  coordinates (``rel_coordinates`` defaults True).

  ``add_prompt`` and each propagate step return ``(frame_idx, out)`` with PARALLEL
  arrays: ``out["out_obj_ids"]`` (K,), ``out["out_binary_masks"]`` (K,H,W) at native
  resolution, ``out["out_probs"]`` (K,).

TWO-PASS cache seeding (same reason as the multiplex capture): ``_build_tracker_output``
only surfaces an object on frames already present in ``inference_state["cached_frame_
outputs"]``, which is normally seeded by a prior concept propagation. With no concept
the cache is empty, so a single propagate surfaces the clicked object only on the seed
frame.
  Pass 1: click on frame 0, ``propagate_in_video(start_frame_idx=0)`` -> seed-frame
          mask, and every frame lands in the cache.
  Pass 2: ``propagate_in_video(start_frame_idx=0)`` again, with NO second click ->
          the tracker-propagated mask surfaces on frames 1..N-1.
  Both passes use ONLY the public add_prompt / propagate_in_video API.

  Do NOT re-issue the click to drive pass 2 (the multiplex capture does, because its
  ``add_prompt`` takes ``clear_old_points``). On the base path a second click hits
  ``use_stateless_refinement``: the object is REMOVED and re-added, and the re-seeded
  click resolves to a different mask -- measured 8736 px (the girl's skirt) against
  31889 px (the whole girl) for the identical click. That mask then propagates, so a
  re-click pass yields a golden that tracks the skirt.

  NOTE the emitted-id record comes from pass 2, which is the run that can show the
  object disappearing. A frame whose ``out_obj_ids`` omits obj 2 is upstream declining
  to emit it there.

Determinism: seed 0, cuDNN deterministic, TF32 off; inference_mode + bf16 autocast.

Env concessions (kernel/dep dispatch ONLY -- no weights/logic change) are available
behind ``--patches`` for hosts where the triton/FA3 kernels are absent: edt scipy stub
+ all-backend SDPA + CPU NMS/CC. The base lineage generally does not need them.
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
    """Force all-backend SDPA in the decoder (FA kernels may be absent)."""
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
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sam3-root", default=".")
    ap.add_argument("--bpe-path", default=None)
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

    from sam3.model_builder import build_sam3_video_model

    # model_builder._setup_tf32() runs at import time and re-enables TF32.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    bpe_path = args.bpe_path or str(
        Path(args.sam3_root) / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    )
    ckpt_path = str(Path(args.ckpt).resolve())
    print(f"[capture] Building base SAM 3 video model (ckpt={ckpt_path})")
    model = build_sam3_video_model(
        checkpoint_path=ckpt_path,
        load_from_HF=False,
        bpe_path=bpe_path,
        device="cuda",
        apply_temporal_disambiguation=True,  # the default / notebook config
    )
    model.eval()
    _determinism()

    src_dir = Path(args.frames)
    src_paths = sorted(src_dir.glob("*.jpg"))[: args.n]
    assert len(src_paths) == args.n, f"need {args.n} frames, found {len(src_paths)}"
    w0, h0 = Image.open(src_paths[0]).size
    print(f"[capture] native frame size (w, h) = ({w0}, {h0})")

    x_rel, y_rel = CLICK_XY[0] / w0, CLICK_XY[1] / h0
    print(f"[capture] click native=({CLICK_XY[0]}, {CLICK_XY[1]}) "
          f"rel=({x_rel:.4f}, {y_rel:.4f}) obj_id={OBJ_ID} label={LABEL}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="sam3_interactive_ref_"))
    masklets = np.zeros((args.n, h0, w0), dtype=bool)
    emitted_ids = {}  # frame_idx -> list of obj_ids upstream emitted
    try:
        for i, p in enumerate(src_paths):
            shutil.copyfile(p, tmp_dir / f"{i:05d}.jpg")

        def _collect(frame_index, out):
            """Record the emitted ids and obj OBJ_ID's mask for one frame."""
            if not isinstance(out, dict) or "out_obj_ids" not in out:
                print(f"  frame {frame_index}: <no output dict>")
                emitted_ids[frame_index] = []
                return
            obj_ids = np.asarray(out["out_obj_ids"], dtype=np.int64)
            masks = np.asarray(out["out_binary_masks"])
            probs = np.asarray(out.get("out_probs", []), dtype=np.float32)
            emitted_ids[frame_index] = obj_ids.tolist()
            hit = np.where(obj_ids == OBJ_ID)[0]
            npix, score = 0, None
            if hit.size:
                j = int(hit[0])
                m = masks[j].astype(bool)
                if 0 <= frame_index < args.n:
                    masklets[frame_index] = m
                npix = int(m.sum())
                if j < probs.size:
                    score = round(float(probs[j]), 3)
            print(f"  frame {frame_index}: obj_ids={obj_ids.tolist()} "
                  f"obj{OBJ_ID}_pix={npix} obj{OBJ_ID}_score={score}")

        def _click():
            return (
                torch.tensor([[x_rel, y_rel]], dtype=torch.float32),
                torch.tensor([LABEL], dtype=torch.int32),
            )

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                state = model.init_state(resource_path=str(tmp_dir))
                assert state["num_frames"] == args.n, (
                    f"loaded {state['num_frames']} frames, expected {args.n}"
                )
                assert (state["orig_height"], state["orig_width"]) == (h0, w0)

                # Pass 1: click on frame 0, propagate FROM 0 -> authoritative seed-frame
                # mask, and cached_frame_outputs gets an entry for every frame.
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
                    if f_idx == 0:
                        _collect(f_idx, out)

                # Pass 2: propagate again, NO second click (see the module docstring
                # -- a re-click re-seeds the object to the skirt). Every frame is
                # cached now, so the tracked mask surfaces on frames 1..N-1.
                print("[capture] pass 2: propagate_in_video(start_frame_idx=0):")
                for f_idx, out in model.propagate_in_video(
                    state, start_frame_idx=0, max_frame_num_to_track=None,
                    reverse=False,
                ):
                    if f_idx >= args.n:
                        break
                    if f_idx == 0:  # keep add_prompt's authoritative click mask
                        continue
                    _collect(f_idx, out)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    payload = {"masklets": masklets, "num_frames": np.int64(args.n)}
    for i in range(args.n):
        payload[f"frame{i}_obj_ids"] = np.asarray(
            emitted_ids.get(i, []), dtype=np.int64
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "interactive_noconcept.npz", **payload)
    (out_dir / "interactive_scenario.json").write_text(json.dumps({
        "frame0_click_xy": list(CLICK_XY),
        "label": LABEL,
        "obj_id": OBJ_ID,
        "num_frames": args.n,
        "hw": [h0, w0],
        "frames_dir": "notebooks/videos/bedroom",
    }, indent=2))

    per_frame_pix = [int(masklets[i].sum()) for i in range(args.n)]
    print(f"\n[capture] per-frame obj{OBJ_ID} pixel counts: {per_frame_pix}")
    alive = [i for i in range(args.n) if OBJ_ID in emitted_ids.get(i, [])]
    print(f"[capture] frames emitting obj{OBJ_ID}: {len(alive)}/{args.n}"
          f"{'' if not alive else f' (last = {max(alive)})'}")
    print(f"[capture] saved masklets {masklets.shape} bool -> {out_dir}")


if __name__ == "__main__":
    main()
