# SPDX-License-Identifier: LicenseRef-SAM
"""Capture the model-find text-tracking golden from the OFFICIAL SAM 3.1 multiplex.

Run ONCE in the isolated reference env (NOT this repo's pixi env):

    cd C:/Users/javerlia/PycharmProjects/sam3_reference
    ./.venv/Scripts/python.exe \
        ../sam2/tests/parity/reference_sam3/capture_sam3p1_modelfind_golden.py \
        --frames ../sam2/notebooks/videos/dance \
        --ckpt ../sam2/checkpoints/sam3.1_multiplex.pt \
        --n 101 \
        --phrase person \
        --out ../sam2/tests/parity/fixtures/sam3p1 \
        --patches

Scenario (MODEL-FIND / detector text tracking, NO clicks):
  Stream the first N=101 dance frames (0..100) with a single TEXT concept
  "person". The detector finds people on frame 0; as the camera pans some
  leave and a fresh "person" re-enters mid-stream, spawning a NEW tracklet.
  The golden records, per frame, the upstream out_obj_ids and each object's
  binary mask at native 720x1280.

    model = predictor.model
    state = model.init_state(resource_path=..., async_loading_frames=False)
    model.add_prompt(state, frame_idx=0, text_str="person")  # concept prompt
    for f_idx, out in model.propagate_in_video(
        state, start_frame_idx=1, max_frame_num_to_track=None, reverse=False):
        ...

API path: MODEL API (Sam3MultiplexTrackingWithInteractivity, predictor.model).
  We build the predictor (for correct checkpoint remap + detector assembly),
  then drive the underlying model directly because the predictor's
  handle_request(start_session) path is broken in this reference rev.

  add_prompt and each propagate step return (frame_idx, out) where ``out`` is
  a dict of PARALLEL arrays (NOT a {obj_id: mask} map):
      out["out_obj_ids"]      int64  (K,)
      out["out_binary_masks"] bool   (K, H, W)  at native (orig) resolution
      out["out_probs"]        float32(K,)

  For a text concept the demo cache is seeded by add_prompt itself, so a single
  forward propagate from frame 1 surfaces objects on every frame -- no two-pass
  seeding is needed (unlike the interactive click golden).

Determinism: seed 0, cuDNN deterministic, TF32 off; inference_mode + bf16 autocast.
use_fa3=False, use_rope_real=False (triton/FA3 kernels absent in this venv).

Env concessions (kernel/dep dispatch ONLY -- no weights/logic change), REQUIRED
here because triton/FA3 kernels are absent (SDPA raises "No available kernel"):
edt scipy stub + all-backend SDPA + CPU NMS/CC. Pass --patches to enable.
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
    ap.add_argument("--n", type=int, default=101)
    ap.add_argument("--phrase", default="person")
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

    tmp_dir = Path(tempfile.mkdtemp(prefix="sam3p1_modelfind_ref_"))
    per_frame = {}  # frame_idx -> (obj_ids list, {oid: uint8 mask})
    try:
        for i, p in enumerate(src_paths):
            shutil.copyfile(p, tmp_dir / f"{i:05d}.jpg")

        def _collect(frame_index, out):
            if not isinstance(out, dict) or "out_obj_ids" not in out:
                print(f"  frame {frame_index}: <no output dict>")
                per_frame[frame_index] = ([], {})
                return
            obj_ids = np.asarray(out["out_obj_ids"], dtype=np.int64)
            masks = np.asarray(out["out_binary_masks"])
            probs = np.asarray(out.get("out_probs", []), dtype=np.float32)
            masks_by_id = {}
            pix = []
            for j, oid in enumerate(obj_ids.tolist()):
                m = masks[j].astype(np.uint8)
                masks_by_id[int(oid)] = m
                pix.append(int(m.sum()))
            per_frame[frame_index] = (obj_ids.tolist(), masks_by_id)
            print(f"  frame {frame_index}: n={obj_ids.size} "
                  f"obj_ids={obj_ids.tolist()} pix={pix} "
                  f"probs={[round(float(x), 3) for x in probs.tolist()]}")

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                state = model.init_state(
                    resource_path=str(tmp_dir), async_loading_frames=False
                )
                assert state["num_frames"] == args.n, (
                    f"loaded {state['num_frames']} frames, expected {args.n}"
                )
                assert (state["orig_height"], state["orig_width"]) == (h0, w0)

                print(f"[capture] add_prompt(frame_idx=0, text_str={args.phrase!r})")
                f0, out0 = model.add_prompt(
                    state, frame_idx=0, text_str=args.phrase
                )
                _collect(0, out0)

                print("[capture] propagate_in_video(start_frame_idx=1):")
                for f_idx, out in model.propagate_in_video(
                    state, start_frame_idx=1, max_frame_num_to_track=None,
                    reverse=False,
                ):
                    if f_idx >= args.n:
                        break
                    _collect(f_idx, out)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Assemble the npz payload.
    payload = {"num_frames": np.int64(args.n)}
    for i in range(args.n):
        obj_ids, masks_by_id = per_frame.get(i, ([], {}))
        payload[f"frame{i}_obj_ids"] = np.asarray(obj_ids, dtype=np.int64)
        for oid, m in masks_by_id.items():
            payload[f"frame{i}_obj{oid}"] = m.astype(np.uint8)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "modelfind_dance.npz", **payload)
    (out_dir / "modelfind_scenario.json").write_text(json.dumps({
        "phrase": args.phrase,
        "num_frames": args.n,
        "hw": [h0, w0],
        "frames_dir": "notebooks/videos/dance",
    }, indent=2))

    # Console summary table + spawn detection.
    print("\n[capture] per-frame id table:")
    seen = set()
    spawn_events = []
    for i in range(args.n):
        obj_ids, masks_by_id = per_frame.get(i, ([], {}))
        new_ids = [o for o in obj_ids if o not in seen]
        for o in new_ids:
            spawn_events.append((i, o))
        seen.update(obj_ids)
        tag = f"  NEW={new_ids}" if new_ids else ""
        pix = {o: int(masks_by_id[o].sum()) for o in obj_ids}
        print(f"  frame {i:3d}: n={len(obj_ids)} ids={obj_ids} pix={pix}{tag}")

    print("\n[capture] spawn events (frame, first-seen id):")
    for f, o in spawn_events:
        print(f"    frame {f}: id {o}")
    print(f"[capture] saved -> {out_dir / 'modelfind_dance.npz'}")


if __name__ == "__main__":
    main()
