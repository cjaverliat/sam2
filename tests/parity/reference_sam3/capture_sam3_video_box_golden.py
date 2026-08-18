# SPDX-License-Identifier: LicenseRef-SAM
"""Capture the video BOX-prompt golden from the OFFICIAL base SAM 3 (``sam3.pt``).

The base counterpart of ``capture_sam3p1_video_box_golden.py`` (which captures the
SAM 3.1 multiplex). Run ONCE in the isolated reference env (NOT this repo's pixi env):

    cd C:/Users/javerlia/PycharmProjects/sam3_reference
    ./.venv/Scripts/python.exe \
        ../sam2/tests/parity/reference_sam3/capture_sam3_video_box_golden.py \
        --frames ../sam2/notebooks/videos/bedroom \
        --ckpt ../sam2/checkpoints/sam3.pt \
        --n 8 \
        --out ../sam2/tests/parity/fixtures/sam3

Scenario (video BOX prompt, NO text concept) -- identical to the sam3.1 one so the
two lineages are comparable: stream the first N=8 bedroom frames (0..7) and on frame 0
add ONE BOX prompt at NATIVE pixel xyxy [300, 150, 470, 420] on the 960x540 video,
converted to the normalized ``[xmin, ymin, width, height]`` the upstream box API takes.

API path: MODEL API (``build_sam3_video_model`` -> ``Sam3VideoInference``), driven
directly:

    model = build_sam3_video_model(checkpoint_path=..., load_from_HF=False,
                                   bpe_path=..., device="cuda",
                                   apply_temporal_disambiguation=True)
    state = model.init_state(resource_path=<dir of N frames>)
    model.add_prompt(state, frame_idx=0, boxes_xywh=<(1,4) normalized>,
                     box_labels=<(1,) long>)          # NO text_str
    for f_idx, out in model.propagate_in_video(state, start_frame_idx=0, ...):
        ...

  ``add_prompt`` asserts only that text OR boxes are given, so a box-only session is
  legal. Upstream routes the box through ``_get_visual_prompt``
  (sam3_video_inference.py:181-222): the FIRST box on a frame with no prior inference
  becomes that frame's GEOMETRIC prompt (a ``Prompt`` carrying ``box_embeddings`` +
  ``box_labels``), stored in ``per_frame_geometric_prompt`` and consumed by the
  detector's geometry encoder. ``visual_prompt_embed`` stays None throughout -- the
  name "visual prompt" in that helper is about UI provenance, not the VISUAL slot.
  Boxes therefore bias DETECTION; they are never fed to the SAM 2 tracker as corner
  points.

  Both ``add_prompt`` and each propagate step return ``(frame_idx, out)`` with
  PARALLEL arrays: ``out["out_obj_ids"]`` (K,), ``out["out_binary_masks"]`` (K,H,W)
  at native resolution, ``out["out_probs"]`` (K,).

  Propagate FROM frame 0: the box's geometric prompt exists only on frame 0, so a run
  starting at frame 1 would skip the box entirely. Frame 0 is taken from add_prompt's
  authoritative output and frames 1..N-1 from propagation.

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

BOX_XYXY = [300, 150, 470, 420]  # native 960x540 pixel [xmin, ymin, xmax, ymax]
BOX_LABEL = 1


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
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bpe-path", default=None)
    ap.add_argument("--sam3-root", default=".")
    ap.add_argument("--patches", action="store_true",
                    help="Apply env concessions (edt/SDPA/NMS/CC) if kernels are absent.")
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

    xmin_px, ymin_px, xmax_px, ymax_px = BOX_XYXY
    box_xywh = [
        xmin_px / w0,
        ymin_px / h0,
        (xmax_px - xmin_px) / w0,
        (ymax_px - ymin_px) / h0,
    ]
    print(f"[capture] box native xyxy={BOX_XYXY} -> normalized xywh="
          f"[{box_xywh[0]:.4f}, {box_xywh[1]:.4f}, {box_xywh[2]:.4f}, {box_xywh[3]:.4f}]")

    tmp_dir = Path(tempfile.mkdtemp(prefix="sam3_video_box_ref_"))
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
            masks_by_id, pix = {}, []
            for j, oid in enumerate(obj_ids.tolist()):
                m = masks[j].astype(np.uint8)
                masks_by_id[int(oid)] = m
                pix.append(int(m.sum()))
            per_frame[frame_index] = (obj_ids.tolist(), masks_by_id)
            print(f"  frame {frame_index}: n={obj_ids.size} obj_ids={obj_ids.tolist()} "
                  f"pix={pix} probs={[round(float(x), 3) for x in probs.tolist()]}")

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                state = model.init_state(resource_path=str(tmp_dir))
                assert state["num_frames"] == args.n, (
                    f"loaded {state['num_frames']} frames, expected {args.n}"
                )
                assert (state["orig_height"], state["orig_width"]) == (h0, w0)

                boxes_xywh = torch.tensor([box_xywh], dtype=torch.float32)  # (1, 4)
                box_labels = torch.tensor([BOX_LABEL], dtype=torch.long)    # (1,)
                print("[capture] add_prompt(frame_idx=0, boxes_xywh=..., "
                      "box_labels=[1])  # NO text_str")
                _f0, out0 = model.add_prompt(
                    state, frame_idx=0,
                    boxes_xywh=boxes_xywh, box_labels=box_labels,
                )
                _collect(0, out0)

                print("[capture] propagate_in_video(start_frame_idx=0):")
                for f_idx, out in model.propagate_in_video(
                    state, start_frame_idx=0, max_frame_num_to_track=None,
                    reverse=False,
                ):
                    if f_idx >= args.n:
                        break
                    if f_idx == 0:  # keep add_prompt's authoritative box detection
                        continue
                    _collect(f_idx, out)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    payload = {"num_frames": np.int64(args.n)}
    for i in range(args.n):
        obj_ids, masks_by_id = per_frame.get(i, ([], {}))
        payload[f"frame{i}_obj_ids"] = np.asarray(obj_ids, dtype=np.int64)
        for oid, m in masks_by_id.items():
            payload[f"frame{i}_obj{oid}"] = m.astype(np.uint8)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "video_box.npz", **payload)
    (out_dir / "video_box_scenario.json").write_text(json.dumps({
        "box_xyxy": BOX_XYXY,
        "num_frames": args.n,
        "hw": [h0, w0],
        "frames_dir": "notebooks/videos/bedroom",
    }, indent=2))

    print("\n[capture] per-frame id table:")
    seen, spawn_events = set(), []
    for i in range(args.n):
        obj_ids, masks_by_id = per_frame.get(i, ([], {}))
        new_ids = [o for o in obj_ids if o not in seen]
        spawn_events += [(i, o) for o in new_ids]
        seen.update(obj_ids)
        pix = {o: int(masks_by_id[o].sum()) for o in obj_ids}
        tag = f"  NEW={new_ids}" if new_ids else ""
        print(f"  frame {i:3d}: n={len(obj_ids)} ids={obj_ids} pix={pix}{tag}")

    print("\n[capture] spawn events (frame, first-seen id):")
    for f, o in spawn_events:
        print(f"    frame {f}: id {o}")
    print(f"[capture] saved -> {out_dir / 'video_box.npz'}")


if __name__ == "__main__":
    main()
