# SPDX-License-Identifier: LicenseRef-SAM
"""Instrumented rerun of the video-box golden: WHY are frames 1-4 empty upstream?

Prints, per frame: the pre-filter obj_id -> mask pixel counts, and the three hide
sets (removed / suppressed / unconfirmed) plus the masklet-confirmation counters.
Writes NOTHING (does not touch the golden fixtures).

Run in the reference env (NOT this repo's pixi env):
    cd C:/Users/javerlia/PycharmProjects/sam3_reference
    ./.venv/Scripts/python.exe \
        ../sam2/tests/parity/reference_sam3/debug_sam3p1_video_box_hotstart.py \
        --frames ../sam2/notebooks/videos/bedroom \
        --ckpt ../sam2/checkpoints/sam3.1_multiplex.pt --n 8 --patches

Captured 2026-08-17 (obj 0, 8 bedroom frames), which established that upstream's
multiplex hide-set is `unconfirmed(min(f + thresh-1, last))` + empty-mask and nothing
else -- keep-alive suppression never fires there. Per-frame trace is in the ledger
(`docs/superpowers/plans/2026-06-26-phase1-sam3-torch-inference.md`).
"""
import argparse
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BOX_XYXY = [300, 150, 470, 420]
BOX_LABEL = 1


def _install_edt_stub():
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


def _instrument(model):
    """Log per-frame masks + hide sets from _run_single_frame_inference."""
    cls = type(model)
    orig = cls._run_single_frame_inference

    def patched(self, inference_state, frame_idx, reverse, *a, **kw):
        out = orig(self, inference_state, frame_idx, reverse, *a, **kw)
        meta = inference_state["tracker_metadata"]
        rank0 = meta["rank0_metadata"]
        conf = rank0.get("masklet_confirmation", {})
        pix = {
            int(o): int(m.sum().item()) for o, m in out["obj_id_to_mask"].items()
        }
        gpu_meta = meta.get("gpu_metadata", {})
        keep_alive = gpu_meta.get("trk_keep_alive", None)
        unmatch = gpu_meta.get("consecutive_unmatch_count", None)
        print(
            f"[frame {frame_idx}] pix={pix} "
            f"removed={sorted(rank0.get('removed_obj_ids', []))} "
            f"suppressed={sorted(rank0.get('suppressed_obj_ids', {}).get(frame_idx, []))} "
            f"unconfirmed={out.get('unconfirmed_obj_ids')} "
            f"obj_ids_all={list(meta.get('obj_ids_all_gpu', []))} "
            f"consec_det={list(conf.get('consecutive_det_num', []))} "
            f"status={list(conf.get('status', []))} "
            f"keep_alive={keep_alive.tolist() if keep_alive is not None else None} "
            f"consec_unmatch={unmatch.tolist() if unmatch is not None else None} "
            f"sam2_score={ {int(k): round(float(v), 3) for k, v in out.get('obj_id_to_sam2_score', {}).items()} }",
            flush=True,
        )
        return out

    cls._run_single_frame_inference = patched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--patches", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    _determinism()
    if args.patches:
        _install_edt_stub()
        _patch_sdpa_all_backends()
        _patch_nms_cpu()
        _patch_connected_components_cpu()
        print("[debug] env concessions applied")

    from sam3.model_builder import build_sam3_multiplex_video_predictor

    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path=str(Path(args.ckpt).resolve()),
        use_fa3=False,
        use_rope_real=False,
        compile=False,
        warm_up=False,
        async_loading_frames=False,
    )
    model = predictor.model
    model.eval()
    _determinism()
    print(
        f"[debug] cfg: hotstart_delay={model.hotstart_delay} "
        f"unmatch_thresh={model.hotstart_unmatch_thresh} "
        f"suppress_only_within_hotstart={model.suppress_unmatched_only_within_hotstart} "
        f"init/min/max_keep_alive={model.init_trk_keep_alive}/"
        f"{model.min_trk_keep_alive}/{model.max_trk_keep_alive} "
        f"masklet_confirmation={model.masklet_confirmation_enable} "
        f"conf_thresh={model.masklet_confirmation_consecutive_det_thresh}"
    )
    _instrument(model)

    src_dir = Path(args.frames)
    src_paths = sorted(src_dir.glob("*.jpg"))[: args.n]
    assert len(src_paths) == args.n
    w0, h0 = Image.open(src_paths[0]).size
    xmin, ymin, xmax, ymax = BOX_XYXY
    box_xywh = [xmin / w0, ymin / h0, (xmax - xmin) / w0, (ymax - ymin) / h0]

    tmp_dir = Path(tempfile.mkdtemp(prefix="sam3p1_box_debug_"))
    try:
        for i, p in enumerate(src_paths):
            shutil.copyfile(p, tmp_dir / f"{i:05d}.jpg")
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                state = model.init_state(
                    resource_path=str(tmp_dir), async_loading_frames=False
                )
                print("[debug] add_prompt(frame 0, box)")
                f0, out0 = model.add_prompt(
                    state,
                    frame_idx=0,
                    boxes_xywh=torch.tensor([box_xywh], dtype=torch.float32),
                    box_labels=torch.tensor([BOX_LABEL], dtype=torch.long),
                )
                ids0 = np.asarray(out0["out_obj_ids"], dtype=np.int64).tolist()
                m0 = np.asarray(out0["out_binary_masks"])
                print(f"[debug] add_prompt OUT ids={ids0} "
                      f"pix={[int(m0[j].sum()) for j in range(len(ids0))]}")
                print("[debug] propagate_in_video(start=0):")
                for f_idx, out in model.propagate_in_video(
                    state, start_frame_idx=0, max_frame_num_to_track=None,
                    reverse=False,
                ):
                    if f_idx >= args.n:
                        break
                    ids = np.asarray(out["out_obj_ids"], dtype=np.int64).tolist()
                    masks = np.asarray(out["out_binary_masks"])
                    print(f"[debug] YIELD frame {f_idx}: ids={ids} "
                          f"pix={[int(masks[j].sum()) for j in range(len(ids))]}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
