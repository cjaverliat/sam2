# SPDX-License-Identifier: LicenseRef-SAM
"""Capture golden masks from the OFFICIAL EfficientSAM3 base-lineage (NON-multiplex) VIDEO model.

Run ONCE per variant in the isolated reference venv (NOT this repo's pixi env):

    C:\\Users\\javerlia\\PycharmProjects\\efficientsam3_reference\\.venv\\Scripts\\python.exe \\
        tests/parity/reference_efficientsam3/capture_efficientsam3_video_golden.py \\
        --backbone repvit --model-name m1.1 \\
        --ckpt checkpoints/_esam3_validate/stage1_all_converted/efficient_sam3_repvit_m_geo_ft.pt \\
        --out efficientsam3_repvit_video

Upstream commit: d063e00b1837f8dd285eb517d2dd40faabc34555 (efficientsam3 `main`).

This is the BASE SAM 3 lineage (per-object, 309 Sam3Tracker) distilled VIDEO model:
distilled EfficientSam3Trunk vision + PE text tower (text_encoder_type=None) + trained
geometry. Built via the upstream `build_efficientsam3_video_model`. UNLIKE the LiteText
capture, there is NO ctx monkeypatch (PE text, not a MobileCLIP student) and geometry is
KEPT (trained, part of the checkpoint).

Determinism: seed=0, deterministic algos (warn_only), cuDNN deterministic, TF32 OFF.
Precision: bf16 autocast. Capture-env concessions: NMS + connected-components CPU
fallbacks (reference env has no triton/cc_torch) -- no weights/logic change.
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

UPSTREAM_COMMIT = "d063e00b1837f8dd285eb517d2dd40faabc34555"
VIDEO_HW = (288, 512)
VIDEO_NUM_FRAMES = 4
VIDEO_PHRASE = "person"

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
REF_ROOT = Path(r"C:\Users\javerlia\PycharmProjects\efficientsam3_reference\sam3")
sys.path.insert(0, str(REF_ROOT))
BPE_PATH = REF_ROOT / "assets" / "bpe_simple_vocab_16e6.txt.gz"
VIDEO_DIR = REF_ROOT / "assets" / "videos" / "0001"


def _load_rgb(path, hw):
    img = Image.open(path).convert("RGB").resize((hw[1], hw[0]))
    return np.asarray(img, dtype=np.uint8)


def _run_section(run):
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return run(), "bf16_autocast"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, help="repvit | tinyvit | efficientvit")
    ap.add_argument("--model-name", required=True, help="m1.1 | 11m | b1")
    ap.add_argument("--ckpt", required=True, help="path to efficient_sam3_<bb>_m_geo_ft.pt")
    ap.add_argument("--out", required=True, help="golden npz basename (no extension)")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    out_npz = HERE / "golden" / f"{args.out}_video.npz"
    assert torch.cuda.is_available(), "CUDA required"
    assert ckpt_path.is_file(), f"Checkpoint absent: {ckpt_path}"
    assert BPE_PATH.is_file() and VIDEO_DIR.is_dir()

    # ------------------------------------------------------------------ determinism
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    import sam3.model_builder as mb  # noqa: E402 (fires _setup_tf32 -> re-disable below)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # ------------------------------------------------------------------ capture-env concessions
    # No triton/cc_torch in the reference env -> CPU fallbacks (same algorithm, no weights change).
    import sam3.perflib.nms as _nms_mod  # noqa: E402
    from sam3.perflib.nms import generic_nms_cpu as _generic_nms_cpu  # noqa: E402

    _nms_mod.generic_nms = _generic_nms_cpu

    import sam3.perflib.connected_components as _cc_mod  # noqa: E402
    from sam3.perflib.connected_components import connected_components_cpu as _cc_cpu  # noqa: E402

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

    from sam3.model_builder import build_efficientsam3_video_model  # noqa: E402

    # ------------------------------------------------------------------ build upstream model
    # text_encoder_type=None -> PE text tower (NOT a MobileCLIP student); no ctx monkeypatch.
    print(f"[capture] {args.backbone}/{args.model_name}  ckpt={ckpt_path}")
    model = build_efficientsam3_video_model(
        checkpoint_path=str(ckpt_path),
        load_from_HF=False,
        bpe_path=str(BPE_PATH),
        device="cuda",
        backbone_type=args.backbone,
        model_name=args.model_name,
        text_encoder_type=None,
        apply_temporal_disambiguation=True,
    )
    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # ------------------------------------------------------------------ assert clean load
    from sam3.model_builder import _load_state_dict_from_path  # noqa: E402

    ckpt_raw = _load_state_dict_from_path(str(ckpt_path))
    cleaned = {k.replace("student_trunk.", ""): v for k, v in ckpt_raw.items()}
    model_sd_keys = set(model.state_dict().keys())
    ckpt_keys = set(cleaned.keys())
    missing = sorted(model_sd_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_sd_keys)
    # The upstream builder constructs the distilled backbone WITH its (unused) classification
    # head (num_classes>0); the checkpoint -- and our EfficientSam3Trunk (num_classes=0) -- omit
    # it. The classifier is never touched by the feature-extraction forward, so it stays at
    # random init in the reference model and does NOT affect the captured masks. Allow exactly
    # those keys as missing; everything else must load cleanly.
    _UNUSED_HEAD = (".classifier.", ".head.", ".norm_head.")  # dropped by our num_classes=0 trunks
    classifier_missing = [k for k in missing if any(p in k for p in _UNUSED_HEAD)]
    real_missing = [k for k in missing if not any(p in k for p in _UNUSED_HEAD)]
    print(f"[capture] missing ({len(missing)}): "
          f"{len(classifier_missing)} unused-head + {len(real_missing)} other")
    print(f"[capture] real missing: {real_missing[:5] if real_missing else '(none)'}")
    print(f"[capture] unexpected ({len(unexpected)}): {unexpected[:5] if unexpected else '(none)'}")
    assert not real_missing, f"STOP: non-classifier model keys absent from ckpt: {real_missing}"
    assert not unexpected, f"STOP: ckpt keys absent from model: {unexpected}"
    print(f"[capture] Load assertion PASSED (0 unexpected; {len(classifier_missing)} "
          "unused-classifier keys random-init by design, off the feature path)")
    del ckpt_raw, cleaned

    # ------------------------------------------------------------------ frames
    tmp_dir = Path(tempfile.mkdtemp(prefix="esam3_video_ref_"))
    frames_rgb = []
    try:
        for i in range(VIDEO_NUM_FRAMES):
            rgb = _load_rgb(VIDEO_DIR / f"{i}.jpg", VIDEO_HW)
            frames_rgb.append(rgb)
            Image.fromarray(rgb).save(tmp_dir / f"{i}.png")
        frames_rgb = np.stack(frames_rgb)

        def run():
            state = model.init_state(resource_path=str(tmp_dir))
            model.add_prompt(state, frame_idx=0, text_str=VIDEO_PHRASE)
            masklets = {}
            for f_idx, fout in model.propagate_in_video(
                state, start_frame_idx=0,
                max_frame_num_to_track=VIDEO_NUM_FRAMES - 1, reverse=False,
            ):
                masklets[f_idx] = fout
            return masklets

        masklets, precision_mode = _run_section(run)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------ build npz (Phase-1 schema)
    out = {}
    print("[capture] Per-frame object counts:")
    for f_idx, fout in sorted(masklets.items()):
        obj_ids = np.asarray(fout["out_obj_ids"], np.int64)
        masks = np.asarray(fout["out_binary_masks"])
        probs = np.asarray(fout["out_probs"], np.float32)
        out[f"frame{f_idx}_obj_ids"] = obj_ids
        out[f"frame{f_idx}_scores"] = probs
        for j, oid in enumerate(obj_ids.tolist()):
            out[f"frame{f_idx}_obj{oid}"] = masks[j].astype(np.uint8)
        print(f"  frame {f_idx}: {len(obj_ids)} obj ids={obj_ids.tolist()} "
              f"scores={[round(float(p),3) for p in probs]}")

    out["video_frames_rgb"] = frames_rgb
    out["video_phrase"] = np.array(VIDEO_PHRASE)
    out["video_hw"] = np.array(VIDEO_HW, np.int64)
    out["video_frame_indices"] = np.arange(VIDEO_NUM_FRAMES, dtype=np.int64)
    out["precision_mode"] = np.array(precision_mode)
    out["upstream_commit"] = np.array(UPSTREAM_COMMIT)
    out["backbone"] = np.array(args.backbone)
    out["model_name"] = np.array(args.model_name)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **out)
    print(f"\n[capture] Wrote {out_npz} ({out_npz.stat().st_size/1e6:.2f} MB)")
    del model
    torch.cuda.empty_cache()
    print("[capture] DONE")


if __name__ == "__main__":
    main()
