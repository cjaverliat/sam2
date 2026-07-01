# SPDX-License-Identifier: LicenseRef-SAM
"""Capture golden reference masks from the OFFICIAL SAM3-LiteText (s0/ctx16) video model.

Run ONCE in the isolated reference venv (NOT this repo's pixi env):

    C:\\Users\\javerlia\\PycharmProjects\\efficientsam3_reference\\.venv\\Scripts\\python.exe \\
        tests/parity/reference_efficientsam3/capture_litetext_video_golden.py

Writes: tests/parity/reference_efficientsam3/golden/sam3_litetext_s0_ctx16_video.npz

Upstream commit: d063e00b1837f8dd285eb517d2dd40faabc34555

Determinism: seed=0, torch.use_deterministic_algorithms(True, warn_only=True),
cuDNN deterministic, TF32 OFF (re-disabled after model_builder import).
Precision: bf16 autocast (the only viable inference mode for SAM3 -- perflib fused
addmm_act hardcodes .to(bfloat16)).

CTX MONKEYPATCH (required): the merged checkpoint's text-encoder pos_embed is
(1,1,16,512). The upstream build_sam3_video_model hardcodes _create_student_text_encoder
context_length=77 -> pos_embed (1,1,77,512). PyTorch load_state_dict (even with
strict=False) raises a shape-mismatch RuntimeError when the key exists in both model
and checkpoint with different shapes. We patch the module-level function before the
build call to force ctx=16 -> shapes match -> clean load.

Geometry: KEPT (this is a video model; geometry is trained and part of the 1281-key
checkpoint). Do NOT apply the _run_golden_nogeo.py geometry-disable slice.
"""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Scenario constants (must match the parity test verbatim)
# ---------------------------------------------------------------------------
UPSTREAM_COMMIT = "d063e00b1837f8dd285eb517d2dd40faabc34555"
VIDEO_HW = (288, 512)          # (H, W)
VIDEO_NUM_FRAMES = 4           # frames 0..3
VIDEO_PHRASE = "person"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]   # tests/parity/reference_efficientsam3 -> parents[2] = sam2/

REF_ROOT = Path(r"C:\Users\javerlia\PycharmProjects\efficientsam3_reference\sam3")
sys.path.insert(0, str(REF_ROOT))

CKPT_PRIMARY = REPO_ROOT / "checkpoints" / "sam3_litetext_mobileclip_s0_ctx16.pt"
CKPT_VALIDATE = (
    REPO_ROOT / "checkpoints" / "_esam3_validate"
    / "sam3_litetext" / "sam3_litetext_mobileclip_s0_ctx16.pt"
)
CKPT_PATH = CKPT_PRIMARY if CKPT_PRIMARY.is_file() else CKPT_VALIDATE

BPE_PATH = REF_ROOT / "assets" / "bpe_simple_vocab_16e6.txt.gz"
VIDEO_DIR = REF_ROOT / "assets" / "videos" / "0001"
OUT_NPZ = HERE / "golden" / "sam3_litetext_s0_ctx16_video.npz"


# ---------------------------------------------------------------------------
def _load_rgb(path, hw):
    """Load image as (H,W,3) uint8 array resized to hw=(H,W)."""
    img = Image.open(path).convert("RGB").resize((hw[1], hw[0]))  # PIL is (W,H)
    return np.asarray(img, dtype=np.uint8)


def _run_section(run):
    """Run fn() under bf16 autocast + inference_mode (the only viable SAM3 regime)."""
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return run(), "bf16_autocast"


def main():
    assert torch.cuda.is_available(), "CUDA required"
    assert CKPT_PATH.is_file(), f"Checkpoint absent: {CKPT_PATH}"
    assert BPE_PATH.is_file(), f"BPE absent: {BPE_PATH}"
    assert VIDEO_DIR.is_dir(), f"Video dir absent: {VIDEO_DIR}"

    # ------------------------------------------------------------------ determinism
    # Pre-disable TF32 BEFORE the sam3.model_builder import, which calls _setup_tf32()
    # at module level and re-enables TF32. We re-disable again after the import.
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # ------------------------------------------------------------------ CTX MONKEYPATCH
    # Must be applied BEFORE build_sam3_video_model is called.
    # sam3.model_builder import fires _setup_tf32() -> re-disable TF32 after.
    import sam3.model_builder as mb  # noqa: E402

    _CTX = 16
    _orig_text = mb._create_student_text_encoder
    mb._create_student_text_encoder = (
        lambda b, t, context_length=77: _orig_text(b, t, context_length=_CTX)
    )

    # Re-disable TF32 (model_builder._setup_tf32() ran at import time and re-enabled it)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # ------------------------------------------------------------------ NMS TRITON FALLBACK
    # The reference env has no triton package. The upstream generic_nms CUDA path tries
    # `from sam3.perflib.triton.nms import nms_triton` when torch_generic_nms is absent.
    # We patch the module-level function to use the pure-Python CPU fallback instead --
    # same greedy NMS algorithm, identical results (same strict-IoU > threshold suppression).
    # This is a capture-env concession (no weights/logic change); parity gates are IoU-based.
    import sam3.perflib.nms as _nms_mod  # noqa: E402
    from sam3.perflib.nms import generic_nms_cpu as _generic_nms_cpu  # noqa: E402

    _nms_mod.generic_nms = _generic_nms_cpu

    # ------------------------------------------------------------------ CONNECTED COMPONENTS FALLBACK
    # Same pattern: reference env has no triton / cc_torch; patch to CPU (skimage) fallback.
    # Wrapper needed to: (a) match the original's (B,3,H,W)->(B,1,H,W) preprocessing,
    # (b) handle empty batch (B=0) which torch.stack rejects.
    import sam3.perflib.connected_components as _cc_mod  # noqa: E402
    from sam3.perflib.connected_components import connected_components_cpu as _cc_cpu  # noqa: E402

    def _cc_cpu_fallback(input_tensor):
        """CPU (skimage) fallback for connected_components; handles B=0 and 3-D inputs."""
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(1)  # (B,H,W) -> (B,1,H,W)
        assert input_tensor.dim() == 4 and input_tensor.shape[1] == 1
        if input_tensor.shape[0] == 0:
            return (
                torch.zeros_like(input_tensor, dtype=torch.int64),
                torch.zeros_like(input_tensor, dtype=torch.int64),
            )
        return _cc_cpu(input_tensor)

    _cc_mod.connected_components = _cc_cpu_fallback

    from sam3.model_builder import build_sam3_video_model  # noqa: E402 (import cached)

    # ------------------------------------------------------------------ build upstream model
    print(f"[capture] Loading checkpoint: {CKPT_PATH}")
    model = build_sam3_video_model(
        checkpoint_path=str(CKPT_PATH),
        load_from_HF=False,
        bpe_path=str(BPE_PATH),
        device="cuda",
        text_encoder_type="MobileCLIP-S0",
        text_encoder_context_length=16,
        apply_temporal_disambiguation=True,
    )
    model.eval()

    # Re-disable TF32 (model build may re-enable it)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # ------------------------------------------------------------------ assert clean load
    # Compare model state_dict keys vs cleaned checkpoint keys.
    # With the ctx monkeypatch, pos_embed shapes match; missing/unexpected must be empty.
    from sam3.model_builder import _load_state_dict_from_path  # noqa: E402

    ckpt_raw = _load_state_dict_from_path(str(CKPT_PATH))
    cleaned = {k.replace("student_trunk.", ""): v for k, v in ckpt_raw.items()}
    model_sd_keys = set(model.state_dict().keys())
    ckpt_keys = set(cleaned.keys())
    missing = sorted(model_sd_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_sd_keys)
    print(f"[capture] missing keys  ({len(missing)}): {missing[:5] if missing else '(none)'}")
    print(f"[capture] unexpected keys ({len(unexpected)}): {unexpected[:5] if unexpected else '(none)'}")
    assert not missing, f"STOP: keys in model but not in checkpoint: {missing}"
    assert not unexpected, f"STOP: keys in checkpoint but not in model: {unexpected}"
    print("[capture] Load assertion PASSED (0 missing, 0 unexpected)")
    del ckpt_raw, cleaned  # free memory before the forward pass

    # ------------------------------------------------------------------ load frames
    tmp_dir = Path(tempfile.mkdtemp(prefix="litetext_ref_frames_"))
    frames_rgb = []
    try:
        for i in range(VIDEO_NUM_FRAMES):
            rgb = _load_rgb(VIDEO_DIR / f"{i}.jpg", VIDEO_HW)
            frames_rgb.append(rgb)
            # Save as lossless PNG so init_state byte-loads the SAME pixels we captured.
            Image.fromarray(rgb).save(tmp_dir / f"{i}.png")
        frames_rgb = np.stack(frames_rgb)   # (T,H,W,3) uint8
        print(f"[capture] frames shape: {frames_rgb.shape}")

        # ------------------------------------------------------------ run inference
        def run():
            state = model.init_state(resource_path=str(tmp_dir))
            model.add_prompt(state, frame_idx=0, text_str=VIDEO_PHRASE)
            masklets = {}
            for f_idx, fout in model.propagate_in_video(
                state,
                start_frame_idx=0,
                max_frame_num_to_track=VIDEO_NUM_FRAMES - 1,
                reverse=False,
            ):
                masklets[f_idx] = fout
            return masklets

        masklets, precision_mode = _run_section(run)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------ build output dict
    # Schema mirrors Phase-1 video.npz EXACTLY (no trk_f1: LiteText is streaming-only).
    out = {}
    print("[capture] Per-frame object counts:")
    for f_idx, fout in sorted(masklets.items()):
        obj_ids = np.asarray(fout["out_obj_ids"], np.int64)
        masks = np.asarray(fout["out_binary_masks"])      # (N,H,W) bool
        probs = np.asarray(fout["out_probs"], np.float32)
        out[f"frame{f_idx}_obj_ids"] = obj_ids
        out[f"frame{f_idx}_scores"] = probs
        for j, oid in enumerate(obj_ids.tolist()):
            out[f"frame{f_idx}_obj{oid}"] = masks[j].astype(np.uint8)
        print(
            f"  frame {f_idx}: {len(obj_ids)} object(s) "
            f"ids={obj_ids.tolist()}  scores={[round(float(p), 3) for p in probs]}"
        )

    out["video_frames_rgb"] = frames_rgb        # (T,H,W,3) uint8
    out["video_phrase"] = np.array(VIDEO_PHRASE)
    out["video_hw"] = np.array(VIDEO_HW, np.int64)
    out["video_frame_indices"] = np.arange(VIDEO_NUM_FRAMES, dtype=np.int64)
    out["precision_mode"] = np.array(precision_mode)
    out["upstream_commit"] = np.array(UPSTREAM_COMMIT)

    # ------------------------------------------------------------------ save
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ, **out)
    size_mb = OUT_NPZ.stat().st_size / 1e6
    print(f"\n[capture] Wrote {OUT_NPZ} ({size_mb:.2f} MB)")
    print(f"[capture] Keys: {sorted(out)}")

    del model
    torch.cuda.empty_cache()
    print("[capture] DONE")


if __name__ == "__main__":
    main()
