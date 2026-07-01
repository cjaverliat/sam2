# SPDX-License-Identifier: LicenseRef-SAM
"""Capture golden reference masks for SAM3.1-LiteText (PE vision / MobileCLIP-S0 / ctx16).

NATIVE reference (apples-to-apples; efficientsam3's OWN sam3.1, NOT facebook):
  efficientsam3 ships its multiplex sam3.1 code on the ``stage1_sam3.1`` branch (worktree at
  ``C:/Users/javerlia/PycharmProjects/efficientsam3_sam3p1``, commit ``6056958``). Its
  ``build_efficientsam3_multiplex_video_model(backbone_type="sam3", text_encoder_type="MobileCLIP-S0")``
  builds the multiplex video model with the FULL PE-ViT vision encoder + MobileCLIP text NATIVELY;
  the checkpoint ``efficient_sam3p1_litetext_mobileclip_s0_ctx16.pt`` (1439 keys: detector 982 +
  tracker 457) loads STRICT with NO encoder swapping. This REPLACES the earlier facebook-derived
  two-repo oracle (wrong reference: distilled-vs-non-distilled), matching the F1-native treatment.

Run ONCE in the efficientsam3 reference venv:
    C:\\Users\\javerlia\\PycharmProjects\\efficientsam3_reference\\.venv\\Scripts\\python.exe \\
        tests/parity/reference_efficientsam3/capture_sam3p1_litetext_video_golden.py

Writes: tests/parity/reference_efficientsam3/golden/sam3p1_litetext_s0_ctx16_video.npz

Env concessions (kernel/dep dispatch only -- NO weights/logic change): identical to the
EfficientSAM3.1 capture -- edt scipy stub (triton absent) + all-backend SDPA + CPU NMS/CC.
Determinism: seed 0, cuDNN deterministic, TF32 off; bf16 autocast. use_fa3=False.
"""
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image

UPSTREAM_COMMIT = "6056958418438beccd4f0782f9b73a1fbcca3e5a"  # efficientsam3 stage1_sam3.1
VIDEO_HW = (288, 512)
VIDEO_NUM_FRAMES = 4
VIDEO_PHRASE = "person"

# SAM3.1-LiteText: PE-ViT vision (backbone "sam3") + MobileCLIP-S0 text (ctx16).
BACKBONE_TYPE = "sam3"
MODEL_NAME = "b0"            # ignored for backbone "sam3" (PE vision); placeholder
TEXT_ENCODER_TYPE = "MobileCLIP-S0"
TEXT_CTX = 16

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
WORKTREE = Path(r"C:\Users\javerlia\PycharmProjects\efficientsam3_sam3p1")
SAM3_PKG_PARENT = WORKTREE / "sam3"
BPE_PATH = SAM3_PKG_PARENT / "assets" / "bpe_simple_vocab_16e6.txt.gz"
VIDEO_DIR = SAM3_PKG_PARENT / "assets" / "videos" / "0001"
CKPT_PATH = (
    REPO_ROOT / "checkpoints" / "_esam3_validate"
    / "sam3p1_litetext" / "efficient_sam3p1_litetext_mobileclip_s0_ctx16.pt"
)
OUT_NPZ = HERE / "golden" / "sam3p1_litetext_s0_ctx16_video.npz"


def _load_rgb(path, hw):
    img = Image.open(path).convert("RGB").resize((hw[1], hw[0]))
    return np.asarray(img, dtype=np.uint8)


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


def _patch_multiplex_sdpa():
    from torch.nn.attention import SDPBackend
    from torch.nn.attention import sdpa_kernel as _sk
    from sam3.model import decoder as _dec

    _all = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH]
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
            return (torch.zeros_like(input_tensor, dtype=torch.int64),
                    torch.zeros_like(input_tensor, dtype=torch.int64))
        return _cc_cpu(input_tensor)

    _cc_mod.connected_components = _cc_cpu_fallback


def determinism():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _run_section(run):
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return run(), "bf16_autocast"


def main():
    assert torch.cuda.is_available(), "CUDA required"
    assert CKPT_PATH.is_file(), f"LiteText checkpoint absent: {CKPT_PATH}"
    assert BPE_PATH.is_file() and VIDEO_DIR.is_dir()

    determinism()
    if str(SAM3_PKG_PARENT) not in sys.path:
        sys.path.insert(0, str(SAM3_PKG_PARENT))
    _install_edt_stub()
    _patch_multiplex_sdpa()
    _patch_nms_cpu()
    _patch_connected_components_cpu()
    print("[capture] edt stub + SDPA(all) + NMS(cpu) + CC(cpu) applied")

    from sam3.model_builder import build_efficientsam3_multiplex_video_model

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print(f"[capture] Building native sam3.1 LiteText multiplex video (backbone={BACKBONE_TYPE}, "
          f"text={TEXT_ENCODER_TYPE}, ctx={TEXT_CTX})")
    demo = build_efficientsam3_multiplex_video_model(
        checkpoint_path=None, load_from_HF=False, bpe_path=str(BPE_PATH),
        backbone_type=BACKBONE_TYPE, model_name=MODEL_NAME,
        text_encoder_type=TEXT_ENCODER_TYPE, text_encoder_context_length=TEXT_CTX,
        text_encoder_pos_embed_table_size=TEXT_CTX, interpolate_pos_embed=False,
        multiplex_count=16, use_fa3=False, use_rope_real=False, device="cuda", compile=False,
    )
    model = demo._model
    determinism()

    print(f"[capture] Loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    n_total = len(ckpt)
    n_tracker = sum(1 for k in ckpt if k.startswith("tracker."))
    n_detector = sum(1 for k in ckpt if k.startswith("detector."))
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(f"[capture] strict-load: total={n_total} (detector={n_detector}, tracker={n_tracker})  "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"  MISSING (first 5): {missing[:5]}")
    if unexpected:
        print(f"  UNEXPECTED (first 5): {unexpected[:5]}")
    assert not missing, f"FAIL: {len(missing)} missing keys"
    assert not unexpected, f"FAIL: {len(unexpected)} unexpected keys"
    assert n_total == 1439, f"expected 1439 keys, got {n_total}"
    print("[capture] STRICT LOAD PASSED (0 missing / 0 unexpected, 1439/1439)")
    del ckpt
    model.eval()

    tmp_dir = Path(tempfile.mkdtemp(prefix="sam3p1_litetext_ref_"))
    frames_rgb = []
    try:
        for i in range(VIDEO_NUM_FRAMES):
            rgb = _load_rgb(VIDEO_DIR / f"{i}.jpg", VIDEO_HW)
            frames_rgb.append(rgb)
            Image.fromarray(rgb).save(tmp_dir / f"{i}.png")
        frames_rgb = np.stack(frames_rgb)
        print(f"[capture] frames shape: {frames_rgb.shape}")

        def run():
            state = model.init_state(resource_path=str(tmp_dir), async_loading_frames=False)
            f0, ap_out = model.add_prompt(state, frame_idx=0, text_str=VIDEO_PHRASE)
            masklets = {f0: ap_out}
            for f_idx, fout in model.propagate_in_video(
                state, start_frame_idx=1, max_frame_num_to_track=None, reverse=False
            ):
                masklets[f_idx] = fout
            return masklets

        masklets, precision_mode = _run_section(run)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

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
              f"scores={[round(float(p), 3) for p in probs]}")

    out["video_frames_rgb"] = frames_rgb
    out["video_phrase"] = np.array(VIDEO_PHRASE)
    out["video_hw"] = np.array(VIDEO_HW, np.int64)
    out["video_frame_indices"] = np.arange(VIDEO_NUM_FRAMES, dtype=np.int64)
    out["precision_mode"] = np.array(precision_mode)
    out["upstream_commit"] = np.array(UPSTREAM_COMMIT)

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ, **out)
    print(f"\n[capture] Wrote {OUT_NPZ} ({OUT_NPZ.stat().st_size/1e6:.2f} MB)")
    del model, demo
    torch.cuda.empty_cache()
    print("[capture] DONE")


if __name__ == "__main__":
    main()
