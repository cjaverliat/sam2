# SPDX-License-Identifier: LicenseRef-SAM
"""Capture golden reference masks for EfficientSAM3.1 (distilled RepViT-M / s0 / ctx16).

NATIVE reference (apples-to-apples, NOT facebook):
  efficientsam3 ships its OWN sam3.1 multiplex code on the ``stage1_sam3.1`` branch.
  A worktree is checked out at ``C:/Users/javerlia/PycharmProjects/efficientsam3_sam3p1``
  (commit ``6056958``). Its ``build_efficientsam3_multiplex_video_model(...)`` builds the
  multiplex video model with the DISTILLED RepViT trunk + MobileCLIP-S0 text encoder
  NATIVELY and the checkpoint ``efficient_sam3p1_repvit_m_mobileclip_s0_ctx16.pt`` (1672
  keys: detector 1215 + tracker 457) loads STRICT with NO encoder swapping / namespace
  tricks. This is the efficientsam3.1 weights run through efficientsam3's OWN sam3.1
  runtime -- the correct independent oracle for our production predictor.

Run ONCE in the efficientsam3 reference venv (it has torch cu128 + timm + einops + scipy
+ skimage + opencv; it does NOT have triton -- see env concessions below):

    C:\\Users\\javerlia\\PycharmProjects\\efficientsam3_reference\\.venv\\Scripts\\python.exe \\
        tests/parity/reference_efficientsam3/capture_efficientsam3p1_repvit_video_golden.py

Writes: tests/parity/reference_efficientsam3/golden/efficientsam3p1_repvit_m_s0_ctx16_video.npz

Upstream (efficientsam3 stage1_sam3.1) commit: 6056958418438beccd4f0782f9b73a1fbcca3e5a

----------------------------------------------------------------------------------------
ENV CONCESSIONS (kernel/dep dispatch only -- NO weights/logic change)
----------------------------------------------------------------------------------------
This venv has NO ``triton`` package. efficientsam3's multiplex stack imports triton in
four places; each is handled WITHOUT globally stubbing triton (a global fake breaks
torchvision.ops -> torch._dynamo, which inspects triton.language):

  1. ``sam3.model.edt`` does a top-level ``import triton`` with NO fallback, and is pulled
     in by ``sam3_tracker_utils`` (``from sam3.model.edt import edt_triton``). We insert a
     pure-scipy ``edt_triton`` CPU module into ``sys.modules["sam3.model.edt"]`` BEFORE any
     sam3 import. ``edt_triton`` mimics ``cv2.distanceTransform(x, DIST_L2, 0)`` (distance
     to nearest zero pixel); ``scipy.ndimage.distance_transform_edt`` is the exact CPU
     equivalent. NOTE: ``edt_triton`` is only used by ``sample_one_point_from_error_center``
     (RITM point-refinement) which is NOT on the text-concept tracking path exercised here,
     so the fallback is correctness insurance -- it must merely import.
  2. ``sam3.perflib.nms.generic_nms`` CUDA path falls back to a triton kernel when
     ``torch_generic_nms`` is absent. We point it at the bundled ``generic_nms_cpu`` (same
     greedy NMS, identical strict-IoU suppression). (Same concession the base SAM3-LiteText
     D-golden used in this same venv.)
  3. ``sam3.perflib.connected_components.connected_components`` CUDA path falls back to a
     triton kernel when ``cc_torch`` is absent. We point it at the bundled skimage CPU
     fallback (wrapped to handle (B,H,W) and B=0). (Same D-golden concession.)
  4. ``backbones/efficientvit/nn/triton_rms_norm`` and ``train/loss/sigmoid_focal_loss``
     import triton too, but neither is on the repvit inference path, so neither is touched.

Multiplex SDPA: ``decoder.functional_attention`` forces
``sdpa_kernel(SDPBackend.FLASH_ATTENTION)`` when ``use_fa3=False``; flash SDPA may be
unavailable for these shapes on this GPU. ``_patch_multiplex_sdpa`` overrides the decoder's
module-level ``sdpa_kernel`` so the forced-flash context permits all backends and SDPA
auto-selects (math == exact reference). (Same concession E2 used.)

Determinism: seed=0, cuDNN deterministic, TF32 OFF (re-asserted after the model_builder
import, which calls ``_setup_tf32()``). ``torch.use_deterministic_algorithms(True)`` is
FORBIDDEN -- the multiplex memory attention's forced-flash SDPA is non-deterministic and
deterministic mode raises ``No available kernel``.
Precision: bf16 autocast + inference_mode (the only viable SAM3 regime).
Builder flags: ``use_fa3=False``, ``use_rope_real=False`` (CPU/SDPA-friendly, no FA3 dep).

Concept: "head". ("person" yields 0 detections for this stage1 distilled checkpoint;
"head" produces stable objects -- recorded in ``video_phrase``.)
"""
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Oracle identity
# ---------------------------------------------------------------------------
UPSTREAM_COMMIT = "6056958418438beccd4f0782f9b73a1fbcca3e5a"  # efficientsam3 stage1_sam3.1

# ---------------------------------------------------------------------------
# Scenario constants (must match the parity test verbatim)
# ---------------------------------------------------------------------------
VIDEO_HW = (288, 512)       # (H, W)
VIDEO_NUM_FRAMES = 4        # frames 0..3
VIDEO_PHRASE = "head"

# Builder args inferred from the checkpoint name
# ``efficient_sam3p1_repvit_m_mobileclip_s0_ctx16.pt`` -- mirrors
# ``sam3p1_demo_utils.infer_model_args_from_checkpoint`` (inlined here because that module
# imports matplotlib, which is absent from this venv):
#   backbone "repvit" + size "m" -> model_name "m1.1"
#   text "s0" -> "MobileCLIP-S0";  ctx16 -> context_length / pos_embed_table_size 16
BACKBONE_TYPE = "repvit"
MODEL_NAME = "m1.1"
TEXT_ENCODER_TYPE = "MobileCLIP-S0"
TEXT_CTX = 16

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]   # .../sam2/

WORKTREE = Path(r"C:\Users\javerlia\PycharmProjects\efficientsam3_sam3p1")
SAM3_PKG_PARENT = WORKTREE / "sam3"             # parent of the `sam3` package
BPE_PATH = SAM3_PKG_PARENT / "assets" / "bpe_simple_vocab_16e6.txt.gz"
VIDEO_DIR = SAM3_PKG_PARENT / "assets" / "videos" / "0001"

CKPT_PATH = (
    REPO_ROOT / "checkpoints" / "_esam3_validate"
    / "stage1_sam3p1" / "efficient_sam3p1_repvit_m_mobileclip_s0_ctx16.pt"
)
OUT_NPZ = HERE / "golden" / "efficientsam3p1_repvit_m_s0_ctx16_video.npz"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_rgb(path, hw):
    """Load image as (H,W,3) uint8 array resized to hw=(H,W)."""
    img = Image.open(path).convert("RGB").resize((hw[1], hw[0]))  # PIL is (W,H)
    return np.asarray(img, dtype=np.uint8)


def _install_edt_stub():
    """Insert a pure-scipy ``sam3.model.edt`` into sys.modules BEFORE any sam3 import.

    The real ``edt.py`` does a top-level ``import triton`` (absent here). ``edt_triton``
    computes the Euclidean distance transform of a batch of binary images -- the distance
    from each foreground pixel to the nearest zero pixel, i.e. a batched
    ``cv2.distanceTransform(x, cv2.DIST_L2, 0)``. ``scipy.ndimage.distance_transform_edt``
    is the exact CPU equivalent. Contract (verbatim from edt.py): in (B,H,W) -> out (B,H,W)
    float on the same device.
    """
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
    stub.__doc__ = "scipy-EDT CPU stub (triton absent); contract matches sam3.model.edt"
    sys.modules["sam3.model.edt"] = stub


def _patch_multiplex_sdpa():
    """Allow all SDPA backends for the multiplex memory attention (E2 concession).

    ``decoder.functional_attention`` runs ``with sdpa_kernel(SDPBackend.FLASH_ATTENTION):``
    when ``use_fa3=False``; flash SDPA may be unavailable for these shapes on this GPU
    (``No available kernel``). Override the decoder's module-level ``sdpa_kernel`` so the
    forced-flash context permits every backend; SDPA then auto-selects (math == exact
    reference). Kernel-dispatch concession only -- no weights/logic change.
    """
    from torch.nn.attention import SDPBackend
    from torch.nn.attention import sdpa_kernel as _sk
    from sam3.model import decoder as _dec  # noqa: E402

    _all = [
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.MATH,
    ]
    _dec.sdpa_kernel = lambda *a, **k: _sk(_all)


def _patch_nms_cpu():
    """Point ``perflib.nms.generic_nms`` at the bundled CPU greedy NMS (triton absent)."""
    import sam3.perflib.nms as _nms_mod  # noqa: E402

    _nms_mod.generic_nms = _nms_mod.generic_nms_cpu


def _patch_connected_components_cpu():
    """Point ``perflib.connected_components.connected_components`` at the skimage CPU path."""
    import sam3.perflib.connected_components as _cc_mod  # noqa: E402
    from sam3.perflib.connected_components import connected_components_cpu as _cc_cpu

    def _cc_cpu_fallback(input_tensor):
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


def determinism():
    """Seed 0, cuDNN deterministic, TF32 OFF. (No use_deterministic_algorithms -- flash SDPA.)"""
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _run_section(run):
    """Run fn() under bf16 autocast + inference_mode (the only viable SAM3 regime)."""
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return run(), "bf16_autocast"


def main():
    assert torch.cuda.is_available(), "CUDA required"
    assert CKPT_PATH.is_file(), f"Efficient checkpoint absent: {CKPT_PATH}"
    assert BPE_PATH.is_file(), f"BPE absent: {BPE_PATH}"
    assert VIDEO_DIR.is_dir(), f"Video dir absent: {VIDEO_DIR}"

    # ------------------------------------------------------------------ env setup
    determinism()
    if str(SAM3_PKG_PARENT) not in sys.path:
        sys.path.insert(0, str(SAM3_PKG_PARENT))

    # edt stub MUST precede every sam3 import (sam3_tracker_utils imports edt_triton).
    _install_edt_stub()
    print("[capture] edt stub installed (scipy CPU fallback)")

    # SDPA / NMS / CC concessions (import-time-safe; patched before build).
    _patch_multiplex_sdpa()
    _patch_nms_cpu()
    _patch_connected_components_cpu()
    print("[capture] SDPA(all-backends) + NMS(cpu) + CC(cpu) patches applied")

    # ------------------------------------------------------------------ build NATIVE model
    from sam3.model_builder import build_efficientsam3_multiplex_video_model  # noqa: E402

    # model_builder._setup_tf32() ran at import -> re-disable TF32.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print(
        f"[capture] Building native efficientsam3 sam3.1 multiplex video model "
        f"(backbone={BACKBONE_TYPE}/{MODEL_NAME}, text={TEXT_ENCODER_TYPE}, ctx={TEXT_CTX})"
    )
    # Build UNLOADED (checkpoint_path=None) so we can strict-load ourselves and report the
    # exact missing/unexpected lists. (build_efficientsam3_multiplex_video_model with
    # strict_state_dict_loading=True would raise on any mismatch -- equivalent gate.)
    demo = build_efficientsam3_multiplex_video_model(
        checkpoint_path=None,
        load_from_HF=False,
        bpe_path=str(BPE_PATH),
        backbone_type=BACKBONE_TYPE,
        model_name=MODEL_NAME,
        text_encoder_type=TEXT_ENCODER_TYPE,
        text_encoder_context_length=TEXT_CTX,
        text_encoder_pos_embed_table_size=TEXT_CTX,
        interpolate_pos_embed=False,
        multiplex_count=16,
        use_fa3=False,
        use_rope_real=False,
        device="cuda",
        compile=False,
    )
    model = demo._model  # Sam3MultiplexTrackingWithInteractivity (real nn.Module)
    determinism()

    # ------------------------------------------------------------------ strict load (GATE)
    print(f"[capture] Loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    # The native ckpt already uses detector.* / tracker.* namespace (no remap needed).
    needs_remap = any(
        k.startswith("sam3_model.") or k.startswith("sam2_predictor.") for k in ckpt
    )
    assert not needs_remap, "unexpected legacy namespace in checkpoint"
    n_total = len(ckpt)
    n_tracker = sum(1 for k in ckpt if k.startswith("tracker."))
    n_detector = sum(1 for k in ckpt if k.startswith("detector."))

    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(
        f"[capture] strict-load: total={n_total} (detector={n_detector}, tracker={n_tracker})  "
        f"missing={len(missing)}  unexpected={len(unexpected)}"
    )
    if missing:
        print(f"  MISSING (first 5): {missing[:5]}")
    if unexpected:
        print(f"  UNEXPECTED (first 5): {unexpected[:5]}")
    assert not missing, f"FAIL: {len(missing)} missing keys"
    assert not unexpected, f"FAIL: {len(unexpected)} unexpected keys"
    assert n_total == 1672, f"expected 1672 keys, got {n_total}"
    print("[capture] STRICT LOAD PASSED (0 missing / 0 unexpected, 1672/1672 keys)")
    del ckpt

    model.eval()

    # ------------------------------------------------------------------ load frames
    tmp_dir = Path(tempfile.mkdtemp(prefix="esam3p1_repvit_ref_frames_"))
    frames_rgb = []
    try:
        for i in range(VIDEO_NUM_FRAMES):
            rgb = _load_rgb(VIDEO_DIR / f"{i}.jpg", VIDEO_HW)
            frames_rgb.append(rgb)
            # Lossless PNG so init_state byte-loads the SAME pixels we captured.
            Image.fromarray(rgb).save(tmp_dir / f"{i}.png")
        frames_rgb = np.stack(frames_rgb)  # (T,H,W,3) uint8
        print(f"[capture] frames shape: {frames_rgb.shape}")

        # ------------------------------------------------------------ run inference
        # add_prompt(@f0) -> propagate_in_video(start_frame_idx=1). Frame 0 masklet from
        # ap_out; frames 1..3 from propagate_in_video. Per-frame dict has out_obj_ids /
        # out_binary_masks / out_probs.
        def run():
            state = model.init_state(
                resource_path=str(tmp_dir), async_loading_frames=False
            )
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

    # ------------------------------------------------------------------ build output dict
    # Streaming schema (same as the other video goldens).
    out = {}
    print("[capture] Per-frame object counts:")
    for f_idx, fout in sorted(masklets.items()):
        obj_ids = np.asarray(fout["out_obj_ids"], np.int64)
        masks = np.asarray(fout["out_binary_masks"])   # (N,H,W) bool
        probs = np.asarray(fout["out_probs"], np.float32)
        out[f"frame{f_idx}_obj_ids"] = obj_ids
        out[f"frame{f_idx}_scores"] = probs
        for j, oid in enumerate(obj_ids.tolist()):
            out[f"frame{f_idx}_obj{oid}"] = masks[j].astype(np.uint8)
        print(
            f"  frame {f_idx}: {len(obj_ids)} object(s)  "
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

    del model, demo
    torch.cuda.empty_cache()
    print("[capture] DONE")


if __name__ == "__main__":
    main()
