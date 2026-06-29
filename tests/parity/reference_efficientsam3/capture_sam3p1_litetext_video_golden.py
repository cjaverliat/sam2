# SPDX-License-Identifier: LicenseRef-SAM
"""Capture golden reference masks for SAM3.1-LiteText multiplex video (s0/ctx16).

Oracle construction (TWO-repo assembly):
  No single upstream repo can run the efficient_sam3p1_litetext checkpoint:
  - The efficientsam3 repo has NO 457-key multiplex tracker (only the base 309-key tracker).
  - The facebook sam3 reference HAS the multiplex tracker but uses the PE text tower (295 keys).
  Solution: build facebook's multiplex video model (``sam3_reference``) then swap its
  ``language_backbone`` for our de-timm'd MobileClipTextEncoder (111 keys from our ``sam``
  namespace), then load the 1439-key efficient checkpoint STRICT (1328 facebook-arch keys +
  111 MobileCLIP keys = 0 missing / 0 unexpected).

Run ONCE in the FACEBOOK reference venv (NOT the efficientsam3 or pixi env):

    C:\\Users\\javerlia\\PycharmProjects\\sam3_reference\\.venv\\Scripts\\python.exe \\
        tests/parity/reference_efficientsam3/capture_sam3p1_litetext_video_golden.py

Writes: tests/parity/reference_efficientsam3/golden/sam3p1_litetext_s0_ctx16_video.npz

Facebook upstream commit: 5dd401d1c5c1d5c3eedff06d41b77af824517619

Determinism: seed=0, cuDNN deterministic, TF32 OFF.
NOTE: torch.use_deterministic_algorithms(True) is FORBIDDEN here -- the multiplex memory
attention hardcodes ``sdpa_kernel(SDPBackend.FLASH_ATTENTION)`` and deterministic mode
forbids the (nondeterministic) flash SDPA kernel -> RuntimeError: No available kernel.
_patch_multiplex_sdpa() allows all backends so SDPA auto-selects math (exact reference).

Precision: bf16 autocast (required -- SAM3 perflib fused addmm_act hardcodes .to(bfloat16)).

Capture-env concessions (none for this venv):
  sam3_reference venv has triton installed -> NMS and connected_components work natively,
  so no CPU fallback patches are needed (unlike the efficientsam3_reference venv).

CTX: NOT needed. Unlike the efficientsam3 video builder (ctx monkeypatch required for
ctx=77->16), the facebook ``build_sam3_predictor`` does not create a MobileCLIP text encoder
at all -- it creates a VETextEncoder (PE tower). We swap AFTER build, constructing
MobileClipTextEncoder(context_length=16) directly, so there is no ctx shape mismatch.
"""
import importlib.util
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
UPSTREAM_COMMIT = "5dd401d1c5c1d5c3eedff06d41b77af824517619"  # facebookresearch/sam3

# ---------------------------------------------------------------------------
# Scenario constants (must match the parity test verbatim)
# ---------------------------------------------------------------------------
VIDEO_HW = (288, 512)       # (H, W)
VIDEO_NUM_FRAMES = 4        # frames 0..3
VIDEO_PHRASE = "person"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]   # .../sam2/

SAM3_REF_ROOT = REPO_ROOT.parent / "sam3_reference"
SAM2_SAM = REPO_ROOT / "sam"

BPE_PATH = SAM3_REF_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
VIDEO_DIR = SAM3_REF_ROOT / "assets" / "videos" / "0001"
FB_CKPT_PATH = REPO_ROOT / "checkpoints" / "sam3.1_multiplex.pt"

CKPT_PATH = (
    REPO_ROOT / "checkpoints" / "_esam3_validate"
    / "sam3p1_litetext" / "efficient_sam3p1_litetext_mobileclip_s0_ctx16.pt"
)
OUT_NPZ = HERE / "golden" / "sam3p1_litetext_s0_ctx16_video.npz"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_rgb(path, hw):
    """Load image as (H,W,3) uint8 array resized to hw=(H,W)."""
    img = Image.open(path).convert("RGB").resize((hw[1], hw[0]))  # PIL is (W,H)
    return np.asarray(img, dtype=np.uint8)


def _load_mobileclip_text_encoder():
    """Load MobileClipTextEncoder + Sam3Tokenizer from our ``sam`` namespace.

    We can't ``import sam`` directly in the facebook venv because ``sam/__init__.py``
    calls ``hydra.initialize_config_module`` (hydra is not in this venv). We stub
    the ``sam`` package in sys.modules to bypass the __init__.py, then load the three
    leaf modules (pe_vitdet.py, tokenizer.py, mobile_clip.py, mobileclip_text_encoder.py)
    via importlib.util.spec_from_file_location. This is the least-invasive path:
    no file copying, no patching, weights are identical.
    """
    sam2 = str(SAM2_SAM)

    def _stub(name, path):
        m = types.ModuleType(name)
        m.__path__ = [path]
        m.__package__ = name
        sys.modules[name] = m
        return m

    def _load(name, filepath):
        spec = importlib.util.spec_from_file_location(name, filepath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    _sam = _stub("sam", sam2)
    _mod = _stub("sam.modeling", sam2 + "\\modeling")
    _sam.modeling = _mod
    _enc = _stub("sam.modeling.encoders", sam2 + "\\modeling\\encoders")
    _mod.encoders = _enc
    _txt = _stub("sam.modeling.text", sam2 + "\\modeling\\text")
    _mod.text = _txt

    pe = _load(
        "sam.modeling.encoders.pe_vitdet",
        sam2 + "\\modeling\\encoders\\pe_vitdet.py",
    )
    _enc.pe_vitdet = pe

    tok = _load(
        "sam.modeling.text.tokenizer",
        sam2 + "\\modeling\\text\\tokenizer.py",
    )
    _txt.tokenizer = tok

    mc = _load(
        "sam.modeling.text.mobile_clip",
        sam2 + "\\modeling\\text\\mobile_clip.py",
    )
    _txt.mobile_clip = mc

    mte_mod = _load(
        "sam.modeling.text.mobileclip_text_encoder",
        sam2 + "\\modeling\\text\\mobileclip_text_encoder.py",
    )
    _txt.mobileclip_text_encoder = mte_mod

    return mte_mod.MobileClipTextEncoder, tok.Sam3Tokenizer


def determinism_sam31():
    """Determinism for the SAM 3.1 multiplex path.

    Same as the base ``determinism()`` EXCEPT we must NOT call
    ``torch.use_deterministic_algorithms(True)``: the multiplex memory attention
    hardcodes ``sdpa_kernel(SDPBackend.FLASH_ATTENTION)``, and deterministic mode
    forbids the (nondeterministic) flash SDPA kernel -> ``RuntimeError: No available kernel``.
    We keep every other lever (seed 0, cuDNN deterministic, TF32 off); reproducibility
    is verified empirically.
    """
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _patch_multiplex_sdpa():
    """Allow math/efficient SDPA fallback for the multiplex memory attention.

    ``decoder.functional_attention`` runs ``with sdpa_kernel(SDPBackend.FLASH_ATTENTION):``
    when ``use_fa3=False``. On this GPU the flash SDPA backend may be unavailable for these
    inputs -> ``No available kernel``. We override the module-level ``sdpa_kernel`` so the
    forced-flash context instead permits every backend; SDPA then auto-selects (math is the
    exact reference). This is a capture-env kernel-dispatch concession (no weights/logic
    change), analogous to the bf16-autocast precision concession.
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


def _run_section(run):
    """Run fn() under bf16 autocast + inference_mode (the only viable SAM3 regime)."""
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return run(), "bf16_autocast"


def main():
    assert torch.cuda.is_available(), "CUDA required"
    assert CKPT_PATH.is_file(), f"Efficient checkpoint absent: {CKPT_PATH}"
    assert FB_CKPT_PATH.is_file(), f"Facebook checkpoint absent: {FB_CKPT_PATH}"
    assert BPE_PATH.is_file(), f"BPE absent: {BPE_PATH}"
    assert VIDEO_DIR.is_dir(), f"Video dir absent: {VIDEO_DIR}"

    # ------------------------------------------------------------------ load text modules
    # Must happen BEFORE sam3 imports (sam3's model_builder.py runs _setup_tf32() at import
    # time; the text encoder stub must be ready when we swap post-build).
    MobileClipTextEncoder, Sam3Tokenizer = _load_mobileclip_text_encoder()
    print("[capture] MobileClipTextEncoder + Sam3Tokenizer loaded (importlib stub)")

    # ------------------------------------------------------------------ determinism
    # Pre-disable TF32 BEFORE sam3.model_builder import, which calls _setup_tf32() at
    # module level and re-enables TF32.  Re-disable again after import.
    determinism_sam31()

    # ------------------------------------------------------------------ add sam3_reference to sys.path
    sam3_ref_str = str(SAM3_REF_ROOT)
    if sam3_ref_str not in sys.path:
        sys.path.insert(0, sam3_ref_str)

    # ------------------------------------------------------------------ SDPA patch
    # Must be applied BEFORE importing the model (module-level import of sam3.model.decoder
    # fires during build_sam3_predictor).
    _patch_multiplex_sdpa()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # ------------------------------------------------------------------ Step 1: Build facebook oracle
    from sam3.model_builder import build_sam3_predictor  # noqa: E402

    print(f"[capture] Loading facebook checkpoint: {FB_CKPT_PATH}")
    predictor = build_sam3_predictor(
        version="sam3.1",
        checkpoint_path=str(FB_CKPT_PATH),
        bpe_path=str(BPE_PATH),
        multiplex_count=16,
        use_fa3=False,
        use_rope_real=False,
        compile=False,
        warm_up=False,
        async_loading_frames=False,
    )
    # predictor.__init__ enters a global bf16 autocast and re-enables TF32 -> re-assert
    determinism_sam31()
    model = predictor.model  # Sam3MultiplexTrackingWithInteractivity

    # ------------------------------------------------------------------ Step 2: Swap text encoder
    device = next(model.detector.backbone.language_backbone.parameters()).device
    model.detector.backbone.language_backbone = MobileClipTextEncoder(
        tokenizer=Sam3Tokenizer(),
        variant="MobileCLIP-S0",
        context_length=16,
        output_dim=256,
    ).to(device).eval()
    print(
        "[capture] Text encoder swapped: "
        f"{type(model.detector.backbone.language_backbone).__name__}"
    )

    # ------------------------------------------------------------------ Step 3: Load efficient ckpt STRICT
    print(f"[capture] Loading efficient checkpoint: {CKPT_PATH}")
    sd = torch.load(str(CKPT_PATH), weights_only=True, map_location="cpu")["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[capture] missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"  MISSING (first 5): {missing[:5]}")
    if unexpected:
        print(f"  UNEXPECTED (first 5): {unexpected[:5]}")
    assert not missing, f"FAIL: {len(missing)} missing keys"
    assert not unexpected, f"FAIL: {len(unexpected)} unexpected keys"
    print("[capture] STRICT LOAD PASSED (0 missing / 0 unexpected, 1439/1439 keys)")
    del sd  # free memory before the forward pass

    model.eval()

    # ------------------------------------------------------------------ load frames
    tmp_dir = Path(tempfile.mkdtemp(prefix="sam3p1_litetext_ref_frames_"))
    frames_rgb = []
    try:
        for i in range(VIDEO_NUM_FRAMES):
            rgb = _load_rgb(VIDEO_DIR / f"{i}.jpg", VIDEO_HW)
            frames_rgb.append(rgb)
            # Save as lossless PNG so init_state byte-loads the SAME pixels we captured.
            Image.fromarray(rgb).save(tmp_dir / f"{i}.png")
        frames_rgb = np.stack(frames_rgb)  # (T,H,W,3) uint8
        print(f"[capture] frames shape: {frames_rgb.shape}")

        # ------------------------------------------------------------ run inference
        # Multiplex API: add_prompt(@f0) -> propagate_in_video(start_frame_idx=1).
        # Frame 0 masklet comes from ap_out; frames 1..3 from propagate_in_video.
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
    # Schema: streaming-only (per-frame masklets, no multiplex tracker internals).
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

    del model
    torch.cuda.empty_cache()
    print("[capture] DONE")


if __name__ == "__main__":
    main()
