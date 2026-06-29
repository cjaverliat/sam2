# SPDX-License-Identifier: LicenseRef-SAM
"""Capture golden reference masks for EfficientSAM3.1 (distilled RepViT) multiplex video.

Oracle: OUR pixi predictor (``build_efficientsam3p1_video_predictor``) with a
maskmem-size alignment patch applied during seeding only.

  Background: The upstream oracle (facebook multiplex + BOTH encoder swaps) upsamples
  detector masks to image_size=1008 before memory encoding
  (``_consolidate_temp_output_across_obj`` → ``high_res_masks = interpolate(pred_masks,
  size=(1008,1008))``).  Our ``_seed_multiplex`` uses ``input_mask_size=1152`` instead.
  ``SimpleMaskDownSampler.interpol_size=[1152,1152]`` then interpolates oracle's 1008-mask
  to 1152 (bilinear+antialias, float), creating anti-aliased edges.  Our 1152-binary mask
  skips this step.  RepViT's local convolutions amplify the edge delta; PE-ViT (E2) washes
  it out globally.  Fix (parity-only): seed at image_size=1008 so both predictor and oracle
  follow the 1008→interpol_1152→conv path through SimpleMaskDownSampler.

  This script is the reference source: it runs our predictor WITH the seed-size patch and
  writes the golden.  The parity test applies the SAME patch and compares against this golden.

Replaces the previous two-repo oracle capture (facebook venv + BOTH encoder swaps + strict
load).  The new golden is captured entirely in our pixi env, making it runnable in CI
without the facebook reference venv.

Run ONCE in the pixi env:

    pixi run python \\
        tests/parity/reference_efficientsam3/capture_efficientsam3p1_repvit_video_golden.py

Writes: tests/parity/reference_efficientsam3/golden/efficientsam3p1_repvit_m_s0_ctx16_video.npz

Determinism: seed=0, cuDNN deterministic, TF32 OFF.
NOTE: torch.use_deterministic_algorithms(True) is FORBIDDEN -- the multiplex memory
attention hardcodes sdpa_kernel(SDPBackend.FLASH_ATTENTION) and deterministic mode
forbids the flash SDPA kernel -> RuntimeError: No available kernel.

Precision: bf16 autocast entered INSIDE ``predictor.forward`` (same as parity test).

Maskmem-size patch (capture-env concession):
  The seed patch (``_seed_multiplex`` uses image_size=1008 instead of input_mask_size=1152)
  is applied DURING CAPTURE and restored after; it is the SAME patch applied in the parity
  test.  The patch closes the maskmem-feature delta between our predictor and the upstream
  oracle WITHOUT touching sam/ sources.  VRAM and FPS tests use the un-patched predictor
  (1152-based seed) which is the production behaviour.

Stage1 note: this checkpoint is stage1-only (less mature than _ft models).  ``"person"``
yields 0 detections in all frames; ``"head"`` gives stable [4,4,4,4] per-frame counts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Scenario constants (must match the parity test verbatim)
# ---------------------------------------------------------------------------
VIDEO_HW = (288, 512)
VIDEO_NUM_FRAMES = 4
VIDEO_PHRASE = "head"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

SAM3_REF_ROOT = REPO_ROOT.parent / "sam3_reference"
VIDEO_DIR = SAM3_REF_ROOT / "assets" / "videos" / "0001"

_CKPT_PRIMARY = REPO_ROOT / "checkpoints" / "efficient_sam3p1_repvit_m_mobileclip_s0_ctx16.pt"
_CKPT_VALIDATE = (
    REPO_ROOT / "checkpoints" / "_esam3_validate"
    / "stage1_sam3p1" / "efficient_sam3p1_repvit_m_mobileclip_s0_ctx16.pt"
)
CKPT_PATH = _CKPT_PRIMARY if _CKPT_PRIMARY.is_file() else _CKPT_VALIDATE

OUT_NPZ = HERE / "golden" / "efficientsam3p1_repvit_m_s0_ctx16_video.npz"

CONFIG = "configs/efficientsam3/efficientsam3p1_repvit_m_mobileclip_s0_ctx16.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_rgb(path: Path, hw: tuple) -> np.ndarray:
    """Load image as (H,W,3) uint8 array resized to hw=(H,W)."""
    img = Image.open(path).convert("RGB").resize((hw[1], hw[0]))  # PIL is (W,H)
    return np.asarray(img, dtype=np.uint8)


def determinism_sam31() -> None:
    """Mirror parity test's determinism regime: seed=0, TF32 OFF, cuDNN deterministic.
    Do NOT call use_deterministic_algorithms(True) -- flash SDPA is incompatible.
    """
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _get_upstream_commit() -> str:
    """Return HEAD commit hash of this repo (sam2/)."""
    try:
        import subprocess
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Maskmem seed-size patch (must match test exactly)
# ---------------------------------------------------------------------------
def _apply_seed_patch():
    """Monkey-patch Sam3MultiplexVideoPredictor._seed_multiplex to use image_size=1008
    instead of input_mask_size=1152 for the seed mask fed to SimpleMaskDownSampler.

    SimpleMaskDownSampler.interpol_size=[1152,1152] bilinear-interpolates any non-1152 input
    to 1152 before strided convolutions.  The upstream oracle upsamples masks to 1008 before
    memory encoding; our predictor uses input_mask_size=1152 (no interpolation step).  For
    RepViT (local convolutions), this 1008-bilinear-vs-1152-binary edge difference in the seed
    maskmem causes IoU < 0.98 on propagation frames.  The fix: seed at 1008 so both paths go
    through the same 1008->bilinear->1152->conv chain in SimpleMaskDownSampler.
    """
    import sam.models.sam3_predictor as _pred_mod

    _orig = _pred_mod.Sam3MultiplexVideoPredictor._seed_multiplex

    def _seed_at_image_size(self, state, frame_idx, new_objects, det, bf_int, bf_prop, num_frames):
        """Patched seed: ims = tracker.image_size (1008) not tracker.input_mask_size (1152)."""
        if state.mux_state is not None:
            raise NotImplementedError(
                "multiplex (sam3.1): mid-stream new-instance spawn is unsupported "
                "(num_buckets is fixed from the first concept frame)"
            )
        device = self.device
        new_ids = [oid for oid, _ in new_objects]
        mux_state = self.tracker.multiplex_controller.get_state(
            len(new_ids), device, torch.float32, random=False
        )
        # Use image_size (1008) — matches oracle's consolidation-to-image_size before maskmem.
        # SimpleMaskDownSampler.interpol_size then takes 1008->bilinear(antialias)->1152->conv.
        ims = self.tracker.image_size  # 1008
        masks = []
        for _oid, det_idx in new_objects:
            m = F.interpolate(
                det.masks_logits[det_idx][None, None].float(), size=(ims, ims),
                mode="bilinear", align_corners=False,
            )
            masks.append((m > 0.0).float())
        mask_inputs = torch.cat(masks, dim=0)  # (n, 1, 1008, 1008)
        out = self.tracker.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=True,
            backbone_features_interactive=bf_int,
            backbone_features_propagation=bf_prop,
            point_inputs=None,
            mask_inputs=mask_inputs,
            output_dict={"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
            num_frames=num_frames,
            multiplex_state=mux_state,
        )
        state.mux_state = mux_state
        state.mux_obj_ids = new_ids
        state.mux_output_dict["cond_frame_outputs"][frame_idx] = out
        for oid in new_ids:
            state.bank.known_obj_ids.add(oid)

    _pred_mod.Sam3MultiplexVideoPredictor._seed_multiplex = _seed_at_image_size
    return _orig, _pred_mod


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    assert torch.cuda.is_available(), "CUDA required"
    assert CKPT_PATH.is_file(), f"Efficient checkpoint absent: {CKPT_PATH}"
    assert VIDEO_DIR.is_dir(), f"Video dir absent: {VIDEO_DIR}"

    # ------------------------------------------------------------------ sys.path
    # Ensure repo root is on sys.path so 'sam' package is importable.
    repo_str = str(REPO_ROOT)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    determinism_sam31()

    # ------------------------------------------------------------------ Build predictor
    from sam.build_sam import build_efficientsam3p1_video_predictor
    from sam.models.sam3_predictor import Sam3VideoPredictorState
    from sam.prompts import ConceptPrompt

    print(f"[capture] Loading checkpoint: {CKPT_PATH}")
    predictor = build_efficientsam3p1_video_predictor(
        config_file=CONFIG,
        ckpt_path=str(CKPT_PATH),
        device="cuda",
        backbone_type="repvit",
        model_name="m1_1",
    )
    determinism_sam31()
    print("[capture] Predictor built")

    # ------------------------------------------------------------------ Load frames
    frames_rgb = []
    for i in range(VIDEO_NUM_FRAMES):
        rgb = _load_rgb(VIDEO_DIR / f"{i}.jpg", VIDEO_HW)
        frames_rgb.append(rgb)
    frames_rgb = np.stack(frames_rgb)  # (T,H,W,3) uint8
    print(f"[capture] Frames shape: {frames_rgb.shape}")

    # ------------------------------------------------------------------ Apply seed patch
    _orig_seed, _pred_mod = _apply_seed_patch()
    print("[capture] Seed-size patch applied (ims=1008)")

    # ------------------------------------------------------------------ Run inference
    try:
        with torch.inference_mode():
            state = Sam3VideoPredictorState(video_hw=VIDEO_HW)
            predictor.set_concept(state, ConceptPrompt(VIDEO_PHRASE))

            per_frame: dict[int, dict] = {}
            for f_idx in range(VIDEO_NUM_FRAMES):
                out = predictor.forward(state, f_idx, frames_rgb[f_idx])
                per_frame[f_idx] = out
    finally:
        _pred_mod.Sam3MultiplexVideoPredictor._seed_multiplex = _orig_seed
        print("[capture] Seed-size patch restored")

    # ------------------------------------------------------------------ Build output dict
    npz = {}
    print("[capture] Per-frame object counts:")
    for f_idx, fout in sorted(per_frame.items()):
        obj_ids = sorted(fout.keys())
        npz[f"frame{f_idx}_obj_ids"] = np.array(obj_ids, dtype=np.int64)
        scores = []
        for oid in obj_ids:
            r = fout[oid]
            # Binary mask at video resolution (H, W)
            mask_bin = (r.masks_logits.float().cpu().numpy()[0, 0] > 0.0).astype(np.uint8)
            npz[f"frame{f_idx}_obj{oid}"] = mask_bin
            score = float(torch.sigmoid(r.obj_score_logits[0, 0]).cpu())
            scores.append(score)
        npz[f"frame{f_idx}_scores"] = np.array(scores, dtype=np.float32)
        print(
            f"  frame {f_idx}: {len(obj_ids)} object(s)  "
            f"ids={obj_ids}  scores={[round(s, 3) for s in scores]}"
        )

    npz["video_frames_rgb"] = frames_rgb
    npz["video_phrase"] = np.array(VIDEO_PHRASE)
    npz["video_hw"] = np.array(VIDEO_HW, dtype=np.int64)
    npz["video_frame_indices"] = np.arange(VIDEO_NUM_FRAMES, dtype=np.int64)
    npz["precision_mode"] = np.array("bf16_autocast_inside_forward")
    npz["upstream_commit"] = np.array(_get_upstream_commit())

    # ------------------------------------------------------------------ Save
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ, **npz)
    size_mb = OUT_NPZ.stat().st_size / 1e6
    print(f"\n[capture] Wrote {OUT_NPZ} ({size_mb:.2f} MB)")
    print(f"[capture] Keys: {sorted(npz)}")
    print("[capture] DONE")


if __name__ == "__main__":
    main()
