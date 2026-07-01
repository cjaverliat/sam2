# SPDX-License-Identifier: LicenseRef-SAM
"""SAM3-LiteText (s0/ctx16) streaming VIDEO parity vs upstream golden.

Phase D acceptance gate: ``build_sam3_video_predictor`` (our side) must reproduce
the upstream SAM3-LiteText ``propagate_in_video`` per-frame masks within IoU tolerance.

Golden: ``golden/sam3_litetext_s0_ctx16_video.npz`` captured from upstream commit
``d063e00b1837f8dd285eb517d2dd40faabc34555`` (efficientsam3 main branch) via
``capture_litetext_video_golden.py`` -- 4 frames of the upstream dance clip resized
to 288x512, phrase "person", under bf16 autocast.

Acceptance gate (verbatim from spec §D):
  * Per-frame object count == golden (expected: 4 "person" objects per frame).
  * Per-frame Hungarian IoU: min >= 0.98, mean >= 0.99, n_ge_99 >= len(ious) - 1.

Video FPS reference (``test_sam3_litetext_video_fps_reference``): not a hard gate;
warmup 2 frames, then per-frame timing with torch.cuda.synchronize. Results printed
for README provenance (see README.md).

Skips when: CUDA absent (module level), checkpoint absent (per test), or golden
npz absent (fixture level). All three are CI-safe.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("SAM3-LiteText video parity requires CUDA", allow_module_level=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# tests/parity/reference_efficientsam3/test_*.py -> parents[3] = repo root (sam2/)
_REPO = Path(__file__).parents[3]

GOLD_NPZ = Path(__file__).parent / "golden" / "sam3_litetext_s0_ctx16_video.npz"

_CKPT_PRIMARY = _REPO / "checkpoints" / "sam3_litetext_mobileclip_s0_ctx16.pt"
_CKPT_VALIDATE = (
    _REPO / "checkpoints" / "_esam3_validate"
    / "sam3_litetext" / "sam3_litetext_mobileclip_s0_ctx16.pt"
)
CKPT = _CKPT_PRIMARY if _CKPT_PRIMARY.is_file() else _CKPT_VALIDATE


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------
# NOTE: the ``_determinism`` and ``_mask_iou`` helpers now live in tests/parity/conftest.py
# as the ``determinism`` / ``mask_iou`` fixtures (auto-discovered here).
@pytest.fixture(scope="module")
def video_fixture():
    if not GOLD_NPZ.is_file():
        pytest.skip(f"golden npz absent: {GOLD_NPZ}")
    return dict(np.load(GOLD_NPZ))


# ---------------------------------------------------------------------------
# Part 2: Streaming parity test (Phase D acceptance gate)
# ---------------------------------------------------------------------------
def test_sam3_litetext_video_parity(video_fixture, determinism, run_streaming_parity):
    """End-to-end streaming video parity through build_sam3_video_predictor.

    Replicates the upstream SAM3-LiteText golden scenario captured in
    ``sam3_litetext_s0_ctx16_video.npz`` (``capture_litetext_video_golden.py``):
      ``set_concept(ConceptPrompt("person"))`` -> stream frames 0..3 via
      ``predictor.forward(state, frame_idx, frame)`` -> collect per-object masks.

    Gate (spec §D verbatim):
      * Exact object count per frame vs golden ``frame{f}_obj_ids``.
      * Hungarian IoU per frame: min >= 0.98, mean >= 0.99, n_ge_99 >= len-1.

    Regime: bf16 autocast + deterministic TF32-off (matches the capture);
    ``forward`` enters autocast + inference_mode internally.
    """
    if not CKPT.is_file():
        pytest.skip(f"checkpoint absent: {CKPT}")

    from sam.build_sam import build_sam3_video_predictor

    determinism()
    predictor = build_sam3_video_predictor(
        config_file="configs/efficientsam3/sam3_litetext_s0_ctx16.yaml",
        ckpt_path=str(CKPT),
        device="cuda",
    )
    run_streaming_parity(predictor, video_fixture, min_gate=0.98, mean_gate=0.99)


# ---------------------------------------------------------------------------
# Part 3: Video FPS reference (not a hard gate — for README provenance)
# ---------------------------------------------------------------------------
def test_sam3_litetext_video_fps_reference(video_fixture, determinism, fps_reference):
    """Record per-frame FPS reference for SAM3-LiteText s0/ctx16 (RTX 3080 Ti).

    Methodology: text encoded ONCE per concept (cached in state), then per-frame
    vision + detect + track timed with torch.cuda.synchronize() around each forward.
    Warmup: 2 frame-level forward calls (throwaway state). Timed: 1 full 4-frame run.
    Result: median ms/frame + fps. Printed for README provenance; NOT a hard gate.
    """
    if not CKPT.is_file():
        pytest.skip(f"checkpoint absent: {CKPT}")

    from sam.build_sam import build_sam3_video_predictor

    determinism()
    predictor = build_sam3_video_predictor(
        config_file="configs/efficientsam3/sam3_litetext_s0_ctx16.yaml",
        ckpt_path=str(CKPT),
        device="cuda",
    )
    fps_reference(
        predictor,
        video_fixture,
        "SAM3-LiteText s0/ctx16  288x512  phrase='person'  (RTX 3080 Ti, bf16 autocast)",
    )
