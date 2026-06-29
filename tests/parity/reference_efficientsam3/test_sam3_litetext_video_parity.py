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

import statistics
import time
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
# Helpers
# ---------------------------------------------------------------------------
def _determinism() -> None:
    """Mirror the capture's determinism regime (seed=0, TF32 OFF, cuDNN deterministic)."""
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Binary mask IoU (any bool/int dtype)."""
    a = a.astype(bool)
    b = b.astype(bool)
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return 1.0 if union == 0.0 else inter / union


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def video_fixture():
    if not GOLD_NPZ.is_file():
        pytest.skip(f"golden npz absent: {GOLD_NPZ}")
    return dict(np.load(GOLD_NPZ))


# ---------------------------------------------------------------------------
# Part 2: Streaming parity test (Phase D acceptance gate)
# ---------------------------------------------------------------------------
def test_sam3_litetext_video_parity(video_fixture):
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

    from scipy.optimize import linear_sum_assignment

    from sam.build_sam import build_sam3_video_predictor
    from sam.models.sam3_predictor import Sam3VideoPredictorState
    from sam.prompts import ConceptPrompt

    _determinism()
    predictor = build_sam3_video_predictor(
        config_file="configs/efficientsam3/sam3_litetext_s0_ctx16.yaml",
        ckpt_path=str(CKPT),
        device="cuda",
    )

    frames = video_fixture["video_frames_rgb"]    # (T,288,512,3) uint8
    video_h, video_w = (int(v) for v in video_fixture["video_hw"])
    phrase = str(video_fixture["video_phrase"])   # "person"
    n_frames = int(frames.shape[0])

    state = Sam3VideoPredictorState(video_hw=(video_h, video_w))
    predictor.set_concept(state, ConceptPrompt(phrase))

    # Stream all frames; collect per-frame {obj_id: (H,W) uint8 binary mask}
    per_frame_masks: dict[int, dict[int, np.ndarray]] = {}
    for f_idx in range(n_frames):
        out = predictor.forward(state, f_idx, frames[f_idx])
        per_frame_masks[f_idx] = {
            oid: (r.masks_logits.float().cpu().numpy()[0, 0] > 0.0).astype(np.uint8)
            for oid, r in out.items()
        }

    all_ious: list[float] = []
    for f_idx in range(n_frames):
        g_ids = video_fixture[f"frame{f_idx}_obj_ids"].tolist()
        my = per_frame_masks[f_idx]
        my_ids = list(my.keys())

        # Count check
        assert len(my_ids) == len(g_ids), (
            f"frame {f_idx}: object count {len(my_ids)} != golden {len(g_ids)} "
            f"(mine={my_ids}, golden={g_ids})"
        )

        # Hungarian IoU matching
        iou_mat = np.zeros((len(g_ids), len(my_ids)), np.float64)
        for i, gid in enumerate(g_ids):
            g = video_fixture[f"frame{f_idx}_obj{gid}"].astype(np.uint8)
            for j, mid in enumerate(my_ids):
                iou_mat[i, j] = _mask_iou(my[mid], g)

        row, col = linear_sum_assignment(-iou_mat)
        matched = {
            int(g_ids[r]): (int(my_ids[c]), float(iou_mat[r, c]))
            for r, c in zip(row, col)
        }
        assert len(set(col)) == len(g_ids), (
            f"frame {f_idx}: id-mapping is not a bijection ({matched})"
        )

        ious = [v[1] for v in matched.values()]
        min_iou = min(ious)
        mean_iou = sum(ious) / len(ious)
        n_ge_99 = sum(1 for x in ious if x >= 0.99)

        print(
            f"\n  frame {f_idx}: matched={matched} "
            f"min={min_iou:.4f}  mean={mean_iou:.4f}  n_ge_99={n_ge_99}/{len(ious)}"
        )

        assert min_iou >= 0.98, (
            f"frame {f_idx}: min per-object IoU {min_iou:.4f} < 0.98 ({matched})"
        )
        assert mean_iou >= 0.99, (
            f"frame {f_idx}: mean per-object IoU {mean_iou:.4f} < 0.99 ({matched})"
        )
        assert n_ge_99 >= len(ious) - 1, (
            f"frame {f_idx}: only {n_ge_99}/{len(ious)} objects >= 0.99 IoU ({matched})"
        )
        all_ious.extend(ious)

    min_overall = min(all_ious)
    mean_overall = sum(all_ious) / len(all_ious)
    print(f"\n[parity] PASSED  overall min={min_overall:.4f}  mean={mean_overall:.4f}")


# ---------------------------------------------------------------------------
# Part 3: Video FPS reference (not a hard gate — for README provenance)
# ---------------------------------------------------------------------------
def test_sam3_litetext_video_fps_reference(video_fixture):
    """Record per-frame FPS reference for SAM3-LiteText s0/ctx16 (RTX 3080 Ti).

    Methodology: text encoded ONCE per concept (cached in state), then per-frame
    vision + detect + track timed with torch.cuda.synchronize() around each forward.
    Warmup: 2 frame-level forward calls (throwaway state). Timed: 1 full 4-frame run.
    Result: median ms/frame + fps. Printed for README provenance; NOT a hard gate.
    """
    if not CKPT.is_file():
        pytest.skip(f"checkpoint absent: {CKPT}")

    from sam.build_sam import build_sam3_video_predictor
    from sam.models.sam3_predictor import Sam3VideoPredictorState
    from sam.prompts import ConceptPrompt

    _determinism()
    predictor = build_sam3_video_predictor(
        config_file="configs/efficientsam3/sam3_litetext_s0_ctx16.yaml",
        ckpt_path=str(CKPT),
        device="cuda",
    )

    frames = video_fixture["video_frames_rgb"]    # (T,288,512,3) uint8
    video_h, video_w = (int(v) for v in video_fixture["video_hw"])
    phrase = str(video_fixture["video_phrase"])
    n_frames = int(frames.shape[0])

    # Warmup: 2 frame-level forward calls (GPU kernel warm-up)
    ws = Sam3VideoPredictorState(video_hw=(video_h, video_w))
    predictor.set_concept(ws, ConceptPrompt(phrase))
    for f_idx in range(min(2, n_frames)):
        predictor.forward(ws, f_idx, frames[f_idx])
    torch.cuda.synchronize()

    # Timed: fresh state, 1 full run of all frames
    ts_state = Sam3VideoPredictorState(video_hw=(video_h, video_w))
    predictor.set_concept(ts_state, ConceptPrompt(phrase))
    frame_times: list[float] = []
    for f_idx in range(n_frames):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        predictor.forward(ts_state, f_idx, frames[f_idx])
        torch.cuda.synchronize()
        frame_times.append(time.perf_counter() - t0)

    median_ms = statistics.median(frame_times) * 1e3
    fps = 1000.0 / median_ms
    per_frame_str = "  ".join(f"f{i}={t*1e3:.1f}ms" for i, t in enumerate(frame_times))
    print(
        f"\n[fps_ref] SAM3-LiteText s0/ctx16  288x512  phrase='person'\n"
        f"  warmup=2frames  timed={n_frames}frames\n"
        f"  {per_frame_str}\n"
        f"  median={median_ms:.1f} ms/frame  fps={fps:.1f}  (RTX 3080 Ti, bf16 autocast)"
    )
    # Not a hard gate — just verify the model ran
    assert fps > 0
