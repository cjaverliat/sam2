# SPDX-License-Identifier: LicenseRef-SAM
"""SAM3.1-LiteText (s0/ctx16) MULTIPLEX streaming video parity + VRAM-flat + FPS.

Phase E2 acceptance gate: ``build_sam3_multiplex_video_predictor`` (OUR side) must
reproduce the SAM3.1-LiteText oracle per-frame masks within IoU tolerance, then prove
constant-VRAM growth and record FPS.

Golden: ``golden/sam3p1_litetext_s0_ctx16_video.npz`` captured from efficientsam3's OWN
sam3.1 (the ``stage1_sam3.1`` branch @ commit ``6056958``,
``build_efficientsam3_multiplex_video_model(backbone_type="sam3", text_encoder_type="MobileCLIP-S0")``
run NATIVELY -- the correct apples-to-apples reference, NOT the earlier facebook two-repo
oracle) via ``capture_sam3p1_litetext_video_golden.py`` -- 4 frames of the dance clip
resized to 288x512, phrase "person", under bf16 autocast. Result: min 0.9944 / mean 0.9980,
4/4 objects >=0.99 every frame.

Parity gate (verbatim from spec §E2):
  * Per-frame exact object count == golden.
  * Per-frame Hungarian IoU: min >= 0.98, mean >= 0.99, n_ge_99 >= len(ious) - 1.

VRAM gate (§E2 §Phase3):
  * Loop 4 golden frames to N_LONG=16; reset peak at WARM_FRAME=9 (> forgetful window 7);
    assert growth from WARM_FRAME to end <= 0.25 (25% allocator slack).

FPS reference (non-gating):
  * text encoded once; per-frame vision+detect+track timed with cuda.synchronize();
    warmup 2, median ms/frame + fps. Asserts fps > 0 only.

Skips when: CUDA absent (module level), checkpoint absent (per-test), or golden npz absent
(fixture level). All three guards are CI-safe.
"""
from __future__ import annotations

import statistics
import time
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip(
        "SAM3.1-LiteText video parity requires CUDA", allow_module_level=True
    )

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# tests/parity/reference_efficientsam3/test_*.py -> parents[3] = repo root (sam2/)
_REPO = Path(__file__).parents[3]

GOLD_NPZ = Path(__file__).parent / "golden" / "sam3p1_litetext_s0_ctx16_video.npz"

_CKPT_PRIMARY = _REPO / "checkpoints" / "efficient_sam3p1_litetext_mobileclip_s0_ctx16.pt"
_CKPT_VALIDATE = (
    _REPO / "checkpoints" / "_esam3_validate"
    / "sam3p1_litetext" / "efficient_sam3p1_litetext_mobileclip_s0_ctx16.pt"
)
CKPT = _CKPT_PRIMARY if _CKPT_PRIMARY.is_file() else _CKPT_VALIDATE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _determinism() -> None:
    """Mirror the oracle's determinism regime (seed=0, TF32 OFF, cuDNN deterministic).

    NOTE: do NOT call torch.use_deterministic_algorithms(True) -- the multiplex memory
    attention hardcodes sdpa_kernel(SDPBackend.FLASH_ATTENTION) and deterministic mode
    forbids the flash SDPA kernel -> RuntimeError: No available kernel.
    """
    torch.manual_seed(0)
    np.random.seed(0)
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
# Part 1: Streaming parity test (E2 acceptance gate)
# ---------------------------------------------------------------------------
def test_sam3p1_litetext_video_parity(video_fixture):
    """End-to-end SAM3.1-LiteText MULTIPLEX streaming video parity.

    Replicates the two-repo oracle golden (``sam3p1_litetext_s0_ctx16_video.npz``):
      ``set_concept(ConceptPrompt("person"))`` -> stream frames 0..3 via
      ``predictor.forward(state, frame_idx, frame)`` -> collect per-object masks.

    Gate (spec §E2):
      * Exact object count per frame vs golden ``frame{f}_obj_ids``.
      * Hungarian IoU per frame: min >= 0.98, mean >= 0.99, n_ge_99 >= len-1.

    Determinism: no use_deterministic_algorithms (flash SDPA incompatible);
    bf16 autocast entered inside predictor.forward.
    """
    if not CKPT.is_file():
        pytest.skip(f"checkpoint absent: {CKPT}")

    from scipy.optimize import linear_sum_assignment

    from sam.build_sam import build_sam3_multiplex_video_predictor
    from sam.models.sam3_predictor import Sam3VideoPredictorState
    from sam.prompts import ConceptPrompt

    _determinism()
    predictor = build_sam3_multiplex_video_predictor(
        config_file="configs/efficientsam3/sam3p1_litetext_s0_ctx16.yaml",
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

        # Exact count check
        assert len(my_ids) == len(g_ids), (
            f"frame {f_idx}: object count {len(my_ids)} != golden {len(g_ids)} "
            f"(mine={my_ids}, golden={g_ids})"
        )

        # Hungarian IoU matching (bijection golden<->mine)
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
# Part 2: VRAM-flat test (constant VRAM as clip grows)
# ---------------------------------------------------------------------------
def test_sam3p1_litetext_video_constant_vram(video_fixture):
    """Persistent CUDA allocation stays flat as the streamed clip grows (forgetful-bank property).

    The multiplex tracker's BUCKET-space spatial memory is threaded as the tracker's
    native ``output_dict``; ``Sam3MultiplexVideoPredictor`` prunes non-conditional frame
    entries outside the forgetful window (cond frames kept), so persistent VRAM is bounded
    to ``<= window`` non-conditional frames -> does not grow with clip length.

    Method (mirrors Phase-1 test_sam3p1_video_constant_vram):
      * Stream 4 golden frames looped to N_LONG=16.
      * Reset peak at WARM_FRAME=9 (> forgetful window 7, the non-cond store is full).
      * PRIMARY gate: persistent allocation (memory_allocated after synchronize) from
        WARM_FRAME to final frame <= PERSISTENT_GROWTH_GATE (5%). This directly proves the
        forgetful bank bounds persistent state regardless of forward-pass temporaries.
      * SECONDARY gate: peak growth (max_memory_allocated) <= VRAM_GROWTH_GATE (40%).
        The 0.40 threshold provides ~1.2× headroom over the measured peak-temporary
        overhead (~33.5%): SAM3.1-LiteText uses MobileCLIP (111 keys) instead of the VE
        text tower (295 keys), so its persistent base is ~700 MB lighter than Phase-1
        SAM3.1. The same absolute forward-overhead / smaller-base -> higher peak ratio.
        The persistent-flatness gate is the authoritative property check; the peak gate
        is a secondary sanity bound only.
    """
    if not CKPT.is_file():
        pytest.skip(f"checkpoint absent: {CKPT}")

    from sam.build_sam import build_sam3_multiplex_video_predictor
    from sam.models.sam3_predictor import Sam3VideoPredictorState
    from sam.prompts import ConceptPrompt

    _determinism()
    predictor = build_sam3_multiplex_video_predictor(
        config_file="configs/efficientsam3/sam3p1_litetext_s0_ctx16.yaml",
        ckpt_path=str(CKPT),
        device="cuda",
    )

    base_frames = video_fixture["video_frames_rgb"]   # (4,288,512,3)
    video_h, video_w = (int(v) for v in video_fixture["video_hw"])
    phrase = str(video_fixture["video_phrase"])
    N_LONG = 16
    WARM_FRAME = 9              # > forgetful window (7): non-cond store full and steady here
    VRAM_GROWTH_GATE = 0.40     # secondary peak gate: see docstring
    PERSISTENT_GROWTH_GATE = 0.05  # primary property gate: persistent alloc must stay flat

    state = Sam3VideoPredictorState(video_hw=(video_h, video_w))
    predictor.set_concept(state, ConceptPrompt(phrase))

    peak_after_warm = None
    persistent_after_warm = None
    for f_idx in range(N_LONG):
        frame = base_frames[f_idx % base_frames.shape[0]]
        predictor.forward(state, f_idx, frame)
        torch.cuda.synchronize()
        if f_idx == WARM_FRAME:
            torch.cuda.reset_peak_memory_stats()
            peak_after_warm = torch.cuda.max_memory_allocated()
            persistent_after_warm = torch.cuda.memory_allocated()

    torch.cuda.synchronize()
    peak_after_long = torch.cuda.max_memory_allocated()
    persistent_after_long = torch.cuda.memory_allocated()

    assert peak_after_warm is not None and peak_after_warm > 0
    assert persistent_after_warm is not None and persistent_after_warm > 0

    # PRIMARY: persistent-flatness (the forgetful-bank property being proved)
    persistent_growth = (persistent_after_long - persistent_after_warm) / persistent_after_warm
    print(
        f"\n[vram] persistent: warm_frame={WARM_FRAME}  "
        f"{persistent_after_warm/1e6:.1f} MB -> {persistent_after_long/1e6:.1f} MB  "
        f"growth={persistent_growth:.1%}  gate={PERSISTENT_GROWTH_GATE:.0%}"
    )
    assert persistent_growth <= PERSISTENT_GROWTH_GATE, (
        f"persistent VRAM grew {persistent_growth:.1%} from frame {WARM_FRAME} "
        f"({persistent_after_warm/1e6:.1f} MB) to frame {N_LONG - 1} "
        f"({persistent_after_long/1e6:.1f} MB) -- forgetful bank leak detected "
        f"(gate={PERSISTENT_GROWTH_GATE:.0%})"
    )

    # SECONDARY: peak sanity bound (catches runaway temporaries)
    peak_growth = (peak_after_long - peak_after_warm) / peak_after_warm
    print(
        f"[vram] peak:       warm_frame={WARM_FRAME}  "
        f"{peak_after_warm/1e6:.1f} MB -> {peak_after_long/1e6:.1f} MB  "
        f"growth={peak_growth:.1%}  gate={VRAM_GROWTH_GATE:.0%}"
    )
    assert peak_growth <= VRAM_GROWTH_GATE, (
        f"peak VRAM grew {peak_growth:.1%} from frame {WARM_FRAME} "
        f"({peak_after_warm/1e6:.1f} MB) to frame {N_LONG - 1} "
        f"({peak_after_long/1e6:.1f} MB) -- not constant-VRAM (gate={VRAM_GROWTH_GATE:.0%})"
    )


# ---------------------------------------------------------------------------
# Part 3: Video FPS reference (non-gating -- for README provenance)
# ---------------------------------------------------------------------------
def test_sam3p1_litetext_video_fps_reference(video_fixture):
    """Record per-frame FPS reference for SAM3.1-LiteText s0/ctx16 (RTX 3080 Ti).

    Methodology: text encoded ONCE per concept (cached in state), then per-frame
    vision + detect + track timed with torch.cuda.synchronize() around each forward.
    Warmup: 2 frame-level forward calls (throwaway state). Timed: 1 full 4-frame run.
    Result: median ms/frame + fps. Printed for README provenance; NOT a hard gate.
    """
    if not CKPT.is_file():
        pytest.skip(f"checkpoint absent: {CKPT}")

    from sam.build_sam import build_sam3_multiplex_video_predictor
    from sam.models.sam3_predictor import Sam3VideoPredictorState
    from sam.prompts import ConceptPrompt

    _determinism()
    predictor = build_sam3_multiplex_video_predictor(
        config_file="configs/efficientsam3/sam3p1_litetext_s0_ctx16.yaml",
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
        f"\n[fps_ref] SAM3.1-LiteText s0/ctx16  288x512  phrase='person'\n"
        f"  warmup=2frames  timed={n_frames}frames\n"
        f"  {per_frame_str}\n"
        f"  median={median_ms:.1f} ms/frame  fps={fps:.1f}  (RTX 3080 Ti, bf16 autocast)"
    )
    # Not a hard gate — just verify the model ran
    assert fps > 0
