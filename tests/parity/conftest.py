# SPDX-License-Identifier: LicenseRef-SAM
"""Shared pytest fixtures for the SAM 3 / EfficientSAM3 parity suite.

Auto-discovered by tests under ``tests/parity/`` AND the ``reference_efficientsam3/``
subdirectory. Exposes the determinism-regime setters and the binary-mask IoU helper that
were previously byte-duplicated across the parity modules as fixtures returning callables.

Determinism note: two regimes exist and MUST stay distinct (they are not interchangeable).
  * ``determinism`` calls ``torch.use_deterministic_algorithms(True, warn_only=True)``
    (used by test_sam3_parity, test_efficientsam3_parity, test_sam3_litetext_video_parity).
  * ``determinism_no_det_algos`` does NOT -- the multiplex memory attention hardcodes the
    flash SDPA kernel, which deterministic mode forbids (RuntimeError: No available kernel);
    the base-lineage distilled-video parity omits it too (used by test_efficientsam3_video_parity,
    test_efficientsam3p1_repvit_video_parity, test_sam3p1_litetext_video_parity). Merging the
    two would change kernel selection / raise, so they are separate fixtures.

numpy/torch are imported inside the fixture bodies so collection stays clean when torch is
absent (the parity modules skip at module level via ``pytest.importorskip('torch')``).
"""
import pytest


@pytest.fixture
def determinism():
    """Determinism regime WITH torch.use_deterministic_algorithms (seed 0, TF32 off, cuDNN det)."""
    import numpy as np
    import torch

    def _determinism():
        torch.manual_seed(0)
        np.random.seed(0)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    return _determinism


@pytest.fixture
def determinism_no_det_algos():
    """Determinism regime WITHOUT torch.use_deterministic_algorithms (seed 0, TF32 off, cuDNN det).

    Required by the multiplex / distilled-video parity tests: the multiplex memory attention
    hardcodes sdpa_kernel(SDPBackend.FLASH_ATTENTION), which use_deterministic_algorithms(True)
    forbids -> RuntimeError: No available kernel.
    """
    import numpy as np
    import torch

    def _determinism():
        torch.manual_seed(0)
        np.random.seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    return _determinism


@pytest.fixture
def mask_iou():
    """Binary-mask IoU of two arrays (any bool/int dtype): astype(bool) intersection-over-union."""
    import numpy as np

    def _mask_iou(a, b):
        a = a.astype(bool)
        b = b.astype(bool)
        inter = float(np.logical_and(a, b).sum())
        union = float(np.logical_or(a, b).sum())
        return 1.0 if union == 0.0 else inter / union

    return _mask_iou


@pytest.fixture
def run_streaming_parity(mask_iou):
    """Pure extraction of the streaming-video parity loop shared by the base + multiplex
    per-frame parity tests.

    Given an already-built ``predictor`` and the golden npz ``fixture``, this:
      * ``set_concept(ConceptPrompt(fixture['video_phrase']))`` on a fresh state,
      * streams every frame via ``predictor.forward`` and binarises each object mask,
      * per frame: asserts exact object COUNT vs ``frame{f}_obj_ids``, a bijective
        Hungarian golden<->mine id-mapping on ``mask_iou``, and the gates
        ``min >= min_gate``, ``mean >= mean_gate``, ``n_ge_99 >= len(ious) - 1``.

    The ``n_ge_99`` threshold (0.99) is fixed (identical across all call sites); the
    ``min_gate`` / ``mean_gate`` are passed per call site to preserve exact values. NOT used
    by test_efficientsam3_video_parity, whose loop aggregates gates over all frames and uses a
    >0.5-threshold IoU (kept local there).
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    from sam.models.sam3_predictor import Sam3VideoPredictorState
    from sam.prompts import ConceptPrompt
    from sam.results import Emit

    def _run(predictor, fixture, *, min_gate=0.98, mean_gate=0.99):
        frames = fixture["video_frames_rgb"]  # (T,H,W,3) uint8
        video_h, video_w = (int(v) for v in fixture["video_hw"])
        phrase = str(fixture["video_phrase"])
        n_frames = int(frames.shape[0])

        # The goldens captured upstream's OBSERVABLE, which shows an object from its
        # first frame (upstream decides visibility non-causally). Emit.CONFIRMED -- the
        # shipped default -- cannot show those birth frames, so parity is measured in
        # Emit.VISIBLE mode.
        predictor.emit = Emit.VISIBLE
        state = Sam3VideoPredictorState(video_hw=(video_h, video_w))
        predictor.set_concept(state, ConceptPrompt(phrase))

        per_frame_masks = {}  # frame_idx -> {obj_id: (H,W) uint8}
        for f_idx in range(n_frames):
            out = predictor.forward(state, f_idx, frames[f_idx])
            per_frame_masks[f_idx] = {
                oid: (r.masks_logits.float().cpu().numpy()[0, 0] > 0.0).astype(np.uint8)
                for oid, r in out.items()
            }

        for f_idx in range(n_frames):
            g_ids = fixture[f"frame{f_idx}_obj_ids"].tolist()
            my = per_frame_masks[f_idx]
            my_ids = list(my.keys())
            assert len(my_ids) == len(g_ids), (
                f"frame {f_idx}: object count {len(my_ids)} != golden {len(g_ids)} "
                f"(mine={my_ids}, golden={g_ids})"
            )
            # Hungarian-match golden<->mine on IoU; require every golden object reproduced.
            iou_mat = np.zeros((len(g_ids), len(my_ids)), np.float64)
            for i, gid in enumerate(g_ids):
                g = fixture[f"frame{f_idx}_obj{gid}"].astype(np.uint8)
                for j, mid in enumerate(my_ids):
                    iou_mat[i, j] = mask_iou(my[mid], g)
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
            assert min_iou >= min_gate, (
                f"frame {f_idx}: min per-object IoU {min_iou:.4f} < {min_gate} ({matched})"
            )
            assert mean_iou >= mean_gate, (
                f"frame {f_idx}: mean per-object IoU {mean_iou:.4f} < {mean_gate} ({matched})"
            )
            assert n_ge_99 >= len(ious) - 1, (
                f"frame {f_idx}: only {n_ge_99}/{len(ious)} objects >= 0.99 IoU ({matched})"
            )

    return _run


@pytest.fixture
def assert_constant_vram():
    """Pure extraction of the constant-VRAM streaming loop shared by the base + multiplex
    video predictors.

    Streams ``fixture['video_frames_rgb']`` looped to ``n_long`` frames; resets the peak at
    ``warm_frame`` (past the forgetful window) and asserts peak growth from there to the final
    frame is ``<= peak_gate``. When ``persistent_gate`` is given (multiplex call sites) it also
    asserts persistent-allocation growth ``<= persistent_gate`` (the authoritative forgetful-bank
    property); the base call sites pass only ``peak_gate`` (peak-only, no persistent check),
    reproducing their single-gate form exactly. All gate values / n_long / warm_frame are passed
    per call site.
    """
    import torch

    from sam.models.sam3_predictor import Sam3VideoPredictorState
    from sam.prompts import ConceptPrompt

    def _run(predictor, fixture, *, peak_gate, persistent_gate=None, n_long=16, warm_frame=9):
        base_frames = fixture["video_frames_rgb"]
        video_h, video_w = (int(v) for v in fixture["video_hw"])
        phrase = str(fixture["video_phrase"])

        state = Sam3VideoPredictorState(video_hw=(video_h, video_w))
        predictor.set_concept(state, ConceptPrompt(phrase))

        peak_after_warm = None
        persistent_after_warm = None
        for f_idx in range(n_long):
            frame = base_frames[f_idx % base_frames.shape[0]]
            predictor.forward(state, f_idx, frame)
            torch.cuda.synchronize()
            if f_idx == warm_frame:
                torch.cuda.reset_peak_memory_stats()
                peak_after_warm = torch.cuda.max_memory_allocated()
                if persistent_gate is not None:
                    persistent_after_warm = torch.cuda.memory_allocated()

        torch.cuda.synchronize()
        peak_after_long = torch.cuda.max_memory_allocated()

        assert peak_after_warm is not None and peak_after_warm > 0
        if persistent_gate is not None:
            persistent_after_long = torch.cuda.memory_allocated()
            assert persistent_after_warm is not None and persistent_after_warm > 0
            persistent_growth = (
                persistent_after_long - persistent_after_warm
            ) / persistent_after_warm
            assert persistent_growth <= persistent_gate, (
                f"persistent VRAM grew {persistent_growth:.1%} from frame {warm_frame} "
                f"({persistent_after_warm/1e6:.1f} MB) to frame {n_long - 1} "
                f"({persistent_after_long/1e6:.1f} MB) -- forgetful bank leak "
                f"(gate={persistent_gate:.0%})"
            )

        peak_growth = (peak_after_long - peak_after_warm) / peak_after_warm
        assert peak_growth <= peak_gate, (
            f"peak VRAM grew {peak_growth:.1%} from frame {warm_frame} "
            f"({peak_after_warm/1e6:.1f} MB) to frame {n_long - 1} "
            f"({peak_after_long/1e6:.1f} MB) -- not constant-VRAM (gate={peak_gate:.0%})"
        )

    return _run


@pytest.fixture
def fps_reference():
    """Pure extraction of the (non-gating) per-frame FPS-reference loop.

    Warms up 2 frame-level forwards on a throwaway state, then times a full run with
    ``torch.cuda.synchronize()`` around each forward, prints the median ms/frame + fps under
    ``label``, and asserts only ``fps > 0`` (the sole assertion at every call site). ``label``
    is passed per call site (model name / device / precision note).
    """
    import statistics
    import time

    import torch

    from sam.models.sam3_predictor import Sam3VideoPredictorState
    from sam.prompts import ConceptPrompt

    def _run(predictor, fixture, label):
        frames = fixture["video_frames_rgb"]
        video_h, video_w = (int(v) for v in fixture["video_hw"])
        phrase = str(fixture["video_phrase"])
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
        frame_times = []
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
            f"\n[fps_ref] {label}\n"
            f"  warmup=2frames  timed={n_frames}frames\n"
            f"  {per_frame_str}\n"
            f"  median={median_ms:.1f} ms/frame  fps={fps:.1f}"
        )
        # Not a hard gate -- just verify the model ran
        assert fps > 0

    return _run
