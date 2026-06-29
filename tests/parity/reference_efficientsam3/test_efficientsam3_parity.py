# SPDX-License-Identifier: LicenseRef-SAM
"""EfficientSAM3 end-to-end image parity vs upstream golden (RepViT + TinyViT).

Phase A/B acceptance gate: the first full forward pass of each integrated variant
(build_efficientsam3 -> Sam3Predictor.predict) validated numerically against the
upstream EfficientSAM3 reference captured in ``golden/efficientsam3_{variant}_*``.

Each integrated model is TEXT-ONLY (geometry encoder disabled; the non-geo checkpoint
carries no trained geometry weights -- see the configs). The golden was captured from
upstream with the geometry CLS token likewise disabled, so this is an apples-to-apples
comparison of the shared (trunk + neck + text + detector) pipeline.

Acceptance criteria (per variant/prompt):
  * Instance count == golden (RepViT: 4 dog / 9 person; TinyViT: 5 dog / 7 person).
  * Every matched instance has mask IoU >= 0.99 (Hungarian-matched by mask IoU
    to handle any ordering difference between our detector output and the golden).

Skip conditions (CI-safe): if the checkpoint or the test image is absent for a given
variant, that variant/prompt is skipped automatically.  All assertions run only when
both are present.

Regime: float32 (no autocast), matching the upstream golden capture. ``predict`` is called
with ``dtype=torch.float32``; bf16 would round borderline scores and change the count.
"""
from __future__ import annotations

import json
import numpy as np
import pytest
from pathlib import Path

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("EfficientSAM3 parity requires CUDA", allow_module_level=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# tests/parity/reference_efficientsam3/test_*.py ->
#   parents[0] = tests/parity/reference_efficientsam3
#   parents[3] = repo root (sam2)
#   parents[4] = workspace (PycharmProjects)
_REPO = Path(__file__).parents[3]
_WORKSPACE = Path(__file__).parents[4]

GOLD_DIR = Path(__file__).parent / "golden"
IMG = _WORKSPACE / "efficientsam3_reference/sam3/assets/dog_person.jpeg"

# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------
VARIANTS: dict[str, dict] = {
    "repvit": dict(
        config="configs/efficientsam3/efficientsam3_repvit.yaml",
        ckpts=[
            _REPO / "checkpoints/efficientsam3_repvit.pt",
            _REPO / "checkpoints/_esam3_validate/efficientsam3_ft/efficientsam3_repvit.pt",
        ],
        golden="efficientsam3_repvit",
    ),
    "tinyvit": dict(
        config="configs/efficientsam3/efficientsam3_tinyvit.yaml",
        ckpts=[
            _REPO / "checkpoints/efficientsam3_tinyvit.pt",
            _REPO / "checkpoints/_esam3_validate/efficientsam3_ft/efficientsam3_tinyvit.pt",
        ],
        golden="efficientsam3_tinyvit",
    ),
}


def _resolve_ckpt(ckpts: list) -> Path | None:
    """Return the first existing checkpoint path, or None."""
    return next((p for p in ckpts if Path(p).is_file()), None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _determinism() -> None:
    """Mirror the determinism regime used by the upstream golden capture."""
    np.random.seed(0)
    torch.manual_seed(0)
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


def _pairwise_iou(masks_a: np.ndarray, masks_b: np.ndarray) -> np.ndarray:
    """Return (N, M) float64 IoU matrix between two sets of (H, W) masks."""
    N, M = len(masks_a), len(masks_b)
    mat = np.zeros((N, M), dtype=np.float64)
    for i in range(N):
        for j in range(M):
            mat[i, j] = _mask_iou(masks_a[i], masks_b[j])
    return mat


# ---------------------------------------------------------------------------
# Parity test (parametrized over variant x prompt)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("prompt", ["dog", "person"])
def test_efficientsam3_parity(variant: str, prompt: str) -> None:
    """End-to-end EfficientSAM3 image parity vs upstream golden (Phase A/B gate).

    Loads the model via ``build_efficientsam3``, runs ``Sam3Predictor.predict`` on
    dog_person.jpeg (2048x1365), then Hungarian-matches the predicted masks to the
    committed golden masks (``efficientsam3_{variant}_masks_{prompt}.npz``).

    Asserts:
      1. Instance count == golden ``num_instances`` (RepViT: 4 dog / 9 person;
         TinyViT: 5 dog / 7 person).
      2. Every Hungarian-matched pair has mask IoU >= 0.99 (binary at logit=0 /
         prob=0.5 for ours; binary at prob=0.5 for the golden sigmoid outputs).

    Skips per-variant when the checkpoint or test image is absent (CI-safe).
    """
    from scipy.optimize import linear_sum_assignment
    from PIL import Image

    from sam.build_sam import build_efficientsam3
    from sam.prompts import ConceptPrompt

    cfg = VARIANTS[variant]
    ckpt = _resolve_ckpt(cfg["ckpts"])
    if ckpt is None or not IMG.is_file():
        pytest.skip(
            f"[{variant}] checkpoint or test image absent (CI-safe skip). "
            f"Tried: {[str(p) for p in cfg['ckpts']]}"
        )

    # ------------------------------------------------------------------ golden
    golden_prefix = cfg["golden"]
    summ_path = GOLD_DIR / f"{golden_prefix}_summary.json"
    summ = json.loads(summ_path.read_text())
    g_info = summ["prompts"][prompt]
    n_expected = int(g_info["num_instances"])

    # golden shape (N, 1, H, W) sigmoid probs; binarise at 0.5
    gold_npz = np.load(GOLD_DIR / f"{golden_prefix}_masks_{prompt}.npz")
    gold_masks_raw = gold_npz["masks"]               # (N, 1, 1365, 2048) float
    gold_masks_bin = (gold_masks_raw[:, 0] > 0.5).astype(np.uint8)  # (N, 1365, 2048)

    _determinism()

    # ------------------------------------------------------------------ model
    model = build_efficientsam3(
        config_file=cfg["config"], ckpt_path=str(ckpt), device="cuda", mode="eval"
    )

    # predict() expects (H, W, 3) uint8 RGB numpy array
    image_rgb = np.array(Image.open(IMG).convert("RGB"))   # (1365, 2048, 3)

    threshold = float(summ["threshold"])   # 0.1 (matches the golden capture)
    # The upstream golden was captured in float32 (no autocast); pass dtype=float32
    # so our predict() matches the golden's precision regime.  bfloat16 rounds
    # borderline scores below the threshold, changing the instance count.
    result = model.predict(
        image_rgb, ConceptPrompt(text=prompt), confidence_threshold=threshold,
        dtype=torch.float32,
    )

    # ------------------------------------------------------------------ count
    n_actual = int(result.masks_logits.shape[0])
    assert n_actual == n_expected, (
        f"[{variant}/{prompt}] instance count {n_actual} != golden {n_expected}; "
        f"threshold={threshold}, scores={result.scores.tolist()}"
    )

    # ------------------------------------------------------------------ IoU
    # Our masks_logits are raw logits; binarise at 0.0 (== sigmoid > 0.5).
    our_masks_bin = (
        result.masks_logits.float().cpu().numpy() > 0.0
    ).astype(np.uint8)  # (N, 1365, 2048)

    # Pairwise IoU matrix (N_ours x N_gold); counts are already equal.
    iou_mat = _pairwise_iou(our_masks_bin, gold_masks_bin)
    row_ind, col_ind = linear_sum_assignment(-iou_mat)

    matched_ious = [float(iou_mat[r, c]) for r, c in zip(row_ind, col_ind)]
    min_iou = min(matched_ious) if matched_ious else 1.0

    # Tolerance: 0.99 (the spec acceptance gate).
    #
    # Text-only (geometry disabled): all dog instances match at IoU 1.0, and most
    # person instances at 1.0; occasional worst-case is ~0.996. The residual sub-1.0
    # gap is inherent float32 non-determinism in the MobileCLIP text transformer --
    # two independent process runs on the same GPU accumulate ~0.001 max diff in text
    # embeddings (even with cuDNN deterministic + strict algorithm mode), flipping a
    # few boundary pixels. Backbone / neck features are bit-exact; a genuinely broken
    # model would show < 0.90 IoU.
    assert min_iou >= 0.99, (
        f"[{variant}/{prompt}] min matched mask IoU {min_iou:.4f} < 0.99. "
        f"Per-instance IoUs (sorted desc): {sorted(matched_ious, reverse=True)}"
    )
