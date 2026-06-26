# SPDX-License-Identifier: Apache-2.0
"""Validate that the post-Phase-0 `sam` package reproduces the notebook pipelines
(image-predict + streaming video) vs the pre-Phase-0 `sam2` code.

Golden was captured once from the pre-refactor worktree (commit 804369f):
    PYTHONPATH=../sam2_prephase0 \
      pixi run python tests/parity/run_pipelines.py --api old \
      --out tests/parity/fixtures/golden_prephase0.npz
"""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("parity pipelines require CUDA", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).parent))
import run_pipelines  # noqa: E402

GOLDEN = Path(__file__).parent / "fixtures" / "golden_prephase0.npz"


def _iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    union = (a | b).sum()
    return 1.0 if union == 0 else float((a & b).sum()) / float(union)


@pytest.fixture(scope="module")
def outputs():
    if not GOLDEN.is_file():
        pytest.skip(f"golden fixture missing: {GOLDEN}")
    return dict(np.load(GOLDEN)), run_pipelines.run_all("new")


def test_keys_match(outputs):
    golden, new = outputs
    assert set(golden) == set(new)


@pytest.mark.parametrize("key", ["img_pt", "img_box", "img_ptbox"])
def test_image_parity(outputs, key):
    golden, new = outputs
    gm, nm = golden[f"{key}_masks"], new[f"{key}_masks"]
    assert gm.shape == nm.shape, f"{key} mask shape {nm.shape} != golden {gm.shape}"
    for i in range(gm.shape[0]):
        assert _iou(gm[i], nm[i]) >= 0.999, f"{key}[{i}] mask IoU below 0.999 vs pre-Phase-0"
    np.testing.assert_allclose(golden[f"{key}_scores"], new[f"{key}_scores"], atol=1e-3)


@pytest.mark.parametrize("f_idx", [0, 1, 2])
def test_video_parity(outputs, f_idx):
    golden, new = outputs
    g, n = golden[f"vid_f{f_idx}_mask"], new[f"vid_f{f_idx}_mask"]
    assert _iou(g, n) >= 0.999, f"video frame {f_idx} mask IoU below 0.999 vs pre-Phase-0"


def test_amg_parity(outputs):
    golden, new = outputs
    assert int(golden["amg_count"][0]) == int(new["amg_count"][0]), (
        "automatic-mask-generator mask count differs vs pre-Phase-0"
    )
    np.testing.assert_allclose(golden["amg_areas"], new["amg_areas"], atol=2)
