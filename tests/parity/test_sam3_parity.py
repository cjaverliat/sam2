# SPDX-License-Identifier: LicenseRef-SAM
"""Parity tests for the vendored SAM 3 components vs the golden oracle.

The golden fixtures (``tests/parity/fixtures/sam3/``) were captured from the OFFICIAL
SAM 3 (``facebook/sam3``) under bf16 autocast + determinism by
``reference_sam3/capture_sam3_golden.py`` (Phase 1, Task 1). Each test skips cleanly
when torch / CUDA, the local checkpoint, or the fixture is absent -- mirroring
``test_notebook_parity.py``.

Regime (must match the capture or parity spuriously fails): seed 0, deterministic
algorithms, cuDNN deterministic, TF32 OFF, forward under ``autocast(cuda, bfloat16)`` +
``inference_mode``. Encoder features are stored fp16, so the gate is ``atol=1e-2`` on
the fp32-upcast compare, not bitwise equality.
"""
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("SAM 3 parity requires CUDA", allow_module_level=True)

FIXTURES = Path(__file__).parent / "fixtures" / "sam3"
CKPT = Path(__file__).parents[2] / "checkpoints" / "sam3.pt"


def _determinism():
    """Mirror run_pipelines._determinism / the capture's determinism() regime."""
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


from sam.utils.sam3_transforms import preprocess_to_1008 as _preprocess_to_1008


@pytest.fixture(scope="module")
def image_fixture():
    f = FIXTURES / "image.npz"
    if not f.is_file():
        pytest.skip(f"fixture absent: {f}")
    return dict(np.load(f))


def test_encoder_parity(image_fixture):
    """The PE vision encoder's principal (stride-14, 72x72/256ch) level matches the golden
    ``enc_feat_lastlevel`` within atol=1e-2."""
    if not CKPT.is_file():
        pytest.skip(f"checkpoint absent: {CKPT}")
    from sam.build_sam import build_sam3_vision_encoder

    _determinism()
    encoder = build_sam3_vision_encoder(ckpt_path=str(CKPT), device="cuda")

    image_rgb = image_fixture["image_input_rgb"]  # (384,512,3) uint8
    x = _preprocess_to_1008(image_rgb, device="cuda")

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            feats, pos = encoder(x)

    golden = image_fixture["enc_feat_lastlevel"].astype(np.float32)  # (1,256,72,72)
    last = feats[-1]
    assert tuple(last.shape) == golden.shape, (
        f"last-level shape {tuple(last.shape)} != golden {golden.shape}"
    )
    got = last.float().cpu().numpy()
    max_abs = float(np.max(np.abs(got - golden)))
    np.testing.assert_allclose(
        got, golden, atol=1e-2,
        err_msg=f"encoder principal-level parity failed: max|delta|={max_abs:.4g}",
    )


# SPDX-License-Identifier: LicenseRef-SAM
def test_text_parity(image_fixture):
    """The SAM 3 text encoder's output matches the golden ``text_emb`` within atol=1e-2.

    Target key: ``text_emb`` (32, 1, 256) = ``language_features`` = ``text_memory_resized``
    from ``VETextEncoder.forward()`` (transformer output transposed + resizer Linear).
    This is a PURE-TEXT quantity — no vision fusion is involved; the early-fusion encoder
    (``TransformerEncoderFusion``) is a separate module that consumes ``language_features``
    together with vision features in Task 4. ``text_embeds_pre`` (1024) was rejected as the
    target because it is merely the token embedding lookup (pre-transformer), which would be
    a weaker check. Targeting ``text_emb`` exercises the full transformer stack + projection.
    """
    if not CKPT.is_file():
        pytest.skip(f"checkpoint absent: {CKPT}")
    from sam.build_sam import build_sam3_text_encoder  # noqa: F401 — will fail RED

    _determinism()
    text_encoder = build_sam3_text_encoder(ckpt_path=str(CKPT), device="cuda")

    phrase = str(image_fixture["image_phrase"])  # "truck"

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            text_emb = text_encoder.encode([phrase])  # (seq=32, 1, 256)

    golden = image_fixture["text_emb"].astype(np.float32)  # (32, 1, 256) f16 → f32
    assert tuple(text_emb.shape) == golden.shape, (
        f"text_emb shape {tuple(text_emb.shape)} != golden {golden.shape}"
    )
    got = text_emb.float().cpu().numpy()
    max_abs = float(np.max(np.abs(got - golden)))
    np.testing.assert_allclose(
        got, golden, atol=1e-2,
        err_msg=f"text encoder parity failed: max|delta|={max_abs:.4g}",
    )
