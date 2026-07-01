# SPDX-License-Identifier: Apache-2.0
"""ONNX-export-aware drop-ins for rotary embedding and attention.

Both modules run normally in eager mode and only matter while a graph is captured for
ONNX export. We **always export the decomposed form** (real rotate-half RoPE + plain
SDPA primitives) and target **opset 18**, because the native opset-23 ONNX ops break on
the ONNX Runtime TensorRT EP (TRT 10.16.x):

* Native ``RotaryEmbedding`` op — the dynamic-shape memory-attention path fails the TRT
  engine build with a myelin SSA error ("producer does not dominate the use"); see
  ``RotaryEmbedding.forward``.
* Native ``Attention`` op — the ORT TensorRT EP does **not** place it on TensorRT; every
  Attention node falls back to the CUDA EP, so a Hiera encoder with 24 attention blocks
  shatters into ~25 TRT subgraphs.

  Note: this op is NOT emitted by ``ScaledDotProductAttention`` below — at opset >= 23
  ``torch.onnx`` auto-fuses *any* ``F.scaled_dot_product_attention`` (incl. the backbones'
  direct calls) into the native ``Attention`` op. The only reliable way to keep attention
  decomposed is to export at **opset < 23**, which is why opset 18 is the target.

  Trade-off: decomposing loses the fused/flash attention kernel, so the decomposed TRT
  engine is actually a bit *slower* than letting the native op run on CUDA — but it's the
  only path that builds reliably. (Refs: NVIDIA/TensorRT #4739, #4705, #4743, #4715.)
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _rope_interleaved(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Real-valued interleaved rotary encoding, identical to the complex
    ``view_as_complex`` formulation. ``x`` is ``(B, H, S, D)``; ``cos``/``sin`` are
    ``(S, D/2)`` and rotate adjacent element pairs."""
    pair = x.float().reshape(*x.shape[:-1], -1, 2)
    xe, xo = pair[..., 0], pair[..., 1]
    cos = cos.view(1, 1, *cos.shape)
    sin = sin.view(1, 1, *sin.shape)
    out = torch.stack((xe * cos - xo * sin, xe * sin + xo * cos), dim=-1)
    return out.flatten(-2).type_as(x)


class ScaledDotProductAttention(nn.Module):
    """SDPA. Always exports as decomposed primitives (``F.scaled_dot_product_attention``
    lowers to MatMul/Softmax/MatMul); the native opset-23 ONNX ``Attention`` op is
    disabled (see ``forward``)."""

    def forward(self, q: Tensor, k: Tensor, v: Tensor, dropout_p: float = 0.0) -> Tensor:
        # Always use F.scaled_dot_product_attention (flash in eager; decomposed
        # MatMul/Softmax/MatMul when exported), at every opset.
        #
        # The native opset-23 ONNX Attention op is not placed on TensorRT by the ORT
        # TensorRT EP (TRT 10.16.x): every Attention node falls back to the CUDA EP, so a
        # Hiera image encoder with 24 attention blocks shatters into ~25 TRT subgraphs
        # (TRT<->CUDA boundary copies + per-engine overhead) — and on the dynamic-shape
        # memory-attention path it also hits the myelin SSA build bug. Same open
        # opset-23-fused-op gap as RotaryEmbedding above.
        # Refs: NVIDIA/TensorRT #4739, #4705, #4743, #4715.
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)


class RotaryEmbedding(nn.Module):
    """Interleaved RoPE applied to ``q`` and the first ``k.size(-2) -
    num_k_exclude_rope`` keys (trailing object-pointer tokens stay un-rotated), then k
    is rejoined. ``freqs_cis`` is the complex frequency table; its real/imag parts are
    the cos/sin caches. Always uses the decomposed real rotate-half form: the native
    ONNX ``RotaryEmbedding`` op currently breaks the TensorRT engine build (see
    ``forward`` for the issue refs)."""

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        freqs_cis: Tensor,
        num_k_exclude_rope: int = 0,
        rope_k_repeat: bool = False,
    ):
        n = k.size(-2) - num_k_exclude_rope
        cos = freqs_cis.real.contiguous()
        sin = freqs_cis.imag.contiguous()
        cos_k, sin_k = cos, sin
        if rope_k_repeat:
            r = n // freqs_cis.size(0)
            cos_k, sin_k = cos.repeat(r, 1), sin.repeat(r, 1)

        k_rope = k[:, :, :n]
        # Always use the decomposed (real rotate-half) RoPE, at every opset.
        #
        # The native opset-23 ONNX RotaryEmbedding op makes TensorRT 10.16.x fail the
        # engine build with a myelin SSA error ("producer does not dominate the use") on
        # the memory-attention block, where the memory length is dynamic: the op is fed
        # shape-data-dependent position_ids (arange(n)) plus a dynamically-repeated
        # (rope_k_repeat) cos/sin cache, which TensorRT's importer lowers into unnamed
        # internal helper layers that myelin cannot resolve. It's an open, unfixed
        # TensorRT bug on our version (not fixed in 10.17 / 10.18 / 11.0); the
        # static-shape image encoder is unaffected.
        # Refs: NVIDIA/TensorRT #4739, #4705, #4743, #4715.
        q = _rope_interleaved(q, cos, sin)
        k_rope = _rope_interleaved(k_rope, cos_k, sin_k)
        return q, torch.cat([k_rope, k[:, :, n:]], dim=-2)
