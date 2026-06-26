# SPDX-License-Identifier: LicenseRef-SAM
# Vendored from facebookresearch/sam3 @ 5dd401d (sam3/model/decoder.py): the SAM 3.1 multiplex
# tracker's DECOUPLED memory-attention transformer -- ``functional_attention`` +
# ``SimpleRoPEAttention`` (projection-free RoPE attention) + ``DecoupledTransformerDecoderLayerv2``
# (a memory-conditioning layer that cross-attends to the stored spatial memory AND, separately,
# the stored image features -- the "decoupled" image path, ``image_cross_attn_*``) +
# ``TransformerEncoderDecoupledCrossAttention``. This is the sam3.1 analogue of the base SAM 3
# per-object memory attention (``decoders/sam3_transformer.py``); kept DISTINCT because the
# checkpoint subtree differs (flat ``self_attn_*`` / ``cross_attn_*`` / ``image_cross_attn_*``
# projections, ``mem_dim == hidden_dim == 256``, 122 keys vs the base 106). flash-attn-3 / the
# forced ``sdpa_kernel(FLASH_ATTENTION)`` context -> plain torch SDPA (auto-select), matching the
# sam3.1 capture regime (task 1b). The RoPE helpers / ``get_clones`` / ``get_activation_fn`` are
# reused from the vendored ``rope.py`` and the shared Apache ``sam.modeling.utils``; training-only
# activation-checkpointing is stripped (inert at inference).
"""SAM 3.1 multiplex decoupled memory attention (RoPE, image + spatial cross-attn)."""

import math
from functools import partial
from typing import Any, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from sam.modeling.decoders.rope import (
    apply_rotary_enc,
    apply_rotary_enc_real,
    compute_axial_cis,
)
from sam.modeling.utils import get_activation_fn, get_clones


def functional_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    dropout: float,
    num_heads: int,
    num_k_exclude_rope: int = 0,
    freqs_cis: Optional[Tensor] = None,
    freqs_cis_real: Optional[Tensor] = None,
    freqs_cis_imag: Optional[Tensor] = None,
    use_rope_real: bool = False,
    rope_k_repeat: bool,
) -> Union[Tensor, tuple]:
    """Multi-head attention with axial RoPE applied to q and the first ``len(k) -
    num_k_exclude_rope`` keys (the trailing ``num_k_exclude_rope`` object-pointer tokens are not
    rotated). flash-attn-3 / forced-flash SDPA stripped -> plain SDPA (auto backend)."""
    b, n, cq = q.shape
    _, m, ck = k.shape
    _, _, cv = v.shape
    if b > 1:
        assert k.shape[0] == v.shape[0] == b
    else:
        assert k.shape[0] == b == 1, f"{q.shape=} {k.shape=} {v.shape=}"
    assert v.shape[1] == m

    q = q.reshape(b, n, num_heads, cq // num_heads).transpose(1, 2)
    k = k.reshape(b, m, num_heads, ck // num_heads).transpose(1, 2)
    v = v.reshape(v.shape[0], m, num_heads, cv // num_heads).transpose(1, 2)

    if freqs_cis is not None:
        num_k_rope = k.size(-2) - num_k_exclude_rope
        if use_rope_real:
            q, k[:, :, :num_k_rope] = apply_rotary_enc_real(
                q, k[:, :, :num_k_rope],
                freqs_cis_real=freqs_cis_real, freqs_cis_imag=freqs_cis_imag,
                repeat_freqs_k=rope_k_repeat,
            )
        else:
            q, k[:, :, :num_k_rope] = apply_rotary_enc(
                q, k[:, :, :num_k_rope], freqs_cis, repeat_freqs_k=rope_k_repeat,
            )

    out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout)
    out = out.transpose(1, 2)  # B * n * n_heads * (cv // num_heads)
    return out.reshape(b, n, cv)


class SimpleRoPEAttention(nn.Module):
    """Attention with rotary position encoding. "Simple" because it does NOT perform q/k/v/out
    projections (the owning layer holds the projections)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout_p: float,
        rope_theta=10000.0,
        rope_k_repeat=False,
        feat_sizes=(64, 64),
        use_rope_real: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.compute_cis = partial(
            compute_axial_cis, dim=d_model // num_heads, theta=rope_theta
        )
        device = torch.device("cuda") if torch.cuda.is_available() else None
        self.freqs_cis = self.compute_cis(
            end_x=feat_sizes[0], end_y=feat_sizes[1], device=device
        )
        self.use_rope_real = use_rope_real
        if self.use_rope_real:
            self.freqs_cis_real = self.freqs_cis.real
            self.freqs_cis_imag = self.freqs_cis.imag
        self.rope_k_repeat = rope_k_repeat

    def forward(self, q: Tensor, k: Tensor, v: Tensor, num_k_exclude_rope: int = 0) -> Tensor:
        w = h = math.sqrt(q.shape[-2])
        self.freqs_cis = self.freqs_cis.to(q.device)
        if self.freqs_cis.shape[0] != q.shape[-2]:
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h, device=q.device)
            if self.use_rope_real:
                self.freqs_cis_real = self.freqs_cis.real
                self.freqs_cis_imag = self.freqs_cis.imag
        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat

        dropout_p = self.dropout_p if self.training else 0.0
        return functional_attention(
            q, k, v,
            dropout=dropout_p, num_heads=self.num_heads,
            num_k_exclude_rope=num_k_exclude_rope, freqs_cis=self.freqs_cis,
            freqs_cis_real=self.freqs_cis_real if self.use_rope_real else None,
            freqs_cis_imag=self.freqs_cis_imag if self.use_rope_real else None,
            use_rope_real=self.use_rope_real, rope_k_repeat=self.rope_k_repeat,
        )


class DecoupledTransformerDecoderLayerv2(nn.Module):
    def __init__(
        self,
        *,
        activation: str,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        pre_norm: bool,
        cross_attention_first: bool = False,
        self_attention_rope: SimpleRoPEAttention,
        cross_attention_rope: SimpleRoPEAttention,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout

        self.self_attn_q_proj = nn.Linear(d_model, d_model)
        self.self_attn_k_proj = nn.Linear(d_model, d_model)
        self.self_attn_v_proj = nn.Linear(d_model, d_model)
        self.self_attn_out_proj = nn.Linear(d_model, d_model)

        self.cross_attn_q_proj = nn.Linear(d_model, d_model)
        self.cross_attn_k_proj = nn.Linear(d_model, d_model)
        self.cross_attn_v_proj = nn.Linear(d_model, d_model)
        self.cross_attn_out_proj = nn.Linear(d_model, d_model)

        self.image_cross_attn_q_proj = nn.Linear(d_model, d_model)
        self.image_cross_attn_k_proj = nn.Linear(d_model, d_model)

        self.self_attention_rope = self_attention_rope
        self.cross_attention_rope = cross_attention_rope

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation_str = activation
        self.activation = get_activation_fn(activation)
        self.pre_norm = pre_norm

        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys
        self.cross_attention_first = cross_attention_first

    def _forward_sa(self, tgt, query_pos):
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        q = self.self_attn_q_proj(q)
        k = self.self_attn_k_proj(k)
        v = self.self_attn_v_proj(tgt2)
        out = self.self_attention_rope(q, k, v)
        tgt2 = self.self_attn_out_proj(out)
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(
        self, *, image, tgt, memory_image, memory, query_pos, memory_image_pos,
        num_k_exclude_rope=0,
    ):
        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attention_rope, SimpleRoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}

        tgt2 = self.norm2(tgt)
        q = self.image_cross_attn_q_proj(image) + self.cross_attn_q_proj(tgt2)
        if self.pos_enc_at_cross_attn_queries:
            q = q + query_pos
        k = self.image_cross_attn_k_proj(memory_image) + self.cross_attn_k_proj(memory)
        if self.pos_enc_at_cross_attn_keys:
            k = k + memory_image_pos
        v = self.cross_attn_v_proj(memory)

        out = self.cross_attention_rope(q, k, v, **kwds)
        tgt2 = self.cross_attn_out_proj(out)
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward_pre(
        self, *, image, tgt, memory_image, memory,
        image_pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None,
        memory_image_pos: Optional[Tensor] = None, memory_pos: Optional[Tensor] = None,
        num_k_exclude_rope: int = 0,
    ):
        if self.cross_attention_first:
            tgt = self._forward_ca(
                image=image, tgt=tgt, memory_image=memory_image, memory=memory,
                query_pos=query_pos, memory_image_pos=memory_image_pos,
                num_k_exclude_rope=num_k_exclude_rope,
            )
            tgt = self._forward_sa(tgt, query_pos)
        else:
            tgt = self._forward_sa(tgt, query_pos)
            tgt = self._forward_ca(
                image=image, tgt=tgt, memory_image=memory_image, memory=memory,
                query_pos=query_pos, memory_image_pos=memory_image_pos,
                num_k_exclude_rope=num_k_exclude_rope,
            )

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return image, tgt

    def forward(self, *args: Any, **kwds: Any) -> torch.Tensor:
        if self.pre_norm:
            return self.forward_pre(*args, **kwds)
        raise NotImplementedError


class TransformerEncoderDecoupledCrossAttention(nn.Module):
    def __init__(
        self, d_model: int, frozen: bool, pos_enc_at_input: bool, layer, num_layers: int,
        use_act_checkpoint: bool = False, batch_first: bool = False,
        use_image_in_output: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.use_act_checkpoint = use_act_checkpoint
        self.use_image_in_output = use_image_in_output
        if frozen:
            for p in self.parameters():
                p.requires_grad_(False)
        self.batch_first = batch_first

    def forward(
        self,
        image: Tensor,
        src: Tensor,
        memory_image: Tensor,
        memory: Tensor,
        image_pos: Optional[Tensor] = None,
        src_pos: Optional[Tensor] = None,
        memory_image_pos: Optional[Tensor] = None,
        memory_pos: Optional[Tensor] = None,
        num_obj_ptr_tokens: int = 0,
    ):
        assert src.shape[1] == memory.shape[1], "Batch size must match for src and memory"
        assert image.shape[1] == memory_image.shape[1], (
            "Batch size must match for image and memory_image"
        )

        output = src
        if self.pos_enc_at_input and src_pos is not None:
            output = output + 0.1 * src_pos

        if self.batch_first:
            output = output.transpose(0, 1)
            src_pos = src_pos.transpose(0, 1)
            image = image.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_pos = memory_pos.transpose(0, 1)
            memory_image = memory_image.transpose(0, 1)
            memory_image_pos = memory_image_pos.transpose(0, 1)

        if memory_image.shape[1] != memory.shape[1]:
            # Pad memory_image with zeros to accommodate the object pointers
            assert (memory.shape[1] - memory_image.shape[1]) == num_obj_ptr_tokens, (
                f"{memory.shape[1]} - {memory_image.shape[1]} != {num_obj_ptr_tokens}"
            )
            memory_image = torch.cat([
                memory_image,
                torch.zeros(
                    (memory_image.shape[0], num_obj_ptr_tokens) + memory_image.shape[2:],
                    dtype=memory_image.dtype, device=memory_image.device,
                ),
            ], dim=1)
            if memory_image_pos is not None:
                assert (memory_pos.shape[1] - memory_image_pos.shape[1]) == num_obj_ptr_tokens, (
                    f"{memory_pos.shape[1]} - {memory_image_pos.shape[1]} != {num_obj_ptr_tokens}"
                )
                memory_image_pos = torch.cat([
                    memory_image_pos, memory_pos[0:1, -num_obj_ptr_tokens:],
                ], dim=1)

        for layer in self.layers:
            image, output = layer(
                image=image, tgt=output, memory_image=memory_image, memory=memory,
                image_pos=image_pos, query_pos=src_pos,
                memory_image_pos=memory_image_pos, memory_pos=memory_pos,
                num_k_exclude_rope=num_obj_ptr_tokens,
            )

        if self.use_image_in_output:
            normed_output = self.norm(output + image)
        else:
            normed_output = self.norm(output)

        if self.batch_first:
            normed_output = normed_output.transpose(0, 1)
            src_pos = src_pos.transpose(0, 1)

        return {"memory": normed_output, "pos_embed": src_pos}
