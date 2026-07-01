# SPDX-License-Identifier: LicenseRef-SAM
# Vendored from facebookresearch/sam3 @ 5dd401d. This is the SAM 3 tracker's attention stack,
# kept under a DISTINCT name from the shared (SAM 2, Apache) ``decoders/transformer.py`` so the
# two coexist (the SAM 3 RoPE variant is not a drop-in for the SAM 2 one):
#   * ``Attention`` / ``RoPEAttention`` / ``TwoWayTransformer`` / ``TwoWayAttentionBlock`` /
#     ``MLPBlock`` / ``LayerNorm2d``  <- sam3/sam/transformer.py + sam3/sam/common.py
#   * ``TransformerWrapper``                                       <- sam3/model/model_misc.py
#   * ``TransformerDecoderLayerv1`` / ``TransformerDecoderLayerv2`` /
#     ``TransformerEncoderCrossAttention``                         <- sam3/model/decoder.py
# Stripped: flash-attn-3 (``use_fa3`` -> torch SDPA, like Task 4), training-only ``dac`` /
# dropout-scheduling branches (kept as inert asserts), activation checkpointing (direct calls),
# and ``torch.compile`` hooks. ``get_clones`` / ``get_activation_fn`` are reused from the shared
# Apache ``sam.modeling.utils``.
"""SAM 3 tracker attention: RoPE memory attention + the SAM mask-decoder two-way transformer."""

import math
from functools import partial
from typing import Any, Optional, Tuple, Type

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from sam.modeling.decoders.rope import (
    apply_rotary_enc,
    apply_rotary_enc_real,
    compute_axial_cis,
)
from sam.modeling.utils import get_activation_fn, get_clones


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def _sdpa(q: Tensor, k: Tensor, v: Tensor, dropout_p: float) -> Tensor:
    # Mirror upstream: enable all backends and let the dispatcher pick (flash-attn-3 path
    # removed; FA3 is absent in this env). Under the parity determinism regime SDPA selects
    # the same backend as the golden capture.
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)


class Attention(nn.Module):
    """An attention layer that allows downscaling the embedding after q/k/v projection."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        kv_in_dim: int = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.dropout_p = dropout

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        dropout_p = self.dropout_p if self.training else 0.0
        out = _sdpa(q, k, v, dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out


class RoPEAttention(Attention):
    """Attention with rotary position encoding (SAM 3 tracker memory attention)."""

    def __init__(
        self,
        *args,
        rope_theta=10000.0,
        rope_k_repeat=False,
        feat_sizes=(64, 64),  # [w, h] for stride 16 feats at 1024 resolution
        use_rope_real=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_rope_real = use_rope_real
        self.compute_cis = partial(
            compute_axial_cis, dim=self.internal_dim // self.num_heads, theta=rope_theta
        )
        device = torch.device("cuda") if torch.cuda.is_available() else None
        self.freqs_cis = self.compute_cis(
            end_x=feat_sizes[0], end_y=feat_sizes[1], device=device
        )
        if self.use_rope_real:
            self.freqs_cis_real = self.freqs_cis.real
            self.freqs_cis_imag = self.freqs_cis.imag
        self.rope_k_repeat = rope_k_repeat

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, num_k_exclude_rope: int = 0
    ) -> Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Apply rotary position encoding
        w = h = math.sqrt(q.shape[-2])
        if self.freqs_cis.shape[0] != q.shape[-2]:
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h, device=q.device)
            self.freqs_cis_real = self.freqs_cis.real
            self.freqs_cis_imag = self.freqs_cis.imag
        self.freqs_cis = self.freqs_cis.to(q.device)
        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat

        num_k_rope = k.size(-2) - num_k_exclude_rope
        if self.use_rope_real:
            self.freqs_cis_real = self.freqs_cis_real.to(q.device)
            self.freqs_cis_imag = self.freqs_cis_imag.to(q.device)
            q, k[:, :, :num_k_rope] = apply_rotary_enc_real(
                q,
                k[:, :, :num_k_rope],
                freqs_cis_real=self.freqs_cis_real,
                freqs_cis_imag=self.freqs_cis_imag,
                repeat_freqs_k=self.rope_k_repeat,
            )
        else:
            q, k[:, :, :num_k_rope] = apply_rotary_enc(
                q,
                k[:, :, :num_k_rope],
                self.freqs_cis,
                repeat_freqs_k=self.rope_k_repeat,
            )

        dropout_p = self.dropout_p if self.training else 0.0
        out = _sdpa(q, k, v, dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """A transformer decoder that attends to an input image using queries whose positional
        embedding is supplied (SAM mask decoder)."""
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        queries = point_embedding
        keys = image_embedding

        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class TransformerWrapper(nn.Module):
    """Thin container holding the tracker's memory-attention ``encoder`` (``decoder`` is None
    for the per-object tracker). ``d_model`` is read by ``Sam3Tracker.hidden_dim``."""

    def __init__(self, encoder, decoder, d_model: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_queries = decoder.num_queries if decoder is not None else None
        self._reset_parameters()
        self.d_model = d_model

    def _reset_parameters(self):
        for n, p in self.named_parameters():
            if p.dim() > 1:
                if (
                    "box_embed" not in n
                    and "query_embed" not in n
                    and "reference_points" not in n
                ):
                    nn.init.xavier_uniform_(p)


class TransformerDecoderLayerv1(nn.Module):
    def __init__(
        self,
        activation: str,
        cross_attention: nn.Module,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        pre_norm: bool,
        self_attention: nn.Module,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attention
        self.cross_attn_image = cross_attention

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


class TransformerDecoderLayerv2(TransformerDecoderLayerv1):
    """SAM 3 tracker memory-attention layer (pre-norm self-attn + RoPE cross-attn + FFN)."""

    def __init__(self, cross_attention_first=False, *args: Any, **kwds: Any):
        super().__init__(*args, **kwds)
        self.cross_attention_first = cross_attention_first

    def _forward_sa(self, tgt, query_pos):
        # Self-Attention
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2)
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0):
        if self.cross_attn_image is None:
            return tgt

        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}

        # Cross-Attention
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory,
            **kwds,
        )
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        dac: bool,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        attn_bias: Optional[Tensor] = None,
        num_k_exclude_rope: int = 0,
    ):
        assert dac is False
        assert tgt_mask is None
        assert memory_mask is None
        assert tgt_key_padding_mask is None
        assert memory_key_padding_mask is None
        assert attn_bias is None

        if self.cross_attention_first:
            tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
            tgt = self._forward_sa(tgt, query_pos)
        else:
            tgt = self._forward_sa(tgt, query_pos)
            tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)

        # MLP
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, *args: Any, **kwds: Any) -> torch.Tensor:
        if self.pre_norm:
            return self.forward_pre(*args, **kwds)
        raise NotImplementedError


class TransformerEncoderCrossAttention(nn.Module):
    """SAM 3 tracker memory attention: ``num_layers`` self+cross-attention layers fusing the
    current frame features (``src``) with the memory prompt (``prompt``)."""

    def __init__(
        self,
        d_model: int,
        frozen: bool,
        pos_enc_at_input: bool,
        layer,
        num_layers: int,
        use_act_checkpoint: bool = False,
        batch_first: bool = False,
        remove_cross_attention_layers: Optional[list] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.use_act_checkpoint = use_act_checkpoint

        if frozen:
            for p in self.parameters():
                p.requires_grad_(False)

        self.batch_first = batch_first

        self.remove_cross_attention_layers = [False] * self.num_layers
        if remove_cross_attention_layers is not None:
            for i in remove_cross_attention_layers:
                self.remove_cross_attention_layers[i] = True
        assert len(self.remove_cross_attention_layers) == len(self.layers)

        for i, remove_cross_attention in enumerate(self.remove_cross_attention_layers):
            if remove_cross_attention:
                self.layers[i].cross_attn_image = None
                self.layers[i].norm2 = None
                self.layers[i].dropout2 = None

    def forward(
        self,
        src,  # self-attention inputs
        prompt,  # cross-attention inputs
        src_mask: Optional[Tensor] = None,
        prompt_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        prompt_key_padding_mask: Optional[Tensor] = None,
        src_pos: Optional[Tensor] = None,
        prompt_pos: Optional[Tensor] = None,
        feat_sizes: Optional[list] = None,
        num_obj_ptr_tokens: int = 0,
    ):
        if isinstance(src, list):
            assert isinstance(src_key_padding_mask, list) and isinstance(src_pos, list)
            assert len(src) == len(src_key_padding_mask) == len(src_pos) == 1
            src, src_key_padding_mask, src_pos = (
                src[0],
                src_key_padding_mask[0],
                src_pos[0],
            )

        assert src.shape[1] == prompt.shape[1], (
            "Batch size must be the same for src and prompt"
        )

        output = src

        if self.pos_enc_at_input and src_pos is not None:
            output = output + 0.1 * src_pos

        if self.batch_first:
            output = output.transpose(0, 1)
            src_pos = src_pos.transpose(0, 1)
            prompt = prompt.transpose(0, 1)
            prompt_pos = prompt_pos.transpose(0, 1)

        for layer in self.layers:
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}

            output = layer(
                tgt=output,
                memory=prompt,
                tgt_mask=src_mask,
                memory_mask=prompt_mask,
                tgt_key_padding_mask=src_key_padding_mask,
                memory_key_padding_mask=prompt_key_padding_mask,
                pos=prompt_pos,
                query_pos=src_pos,
                dac=False,
                attn_bias=None,
                **kwds,
            )
            normed_output = self.norm(output)

        if self.batch_first:
            normed_output = normed_output.transpose(0, 1)
            src_pos = src_pos.transpose(0, 1)

        return {
            "memory": normed_output,
            "pos_embed": src_pos,
            "padding_mask": src_key_padding_mask,
        }
