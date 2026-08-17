# SPDX-License-Identifier: LicenseRef-SAM
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# Vendored/adapted from facebookresearch/sam3 @ 5dd401d (Phase 1, Task 4): the base
# (per-object) SAM 3 DETR detector that grounds a text phrase into detections. Sources:
#   * sam3/model/encoder.py            -> TransformerEncoderFusion (VL early-fusion encoder)
#   * sam3/model/decoder.py            -> TransformerDecoder/Layer (200-query set decoder
#                                         with box-refine, log-boxRPB, and a presence token)
#   * sam3/model/model_misc.py         -> MultiheadAttention, DotProductScoring, MLP, helpers
#   * sam3/model/geometry_encoders.py  -> SequenceGeometryEncoder (cls token + the box/point
#                                         encoders; the mask encoder stays dormant for strict
#                                         state_dict loading -- no mask_encoder weights exist
#                                         in either checkpoint)
#   * sam3/model/sam3_image.py         -> Sam3Image.forward_grounding (the reference forward
#                                         this module's forward_grounding/detect reproduce)
#
# Stripped (not needed for the base text-only image inference path): the SAM 3.1 multiplex
# decoder, the mask geometry encoder, training (matcher/DAC/aux losses), multi-GPU,
# activation checkpointing, torch.compile, and the FA3/flash-attn fast paths. The hardcoded
# flash-attn is replaced by torch SDPA (flash+math+mem-efficient enabled, auto-selected) --
# numerically equivalent under bf16 autocast within the parity tolerance.
#
# Attribute names are preserved verbatim so the checkpoint subtree (``detector.*`` minus the
# vision/language backbones) loads with ``strict=True``.
"""SAM 3 DETR detector (base, per-object): fusion encoder + set decoder + presence."""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import Tensor

from sam.modeling.position_encoding import PositionEmbeddingSine
from sam.modeling.utils import get_activation_fn, get_clones


def _concat_padded_sequences(seq1, mask1, seq2, mask2):
    """Concatenate two right-padded seq-first sequences -> one right-padded sequence.

    ``seq*`` are ``(L, B, C)``; ``mask*`` are ``(B, L)`` with True at padded positions.
    Faithful port of upstream ``geometry_encoders.concat_padded_sequences``.
    """
    l1, bs, c = seq1.shape
    l2 = seq2.shape[0]
    len1 = (~mask1).sum(dim=-1)
    len2 = (~mask2).sum(dim=-1)
    max_len = l1 + l2
    cat_mask = (
        torch.arange(max_len, device=seq2.device)[None].repeat(bs, 1)
        >= (len1 + len2)[:, None]
    )
    cat_seq = torch.zeros((max_len, bs, c), device=seq2.device, dtype=seq2.dtype)
    cat_seq[:l1] = seq1
    index = torch.arange(l2, device=seq2.device)[:, None].repeat(1, bs) + len1[None]
    cat_seq = cat_seq.scatter(0, index[:, :, None].expand(-1, -1, c), seq2)
    return cat_seq, cat_mask


# =====================================================================================
# Helpers (vendored from sam3/model/model_misc.py and box_ops.py)
# =====================================================================================
def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def get_valid_ratio(mask):
    _, H, W = mask.shape
    valid_H = torch.sum(~mask[:, :, 0], 1)
    valid_W = torch.sum(~mask[:, 0, :], 1)
    valid_ratio_h = valid_H.float() / H
    valid_ratio_w = valid_W.float() / W
    return torch.stack([valid_ratio_w, valid_ratio_h], -1)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def gen_sineembed_for_position(pos_tensor, num_feats=256):
    assert num_feats % 2 == 0
    num_feats = num_feats // 2
    scale = 2 * math.pi
    dim_t = torch.arange(num_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (torch.div(dim_t, 2, rounding_mode="floor")) / num_feats)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)
        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError(f"Unknown pos_tensor shape(-1):{pos_tensor.size(-1)}")
    return pos


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float = 0.0,
        residual: bool = False,
        out_norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if residual and input_dim != output_dim:
            raise ValueError("residual is only supported if input_dim == output_dim")
        self.residual = residual
        assert isinstance(out_norm, nn.Module) or out_norm is None
        self.out_norm = out_norm or nn.Identity()

    def forward(self, x):
        orig_x = x
        for i, layer in enumerate(self.layers):
            x = self.drop(F.relu(layer(x))) if i < self.num_layers - 1 else layer(x)
        if self.residual:
            x = x + orig_x
        return self.out_norm(x)


# =====================================================================================
# Multi-head attention (vendored from model_misc.py; Vanilla dot-product via torch SDPA).
# The upstream forward enables flash+math+mem-efficient SDP and lets SDPA auto-select the
# kernel (flash for unmasked, math/mem-efficient for the additive boxRPB / padding masks).
# The xformers / sparse / deformable / FA3 paths and the never-hit need_weights branch are
# stripped. State_dict keys (in_proj_weight, in_proj_bias, out_proj.{weight,bias}) are
# unchanged, so this is a drop-in for the checkpoint's MultiheadAttentionWrapper modules.
# =====================================================================================
def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
) -> Tensor:
    tgt_len, bsz, embed_dim = query.shape
    head_dim = embed_dim // num_heads

    # in-projection
    if not use_separate_proj_weight:
        q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = F._in_projection(
            query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v
        )

    # prep attention mask -> ensure dim 3
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            attn_mask = attn_mask.to(torch.bool)
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)

    # add bias along batch dimension
    if bias_k is not None and bias_v is not None:
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    # reshape q, k, v for multihead attention and make them batch first
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        key_padding_mask = (
            key_padding_mask.view(bsz, 1, 1, src_len)
            .expand(-1, num_heads, -1, -1)
            .reshape(bsz * num_heads, 1, src_len)
        )
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert bool mask to float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    if not training:
        dropout_p = 0.0

    if attn_mask is not None:
        if attn_mask.size(0) == 1:
            attn_mask = attn_mask.unsqueeze(0)
        else:
            attn_mask = attn_mask.view(bsz, num_heads, -1, src_len)

    q = q.view(bsz, num_heads, tgt_len, head_dim)
    k = k.view(bsz, num_heads, src_len, head_dim)
    v = v.view(bsz, num_heads, src_len, head_dim)

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal=False)

    attn_output = (
        attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
    )
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
    return attn_output


class MultiheadAttention(nn.Module):
    """nn.MultiheadAttention-compatible module backed by the trimmed SDPA forward above."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        **kwargs,  # absorb upstream-only knobs (attn_type, sparsity, use_fa3, ...)
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = nn.Parameter(torch.empty((embed_dim, embed_dim)))
            self.k_proj_weight = nn.Parameter(torch.empty((embed_dim, self.kdim)))
            self.v_proj_weight = nn.Parameter(torch.empty((embed_dim, self.vdim)))
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty((1, 1, embed_dim)))
            self.bias_v = nn.Parameter(torch.empty((1, 1, embed_dim)))
        else:
            self.bias_k = self.bias_v = None
        self.add_zero_attn = add_zero_attn

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, None]:
        assert not need_weights, "need_weights is not supported (inference path)"
        if self.batch_first:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        attn_output = multi_head_attention_forward(
            query,
            key,
            value,
            self.num_heads,
            self.in_proj_weight,
            self.in_proj_bias,
            self.bias_k,
            self.bias_v,
            self.add_zero_attn,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            use_separate_proj_weight=not self._qkv_same_embed_dim,
            q_proj_weight=self.q_proj_weight,
            k_proj_weight=self.k_proj_weight,
            v_proj_weight=self.v_proj_weight,
        )
        if self.batch_first:
            return attn_output.transpose(1, 0), None
        return attn_output, None


# =====================================================================================
# Dot-product scoring (vendored from model_misc.py::DotProductScoring)
# =====================================================================================
class DotProductScoring(nn.Module):
    def __init__(self, d_model, d_proj, prompt_mlp=None, clamp_logits=True, clamp_max_val=12.0):
        super().__init__()
        self.d_proj = d_proj
        self.prompt_mlp = prompt_mlp
        self.prompt_proj = nn.Linear(d_model, d_proj)
        self.hs_proj = nn.Linear(d_model, d_proj)
        self.scale = float(1.0 / np.sqrt(d_proj))
        self.clamp_logits = clamp_logits
        if self.clamp_logits:
            self.clamp_max_val = clamp_max_val

    def mean_pool_text(self, prompt, prompt_mask):
        is_valid = (~prompt_mask).float().permute(1, 0)[..., None]
        num_valid = torch.clamp(torch.sum(is_valid, dim=0), min=1.0)
        return (prompt * is_valid).sum(dim=0) / num_valid

    def forward(self, hs, prompt, prompt_mask):
        # hs: (num_layer, bs, num_query, d_model); prompt: (seq, bs, d_model); mask: (bs, seq)
        assert hs.dim() == 4 and prompt.dim() == 3 and prompt_mask.dim() == 2
        if self.prompt_mlp is not None:
            prompt = self.prompt_mlp(prompt)
        pooled_prompt = self.mean_pool_text(prompt, prompt_mask)
        proj_pooled_prompt = self.prompt_proj(pooled_prompt)
        proj_hs = self.hs_proj(hs)
        scores = torch.matmul(proj_hs, proj_pooled_prompt.unsqueeze(-1))
        scores = scores * self.scale
        if self.clamp_logits:
            scores = scores.clamp(min=-self.clamp_max_val, max=self.clamp_max_val)
        return scores


# =====================================================================================
# VL early-fusion encoder (vendored from encoder.py)
# =====================================================================================
class TransformerEncoderLayer(nn.Module):
    """Self-attention over ``tgt`` then cross-attention to ``memory`` (pre/post-norm)."""

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
        self.activation = get_activation_fn(activation)
        self.pre_norm = pre_norm
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys
        self.layer_idx = None

    def forward_post(
        self, tgt, memory, tgt_mask=None, memory_mask=None,
        tgt_key_padding_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None,
    ):
        q = k = tgt + query_pos if self.pos_enc_at_attn else tgt
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.cross_attn_image(
            query=tgt + query_pos if self.pos_enc_at_cross_attn_queries else tgt,
            key=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self, tgt, memory, dac=False, tgt_mask=None, memory_mask=None,
        tgt_key_padding_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None,
    ):
        assert not dac, "DAC is training-only"
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            query=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            key=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory, dac=False, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        if self.pre_norm:
            return self.forward_pre(
                tgt, memory, dac=dac, tgt_mask=tgt_mask, memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask, pos=pos, query_pos=query_pos,
            )
        return self.forward_post(
            tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask, pos=pos, query_pos=query_pos,
        )


def pool_text_feat(prompt, prompt_mask, pool_with_mask):
    if not pool_with_mask:
        return prompt.mean(dim=0)
    is_valid = (~prompt_mask).float().permute(1, 0)[..., None]
    num_valid = torch.clamp(torch.sum(is_valid, dim=0), min=1.0)
    return (prompt * is_valid).sum(dim=0) / num_valid


class TransformerEncoder(nn.Module):
    def __init__(self, layer, num_layers, d_model, num_feature_levels, frozen=False, **kwargs):
        super().__init__()
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.num_feature_levels = num_feature_levels
        self.level_embed = None
        if num_feature_levels > 1:
            self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        for layer_idx, lay in enumerate(self.layers):
            lay.layer_idx = layer_idx

    def _prepare_multilevel_features(self, srcs, masks, pos_embeds):
        assert len(srcs) == self.num_feature_levels
        src_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes = [], [], [], []
        has_mask = masks is not None and masks[0] is not None
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shapes.append((h, w))
            src = src.flatten(2).transpose(1, 2)
            if has_mask:
                mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            if self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            if has_mask:
                mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1) if has_mask else None
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        if has_mask:
            valid_ratios = torch.stack([get_valid_ratio(m) for m in masks], 1)
        else:
            valid_ratios = torch.ones(
                (src_flatten.shape[0], self.num_feature_levels, 2), device=src_flatten.device
            )
        return (src_flatten, mask_flatten, lvl_pos_embed_flatten, level_start_index,
                valid_ratios, spatial_shapes)

    def forward(self, src, src_key_padding_masks=None, pos=None, prompt=None,
                prompt_key_padding_mask=None):
        (src_flatten, key_padding_masks_flatten, lvl_pos_embed_flatten, level_start_index,
         valid_ratios, spatial_shapes) = self._prepare_multilevel_features(
            src, src_key_padding_masks, pos
        )
        output = src_flatten
        for layer in self.layers:
            output = layer(
                tgt=output,
                memory=prompt,
                memory_key_padding_mask=prompt_key_padding_mask,
                query_pos=lvl_pos_embed_flatten,
                tgt_key_padding_mask=key_padding_masks_flatten,
            )
        return (
            output.transpose(0, 1),
            (key_padding_masks_flatten.transpose(0, 1)
             if key_padding_masks_flatten is not None else None),
            lvl_pos_embed_flatten.transpose(0, 1),
            level_start_index,
            spatial_shapes,
            valid_ratios,
        )


class TransformerEncoderFusion(TransformerEncoder):
    """Fuses text (``prompt``) into the image features via per-layer cross-attention."""

    def __init__(self, layer, num_layers, d_model, num_feature_levels,
                 add_pooled_text_to_img_feat=True, pool_text_with_mask=False, **kwargs):
        super().__init__(layer, num_layers, d_model, num_feature_levels, **kwargs)
        self.add_pooled_text_to_img_feat = add_pooled_text_to_img_feat
        if self.add_pooled_text_to_img_feat:
            self.text_pooling_proj = nn.Linear(d_model, d_model)
        self.pool_text_with_mask = pool_text_with_mask

    def forward(self, src, prompt, src_key_padding_mask=None, src_pos=None,
                prompt_key_padding_mask=None, prompt_pos=None, feat_sizes=None):
        bs = src[0].shape[1]  # seq first
        assert feat_sizes is not None and len(feat_sizes) == len(src)
        if src_key_padding_mask is None:
            src_key_padding_mask = [None] * len(src)
        for i, (h, w) in enumerate(feat_sizes):
            src[i] = src[i].reshape(h, w, bs, -1).permute(2, 3, 0, 1)
            src_pos[i] = src_pos[i].reshape(h, w, bs, -1).permute(2, 3, 0, 1)
            src_key_padding_mask[i] = (
                src_key_padding_mask[i].reshape(h, w, bs).permute(2, 0, 1)
                if src_key_padding_mask[i] is not None else None
            )

        if self.add_pooled_text_to_img_feat:
            pooled_text = pool_text_feat(prompt, prompt_key_padding_mask, self.pool_text_with_mask)
            pooled_text = self.text_pooling_proj(pooled_text)[..., None, None]
            src = [x.add_(pooled_text) for x in src]

        (out, key_padding_masks_flatten, lvl_pos_embed_flatten, level_start_index,
         spatial_shapes, valid_ratios) = super().forward(
            src, src_key_padding_masks=src_key_padding_mask, pos=src_pos,
            prompt=prompt.transpose(0, 1), prompt_key_padding_mask=prompt_key_padding_mask,
        )
        return {
            "memory": out,
            "padding_mask": key_padding_masks_flatten,
            "pos_embed": lvl_pos_embed_flatten,
            "memory_text": prompt,
            "level_start_index": level_start_index,
            "spatial_shapes": spatial_shapes,
            "valid_ratios": valid_ratios,
        }


# =====================================================================================
# DETR set decoder (vendored from decoder.py)
# =====================================================================================
class TransformerDecoderLayer(nn.Module):
    def __init__(self, activation, d_model, dim_feedforward, dropout, cross_attention,
                 n_heads, use_text_cross_attention=False):
        super().__init__()
        self.cross_attn = cross_attention
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm1 = nn.LayerNorm(d_model)
        self.use_text_cross_attention = use_text_cross_attention
        if use_text_cross_attention:
            self.ca_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.catext_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.catext_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model)
        self.layer_idx = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        with torch.amp.autocast(device_type="cuda", enabled=False):
            tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, tgt_query_pos=None, tgt_key_padding_mask=None,
                memory_text=None, text_attention_mask=None, memory=None,
                memory_key_padding_mask=None, memory_pos=None, self_attn_mask=None,
                cross_attn_mask=None, presence_token=None):
        # self attention (+ optional presence token prepended)
        tgt_o2o = tgt
        tgt_query_pos_o2o = tgt_query_pos
        if presence_token is not None:
            tgt_o2o = torch.cat([presence_token, tgt_o2o], dim=0)
            tgt_query_pos_o2o = torch.cat(
                [torch.zeros_like(presence_token), tgt_query_pos_o2o], dim=0
            )
            tgt_query_pos = torch.cat(
                [torch.zeros_like(presence_token), tgt_query_pos], dim=0
            )
        q = k = self.with_pos_embed(tgt_o2o, tgt_query_pos_o2o)
        tgt2 = self.self_attn(q, k, tgt_o2o, attn_mask=self_attn_mask)[0]
        tgt_o2o = tgt_o2o + self.dropout2(tgt2)
        tgt = self.norm2(tgt_o2o)

        if self.use_text_cross_attention:
            tgt2 = self.ca_text(
                self.with_pos_embed(tgt, tgt_query_pos),
                memory_text,
                memory_text,
                key_padding_mask=text_attention_mask,
            )[0]
            tgt = tgt + self.catext_dropout(tgt2)
            tgt = self.catext_norm(tgt)

        if presence_token is not None:
            presence_token_mask = torch.zeros_like(cross_attn_mask[:, :1, :])
            cross_attn_mask = torch.cat([presence_token_mask, cross_attn_mask], dim=1)

        tgt2 = self.cross_attn(
            query=self.with_pos_embed(tgt, tgt_query_pos),
            key=self.with_pos_embed(memory, memory_pos),
            value=memory,
            attn_mask=cross_attn_mask,
            key_padding_mask=(
                memory_key_padding_mask.transpose(0, 1)
                if memory_key_padding_mask is not None else None
            ),
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt = self.forward_ffn(tgt)

        presence_token_out = None
        if presence_token is not None:
            presence_token_out = tgt[:1]
            tgt = tgt[1:]
        return tgt, presence_token_out


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, layer, num_layers, num_queries, return_intermediate,
                 box_refine=False, num_o2m_queries=0, dac=False, boxRPB="none",
                 dac_use_selfatt_ln=True, presence_token=False, clamp_presence_logits=True,
                 clamp_presence_logit_max_val=10.0, use_normed_output_consistently=True,
                 resolution=None, stride=None, frozen=False, interaction_layer=None, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.num_queries = num_queries
        self.dac = dac
        if dac:
            self.num_o2m_queries = num_queries
            tot_num_queries = num_queries
        else:
            self.num_o2m_queries = num_o2m_queries
            tot_num_queries = num_queries + num_o2m_queries
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate
        self.bbox_embed = MLP(d_model, d_model, 4, 3)
        self.query_embed = nn.Embedding(tot_num_queries, d_model)
        self.instance_bbox_embed = None
        self.instance_norm = None
        self.use_normed_output_consistently = use_normed_output_consistently
        self.box_refine = box_refine
        if box_refine:
            self.reference_points = nn.Embedding(num_queries, 4)
        assert boxRPB in ["none", "log", "linear", "both"]
        self.boxRPB = boxRPB
        if boxRPB != "none":
            try:
                nheads = self.layers[0].cross_attn_image.num_heads
            except AttributeError:
                nheads = self.layers[0].cross_attn.num_heads
            n_input = 4 if boxRPB == "both" else 2
            self.boxRPB_embed_x = MLP(n_input, d_model, nheads, 2)
            self.boxRPB_embed_y = MLP(n_input, d_model, nheads, 2)
            self.compilable_cord_cache = None
            self.compilable_stored_size = None
            self.coord_cache = {}
            # resolution/stride define the principal feature grid; the coord cache is filled
            # lazily on the first forward (on the right device) -- identical to precomputing.
        self.presence_token = None
        self.clamp_presence_logits = clamp_presence_logits
        self.clamp_presence_logit_max_val = clamp_presence_logit_max_val
        if presence_token:
            self.presence_token = nn.Embedding(1, d_model)
            self.presence_token_head = MLP(d_model, d_model, 1, 3)
            self.presence_token_out_norm = nn.LayerNorm(d_model)
        self.ref_point_head = MLP(2 * self.d_model, self.d_model, self.d_model, 2)
        self.dac_use_selfatt_ln = dac_use_selfatt_ln
        assert self.return_intermediate, "support return_intermediate only"
        assert self.box_refine, "support box refine only"
        for layer_idx, lay in enumerate(self.layers):
            lay.layer_idx = layer_idx

    @staticmethod
    def _get_coords(H, W, device):
        coords_h = torch.arange(0, H, device=device, dtype=torch.float32) / H
        coords_w = torch.arange(0, W, device=device, dtype=torch.float32) / W
        return coords_h, coords_w

    def _get_rpb_matrix(self, reference_boxes, feat_size):
        H, W = feat_size
        boxes_xyxy = box_cxcywh_to_xyxy(reference_boxes).transpose(0, 1)
        bs, num_queries, _ = boxes_xyxy.shape
        if self.compilable_cord_cache is None or self.compilable_stored_size != (H, W):
            self.compilable_cord_cache = self._get_coords(H, W, reference_boxes.device)
            self.compilable_stored_size = (H, W)
        coords_h, coords_w = self.compilable_cord_cache
        deltas_y = coords_h.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 1:4:2]
        deltas_y = deltas_y.view(bs, num_queries, -1, 2)
        deltas_x = coords_w.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 0:3:2]
        deltas_x = deltas_x.view(bs, num_queries, -1, 2)
        if self.boxRPB in ["log", "both"]:
            deltas_x_log = deltas_x * 8
            deltas_x_log = (
                torch.sign(deltas_x_log) * torch.log2(torch.abs(deltas_x_log) + 1.0)
                / np.log2(8)
            )
            deltas_y_log = deltas_y * 8
            deltas_y_log = (
                torch.sign(deltas_y_log) * torch.log2(torch.abs(deltas_y_log) + 1.0)
                / np.log2(8)
            )
            if self.boxRPB == "log":
                deltas_x = deltas_x_log
                deltas_y = deltas_y_log
            else:
                deltas_x = torch.cat([deltas_x, deltas_x_log], dim=-1)
                deltas_y = torch.cat([deltas_y, deltas_y_log], dim=-1)
        deltas_x = self.boxRPB_embed_x(deltas_x)  # bs, num_queries, W, n_heads
        deltas_y = self.boxRPB_embed_y(deltas_y)  # bs, num_queries, H, n_heads
        B = deltas_y.unsqueeze(3) + deltas_x.unsqueeze(2)  # bs, num_queries, H, W, n_heads
        B = B.flatten(2, 3)  # bs, num_queries, H*W, n_heads
        B = B.permute(0, 3, 1, 2)  # bs, n_heads, num_queries, H*W
        return B.contiguous()

    def forward(self, tgt, memory, memory_key_padding_mask=None, pos=None,
                reference_boxes=None, level_start_index=None, spatial_shapes=None,
                valid_ratios=None, tgt_mask=None, memory_text=None,
                text_attention_mask=None, apply_dac=None):
        apply_dac = apply_dac if apply_dac is not None else self.dac
        assert not apply_dac, "DAC is training-only"
        bs = tgt.shape[1]
        intermediate = []
        intermediate_presence_logits = []
        presence_feats = None

        if reference_boxes is None:
            reference_boxes = self.reference_points.weight.unsqueeze(1)
            reference_boxes = reference_boxes.repeat(1, bs, 1).sigmoid()
        intermediate_ref_boxes = [reference_boxes]

        output = tgt
        presence_out = None
        if self.presence_token is not None:
            presence_out = self.presence_token.weight[None].expand(1, bs, -1)
        box_head = self.bbox_embed
        out_norm = self.norm

        for layer_idx, layer in enumerate(self.layers):
            reference_points_input = (
                reference_boxes[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
            )
            query_sine_embed = gen_sineembed_for_position(
                reference_points_input[:, :, 0, :], self.d_model
            )
            query_pos = self.ref_point_head(query_sine_embed)

            memory_mask = None
            if self.boxRPB != "none" and reference_boxes is not None:
                assert spatial_shapes.shape[0] == 1, "only single scale supported"
                memory_mask = self._get_rpb_matrix(
                    reference_boxes, (spatial_shapes[0, 0], spatial_shapes[0, 1])
                )
                memory_mask = memory_mask.flatten(0, 1)  # (bs*n_heads, nq, H*W)

            output, presence_out = layer(
                tgt=output,
                tgt_query_pos=query_pos,
                tgt_key_padding_mask=None,
                memory_text=memory_text,
                text_attention_mask=text_attention_mask,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_pos=pos,
                self_attn_mask=tgt_mask,
                cross_attn_mask=memory_mask,
                presence_token=presence_out,
            )

            reference_before_sigmoid = inverse_sigmoid(reference_boxes)
            if not self.use_normed_output_consistently:
                delta_unsig = box_head(output)
            else:
                delta_unsig = box_head(out_norm(output))
            outputs_unsig = delta_unsig + reference_before_sigmoid
            new_reference_points = outputs_unsig.sigmoid()
            reference_boxes = new_reference_points.detach()
            if layer_idx != self.num_layers - 1:
                intermediate_ref_boxes.append(new_reference_points)

            intermediate.append(out_norm(output))
            if self.presence_token is not None:
                layer_presence = self.presence_token_head(
                    self.presence_token_out_norm(presence_out)
                ).squeeze(-1)
                if self.clamp_presence_logits:
                    layer_presence.clamp(
                        min=-self.clamp_presence_logit_max_val,
                        max=self.clamp_presence_logit_max_val,
                    )
                intermediate_presence_logits.append(layer_presence)
                presence_feats = presence_out.clone()

        return (
            torch.stack(intermediate),
            torch.stack(intermediate_ref_boxes),
            (torch.stack(intermediate_presence_logits)
             if self.presence_token is not None else None),
            presence_feats,
        )


class TransformerWrapper(nn.Module):
    def __init__(self, encoder, decoder, d_model, pos_enc_at_input_dec=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_queries = decoder.num_queries if decoder is not None else None
        self.pos_enc_at_input_dec = pos_enc_at_input_dec
        self.d_model = d_model


# =====================================================================================
# Geometry encoder (vendored from geometry_encoders.py). For a null geometric prompt
# (text-only), _encode_points / _encode_boxes produce empty sequences, so the only
# contribution is the CLS token, which is image-conditioned by the 3 ``encode``
# cross-attention layers. Box and point prompts are live. The mask submodule is built
# (so the checkpoint subtree loads strictly) but DORMANT -- neither checkpoint carries
# mask_encoder weights, so mask geometry is permanently out.
# =====================================================================================
class Sam3GeometryEncoder(nn.Module):
    def __init__(self, d_model, layer, num_layers, roi_size=7):
        super().__init__()
        self.d_model = d_model
        # dormant prompt encoders (kept for strict state_dict load; not exercised text-only)
        self.label_embed = nn.Embedding(2, d_model)
        self.cls_embed = nn.Embedding(1, d_model)
        self.points_direct_project = nn.Linear(2, d_model)
        self.points_pool_project = nn.Linear(d_model, d_model)
        self.points_pos_enc_project = nn.Linear(d_model, d_model)
        self.boxes_direct_project = nn.Linear(4, d_model)
        self.boxes_pool_project = nn.Conv2d(d_model, d_model, roi_size)
        self.boxes_pos_enc_project = nn.Linear(d_model + 2, d_model)
        self.final_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.img_pre_norm = nn.LayerNorm(d_model)
        self.roi_size = roi_size
        # sinusoidal position encoder for the point/box pos-enc sub-encoders (no params)
        self.pos_enc = PositionEmbeddingSine(num_pos_feats=d_model)
        # active cls-token transformer (cross-attends the tokens to the image features)
        self.encode = get_clones(layer, num_layers)
        self.encode_norm = nn.LayerNorm(d_model)

    def _encode_points(self, points, points_mask, points_labels, img_feats):
        """Point tokens: direct + grid_sample pool + sine pos-enc + label (all summed)."""
        n_points, bs = points.shape[:2]
        emb = self.points_direct_project(points)
        grid = (points.transpose(0, 1).unsqueeze(2) * 2) - 1          # (B,N,1,2) in [-1,1]
        sampled = F.grid_sample(img_feats, grid, align_corners=False)  # (B,C,N,1)
        emb = emb + self.points_pool_project(sampled.squeeze(-1).permute(2, 0, 1))
        x, y = points.unbind(-1)
        enc_x, enc_y = self.pos_enc._encode_xy(x.flatten(), y.flatten())
        enc = torch.cat([enc_x.view(n_points, bs, -1), enc_y.view(n_points, bs, -1)], -1)
        emb = emb + self.points_pos_enc_project(enc)
        return self.label_embed(points_labels.long()) + emb, points_mask

    def _encode_boxes(self, boxes, boxes_mask, boxes_labels, img_feats):
        """Box tokens: direct + roi_align pool + sine pos-enc + label (all summed)."""
        n_boxes, bs = boxes.shape[:2]
        emb = self.boxes_direct_project(boxes)
        h, w = img_feats.shape[-2:]
        boxes_xyxy = box_cxcywh_to_xyxy(boxes)
        scale = torch.tensor([w, h, w, h], device=boxes.device, dtype=boxes_xyxy.dtype)
        boxes_xyxy = boxes_xyxy * scale.view(1, 1, 4)
        sampled = torchvision.ops.roi_align(
            img_feats, boxes_xyxy.float().transpose(0, 1).unbind(0), self.roi_size
        )                                                              # (B*N,C,roi,roi)
        proj = self.boxes_pool_project(sampled).view(bs, n_boxes, self.d_model).transpose(0, 1)
        emb = emb + proj
        cx, cy, bw, bh = boxes.unbind(-1)
        enc = self.pos_enc.encode_boxes(cx.flatten(), cy.flatten(), bw.flatten(), bh.flatten())
        emb = emb + self.boxes_pos_enc_project(enc.view(n_boxes, bs, -1))
        return self.label_embed(boxes_labels.long()) + emb, boxes_mask

    def forward(self, img_feats, img_pos_embeds, bs, img_sizes=None,
                box_coords=None, box_labels=None, point_coords=None, point_labels=None):
        seq_first_img_feats = img_feats[-1]
        seq_first_img_pos_embeds = (
            img_pos_embeds[-1] if img_pos_embeds is not None
            else torch.zeros_like(seq_first_img_feats)
        )
        device = seq_first_img_feats.device
        has_geo = box_coords is not None or point_coords is not None
        if has_geo:
            h, w = img_sizes[-1]
            nchw = self.img_pre_norm(img_feats[-1]).permute(1, 2, 0).view(
                bs, self.d_model, h, w
            )
            if point_coords is not None:
                pm = torch.zeros(bs, point_coords.shape[0], dtype=torch.bool, device=device)
                final_embeds, final_mask = self._encode_points(
                    point_coords, pm, point_labels, nchw
                )
            else:
                final_embeds = seq_first_img_feats.new_zeros(0, bs, self.d_model)
                final_mask = torch.zeros(bs, 0, dtype=torch.bool, device=device)
            if box_coords is not None:
                bm = torch.zeros(bs, box_coords.shape[0], dtype=torch.bool, device=device)
                be, bmask = self._encode_boxes(box_coords, bm, box_labels, nchw)
                final_embeds, final_mask = _concat_padded_sequences(
                    final_embeds, final_mask, be, bmask
                )
            cls = self.cls_embed.weight.view(1, 1, self.d_model).repeat(1, bs, 1)
            cls_mask = torch.zeros(bs, 1, dtype=torch.bool, device=device)
            final_embeds, final_mask = _concat_padded_sequences(
                final_embeds, final_mask, cls, cls_mask
            )
        else:
            # text-only: points/boxes empty -> only the CLS token remains
            final_embeds = self.cls_embed.weight.view(1, 1, self.d_model).repeat(1, bs, 1)
            final_mask = torch.zeros(bs, 1, dtype=torch.bool, device=device)
        final_embeds = self.norm(self.final_proj(final_embeds))
        for lay in self.encode:
            final_embeds = lay(
                tgt=final_embeds,
                memory=seq_first_img_feats,
                tgt_key_padding_mask=final_mask,
                pos=seq_first_img_pos_embeds,
            )
        return self.encode_norm(final_embeds), final_mask


# =====================================================================================
# The detector (vendored from sam3_image.py::Sam3Image, base per-object text-only path)
# =====================================================================================
class Sam3DetrDetector(nn.Module):
    """Grounds a text phrase into per-object detections (boxes/scores/presence/masks).

    ``forward_grounding(feats, pos, text_emb, text_mask)`` reproduces
    ``Sam3Image.forward_grounding`` for the base checkpoint (per-object): encode the prompt
    (text + geometry cls token), run the VL fusion encoder, the 200-query set decoder (with
    a presence token), the dot-product scorer, and the per-object mask head. ``detect`` adds
    the ``Sam3Processor`` post-processing (presence-weighted thresholding, box denorm, mask
    upsample) and returns a :class:`~sam.results.Sam3DetectionResult`.
    """

    def __init__(
        self,
        transformer: TransformerWrapper,
        geometry_encoder: Sam3GeometryEncoder,
        segmentation_head: nn.Module,
        dot_prod_scoring: DotProductScoring,
        num_feature_levels: int = 1,
        o2m_mask_predict: bool = True,
        supervise_joint_box_scores: bool = False,
    ):
        super().__init__()
        self.transformer = transformer
        self.geometry_encoder = geometry_encoder
        self.segmentation_head = segmentation_head
        self.dot_prod_scoring = dot_prod_scoring
        self.hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.o2m_mask_predict = o2m_mask_predict
        # SAM 3.1: fold the presence probability into ``pred_logits`` (joint box-score
        # supervision). Base ``sam3.pt`` leaves them separate (``supervise_joint_box_scores
        # =False``); SAM 3.1 trains the joint score, so the captured ``pred_logits`` already
        # carry presence and ``out_probs = sigmoid(pred_logits)`` (no extra presence mult).
        self.supervise_joint_box_scores = supervise_joint_box_scores

    def _update_scores_and_boxes(self, out, hs, reference_boxes, prompt, prompt_mask,
                                 dec_presence_out):
        num_o2o = hs.size(2)
        out["queries"] = hs[-1][:, :num_o2o]
        outputs_class = self.dot_prod_scoring(hs, prompt, prompt_mask)
        box_head = self.transformer.decoder.bbox_embed
        anchor_box_offsets = box_head(hs)
        reference_boxes_inv_sig = inverse_sigmoid(reference_boxes)
        outputs_coord = (reference_boxes_inv_sig + anchor_box_offsets).sigmoid()
        outputs_boxes_xyxy = box_cxcywh_to_xyxy(outputs_coord)
        if dec_presence_out is not None:
            out["presence_logit_dec"] = dec_presence_out[-1]
        if self.supervise_joint_box_scores:
            # SAM 3.1: pred_logits = inverse_sigmoid(sigmoid(class) * sigmoid(presence)),
            # so sigmoid(pred_logits) is the presence-weighted joint score directly
            # (verbatim from Sam3Image._update_scores_and_boxes).
            assert dec_presence_out is not None, (
                "supervise_joint_box_scores requires the decoder presence token"
            )
            prob_dec_presence_out = dec_presence_out.clone().sigmoid()
            outputs_class = inverse_sigmoid(
                outputs_class.sigmoid() * prob_dec_presence_out.unsqueeze(2)
            ).clamp(min=-10.0, max=10.0)
        out["pred_logits"] = outputs_class[:, :, :num_o2o][-1]
        out["pred_boxes"] = outputs_coord[:, :, :num_o2o][-1]
        out["pred_boxes_xyxy"] = outputs_boxes_xyxy[:, :, :num_o2o][-1]

    def forward_grounding(
        self,
        feats: List[Tensor],
        pos: List[Tensor],
        text_emb: Tensor,
        text_mask: Tensor,
        geo: Optional[dict] = None,
    ) -> Dict[str, Tensor]:
        src = feats[-1]  # principal level (num_feature_levels=1), batch-first (bs,c,H,W)
        bs = src.shape[0]
        h, w = src.shape[-2:]
        img_feat_seq = src.flatten(2).permute(2, 0, 1)  # (H*W, bs, c)
        img_pos_seq = pos[-1].flatten(2).permute(2, 0, 1)

        # --- encode prompt: text features then the (optional) image-conditioned geometry cls
        # token. The base SAM 3 detector cross-attends a learned cls token to the image and
        # appends it to the text prompt; the EfficientSAM3 student detector has NO geometry
        # encoder (geometry_encoder=None) -> the prompt is the text features alone.
        if self.geometry_encoder is not None:
            geo_feats, geo_mask = self.geometry_encoder(
                img_feats=[img_feat_seq], img_pos_embeds=[img_pos_seq], bs=bs,
                img_sizes=[(h, w)], **(geo or {}),
            )
            prompt = torch.cat([text_emb, geo_feats], dim=0)        # (seq+Ngeo, bs, c)
            prompt_mask = torch.cat([text_mask, geo_mask], dim=1)   # (bs, seq+Ngeo)
        else:
            prompt = text_emb                                       # (seq, bs, c)
            prompt_mask = text_mask                                 # (bs, seq)

        # --- VL fusion encoder ---
        enc = self.transformer.encoder(
            src=[img_feat_seq],
            prompt=prompt,
            src_key_padding_mask=None,
            src_pos=[img_pos_seq],
            prompt_key_padding_mask=prompt_mask,
            prompt_pos=torch.zeros_like(prompt),
            feat_sizes=[(h, w)],
        )
        memory = enc["memory"]

        # --- set decoder ---
        out: Dict[str, Tensor] = {}
        query_embed = self.transformer.decoder.query_embed.weight
        tgt = query_embed.unsqueeze(1).repeat(1, bs, 1)
        hs, reference_boxes, dec_presence_out, _ = self.transformer.decoder(
            tgt=tgt,
            memory=memory,
            memory_key_padding_mask=enc["padding_mask"],
            pos=enc["pos_embed"],
            reference_boxes=None,
            level_start_index=enc["level_start_index"],
            spatial_shapes=enc["spatial_shapes"],
            valid_ratios=enc["valid_ratios"],
            tgt_mask=None,
            memory_text=prompt,
            text_attention_mask=prompt_mask,
            apply_dac=False,
        )
        hs = hs.transpose(1, 2)                      # (layers, bs, nq, c)
        reference_boxes = reference_boxes.transpose(1, 2)
        if dec_presence_out is not None:
            dec_presence_out = dec_presence_out.transpose(1, 2)  # (layers, bs, 1)
        self._update_scores_and_boxes(out, hs, reference_boxes, prompt, prompt_mask,
                                      dec_presence_out)

        # --- per-object mask head ---
        image_ids = torch.zeros(bs, dtype=torch.long, device=src.device)
        obj_queries = hs if self.o2m_mask_predict else hs[:, :, : hs.size(2)]
        seg = self.segmentation_head(
            backbone_feats=feats,
            obj_queries=obj_queries,
            image_ids=image_ids,
            encoder_hidden_states=memory,
            prompt=prompt,
            prompt_mask=prompt_mask,
        )
        out["pred_masks"] = seg["pred_masks"][:, : hs.size(2)]
        return out

    @torch.inference_mode()
    def detect(
        self,
        feats: List[Tensor],
        pos: List[Tensor],
        text_emb: Tensor,
        text_mask: Tensor,
        image_hw: Tuple[int, int],
        confidence_threshold: float = 0.5,
        geo: Optional[dict] = None,
    ):
        from sam.results import Sam3DetectionResult

        out = self.forward_grounding(feats, pos, text_emb, text_mask, geo)
        pred_boxes = out["pred_boxes"]              # (P, nq, 4) cxcywh, normalised
        pred_logits = out["pred_logits"]            # (P, nq, 1)
        pred_masks = out["pred_masks"]              # (P, nq, h, w) logits
        presence_logit = out["presence_logit_dec"]  # (P, 1)

        out_probs = pred_logits.sigmoid()
        presence_score = presence_logit.sigmoid().unsqueeze(1)  # (P, 1, 1)
        out_probs = (out_probs * presence_score).squeeze(-1)    # (P, nq)

        keep = out_probs > confidence_threshold
        kept_probs = out_probs[keep]
        kept_masks = pred_masks[keep]
        kept_bbox = pred_boxes[keep]

        boxes = box_cxcywh_to_xyxy(kept_bbox)
        img_h, img_w = int(image_hw[0]), int(image_hw[1])
        scale = torch.tensor([img_w, img_h, img_w, img_h], device=boxes.device, dtype=boxes.dtype)
        boxes = boxes * scale[None, :]

        masks_logits = F.interpolate(
            kept_masks.unsqueeze(1).float(), (img_h, img_w),
            mode="bilinear", align_corners=False,
        ).squeeze(1)  # (N, H, W) logits (binarise at 0 == prob > 0.5)

        # score-weighting uses the bf16 presence (matches Sam3Processor); the reported
        # presence field is upcast to fp32 to match the golden presence (fp32 sigmoid).
        presence = float(presence_logit.float().sigmoid().reshape(-1)[0])
        instance_ids = torch.arange(boxes.shape[0], device=boxes.device)
        return Sam3DetectionResult(
            masks_logits=masks_logits,
            boxes=boxes,
            scores=kept_probs,
            presence=presence,
            instance_ids=instance_ids,
        )
