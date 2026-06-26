# SPDX-License-Identifier: LicenseRef-SAM
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# Vendored from sam3/model/text_encoder_ve.py @ commit 5dd401d (Phase 1, Task 3).
# Extraneous deps stripped: compile_mode / use_act_checkpoint runtime knobs removed
# (inference-only path; no triton/perflib/timm required).
# LayerScale inlined from sam3/model/model_misc.py (used only when ls_init_value!=None,
# which is not the case in the SAM 3 checkpoint — ls_1/ls_2 are nn.Identity() there).
# class VETextEncoder renamed Sam3TextEncoder; encode() convenience method added.
"""SAM 3 text encoder (Perception-Encoder text tower + resizer)."""

from collections import OrderedDict
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Inlined from sam3/model/model_misc.py::LayerScale
# ---------------------------------------------------------------------------
class LayerScale(nn.Module):
    """Per-channel learnable scale (used when ls_init_value is not None)."""

    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


# ---------------------------------------------------------------------------
# Text transformer blocks
# ---------------------------------------------------------------------------

class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: Optional[float] = None,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)

        self.ln_1 = norm_layer(d_model)
        self.ln_2 = norm_layer(d_model)

        self.ls_1 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )
        self.ls_2 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )

        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(mlp_width, d_model)),
                ]
            )
        )

    def attention(
        self,
        q_x: torch.Tensor,
        k_x: Optional[torch.Tensor] = None,
        v_x: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x
        if attn_mask is not None and not attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.to(q_x.dtype)
        return self.attn(q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(
        self,
        q_x: torch.Tensor,
        k_x: Optional[torch.Tensor] = None,
        v_x: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        k_x = (
            self.ln_1_kv(k_x)
            if hasattr(self, "ln_1_kv") and k_x is not None
            else None
        )
        v_x = (
            self.ln_1_kv(v_x)
            if hasattr(self, "ln_1_kv") and v_x is not None
            else None
        )
        x = q_x + self.ls_1(
            self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        )
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        ls_init_value: Optional[float] = None,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    width,
                    heads,
                    mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                for _ in range(layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for r in self.resblocks:
            x = r(x, attn_mask=attn_mask)
        return x


def text_global_pool(
    x: torch.Tensor,
    text: Optional[torch.Tensor] = None,
    pool_type: str = "argmax",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if pool_type == "first":
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == "last":
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == "argmax":
        assert text is not None
        pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
    else:
        pooled = tokens = x
    return pooled, tokens


class TextTransformer(nn.Module):
    def __init__(
        self,
        context_length: int = 77,
        vocab_size: int = 49408,
        width: int = 512,
        heads: int = 8,
        layers: int = 12,
        mlp_ratio: float = 4.0,
        ls_init_value: Optional[float] = None,
        output_dim: int = 512,
        no_causal_mask: bool = False,
        pool_type: str = "none",
        proj_bias: bool = False,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        output_tokens: bool = False,
        use_ln_post: bool = True,
    ):
        super().__init__()
        assert pool_type in ("first", "last", "argmax", "none")
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pool_type = pool_type

        self.token_embedding = nn.Embedding(self.vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(self.num_pos, width))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.ln_final = norm_layer(width) if use_ln_post else nn.Identity()
        if no_causal_mask:
            self.attn_mask = None
        else:
            self.register_buffer(
                "attn_mask", self._build_causal_mask(), persistent=False
            )
        if proj_bias:
            self.text_projection = nn.Linear(width, output_dim)
        else:
            self.text_projection = nn.Parameter(torch.empty(width, output_dim))

    def _build_causal_mask(self) -> torch.Tensor:
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def forward(
        self, text: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_len = text.shape[1]
        x = self.token_embedding(text)  # (batch, seq, width)

        attn_mask = self.attn_mask if hasattr(self, "attn_mask") else None
        if attn_mask is not None:
            attn_mask = attn_mask[:seq_len, :seq_len]

        x = x + self.positional_embedding[:seq_len]
        x = self.transformer(x, attn_mask=attn_mask)
        x = self.ln_final(x)

        pooled, tokens = text_global_pool(x, text, pool_type=self.pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                pooled = self.text_projection(pooled)
            else:
                pooled = pooled @ self.text_projection
        if self.output_tokens:
            return pooled, tokens
        return pooled


# ---------------------------------------------------------------------------
# SAM 3 text encoder (vendored from VETextEncoder, renamed Sam3TextEncoder)
# ---------------------------------------------------------------------------

class Sam3TextEncoder(nn.Module):
    """SAM 3 text encoder: CLIP-style transformer + d_model projection.

    Vendored from ``sam3/model/text_encoder_ve.py::VETextEncoder``.
    Attribute names are preserved verbatim so the checkpoint subtree
    ``detector.backbone.language_backbone.*`` loads with ``strict=True``
    (295 keys, 0 missing, 0 unexpected).

    Architecture (mirrors ``sam3/model_builder.py``):
        - context_length = 32
        - vocab_size     = 49408  (CLIP BPE)
        - width          = 1024
        - heads          = 16
        - layers         = 24
        - d_model        = 256   (DETR decoder width)

    The forward produces ``(text_attention_mask, text_memory_resized, inputs_embeds)``
    matching upstream ``VETextEncoder.forward()`` exactly.  The convenience method
    ``encode(phrases)`` returns ``text_memory_resized`` (= ``language_features`` /
    ``text_emb`` in the parity fixture, shape ``(seq, batch, d_model)``).
    """

    def __init__(
        self,
        d_model: int,
        tokenizer,
        width: int = 1024,
        heads: int = 16,
        layers: int = 24,
        context_length: int = 32,
        vocab_size: int = 49408,
        use_ln_post: bool = True,
    ):
        super().__init__()
        self.context_length = context_length
        self.use_ln_post = use_ln_post
        # tokenizer is NOT an nn.Module — stored as plain attribute (not in state_dict).
        self.tokenizer = tokenizer

        self.encoder = TextTransformer(
            context_length=self.context_length,
            vocab_size=vocab_size,
            width=width,
            heads=heads,
            layers=layers,
            output_tokens=True,   # forward returns (pooled, tokens)
            use_ln_post=use_ln_post,
        )
        self.resizer = nn.Linear(self.encoder.width, d_model)

    def forward(
        self,
        text: Union[List[str], Tuple[torch.Tensor, torch.Tensor, dict]],
        input_boxes: Optional[List] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Upstream-compatible forward (mirrors VETextEncoder.forward).

        Returns:
            text_attention_mask: (batch, seq) bool — True where token is PAD
            text_memory_resized: (seq, batch, d_model=256) — language_features
            inputs_embeds_T:     (seq, batch, width=1024) — pre-transformer embeds
        """
        if isinstance(text[0], str):
            assert input_boxes is None or len(input_boxes) == 0, "not supported"
            tokenized = self.tokenizer(text, context_length=self.context_length).to(
                device
            )  # (batch, seq)
            text_attention_mask = (tokenized != 0).bool()
            inputs_embeds = self.encoder.token_embedding(tokenized)  # (batch, seq, width)
            _, text_memory = self.encoder(tokenized)                 # (batch, seq, width)
            assert text_memory.shape[1] == inputs_embeds.shape[1]
            text_attention_mask = text_attention_mask.ne(1)
            text_memory = text_memory.transpose(0, 1)                # (seq, batch, width)
            text_memory_resized = self.resizer(text_memory)          # (seq, batch, d_model)
        else:
            text_attention_mask, text_memory_resized, tokenized = text
            inputs_embeds = tokenized["inputs_embeds"]
            assert input_boxes is None or len(input_boxes) == 0, (
                "Can't replace boxes in text if it's already encoded"
            )
        return (
            text_attention_mask,
            text_memory_resized,
            inputs_embeds.transpose(0, 1),
        )

    def encode(self, phrases: List[str]) -> torch.Tensor:
        """Encode text phrases and return ``language_features``.

        This is the clean seam for the DETR detector (Task 4) and for parity testing.
        No vision input is needed — the text tower runs independently.

        Args:
            phrases: list of N strings to encode.

        Returns:
            Tensor of shape ``(context_length, N, d_model)`` = ``language_features``
            (seq-first, matching ``text_emb`` in the parity fixture).
        """
        device = next(self.parameters()).device
        _, text_memory_resized, _ = self.forward(phrases, device=device)
        return text_memory_resized
