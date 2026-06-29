# SPDX-License-Identifier: LicenseRef-SAM
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# Vendored from efficientsam3_reference/sam3/sam3/model/text_encoder_student.py
# (class TextStudentEncoder) @ d063e00. Adapted for the sam repo:
#   * tokenizer is Sam3Tokenizer (not SimpleTokenizer) — stored as plain attribute
#   * forward returns same 3-tuple as Sam3TextEncoder
#   * submodule names preserved for strict checkpoint load (language_backbone.*)
"""MobileCLIP text encoder for EfficientSAM3 (Phase A5)."""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from sam.modeling.text.mobile_clip import MobileCLIPTextTransformer


# ---------------------------------------------------------------------------
# Variant configs (mirrors _create_student_text_encoder in efficientsam3_reference
# model_builder.py, § MobileCLIP-S0 block)
# ---------------------------------------------------------------------------

_MOBILECLIP_CONFIGS = {
    "MobileCLIP-S0": {
        "dim": 512,
        "n_transformer_layers": 4,
        "n_heads_per_layer": 8,
        "model_name": "mct",
        "ffn_multiplier_per_layer": 4.0,
        "norm_layer": "layer_norm_fp32",
        "causal_masking": False,
        "vocab_size": 49408,
        "embed_dropout": 0.0,
        "no_scale_embedding": False,
        "no_pos_embedding": False,
    },
    "MobileCLIP-S1": {
        "dim": 512,
        "n_transformer_layers": 12,
        "n_heads_per_layer": 8,
        "model_name": "base",
        "ffn_multiplier_per_layer": 4.0,
        "norm_layer": "layer_norm_fp32",
        "causal_masking": False,
        "vocab_size": 49408,
        "embed_dropout": 0.0,
        "no_scale_embedding": False,
        "no_pos_embedding": False,
    },
    "MobileCLIP-B": {
        "dim": 512,
        "n_transformer_layers": 12,
        "n_heads_per_layer": 8,
        "model_name": "base",
        "ffn_multiplier_per_layer": 4.0,
        "norm_layer": "layer_norm_fp32",
        "causal_masking": True,
        "vocab_size": 49408,
        "embed_dropout": 0.0,
        "no_scale_embedding": False,
        "no_pos_embedding": False,
    },
}


def _get_mobileclip_cfg(variant: str, context_length: int) -> dict:
    """Return the MobileCLIP config dict for *variant* at *context_length*."""
    if variant not in _MOBILECLIP_CONFIGS:
        raise ValueError(
            f"Unknown MobileCLIP variant {variant!r}. "
            f"Available: {list(_MOBILECLIP_CONFIGS)}"
        )
    cfg = dict(_MOBILECLIP_CONFIGS[variant])
    cfg["context_length"] = context_length
    return cfg


# ---------------------------------------------------------------------------
# MobileClipTextEncoder
# ---------------------------------------------------------------------------

class MobileClipTextEncoder(nn.Module):
    """EfficientSAM3 text encoder: MobileCLIP transformer + d_model projector.

    Submodule names are preserved verbatim so the checkpoint subtree
    ``detector.backbone.language_backbone.*`` loads with ``strict=True``:

        encoder.embedding_layer.weight              (49408, 512)
        encoder.positional_embedding.pos_embed.pos_embed  (1, 1, ctx, 512)
        encoder.transformer.*                       (104 keys for MobileCLIP-S0)
        encoder.final_layer_norm.*
        encoder.projection_layer                    (512, 512)
        projector.weight                            (output_dim, 512)
        projector.bias                              (output_dim,)

    The ``forward`` produces the same 3-tuple as ``Sam3TextEncoder.forward``:
        (text_attention_mask, text_memory_resized, inputs_embeds_T)
    where mask is (batch, seq) bool (True = padding), memory is
    (seq, batch, output_dim), and embeds is (seq, batch, encoder_dim).

    Args:
        tokenizer: ``Sam3Tokenizer`` instance (plain attribute, not nn.Module).
        variant: MobileCLIP architecture name (default ``"MobileCLIP-S0"``).
        context_length: Sequence length — must match the checkpoint's positional
            embedding table (16 for the EfficientSAM3 checkpoint).
        output_dim: Projection output size (default 256 = DETR d_model).
    """

    def __init__(
        self,
        tokenizer,
        variant: str = "MobileCLIP-S0",
        context_length: int = 16,
        output_dim: int = 256,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        # tokenizer is NOT an nn.Module — stored as plain attribute (same as Sam3TextEncoder).
        self.tokenizer = tokenizer

        cfg = _get_mobileclip_cfg(variant, context_length)
        encoder_dim: int = cfg["dim"]

        # Checkpoint key prefix: encoder.*
        self.encoder = MobileCLIPTextTransformer(cfg=cfg, projection_dim=encoder_dim)

        # Checkpoint keys: projector.weight, projector.bias
        self.projector = nn.Linear(encoder_dim, output_dim)

    def set_context_length(self, context_length: int) -> None:
        """Resize positional embeddings to a new context length (post-load truncation)."""
        self.context_length = context_length
        if hasattr(self.encoder, "resize_pos_embed"):
            self.encoder.resize_pos_embed(context_length)

    def forward(
        self,
        text: Union[List[str], Tuple[Tensor, Tensor, dict]],
        input_boxes: Optional[List] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Upstream-compatible forward (mirrors Sam3TextEncoder.forward / TextStudentEncoder.forward).

        Returns:
            text_attention_mask: (batch, seq) bool — True where token is PAD
            text_memory_resized: (seq, batch, output_dim) — language_features
            inputs_embeds_T:     (seq, batch, encoder_dim) — pre-transformer embeds
        """
        if isinstance(text[0], str):
            assert input_boxes is None or len(input_boxes) == 0, "not supported"
            tokenized = self.tokenizer(text, context_length=self.context_length).to(device)  # (batch, seq)

            # 1. Get input embeddings via the MobileCLIP embedding layer
            input_embeds = self.encoder.forward_embedding(tokenized)  # (batch, seq, encoder_dim)

            # 2. Run MobileCLIP transformer (pass embeddings to skip redundant lookup)
            text_memory = self.encoder(
                input_embeds, return_all_tokens=True, input_is_embeddings=True
            )  # (batch, seq, encoder_dim)

            # 3. Post-project to output_dim (d_model)
            text_memory = self.projector(text_memory)  # (batch, seq, output_dim)

            # 4. Attention mask: True = padding (same convention as Sam3TextEncoder)
            text_attention_mask = (tokenized != 0).bool().ne(1)  # (batch, seq)

            return (
                text_attention_mask,
                text_memory.transpose(0, 1),       # (seq, batch, output_dim)
                input_embeds.transpose(0, 1),      # (seq, batch, encoder_dim)
            )
        else:
            text_attention_mask, text_memory_resized, tokenized_dict = text
            inputs_embeds = tokenized_dict["inputs_embeds"]
            assert input_boxes is None or len(input_boxes) == 0, (
                "Can't replace boxes in text if it's already encoded"
            )
            return (
                text_attention_mask,
                text_memory_resized,
                inputs_embeds.transpose(0, 1),
            )

    def encode(self, phrases: List[str]) -> Tensor:
        """Encode text phrases and return language_features.

        Args:
            phrases: list of N strings to encode.

        Returns:
            Tensor of shape ``(context_length, N, output_dim)`` = ``language_features``
            (seq-first, matching Sam3TextEncoder.encode).
        """
        device = next(self.parameters()).device
        _, text_memory_resized, _ = self.forward(phrases, device=device)
        return text_memory_resized
