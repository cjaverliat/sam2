# SPDX-License-Identifier: LicenseRef-SAM
# Vendored from facebookresearch/sam3 @ 5dd401d (sam3/model/multiplex_mask_decoder.py): the SAM
# 3.1 ``MultiplexMaskDecoder`` -- a SAM-style two-way-transformer mask decoder whose token
# embeddings are sized for ``multiplex_count`` objects, run at ``batch = num_buckets`` so all K
# objects in a bucket are decoded JOINTLY in one transformer pass (spec §8). Kept under a DISTINCT
# filename from the shared (SAM 2, Apache) ``decoders/mask_decoder.py`` and the base SAM 3
# per-object ``decoders/sam3_mask_decoder.py``. ``LayerNorm2d`` is reused from the shared Apache
# ``sam.modeling.utils``; ``TwoWayTransformer`` is the vendored SAM 3 RoPE-free two-way decoder
# (``decoders/sam3_transformer.py``). Nothing else stripped (no Triton / flash / training).
"""SAM 3.1 multiplex mask decoder (K-token joint decode at batch = num_buckets)."""

from typing import List, Optional, Type

import torch
from torch import nn
from torch.nn import functional as F

from sam.modeling.utils import LayerNorm2d


class MultiplexMaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        multiplex_count: int,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid: bool = False,
        dynamic_multimask_via_stability=False,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
        decode_mask_with_shared_tokens: bool = False,
        decode_mask_attribute_with_shared_tokens: bool = False,
        multimask_outputs_only: bool = False,
    ) -> None:
        """Predict masks for ``multiplex_count`` objects from a single (bucket) feature map.

        ``multiplex_count`` objects share one feature map and decode jointly (their tokens
        self-attend). The total number of masks is ``(num_multimask_outputs + 1) *
        multiplex_count`` unless ``multimask_outputs_only`` (then ``num_multimask_outputs``).
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.multiplex_count = multiplex_count
        self.num_multimask_outputs = num_multimask_outputs
        self.multimask_outputs_only = multimask_outputs_only
        self.decode_mask_with_shared_tokens = decode_mask_with_shared_tokens
        self.decode_mask_attribute_with_shared_tokens = (
            decode_mask_attribute_with_shared_tokens
        )

        if self.decode_mask_with_shared_tokens:
            assert multimask_outputs_only, (
                "multimask_outputs_only must be True if decode_mask_with_shared_tokens"
            )

        if self.multimask_outputs_only:
            self.num_mask_output_per_object = num_multimask_outputs
        else:
            self.num_mask_output_per_object = num_multimask_outputs + 1

        if self.decode_mask_with_shared_tokens:
            self.num_mask_tokens = multiplex_count
        else:
            self.num_mask_tokens = multiplex_count * self.num_mask_output_per_object

        self.pred_obj_scores = pred_obj_scores
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

        if not self.decode_mask_attribute_with_shared_tokens:
            self.iou_token = nn.Embedding(multiplex_count, transformer_dim)
            if self.pred_obj_scores:
                self.obj_score_token = nn.Embedding(multiplex_count, transformer_dim)

        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(
                transformer_dim, transformer_dim // 8, kernel_size=1, stride=1
            )
            self.conv_s1 = nn.Conv2d(
                transformer_dim, transformer_dim // 4, kernel_size=1, stride=1
            )

        if self.num_multimask_outputs == 0:
            self.output_hypernetworks_mlp = MLP(
                transformer_dim, transformer_dim, transformer_dim // 8, 3
            )
        else:
            self.output_hypernetworks_mlps = nn.ModuleList(
                [
                    MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                    for _ in range(self.num_mask_output_per_object)
                ]
            )

        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            (
                1
                if (
                    self.decode_mask_attribute_with_shared_tokens
                    and not self.decode_mask_with_shared_tokens
                )
                else self.num_mask_output_per_object
            ),
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
        )

        if self.pred_obj_scores:
            self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        multimask_output: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
        extra_per_object_embeddings: Optional[torch.Tensor] = None,
    ) -> dict:
        """Predict masks given (bucket) image embeddings and positional encoding.

        ``extra_per_object_embeddings`` (``num_buckets, multiplex_count, C``) is added to the
        mask tokens. Returns a dict with ``masks`` / ``iou_pred`` / ``sam_tokens_out`` /
        ``object_score_logits`` in multiplex space ``(num_buckets, multiplex_count, ...)``.
        """
        if self.num_multimask_outputs <= 0:
            assert not multimask_output, (
                f"multimask_output must be False with {self.num_multimask_outputs=}"
            )
        if self.multimask_outputs_only:
            assert multimask_output, (
                f"multimask_output must be True with {self.multimask_outputs_only=}"
            )

        out = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            high_res_features=high_res_features,
            extra_per_object_embeddings=extra_per_object_embeddings,
        )

        masks = out["masks"]
        iou_pred = out["iou_pred"]
        mask_tokens_out = out["mask_tokens_out"]

        if multimask_output:
            if not self.multimask_outputs_only:
                masks = masks[:, :, 1:, :, :]
                iou_pred = iou_pred[:, :, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks = masks[:, :, 0:1, :, :]
            iou_pred = iou_pred[:, :, 0:1]

        if multimask_output and self.use_multimask_token_for_obj_ptr:
            if self.multimask_outputs_only:
                sam_tokens_out = mask_tokens_out
            else:
                sam_tokens_out = mask_tokens_out[:, :, 1:]
        else:
            sam_tokens_out = mask_tokens_out[:, :, 0:1]

        del out["mask_tokens_out"]
        out["masks"] = masks
        out["iou_pred"] = iou_pred
        out["sam_tokens_out"] = sam_tokens_out

        if multimask_output:
            assert masks.shape[2] == self.num_mask_output_per_object
            assert iou_pred.shape[2] == self.num_mask_output_per_object
            if self.use_multimask_token_for_obj_ptr:
                if self.decode_mask_with_shared_tokens:
                    assert sam_tokens_out.shape[2] == 1
                else:
                    assert sam_tokens_out.shape[2] == self.num_mask_output_per_object
        else:
            assert masks.shape[2] == 1
            assert iou_pred.shape[2] == 1
            assert sam_tokens_out.shape[2] == 1

        return out

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        high_res_features: Optional[List[torch.Tensor]] = None,
        extra_per_object_embeddings: Optional[torch.Tensor] = None,
    ) -> dict:
        """Predicts masks. See ``forward`` for more details."""
        B = image_embeddings.shape[0]
        token_list = []
        if self.pred_obj_scores and not self.decode_mask_attribute_with_shared_tokens:
            token_list.append(self.obj_score_token.weight)
        if not self.decode_mask_attribute_with_shared_tokens:
            token_list.append(self.iou_token.weight)

        tokens = torch.cat(token_list, dim=0)
        tokens = tokens.unsqueeze(0).expand(B, -1, -1)

        if extra_per_object_embeddings is not None:
            mask_tokens = self.mask_tokens.weight.view(
                1, self.multiplex_count, self.num_mask_output_per_object, -1
            ).expand(B, -1, -1, -1)
            mask_tokens = mask_tokens + extra_per_object_embeddings.unsqueeze(2)
            mask_tokens = mask_tokens.flatten(1, 2)
        else:
            mask_tokens = self.mask_tokens.weight.unsqueeze(0).expand(B, -1, -1)

        tokens = torch.cat([tokens, mask_tokens], dim=1)

        src = image_embeddings
        assert image_pe.size(0) == 1, (
            "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        )
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)

        if self.decode_mask_attribute_with_shared_tokens:
            assert hs.shape[1] == self.num_mask_tokens
            iou_token_out = mask_tokens_out = hs[:, 0 : self.num_mask_tokens]
            if self.pred_obj_scores:
                obj_score_token_out = mask_tokens_out
        else:
            s = 0
            if self.pred_obj_scores:
                obj_score_token_out = hs[:, s : s + self.multiplex_count, :]
                s += self.multiplex_count
            iou_token_out = hs[:, s : s + self.multiplex_count, :]
            s += self.multiplex_count
            mask_tokens_out = hs[:, s : s + self.num_mask_tokens, :]
            assert hs.shape[1] == s + self.num_mask_tokens

        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        if self.decode_mask_with_shared_tokens:
            mask_tokens_out = mask_tokens_out.view(B, self.multiplex_count, 1, -1)
        else:
            mask_tokens_out = mask_tokens_out.view(
                B, self.multiplex_count, self.num_mask_output_per_object, -1
            )
        if self.num_multimask_outputs == 0:
            hyper_in = self.output_hypernetworks_mlp(mask_tokens_out[:, :, 0, :]).unsqueeze(2)
        else:
            hyper_in_list: List[torch.Tensor] = []
            for i in range(self.num_mask_output_per_object):
                if self.decode_mask_with_shared_tokens:
                    hyper_in_list.append(
                        self.output_hypernetworks_mlps[i](mask_tokens_out[:, :, 0, :])
                    )
                else:
                    hyper_in_list.append(
                        self.output_hypernetworks_mlps[i](mask_tokens_out[:, :, i, :])
                    )
            hyper_in = torch.stack(hyper_in_list, dim=2)

        b, c, h, w = upscaled_embedding.shape
        masks = torch.bmm(
            hyper_in.flatten(1, 2), upscaled_embedding.view(b, c, h * w)
        ).view(b, self.multiplex_count, self.num_mask_output_per_object, h, w)

        iou_pred = self.iou_prediction_head(iou_token_out).view(
            b, self.multiplex_count, self.num_mask_output_per_object
        )

        if self.pred_obj_scores:
            if (
                self.decode_mask_attribute_with_shared_tokens
                and not self.decode_mask_with_shared_tokens
            ):
                object_score_logits = (
                    self.pred_obj_score_head(obj_score_token_out)
                    .view(b, self.multiplex_count, self.num_mask_output_per_object)
                    .sum(-1, keepdim=True)
                )
            else:
                object_score_logits = self.pred_obj_score_head(obj_score_token_out)
        else:
            object_score_logits = 10.0 * iou_pred.new_ones(
                iou_pred.shape[0], iou_pred.shape[1]
            )

        return {
            "masks": masks,
            "iou_pred": iou_pred,
            "mask_tokens_out": mask_tokens_out,
            "object_score_logits": object_score_logits,
        }

    def _get_stability_scores(self, mask_logits):
        mask_logits = mask_logits.flatten(-2)
        stability_delta = self.dynamic_multimask_stability_delta
        area_i = torch.sum(mask_logits > stability_delta, dim=-1).float()
        area_u = torch.sum(mask_logits > -stability_delta, dim=-1).float()
        stability_scores = torch.where(area_u > 0, area_i / area_u, 1.0)
        return stability_scores

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        B, M = all_mask_logits.shape[:2]
        all_mask_logits = all_mask_logits.flatten(0, 1)
        all_iou_scores = all_iou_scores.flatten(0, 1)

        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1)
        batch_inds = torch.arange(multimask_iou_scores.size(0), device=all_iou_scores.device)
        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds].unsqueeze(1)
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds].unsqueeze(1)

        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh

        mask_logits_out = torch.where(
            is_stable[..., None, None].expand_as(singlemask_logits),
            singlemask_logits, best_multimask_logits,
        )
        iou_scores_out = torch.where(
            is_stable.expand_as(singlemask_iou_scores),
            singlemask_iou_scores, best_multimask_iou_scores,
        )
        mask_logits_out = mask_logits_out.unflatten(0, (B, M))
        iou_scores_out = iou_scores_out.unflatten(0, (B, M))
        return mask_logits_out, iou_scores_out


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py
class MLP(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
