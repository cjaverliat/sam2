# SPDX-License-Identifier: LicenseRef-SAM
# Vendored from facebookresearch/sam3 @ 5dd401d (sam3/model/video_tracking_multiplex.py::
# VideoTrackingMultiplex): the SAM 3.1 MULTIPLEX per-frame tracker. It SUBCLASSES the base SAM 3
# per-object ``Sam3Tracker`` to reuse the bit-identical helpers (``frame_filter`` / ``cal_mem_score``
# / ``_apply_non_overlapping_constraints`` / ``_get_tpos_enc`` / ``device``) and SWAPS the SAM head
# path for the multiplex one (spec §8): the per-object tensors are ``mux``ed into ``num_buckets``
# buckets, the ``MultiplexMaskDecoder`` decodes all ``K`` slots of a bucket JOINTLY at
# ``batch = num_buckets``, and the outputs are ``demux``ed back to per-object BEFORE storage -- so
# the bank / streaming loop keep seeing per-object data. The memory attention is the DECOUPLED
# variant (image + spatial cross-attn, mem_dim == hidden_dim == 256). Despite both deriving from the
# SAM-2 tracker lineage, the sam3.1 ``tracker.*`` subtree is NOT bit-shared with base (different
# memory-attention transformer, 16-channel mask encoder, MultiplexMaskDecoder, separate interactive
# heads, linear no-obj-ptr, output-suppression + per-object conditioning embeddings); only
# ``obj_ptr_proj`` is bit-identical. Stripped (vs upstream): training (loss/teacher-forcing/dropout/
# DDP/correction-point sampling), ``torch.compile`` / ``_maybe_clone`` / activation checkpointing,
# CPU-offload + past-output trimming, the on-the-fly per-frame backbone, and the dynamic
# add/remove-object demo plumbing. flash-attn-3 -> torch SDPA lives in the vendored attention.
"""SAM 3.1 multiplex tracker (``Sam3MultiplexTracker``: mux -> joint K-token decode -> demux)."""

import torch
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from sam.modeling.decoders.multiplex_mask_decoder import MLP, MultiplexMaskDecoder
from sam.modeling.decoders.sam3_mask_decoder import MaskDecoder
from sam.modeling.decoders.sam3_prompt_encoder import (
    PositionEmbeddingRandom,
    PromptEncoder,
)
from sam.modeling.decoders.sam3_transformer import TwoWayTransformer
from sam.modeling.multiplex import MultiplexState
from sam.modeling.tracking.sam3_tracker import NO_OBJ_SCORE, Sam3Tracker
from sam.modeling.tracking.sam3_tracker_utils import select_closest_cond_frames


class Sam3MultiplexTracker(Sam3Tracker):
    """SAM 3.1 multiplex tracker network (joint decode of ``K`` objects per bucket).

    Built weights-only (no vision backbone): callers pass the propagation
    (``sam2_backbone_out``) feature pyramid into ``track_step``. Memory attention + the SAM
    mask decode run at ``batch = num_buckets``; outputs are demuxed to per-object before storage.
    """

    def __init__(
        self,
        transformer,
        maskmem_backbone,
        multiplex_controller,
        num_maskmem=7,
        image_size=1008,
        backbone_stride=14,
        max_cond_frames_in_attn=4,
        keep_first_cond_frame=False,
        multimask_output_in_sam=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        multimask_output_for_tracking=True,
        num_multimask_outputs=3,
        use_multimask_token_for_obj_ptr=True,
        memory_temporal_stride_for_eval=1,
        non_overlap_masks_for_mem_enc=False,
        max_obj_ptrs_in_encoder=16,
        pred_obj_scores=True,
        pred_obj_scores_mlp=True,
        sam_mask_decoder_extra_args=None,
        use_memory_selection=False,
        mf_threshold=0.01,
        # multiplex-specific knobs (defaults match sam3.1_multiplex.pt)
        apply_sigmoid_to_mask_logits_for_mem_enc=True,
        sigmoid_scale_for_mem_enc=2.0,
        sigmoid_bias_for_mem_enc=-1.0,
        add_output_suppression_embeddings=True,
        no_obj_embed_spatial=True,
        condition_as_mask_input=True,
        condition_as_mask_input_fg=1.0,
        condition_as_mask_input_bg=0.0,
        use_maskmem_tpos_v2=True,
        save_image_features=True,
        object_score_logit_threshold=0.0,
    ):
        torch.nn.Module.__init__(self)

        # the multiplex SAM mask decoder cannot use dynamic_multimask_via_stability
        interactive_sam_mask_decoder_extra_args = (
            dict(sam_mask_decoder_extra_args) if sam_mask_decoder_extra_args else None
        )
        if sam_mask_decoder_extra_args is not None and sam_mask_decoder_extra_args.get(
            "dynamic_multimask_via_stability", False
        ):
            sam_mask_decoder_extra_args = dict(sam_mask_decoder_extra_args)
            sam_mask_decoder_extra_args["dynamic_multimask_via_stability"] = False

        self.num_feature_levels = 3
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder

        self.multiplex_controller = multiplex_controller
        self.multiplex_count = multiplex_controller.multiplex_count

        assert transformer.decoder is None, "transformer should be encoder-only"
        self.transformer = transformer
        self.hidden_dim = transformer.d_model

        self.maskmem_backbone = maskmem_backbone
        self.mem_dim = self.hidden_dim
        if hasattr(self.maskmem_backbone, "out_proj") and hasattr(
            self.maskmem_backbone.out_proj, "weight"
        ):
            mem_dim = self.maskmem_backbone.out_proj.weight.shape[0]
            assert mem_dim == self.hidden_dim, "no compression of memory embeddings (mem_dim==C)"
        self.num_maskmem = num_maskmem

        # a conv to downsample the GT mask prompt to stride 4 for the interactive obj-ptr head
        self.interactive_mask_downsample = torch.nn.Conv2d(1, 1, kernel_size=4, stride=4)

        # temporal encoding of the spatial memories
        self.use_maskmem_tpos_v2 = use_maskmem_tpos_v2
        self.maskmem_tpos_enc = torch.nn.Parameter(
            torch.zeros(num_maskmem, 1, 1, self.mem_dim)
        )
        trunc_normal_(self.maskmem_tpos_enc, std=0.02)

        # a single token to indicate no memory embedding (the interactive / first-frame path)
        self.interactivity_no_mem_embed = torch.nn.Parameter(
            torch.zeros(1, 1, self.hidden_dim)
        )
        trunc_normal_(self.interactivity_no_mem_embed, std=0.02)

        self.apply_sigmoid_to_mask_logits_for_mem_enc = (
            apply_sigmoid_to_mask_logits_for_mem_enc
        )
        self.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc
        self.binarize_mask_from_pts_for_mem_enc = False
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
        self.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval
        self.object_score_logit_threshold = object_score_logit_threshold

        self.multimask_output_in_sam = multimask_output_in_sam
        self.multimask_min_pt_num = multimask_min_pt_num
        self.multimask_max_pt_num = multimask_max_pt_num
        self.multimask_output_for_tracking = multimask_output_for_tracking
        self.num_multimask_outputs = num_multimask_outputs
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.low_res_mask_size = self.image_size // self.backbone_stride * 4
        self.input_mask_size = self.low_res_mask_size * 4

        # object-presence / no-object handling (use_no_obj_ptr + use_linear_no_obj_ptr +
        # fixed_no_obj_ptr; pred_obj_scores -> obj-score token in the decoder)
        self.pred_obj_scores = pred_obj_scores
        self.pred_obj_scores_mlp = pred_obj_scores_mlp
        self.use_obj_ptrs_in_encoder = True
        self.use_no_obj_ptr = True
        self.use_linear_no_obj_ptr = True
        self.fixed_no_obj_ptr = True
        self.proj_tpos_enc_in_obj_ptrs = True
        self.add_tpos_enc_to_obj_ptrs = True
        self.use_signed_tpos_enc_to_obj_ptrs = False
        self.only_obj_ptrs_in_the_past_for_eval = False
        self.sincos_tpos_enc = True
        self.no_obj_ptr_linear = torch.nn.Linear(self.hidden_dim, self.hidden_dim)

        self.no_obj_embed_spatial = None
        if no_obj_embed_spatial:
            self.no_obj_embed_spatial = torch.nn.Parameter(
                torch.zeros(self.multiplex_count, self.hidden_dim)
            )
            trunc_normal_(self.no_obj_embed_spatial, std=0.02)

        self.add_output_suppression_embeddings = add_output_suppression_embeddings
        if add_output_suppression_embeddings:
            self.output_valid_embed = torch.nn.Parameter(
                torch.zeros(self.multiplex_count, self.hidden_dim)
            )
            self.output_invalid_embed = torch.nn.Parameter(
                torch.zeros(self.multiplex_count, self.hidden_dim)
            )
            trunc_normal_(self.output_valid_embed, std=0.02)
            trunc_normal_(self.output_invalid_embed, std=0.02)

        self.add_object_conditional_embeddings = False
        self.condition_as_mask_input = condition_as_mask_input
        self.condition_as_mask_input_fg = condition_as_mask_input_fg
        self.condition_as_mask_input_bg = condition_as_mask_input_bg

        self.save_image_features = save_image_features

        self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args
        self.interactive_sam_mask_decoder_extra_args = (
            interactive_sam_mask_decoder_extra_args
        )
        self._build_sam_heads()

        self.max_cond_frames_in_attn = max_cond_frames_in_attn
        self.keep_first_cond_frame = keep_first_cond_frame
        self.use_memory_selection = use_memory_selection
        self.mf_threshold = mf_threshold

    # --- head construction ------------------------------------------------------------------
    def _build_sam_heads(self):
        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_image_embedding_size = self.image_size // self.backbone_stride

        self.image_pe_layer = PositionEmbeddingRandom(self.hidden_dim // 2)

        self.interactive_sam_prompt_encoder = PromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(self.sam_image_embedding_size, self.sam_image_embedding_size),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.interactive_sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2, embedding_dim=self.sam_prompt_embed_dim, mlp_dim=2048, num_heads=8
            ),
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=True,
            iou_prediction_use_sigmoid=False,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
            **(self.interactive_sam_mask_decoder_extra_args or {}),
        )

        self.sam_mask_decoder = MultiplexMaskDecoder(
            multiplex_count=self.multiplex_count,
            num_multimask_outputs=self.num_multimask_outputs,
            transformer=TwoWayTransformer(
                depth=2, embedding_dim=self.hidden_dim, mlp_dim=2048, num_heads=8
            ),
            transformer_dim=self.hidden_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=True,
            iou_prediction_use_sigmoid=False,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
            decode_mask_with_shared_tokens=False,
            decode_mask_attribute_with_shared_tokens=False,
            multimask_outputs_only=self.num_multimask_outputs > 0 and self.multimask_output_in_sam,
            **(self.sam_mask_decoder_extra_args or {}),
        )

        # object-pointer projections (MLP, separate interactive + propagation)
        self.obj_ptr_proj = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, 3)
        self.interactive_obj_ptr_proj = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, 3)
        # projection on the temporal positional encoding in object pointers
        self.obj_ptr_tpos_proj = torch.nn.Linear(self.hidden_dim, self.mem_dim)

    def get_propagation_dense_pe(self) -> torch.Tensor:
        return self.image_pe_layer(
            (self.sam_image_embedding_size, self.sam_image_embedding_size)
        ).unsqueeze(0)

    def _get_interactive_pix_mem(self, features, feat_sizes):
        pix_feat_with_mem = features[-1] + self.interactivity_no_mem_embed
        B = features[-1].size(1)
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        return pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)

    def _use_multimask(self, is_init_cond_frame, point_inputs):
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
        return (
            self.multimask_output_in_sam
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
            and self.num_multimask_outputs > 0
        )

    # --- SAM head (mux propagation path + per-object interactive path) -----------------------
    def _forward_sam_heads(
        self,
        backbone_features,
        *,
        point_inputs=None,
        mask_inputs=None,
        interactive_high_res_features=None,
        propagation_high_res_features=None,
        multimask_output=False,
        multiplex_state: MultiplexState = None,
        objects_to_interact=None,
    ):
        device = backbone_features.device
        assert backbone_features.size(1) == self.hidden_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size

        is_interactive = point_inputs is not None or mask_inputs is not None

        if is_interactive:
            assert interactive_high_res_features is not None
            assert objects_to_interact is not None
            if point_inputs is not None:
                sam_point_coords = point_inputs["point_coords"]
                sam_point_labels = point_inputs["point_labels"]
            else:
                sam_point_coords = torch.zeros(mask_inputs.shape[0], 1, 2, device=device)
                sam_point_labels = -torch.ones(
                    mask_inputs.shape[0], 1, dtype=torch.int32, device=device
                )
            if mask_inputs is not None:
                if mask_inputs.shape[-2:] != self.interactive_sam_prompt_encoder.mask_input_size:
                    sam_mask_prompt = F.interpolate(
                        mask_inputs.float(),
                        size=self.interactive_sam_prompt_encoder.mask_input_size,
                        align_corners=False, mode="bilinear", antialias=True,
                    )
                else:
                    sam_mask_prompt = mask_inputs
            else:
                sam_mask_prompt = None

            sparse_embeddings, dense_embeddings = self.interactive_sam_prompt_encoder(
                points=(sam_point_coords, sam_point_labels), boxes=None, masks=sam_mask_prompt,
            )
            image_pe = self.interactive_sam_prompt_encoder.get_dense_pe()
            (
                low_res_multimasks, ious, sam_output_tokens, object_score_logits,
            ) = self.interactive_sam_mask_decoder(
                image_embeddings=backbone_features,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
                repeat_image=True,
                high_res_features=interactive_high_res_features,
            )
        else:
            assert propagation_high_res_features is not None
            assert multiplex_state is not None
            if self.add_output_suppression_embeddings:
                output_valid_embed = self.output_valid_embed.unsqueeze(0)
                output_invalid_embed = self.output_invalid_embed.unsqueeze(0)
                valid_object_mask = multiplex_state.get_valid_object_mask().unsqueeze(-1).to(
                    output_valid_embed.dtype
                )
                output_merged_embed = (
                    valid_object_mask * output_valid_embed
                    + (1 - valid_object_mask) * output_invalid_embed
                )
            else:
                output_merged_embed = None

            image_pe = self.get_propagation_dense_pe()
            out = self.sam_mask_decoder(
                image_embeddings=backbone_features,
                image_pe=image_pe,
                high_res_features=propagation_high_res_features,
                multimask_output=multimask_output,
                extra_per_object_embeddings=output_merged_embed,
            )
            low_res_multimasks = out["masks"]
            ious = out["iou_pred"]
            sam_output_tokens = out["sam_tokens_out"]
            object_score_logits = out["object_score_logits"]

            low_res_multimasks = multiplex_state.demux(low_res_multimasks)
            ious = multiplex_state.demux(ious)
            object_score_logits = multiplex_state.demux(object_score_logits)
            sam_output_tokens = multiplex_state.demux(sam_output_tokens)

        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > self.object_score_logit_threshold
            low_res_multimasks = torch.where(
                is_obj_appearing[:, None, None], low_res_multimasks, NO_OBJ_SCORE,
            )

        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks, size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False,
        )

        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(ious.shape[0], device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        if is_interactive:
            obj_ptr = self.interactive_obj_ptr_proj(sam_output_token)
        else:
            obj_ptr = self.obj_ptr_proj(sam_output_token)

        if self.pred_obj_scores and self.use_no_obj_ptr:
            lambda_is_obj_appearing = is_obj_appearing.float()
            # use_linear_no_obj_ptr: blend with a linear projection of the obj_ptr
            obj_ptr = lambda_is_obj_appearing * obj_ptr + (
                1 - lambda_is_obj_appearing
            ) * self.no_obj_ptr_linear(obj_ptr)

        outputs = {
            "low_res_multimasks": low_res_multimasks,
            "high_res_multimasks": high_res_multimasks,
            "ious": ious,
            "low_res_masks": low_res_masks,
            "high_res_masks": high_res_masks,
            "object_score_logits": object_score_logits,
            "obj_ptr": obj_ptr,
        }
        return outputs

    def _use_mask_as_output(
        self, backbone_features, high_res_features, mask_inputs, multiplex_state,
        objects_in_mask=None,
    ):
        if objects_in_mask is None:
            objects_in_mask = list(range(multiplex_state.total_valid_entries))
        out_scale, out_bias = 20.0, -10.0
        mask_inputs_float = mask_inputs.to(backbone_features.dtype)
        assert mask_inputs.shape[0] == len(objects_in_mask)
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(
            high_res_masks,
            size=(high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4),
            align_corners=False, mode="bilinear", antialias=True,
        )
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1, dtype=backbone_features.dtype)

        sam_outputs = self._forward_sam_heads(
            backbone_features=backbone_features,
            mask_inputs=self.interactive_mask_downsample(mask_inputs_float),
            interactive_high_res_features=high_res_features,
            objects_to_interact=objects_in_mask,
            multiplex_state=multiplex_state,
        )
        obj_ptr = sam_outputs["obj_ptr"]

        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)[..., None]
        lambda_is_obj_appearing = is_obj_appearing.float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        if self.pred_obj_scores and self.use_no_obj_ptr:
            obj_ptr = lambda_is_obj_appearing * obj_ptr + (
                1 - lambda_is_obj_appearing
            ) * self.no_obj_ptr_linear(obj_ptr)

        return {
            "low_res_multimasks": low_res_masks,
            "high_res_multimasks": high_res_masks,
            "ious": ious,
            "low_res_masks": low_res_masks,
            "high_res_masks": high_res_masks,
            "object_score_logits": object_score_logits,
            "obj_ptr": obj_ptr,
        }

    # --- memory conditioning (decoupled image + spatial cross-attn) -------------------------
    def _prepare_memory_conditioned_features(
        self, *, frame_idx, is_init_cond_frame, current_vision_feats, current_vision_masks,
        current_vision_pos_embeds, feat_sizes, output_dict, num_frames,
        track_in_reverse=False, multiplex_state: MultiplexState = None,
    ):
        B = multiplex_state.num_buckets
        vision_feat = current_vision_feats[-1].expand(-1, B, -1)
        vision_mask = (
            current_vision_masks[-1].expand(-1, B, -1)
            if current_vision_masks[-1] is not None else None
        )
        vision_pos_embed = current_vision_pos_embeds[-1].expand(-1, B, -1)

        C = self.hidden_dim
        H, W = feat_sizes[-1]
        device = current_vision_feats[-1].device
        if self.num_maskmem == 0:
            return vision_feat.permute(1, 2, 0).view(B, C, H, W)

        assert not is_init_cond_frame, "init cond frame must use _use_mask_as_output"
        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1

        to_cat_prompt, to_cat_prompt_pos_embed = [], []
        to_cat_image_feat, to_cat_image_pos_embed = [], []
        assert len(output_dict["cond_frame_outputs"]) > 0
        cond_outputs = output_dict["cond_frame_outputs"]
        selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
            frame_idx, cond_outputs, self.max_cond_frames_in_attn,
            keep_first_cond_frame=self.keep_first_cond_frame,
        )
        t_pos_and_prevs = [
            ((frame_idx - t) * tpos_sign_mul, out, True)
            for t, out in selected_cond_outputs.items()
        ]
        r = self.memory_temporal_stride_for_eval
        if self.use_memory_selection:
            valid_indices = self.frame_filter(
                output_dict, track_in_reverse, frame_idx, num_frames, r
            )
        for t_pos in range(1, self.num_maskmem):
            t_rel = self.num_maskmem - t_pos
            if self.use_memory_selection:
                if t_rel > len(valid_indices):
                    continue
                prev_frame_idx = valid_indices[-t_rel]
            else:
                if t_rel == 1:
                    prev_frame_idx = frame_idx + (t_rel if track_in_reverse else -t_rel)
                elif not track_in_reverse:
                    prev_frame_idx = ((frame_idx - 2) // r) * r - (t_rel - 2) * r
                else:
                    prev_frame_idx = -(-(frame_idx + 2) // r) * r + (t_rel - 2) * r
            out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
            if out is None:
                out = unselected_cond_outputs.get(prev_frame_idx, None)
            t_pos_and_prevs.append((t_pos, out, False))

        for t_pos, prev, is_selected_cond_frame in t_pos_and_prevs:
            if prev is None:
                continue
            feats = prev.get("maskmem_features")
            if feats is None:
                continue
            feats = feats.to(device, non_blocking=True)
            if feats.dim() == 5:
                feats = multiplex_state.demux(feats).contiguous()
            if feats.shape[0] == 0:
                continue
            to_cat_prompt.append(feats.flatten(2).permute(2, 0, 1))

            maskmem_pos_list = prev.get("maskmem_pos_enc")
            if not maskmem_pos_list:
                continue
            maskmem_enc = maskmem_pos_list[-1]
            if maskmem_enc is None:
                continue
            maskmem_enc = maskmem_enc.to(device, non_blocking=True)
            if maskmem_enc.dim() == 5:
                maskmem_enc = multiplex_state.demux(maskmem_enc).contiguous()
            maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)

            if self.use_maskmem_tpos_v2:
                if t_pos <= 0 or t_pos >= self.num_maskmem:
                    tpos_enc = self.maskmem_tpos_enc[self.num_maskmem - 1]
                else:
                    tpos_enc = self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
            else:
                t = t_pos if not is_selected_cond_frame else 0
                tpos_enc = self.maskmem_tpos_enc[self.num_maskmem - t - 1]
            maskmem_enc = maskmem_enc + tpos_enc

            if self.save_image_features:
                image_feat = prev["image_features"].to(device)
                image_pos_embed = prev["image_pos_enc"].to(device) + tpos_enc
                to_cat_image_feat.append(image_feat)
                to_cat_image_pos_embed.append(image_pos_embed)

            to_cat_prompt_pos_embed.append(maskmem_enc)

        # past object pointers
        max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
        if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
            ptr_cond_outputs = {
                t: out for t, out in selected_cond_outputs.items()
                if (t >= frame_idx if track_in_reverse else t <= frame_idx)
            }
        else:
            ptr_cond_outputs = selected_cond_outputs
        pos_and_outs_for_ptr = [
            (
                (frame_idx - t) * tpos_sign_mul
                if self.use_signed_tpos_enc_to_obj_ptrs else abs(frame_idx - t),
                out, True,
            )
            for t, out in ptr_cond_outputs.items()
        ]
        for t_diff in range(1, max_obj_ptrs_in_encoder):
            if not self.use_memory_selection:
                t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                if t < 0 or (num_frames is not None and t >= num_frames):
                    break
            else:
                if -t_diff <= -len(valid_indices):
                    break
                t = valid_indices[-t_diff]
            out = output_dict["non_cond_frame_outputs"].get(
                t, unselected_cond_outputs.get(t, None)
            )
            if out is not None:
                pos_and_outs_for_ptr.append((t_diff, out, False))

        if len(pos_and_outs_for_ptr) > 0:
            pos_list, out_list, _ = zip(*pos_and_outs_for_ptr)
            filtered = [(p, o) for p, o in zip(pos_list, out_list) if "obj_ptr" in o]
            if len(filtered) > 0:
                pos_list, out_list = zip(*filtered)
                obj_ptrs = torch.cat([o["obj_ptr"] for o in out_list], dim=1).transpose(0, 1)
                if self.add_tpos_enc_to_obj_ptrs:
                    obj_pos = self._get_tpos_enc(
                        list(pos_list), max_abs_pos=max_obj_ptrs_in_encoder, device=device
                    )
                else:
                    obj_pos = self._get_tpos_enc(list(pos_list), device=device, dummy=True)
                obj_pos = obj_pos.unsqueeze(1).expand(-1, B, -1)
                assert self.mem_dim == C
                obj_pos = obj_pos.repeat_interleave(multiplex_state.multiplex_count, dim=0)
                to_cat_prompt.append(obj_ptrs)
                to_cat_prompt_pos_embed.append(obj_pos)
                num_obj_ptr_tokens = obj_ptrs.shape[0]

        if len(to_cat_prompt) == 0:
            return vision_feat.permute(1, 2, 0).view(B, C, H, W)

        prompt = torch.cat(to_cat_prompt, dim=0)
        prompt_pos_embed = torch.cat(to_cat_prompt_pos_embed, dim=0)

        if self.save_image_features:
            if len(to_cat_image_feat) == 0:
                return vision_feat.permute(1, 2, 0).view(B, C, H, W)
            image_feat = torch.cat(to_cat_image_feat, dim=0)
            image_pos_embed = torch.cat(to_cat_image_pos_embed, dim=0)
            encoder_out = self.transformer.encoder(
                image=current_vision_feats[-1],
                src=vision_feat,
                memory_image=image_feat,
                memory=prompt,
                image_pos=current_vision_pos_embeds[-1],
                src_pos=vision_pos_embed,
                memory_image_pos=image_pos_embed,
                memory_pos=prompt_pos_embed,
                num_obj_ptr_tokens=num_obj_ptr_tokens,
            )
        else:
            encoder_out = self.transformer.encoder(
                src=vision_feat, src_key_padding_mask=vision_mask, src_pos=vision_pos_embed,
                prompt=prompt, prompt_pos=prompt_pos_embed, prompt_key_padding_mask=None,
                feat_sizes=feat_sizes, num_obj_ptr_tokens=num_obj_ptr_tokens,
            )
        return encoder_out["memory"].permute(1, 2, 0).view(B, C, H, W)

    def _encode_new_memory(
        self, *, image, current_vision_feats, feat_sizes, pred_masks_high_res,
        object_score_logits, is_mask_from_pts, conditioning_objects, multiplex_state,
    ):
        B = current_vision_feats[-1].size(1)
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc:
            pred_masks_high_res = self._apply_non_overlapping_constraints(pred_masks_high_res)

        if self.apply_sigmoid_to_mask_logits_for_mem_enc:
            binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
            if binarize and not self.training:
                mask_for_mem = (pred_masks_high_res > 0).float()
            else:
                mask_for_mem = torch.sigmoid(pred_masks_high_res)
            if self.sigmoid_scale_for_mem_enc != 1.0:
                mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
            if self.sigmoid_bias_for_mem_enc != 0.0:
                mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        else:
            mask_for_mem = pred_masks_high_res

        if self.condition_as_mask_input:
            conditioning_objects = sorted(list(conditioning_objects or []))

        mux_mask_for_mem = multiplex_state.mux(mask_for_mem).squeeze(2)

        if self.condition_as_mask_input:
            num_objects = mask_for_mem.shape[0]
            cond_values = torch.full(
                (num_objects,), self.condition_as_mask_input_bg,
                device=mask_for_mem.device, dtype=mask_for_mem.dtype,
            )
            if len(conditioning_objects) > 0:
                cond_values[conditioning_objects] = self.condition_as_mask_input_fg
            embedded_conditions = cond_values.view(-1, 1, 1, 1).expand_as(mask_for_mem)
            embedded_conditions = multiplex_state.mux(embedded_conditions).squeeze(2)
            mux_mask_for_mem = torch.cat([mux_mask_for_mem, embedded_conditions], dim=1)

        maskmem_out = self.maskmem_backbone(pix_feat, mux_mask_for_mem, skip_mask_sigmoid=True)
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]

        if self.no_obj_embed_spatial is not None:
            no_obj_embed_spatial = self.no_obj_embed_spatial.unsqueeze(0).repeat(
                multiplex_state.num_buckets, 1, 1
            )
            if object_score_logits is not None:
                obj_expected = multiplex_state.total_valid_entries
                obj_current = object_score_logits.shape[0]
                if obj_current != obj_expected:
                    if obj_current < obj_expected:
                        pad = object_score_logits.new_zeros(
                            (obj_expected - obj_current,) + tuple(object_score_logits.shape[1:])
                        )
                        object_score_logits = torch.cat([object_score_logits, pad], dim=0)
                    else:
                        object_score_logits = object_score_logits[:obj_expected]
            object_score_logits = multiplex_state.mux(object_score_logits)
            is_obj_appearing = (
                object_score_logits > self.object_score_logit_threshold
            ).float()
            no_obj_embed = ((1 - is_obj_appearing) * no_obj_embed_spatial).sum(dim=1)
            maskmem_features = maskmem_features + no_obj_embed[..., None, None].expand_as(
                maskmem_features
            )

        if maskmem_features.dim() == 5:
            maskmem_features = multiplex_state.demux(maskmem_features).contiguous()
        demuxed_pos_enc = []
        for pos_enc in maskmem_pos_enc:
            if pos_enc is not None and pos_enc.dim() == 5:
                pos_enc = multiplex_state.demux(pos_enc).contiguous()
            demuxed_pos_enc.append(pos_enc)
        return maskmem_features, demuxed_pos_enc

    # --- per-frame step (data-space seam; mux/demux internal) -------------------------------
    def track_step(
        self, *, frame_idx, is_init_cond_frame, backbone_features_interactive,
        backbone_features_propagation, point_inputs, mask_inputs, output_dict, num_frames,
        multiplex_state: MultiplexState, run_mem_encoder=True, track_in_reverse=False, image=None,
    ):
        current_out = {
            "conditioning_objects": set(),
            "point_inputs": point_inputs,
            "mask_inputs": mask_inputs,
        }
        if mask_inputs is not None:
            mode = "mask_as_output"
        elif point_inputs is None:
            mode = "propagation_only"
        else:
            mode = "interaction_only"

        interactive_vision_feats = interactive_feat_sizes = interactive_high_res_features = None
        if backbone_features_interactive is not None:
            interactive_vision_feats = backbone_features_interactive["vision_feats"]
            interactive_feat_sizes = backbone_features_interactive["feat_sizes"]
            if len(interactive_vision_feats) > 1:
                interactive_high_res_features = [
                    x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                    for x, s in zip(interactive_vision_feats[:-1], interactive_feat_sizes[:-1])
                ]

        propagation_vision_feats = propagation_vision_masks = None
        propagation_vision_pos_embeds = propagation_feat_sizes = None
        propagation_high_res_features = None
        if backbone_features_propagation is not None:
            propagation_vision_feats = backbone_features_propagation["vision_feats"]
            propagation_vision_masks = backbone_features_propagation["vision_masks"]
            propagation_vision_pos_embeds = backbone_features_propagation["vision_pos_embeds"]
            propagation_feat_sizes = backbone_features_propagation["feat_sizes"]
            if len(propagation_vision_feats) > 1:
                propagation_high_res_features = [
                    x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                    for x, s in zip(propagation_vision_feats[:-1], propagation_feat_sizes[:-1])
                ]

        if mode == "mask_as_output":
            interactive_pix_feat = self._get_interactive_pix_mem(
                interactive_vision_feats, interactive_feat_sizes
            )
            sam_outputs = self._use_mask_as_output(
                interactive_pix_feat, interactive_high_res_features, mask_inputs, multiplex_state,
            )
            current_out["conditioning_objects"].update(range(mask_inputs.shape[0]))
        elif mode == "propagation_only":
            pix_feat_with_mem = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx, is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=propagation_vision_feats[-1:],
                current_vision_masks=propagation_vision_masks[-1:],
                current_vision_pos_embeds=propagation_vision_pos_embeds[-1:],
                feat_sizes=propagation_feat_sizes[-1:], output_dict=output_dict,
                num_frames=num_frames, track_in_reverse=track_in_reverse,
                multiplex_state=multiplex_state,
            )
            multimask_output = self._use_multimask(is_init_cond_frame, None)
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat_with_mem,
                propagation_high_res_features=propagation_high_res_features,
                multimask_output=multimask_output,
                objects_to_interact=list(range(multiplex_state.total_valid_entries)),
                multiplex_state=multiplex_state,
            )
        else:  # interaction_only
            interactive_pix_feat = self._get_interactive_pix_mem(
                interactive_vision_feats, interactive_feat_sizes
            )
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            sam_outputs = self._forward_sam_heads(
                backbone_features=interactive_pix_feat, point_inputs=point_inputs,
                interactive_high_res_features=interactive_high_res_features,
                multimask_output=multimask_output,
                objects_to_interact=list(range(multiplex_state.total_valid_entries)),
                multiplex_state=multiplex_state,
            )
            current_out["conditioning_objects"].update(
                multiplex_state.get_all_valid_object_idx()
            )

        low_res_masks = sam_outputs["low_res_masks"]
        high_res_masks = sam_outputs["high_res_masks"]
        obj_ptr = sam_outputs["obj_ptr"]
        object_score_logits = sam_outputs["object_score_logits"]
        ious = sam_outputs["ious"]

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = multiplex_state.mux(obj_ptr)
        current_out["object_score_logits"] = object_score_logits
        if self.use_memory_selection:
            iou_score = ious.max(-1)[0]
            current_out["iou_score"] = iou_score
            current_out["eff_iou_score"] = self.cal_mem_score(object_score_logits, iou_score)

        if run_mem_encoder and self.num_maskmem > 0:
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                image=image, current_vision_feats=propagation_vision_feats,
                feat_sizes=propagation_feat_sizes, pred_masks_high_res=high_res_masks,
                object_score_logits=object_score_logits,
                is_mask_from_pts=(point_inputs is not None),
                conditioning_objects=current_out["conditioning_objects"],
                multiplex_state=multiplex_state,
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

        if self.save_image_features:
            current_out["image_features"] = propagation_vision_feats[-1]
            current_out["image_pos_enc"] = propagation_vision_pos_embeds[-1]

        return current_out
