import torch
from sam2.modeling.sam2_base import SAM2Base
from sam2.utils.transforms import SAM2Transforms

from sam2.modeling.sam2_utils import get_1d_sine_pe
from sam2.modeling.sam2_result import SAM2Result
from sam2.modeling.memory import ObjectMemory


class SAM2Generic(SAM2Base):

    def __init__(
        self,
        mask_threshold=0.0,
        max_hole_area=0.0,
        max_sprinkle_area=0.0,
        non_overlap_masks=False,
        **kwargs,
    ) -> None:
        """
        SAM2Generic is a class that extends SAM2Base to provide easier APIs for generic segmentation tasks.

        Arguments:
          mask_threshold (float): The threshold to use when converting mask logits
            to binary masks. Masks are thresholded at 0 by default.
          max_hole_area (int): If max_hole_area > 0, we fill small holes in up to
            the maximum area of max_hole_area in low_res_masks.
          max_sprinkle_area (int): If max_sprinkle_area > 0, we remove small sprinkles up to
            the maximum area of max_sprinkle_area in low_res_masks.
        """
        super().__init__(**kwargs)
        self._transforms = SAM2Transforms(
            resolution=self.image_size,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )
        self.mask_threshold = mask_threshold
        self.non_overlap_masks = non_overlap_masks

        self.empty_prompt_embeddings = self.encode_prompts()

    def _prepare_images(
        self, img: torch.Tensor | list[torch.Tensor], scale: bool = True
    ):

        # If we have a list of images (potentially of different sizes), we apply the transforms to each image
        # and then concatenate them along the batch dimension.
        img_list = [img] if not isinstance(img, (list, tuple)) else img

        for i, img in enumerate(img_list):
            assert img.ndim in [
                3,
                4,
            ], f"Expected image to be of shape (B, C, H, W) or (C, H, W), got {img.shape}"
            if img.ndim == 3:
                img = img.unsqueeze(0)
            if img.dtype == torch.uint8 and scale:
                img = img.float() / 255.0
            img_list[i] = self._transforms.transforms(img)

        return torch.cat(img_list, dim=0)

    @torch.inference_mode()
    def encode_image(
        self, image: torch.Tensor | list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the image for the SAM-2 model.

        Args:
            image (torch.Tensor | list[torch.Tensor]): The image or list of images to encode.

        Returns:
            img_embeddings (torch.Tensor): The image embeddings (the last one being the lowest resolution).
            img_pos_embeddings (torch.Tensor): The image position embeddings (the last one being the lowest resolution).
        """
        img_batch = self._prepare_images(image)

        backbone_out = self.image_encoder(img_batch)

        if self.use_high_res_features_in_sam:
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0]
            )
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1]
            )

        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        img_embeddings = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        img_pos_embeddings = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

        return img_embeddings, img_pos_embeddings

    @torch.inference_mode()
    def encode_memory(
        self,
        low_res_img_embeddings: torch.Tensor,
        masks_logits: torch.Tensor,
        obj_score_logits: torch.Tensor,
        is_prompt: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the image and its prediction into a memory.

        Args:
            low_res_img_embeddings (torch.Tensor): The low-resolution image embeddings.
            masks_high_res_logits (torch.Tensor): The high-resolution mask logits.
            object_score_logits (torch.Tensor): The object score logits.
            is_prompt (torch.BoolTensor): Whether the masks are from a user prompt or from a SAM prediction.

        Returns:
            memory_embeddings (torch.Tensor): The encoded memory embeddings.
            memory_pos_embeddings (torch.Tensor): The encoded memory position embeddings.
        """

        assert (
            low_res_img_embeddings.ndim == 4
        ), f"Expected low_res_img_embeddings to be of shape (B, C, H, W), got {low_res_img_embeddings.shape}"
        B = low_res_img_embeddings.shape[0]
        assert (
            masks_logits.ndim == 4 and masks_logits.shape[0] == B
        ), f"Expected masks_logits to be of shape (B, C, H, W), got {masks_logits.shape}"
        assert obj_score_logits.shape == (
            B,
            1,
        ), f"Expected obj_score_logits to be of shape ({B}, 1), got {obj_score_logits.shape}"
        assert is_prompt.shape == (
            B,
        ), f"Expected is_prompt to be of shape ({B},), got {is_prompt.shape}"

        if self.non_overlap_masks_for_mem_enc and not self.training:
            masks_logits = self._apply_non_overlapping_constraints(masks_logits)

        masks_logits = self._transforms.downscale_masks_logits(masks_logits)

        # Scale the raw mask logits with a temperature before applying sigmoid
        binarize = (
            self.binarize_mask_from_pts_for_mem_enc & is_prompt & (not self.training)
        )

        mask_for_mem = torch.where(
            binarize.reshape((-1, 1, 1, 1)),
            (masks_logits > self.mask_threshold).float(),
            torch.sigmoid(
                masks_logits
            ),  # Apply sigmoid on the raw mask logits to turn them into range (0, 1)
        )

        # Apply scale and bias terms to the sigmoid probabilities
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        maskmem_out = self.memory_encoder(
            low_res_img_embeddings,
            mask_for_mem,
            skip_mask_sigmoid=True,  # sigmoid already applied
        )

        memory_embeddings = maskmem_out["vision_features"]
        memory_pos_embeddings = maskmem_out["vision_pos_enc"][0]

        # Add a no-object embedding to the spatial memory to indicate that the frame
        # is predicted to be occluded (i.e. no object is appearing in the frame)
        if self.no_obj_embed_spatial is not None:
            is_obj_appearing = (obj_score_logits > 0).float()
            memory_embeddings += (
                1 - is_obj_appearing[..., None, None]
            ) * self.no_obj_embed_spatial[..., None, None].expand(
                *memory_embeddings.shape
            )

        return memory_embeddings, memory_pos_embeddings

    def _get_no_mem_pos_enc(self, batch_size: int) -> torch.Tensor:
        no_mem_pos_enc = self.no_mem_pos_enc

        if self.mem_dim < self.hidden_dim:
            # Split a pointer into (self.hidden_dim // self.mem_dim) tokens for self.mem_dim < self.hidden_dim
            no_mem_pos_enc = no_mem_pos_enc.reshape(
                -1,
                1,
                self.hidden_dim // self.mem_dim,
                self.mem_dim,
            )
            no_mem_pos_enc = no_mem_pos_enc.expand(-1, batch_size, -1, -1)
            no_mem_pos_enc = no_mem_pos_enc.permute(0, 2, 1, 3).flatten(0, 1)
        else:
            no_mem_pos_enc = no_mem_pos_enc.expand(-1, batch_size, -1)

        return no_mem_pos_enc

    def _get_no_mem_embed(self, batch_size: int) -> torch.Tensor:
        no_mem_embed = self.no_mem_embed

        if self.mem_dim < self.hidden_dim:
            # Split a pointer into (self.hidden_dim // self.mem_dim) tokens for self.mem_dim < self.hidden_dim
            no_mem_embed = no_mem_embed.reshape(
                -1,
                1,
                self.hidden_dim // self.mem_dim,
                self.mem_dim,
            )
            no_mem_embed = no_mem_embed.expand(-1, batch_size, -1, -1)
            no_mem_embed = no_mem_embed.permute(0, 2, 1, 3).flatten(0, 1)
        else:
            no_mem_embed = no_mem_embed.expand(-1, batch_size, -1)
        return no_mem_embed

    def _prepare_obj_ptrs_memories(
        self,
        current_frame_idx: int,
        obj_ptrs_memories_mask: torch.BoolTensor,
        obj_ptrs_memories: torch.Tensor,
        obj_ptrs_memories_frame_idx: torch.Tensor,
        reverse_tracking: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the object pointers into a memory, adding a temporal positional embedding based
        on how far each object pointer is from the current frame (sine embedding normalized by the max pointer num).
        """
        B, n_obj_ptrs_mem, C = obj_ptrs_memories.shape

        # (B, n_obj_ptrs_mem, C) -> (n_obj_ptrs_mem, B, C)
        obj_ptrs_memories = obj_ptrs_memories.permute(1, 0, 2)

        tpos_sign_mul = -1 if reverse_tracking else 1

        obj_trel = (
            (current_frame_idx - obj_ptrs_memories_frame_idx) * tpos_sign_mul
            if self.use_signed_tpos_enc_to_obj_ptrs
            else torch.abs(current_frame_idx - obj_ptrs_memories_frame_idx)
        )

        if self.add_tpos_enc_to_obj_ptrs:
            t_diff_max = self.max_obj_ptrs_in_encoder - 1
            tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim

            obj_trel = get_1d_sine_pe(obj_trel / t_diff_max, dim=tpos_dim)
            obj_trel = self.obj_ptr_tpos_proj.forward(obj_trel)
            # (B, n_obj_ptrs_mem, C) -> (n_obj_ptrs_mem, B, C)
            obj_trel = obj_trel.permute(1, 0, 2)
        else:
            obj_trel = obj_ptrs_memories.new_zeros(n_obj_ptrs_mem, B, self.mem_dim)

        if self.mem_dim < self.hidden_dim:
            # Split a pointer into (self.hidden_dim // self.mem_dim) tokens for self.mem_dim < self.hidden_dim
            obj_ptrs_memories = obj_ptrs_memories.reshape(
                -1,
                B,
                self.hidden_dim // self.mem_dim,
                self.mem_dim,
            )
            obj_ptrs_memories = obj_ptrs_memories.permute(0, 2, 1, 3).flatten(0, 1)
            obj_trel = obj_trel.repeat_interleave(
                self.hidden_dim // self.mem_dim, dim=0
            )

        no_mem_embed = self._get_no_mem_embed(B)
        no_mem_pos_enc = self._get_no_mem_pos_enc(B)

        obj_ptrs_memories_mask = (
            obj_ptrs_memories_mask.permute(1, 0)
            .repeat(no_mem_embed.shape[0], 1)
            .unsqueeze(-1)
            .expand(-1, -1, self.mem_dim)
        )

        obj_ptrs_memories = torch.where(
            obj_ptrs_memories_mask,
            obj_ptrs_memories,
            no_mem_embed.repeat(n_obj_ptrs_mem, 1, 1),
        )

        obj_trel = torch.where(
            obj_ptrs_memories_mask,
            obj_trel,
            no_mem_pos_enc.repeat(n_obj_ptrs_mem, 1, 1),
        )

        return obj_ptrs_memories, obj_trel

    def _prepare_conditional_memories(
        self,
        conditional_memories_mask: torch.BoolTensor,
        conditional_memories_embeddings: torch.Tensor,
        conditional_memories_pos_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        B, n_cond_mem, C, H, W = conditional_memories_embeddings.shape

        # The temporal positional encoding is always the last one for conditional memories (reserved)
        conditional_memory_pos_embeddings_with_tpos = (
            conditional_memories_pos_embeddings
            + self.maskmem_tpos_enc[self.num_maskmem - 1][None, None, :, :, :]
        )

        # (B, n_cond_mem, C, H, W) -> (n_cond_mem, H, W, B, C) -> (n_cond_mem*H*W, B, C)
        conditional_memories_embeddings = conditional_memories_embeddings.permute(
            1, 3, 4, 0, 2
        ).reshape(n_cond_mem * H * W, B, C)
        # (B, n_cond_mem, C, H, W) -> (n_cond_mem, H, W, B, C) -> (n_cond_mem*H*W, B, C)
        conditional_memory_pos_embeddings_with_tpos = (
            conditional_memory_pos_embeddings_with_tpos.permute(1, 3, 4, 0, 2).reshape(
                n_cond_mem * H * W, B, C
            )
        )

        no_mem_embed = self._get_no_mem_embed(B)
        no_mem_pos_enc = self._get_no_mem_pos_enc(B)

        conditional_memories_mask = (
            conditional_memories_mask.permute(1, 0)
            .repeat(H * W, 1)
            .unsqueeze(-1)
            .expand(-1, -1, self.mem_dim)
        )

        conditional_memories_embeddings = torch.where(
            conditional_memories_mask,
            conditional_memories_embeddings,
            no_mem_embed.repeat((n_cond_mem * H * W) // no_mem_embed.shape[0], 1, 1),
        )

        conditional_memory_pos_embeddings_with_tpos = torch.where(
            conditional_memories_mask,
            conditional_memory_pos_embeddings_with_tpos,
            no_mem_pos_enc.repeat(
                (n_cond_mem * H * W) // no_mem_pos_enc.shape[0], 1, 1
            ),
        )

        return (
            conditional_memories_embeddings,
            conditional_memory_pos_embeddings_with_tpos,
        )

    def _prepare_non_conditional_memories(
        self,
        current_frame_idx: int,
        non_conditional_memories_mask: torch.BoolTensor,
        non_conditional_memories_embeddings: torch.Tensor,
        non_conditional_memories_pos_embeddings: torch.Tensor,
        non_conditional_memories_frame_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare the non-conditional memories for the memory conditioning by adding the temporal positional encoding.

        Args:
            current_frame_idx (int): The index of the current frame.
            non_conditional_memories_embeddings (torch.Tensor): The embeddings of the non-conditional memories.
            non_conditional_memories_pos_embeddings (torch.Tensor): The position embeddings of the non-conditional memories.
            non_conditional_memories_frame_idx (torch.Tensor): The frame indices of the non-conditional memories.

        Returns:
            memory_embeddings (torch.Tensor): The embeddings of the non-conditional memories.
            memory_pos_embeddings (torch.Tensor): The position embeddings of the non-conditional memories with temporal positional encoding.
        """
        B, n_non_cond_mem, C, H, W = non_conditional_memories_embeddings.shape

        # Sort by distance to current frame along the n_non_cond_mem dimension (closest frames first)
        frame_distances = torch.abs(
            non_conditional_memories_frame_idx - current_frame_idx
        )
        sorted_indices = torch.argsort(frame_distances, dim=1)
        batch_indices = torch.arange(
            B, device=non_conditional_memories_embeddings.device
        ).unsqueeze(1)
        memory_embeddings = non_conditional_memories_embeddings[
            batch_indices, sorted_indices
        ]
        memory_pos_embeddings = non_conditional_memories_pos_embeddings[
            batch_indices, sorted_indices
        ]

        # Relative position to current frame, starting from 1 (closest frame) to self.num_maskmem - 2 (farthest frame).
        # The t_rel = self.num_maskmem - 1 is reserved for conditional memories.
        t_rel = torch.arange(
            0, self.num_maskmem - 1, 1, device=memory_embeddings.device
        )

        # Add temporal positional encoding
        memory_tpos_embeddings = memory_pos_embeddings + self.maskmem_tpos_enc[
            t_rel
        ].unsqueeze(0)

        # (B, n_non_cond_mem, C, H, W) -> (n_non_cond_mem, H, W, B, C) -> (n_non_cond_mem*H*W, B, C)
        memory_embeddings = memory_embeddings.permute(1, 3, 4, 0, 2).reshape(
            n_non_cond_mem * H * W, B, C
        )
        # (B, n_non_cond_mem, C, H, W) -> (n_non_cond_mem, H, W, B, C) -> (n_non_cond_mem*H*W, B, C)
        memory_tpos_embeddings = memory_tpos_embeddings.permute(1, 3, 4, 0, 2).reshape(
            n_non_cond_mem * H * W, B, C
        )

        no_mem_embed = self._get_no_mem_embed(B)
        no_mem_pos_enc = self._get_no_mem_pos_enc(B)

        non_conditional_memories_mask = (
            non_conditional_memories_mask.permute(1, 0)
            .repeat(H * W, 1)
            .unsqueeze(-1)
            .expand(-1, -1, self.mem_dim)
        )

        memory_embeddings = torch.where(
            non_conditional_memories_mask,
            memory_embeddings,
            no_mem_embed.repeat(
                (n_non_cond_mem * H * W) // no_mem_embed.shape[0], 1, 1
            ),
        )

        memory_tpos_embeddings = torch.where(
            non_conditional_memories_mask,
            memory_tpos_embeddings,
            no_mem_pos_enc.repeat(
                (n_non_cond_mem * H * W) // no_mem_pos_enc.shape[0], 1, 1
            ),
        )

        return memory_embeddings, memory_tpos_embeddings

    @torch.inference_mode()
    def condition_image_embeddings_on_memories(
        self,
        frame_idx: int,
        low_res_img_embeddings: torch.Tensor,
        low_res_img_pos_embeddings: torch.Tensor,
        conditional_memories_mask: torch.BoolTensor,
        conditional_memories_embeddings: torch.Tensor,
        conditional_memories_pos_embeddings: torch.Tensor,
        non_conditional_memories_mask: torch.BoolTensor,
        non_conditional_memories_embeddings: torch.Tensor,
        non_conditional_memories_pos_embeddings: torch.Tensor,
        non_conditional_memories_frame_idx: torch.Tensor,
        obj_ptrs_memories_mask: torch.BoolTensor,
        obj_ptrs_memories: torch.Tensor,
        obj_ptrs_memories_frame_idx: torch.Tensor,
        reverse_tracking: bool = False,
    ) -> torch.Tensor:
        """
        Condition the image embeddings on the memory embeddings.

        Note: the non conditional memories are ordered by order of importance, i.e. the first non conditional memory is the most important one.
        For example, if you use temporal memory, the first non conditional memory is the one that is closest to the current frame, and the last one is the one that is farthest.

        Args:
            frame_idx (int): The index of the current frame.
            img_embeddings (list[torch.Tensor]): The image embeddings.
            img_pos_embeddings (list[torch.Tensor]): The image position embeddings.
            conditional_memories_pad_mask (torch.BoolTensor): Mask indicating which conditional memories are just padding.
            conditional_memories_embeddings (torch.Tensor): The conditional memories embeddings.
            conditional_memories_pos_embeddings (torch.Tensor): The conditional memories position embeddings.
            non_conditional_memories_pad_mask (torch.BoolTensor): Mask indicating which non conditional memories are just padding.
            non_conditional_memories_embeddings (torch.Tensor): The non conditional memories embeddings.
            non_conditional_memories_pos_embeddings (torch.Tensor): The non conditional memories position embeddings.
            non_conditional_memories_frame_idx (torch.Tensor): The non conditional memories frame index.
            obj_ptrs_memories_mask (torch.BoolTensor): Mask indicating which object pointer memories are just padding.
            obj_ptrs_memories (torch.Tensor): The object pointer memories embeddings.
            obj_ptrs_memories_frame_idx (torch.Tensor): The object pointer memories frame index.
            reverse_tracking (bool): Whether to reverse the tracking.

        Returns:
            torch.Tensor: The conditioned low-res image embeddings.
        """

        M = self.mem_dim

        assert low_res_img_embeddings.ndim == 4 and low_res_img_embeddings.shape[
            -2:
        ] == (
            self.mem_dim,
            self.mem_dim,
        ), f"Expected low_res_img_embeddings to be of shape (B, C, {M}, {M}), got {low_res_img_embeddings.shape}"

        B, C, H, W = low_res_img_embeddings.shape

        assert low_res_img_pos_embeddings.shape == (
            B,
            C,
            self.mem_dim,
            self.mem_dim,
        ), f"Expected low_res_img_pos_embeddings to be of shape {(B, C, M, M)}, got {low_res_img_pos_embeddings.shape}"
        assert conditional_memories_embeddings.shape[
            0
        ] == B and conditional_memories_embeddings.shape[-3:] == (
            self.mem_dim,
            self.mem_dim,
            self.mem_dim,
        ), f"Expected conditional_memories_embeddings to be of shape ({B}, n_cond_mem, {M}, {M}, {M}), got {conditional_memories_embeddings.shape}"

        n_cond_mem = conditional_memories_embeddings.shape[-4]

        assert conditional_memories_mask.shape == (
            B,
            n_cond_mem,
        ), f"Expected conditional_memories_mask to be of shape (B, n_cond_mem), got {conditional_memories_mask.shape}"

        assert conditional_memories_pos_embeddings.shape == (
            B,
            n_cond_mem,
            self.mem_dim,
            self.mem_dim,
            self.mem_dim,
        ), f"Expected conditional_memories_pos_embeddings to be of shape {(B, n_cond_mem, M, M, M)}, got {conditional_memories_pos_embeddings.shape}"
        assert non_conditional_memories_embeddings.shape[
            0
        ] == B and non_conditional_memories_embeddings.shape[-3:] == (
            self.mem_dim,
            self.mem_dim,
            self.mem_dim,
        ), f"Expected non_conditional_memories_embeddings to be of shape ({B}, n_non_cond_mem, {M}, {M}, {M}), got {non_conditional_memories_embeddings.shape}"

        n_non_cond_mem = non_conditional_memories_embeddings.shape[-4]

        assert non_conditional_memories_mask.shape == (
            B,
            n_non_cond_mem,
        ), f"Expected non_conditional_memories_mask to be of shape (B, n_non_cond_mem), got {non_conditional_memories_mask.shape}"

        assert non_conditional_memories_pos_embeddings.shape == (
            B,
            n_non_cond_mem,
            self.mem_dim,
            self.mem_dim,
            self.mem_dim,
        ), f"Expected non_conditional_memories_pos_embeddings to be of shape {(B, n_non_cond_mem, M, M, M)}, got {non_conditional_memories_pos_embeddings.shape}"
        assert (
            non_conditional_memories_frame_idx.shape[0] == B
        ), f"Expected non_conditional_memories_frame_idx to be of shape (B, n_non_cond_mem), got {non_conditional_memories_frame_idx.shape}"
        assert (
            obj_ptrs_memories.shape[0] == B
            and obj_ptrs_memories.shape[-1] == self.hidden_dim
        ), f"Expected obj_ptrs_memories to be of shape ({B}, n_obj_ptrs_mem, {self.hidden_dim}), got {obj_ptrs_memories.shape}"

        n_obj_ptrs_mem = obj_ptrs_memories.shape[-2]

        assert obj_ptrs_memories_mask.shape == (
            B,
            n_obj_ptrs_mem,
        ), f"Expected obj_ptrs_memories_mask to be of shape (B, n_obj_ptrs_mem), got {obj_ptrs_memories_mask.shape}"

        assert obj_ptrs_memories_frame_idx.shape == (
            B,
            n_obj_ptrs_mem,
        ), f"Expected obj_ptrs_memories_frame_idx to be of shape {(B, n_obj_ptrs_mem)}, got {obj_ptrs_memories_frame_idx.shape}"

        if n_cond_mem == 0 and n_non_cond_mem == 0 and n_obj_ptrs_mem == 0:
            # We don't have any memories, we add the no-mem embedding
            if self.directly_add_no_mem_embed:
                # Directly add the no-mem embedding (instead of using the transformer encoder)

                # (1, 1, 256) -> (B, 256, 1, 1)
                no_mem_embed = (
                    self.no_mem_embed.permute(0, 2, 1)
                    .view(1, C, 1, 1)
                    .expand(B, -1, -1, -1)
                )

                low_res_img_embeddings_with_mem = low_res_img_embeddings + no_mem_embed
                return low_res_img_embeddings_with_mem

            # Use a dummy token on the first frame (to avoid empty memory input to tranformer encoder)
            memories = self.no_mem_embed.expand(1, B, self.mem_dim)
            memories_pos_embed = self.no_mem_pos_enc.expand(1, B, self.mem_dim)
            num_obj_ptr_tokens = 0

        else:

            # Add conditional memories (prompt from the user)
            conditional_memories_embeddings, conditional_memories_pos_embeddings = (
                self._prepare_conditional_memories(
                    conditional_memories_mask=conditional_memories_mask,
                    conditional_memories_embeddings=conditional_memories_embeddings,
                    conditional_memories_pos_embeddings=conditional_memories_pos_embeddings,
                )
            )

            # Add non-conditional memories
            (
                non_conditional_memories_embeddings,
                non_conditional_memories_pos_embeddings,
            ) = self._prepare_non_conditional_memories(
                current_frame_idx=frame_idx,
                non_conditional_memories_mask=non_conditional_memories_mask,
                non_conditional_memories_embeddings=non_conditional_memories_embeddings,
                non_conditional_memories_pos_embeddings=non_conditional_memories_pos_embeddings,
                non_conditional_memories_frame_idx=non_conditional_memories_frame_idx,
            )

            memories = torch.cat(
                [conditional_memories_embeddings, non_conditional_memories_embeddings],
                dim=0,
            )
            memories_pos_embed = torch.cat(
                [
                    conditional_memories_pos_embeddings,
                    non_conditional_memories_pos_embeddings,
                ],
                dim=0,
            )

            if self.use_obj_ptrs_in_encoder:

                obj_ptrs_enc, obj_pos_enc = self._prepare_obj_ptrs_memories(
                    current_frame_idx=frame_idx,
                    obj_ptrs_memories_mask=obj_ptrs_memories_mask,
                    obj_ptrs_memories=obj_ptrs_memories,
                    obj_ptrs_memories_frame_idx=obj_ptrs_memories_frame_idx,
                    reverse_tracking=reverse_tracking,
                )
                num_obj_ptr_tokens = obj_ptrs_enc.shape[0]

                memories = torch.cat([memories, obj_ptrs_enc], dim=0)
                memories_pos_embed = torch.cat([memories_pos_embed, obj_pos_enc], dim=0)

        low_res_img_embeddings = low_res_img_embeddings.flatten(2).permute(
            2, 0, 1
        )  # (B, C, H, W) -> (H*W, B, C)
        low_res_img_pos_embeddings = low_res_img_pos_embeddings.flatten(2).permute(
            2, 0, 1
        )  # (B, C, H, W) -> (H*W, B, C)

        low_res_img_embeddings_with_mem: torch.Tensor = self.memory_attention(
            curr=low_res_img_embeddings,
            curr_pos=low_res_img_pos_embeddings,
            memory=memories,
            memory_pos=memories_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        # reshape the output (HW)BC => BCHW
        low_res_img_embeddings_with_mem = low_res_img_embeddings_with_mem.permute(
            1, 2, 0
        ).view(B, C, H, W)

        return low_res_img_embeddings_with_mem

    @torch.inference_mode()
    def encode_prompts(
        self,
        orig_hw: tuple[int, int] | None = None,
        batch_size: int = 1,
        points_coords: torch.Tensor | None = None,
        points_labels: torch.Tensor | None = None,
        boxes: torch.Tensor | None = None,
        masks_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the prompts for the SAM-2 model.

        Args:
            orig_hw (tuple[int, int]): The original height and width of the image.
            batch_size (int): The batch size of the prompts.
            points_coords (torch.Tensor | None): The coordinates of the points to encode. Shape: (B, N, 2) with N being the number of points and the last dimension being (x, y).
            points_labels (torch.Tensor | None): The labels of the points to encode. Shape: (B, N).
            boxes (torch.Tensor | None): The boxes to encode. Shape: (B, 4) with the last dimension being (x1, y1, x2, y2).
            masks_logits (torch.Tensor | None): The masks logits to encode. Shape: (B, H, W).

        Returns:
            prompt_embeddings (tuple[torch.Tensor, torch.Tensor]): The sparse and dense prompt embeddings.
        """

        if points_coords is not None or boxes is not None:
            assert (
                orig_hw is not None
            ), "Expected orig_hw to be provided if points_coords or boxes are provided"

        points = None

        if points_coords is not None:
            assert (
                points_labels is not None
            ), f"Expected points_labels to be provided if points_coords is provided, got None"
            assert (
                points_coords.ndim == 3
                and points_coords.shape[0] == batch_size
                and points_coords.shape[2] == 2
            ), f"Expected points_coords to be of shape (B, N, 2), got {points_coords.shape}"
            assert (
                points_labels.ndim == 2
                and points_labels.shape == points_coords.shape[:2]
            ), f"Expected points_labels to be of shape (B, N), got {points_labels.shape}"
            points_coords = self._transforms.transform_coords(
                points_coords, normalize=True, orig_hw=orig_hw
            )
            points = (points_coords, points_labels)

        masks_low_res_logits = None

        if masks_logits is not None:
            # If mask_inputs is provided, downsize it into low-res mask input if needed
            # and feed it as a dense mask prompt into the SAM mask encoder
            assert masks_logits.ndim == 4 and masks_logits.shape[:2] == (
                batch_size,
                1,
            ), f"Expected masks to be of shape (B, 1, H, W), got {masks_logits.shape}"

            masks_low_res_logits = self._transforms.downscale_masks_logits(
                masks_low_res_logits
            )

        if boxes is not None:
            assert (
                boxes.ndim == 3 and boxes.shape[0] == batch_size and boxes.shape[2] == 4
            ), f"Expected boxes to be of shape (B, N, 4), got {boxes.shape}"
            # Encode the boxes as points with labels 2 and 3
            box_points_coords = boxes.reshape(batch_size, 2, 2)
            box_points_coords = self._transforms.transform_boxes(
                box_points_coords, normalize=True, orig_hw=orig_hw
            )
            box_points_labels = torch.tensor(
                [2, 3], dtype=torch.int32, device=boxes.device
            )
            box_points_labels = box_points_labels.reshape(batch_size, 2)

            # Concatenate the box points with the existing points
            if points is not None:
                points[0] = torch.cat([points[0], box_points_coords], dim=1)
                points[1] = torch.cat([points[1], box_points_labels], dim=1)
            else:
                points = (box_points_coords, box_points_labels)

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder.forward(
            points=points, masks=masks_low_res_logits, boxes=None
        )

        return sparse_embeddings, dense_embeddings

    @torch.inference_mode()
    def generate_masks(
        self,
        orig_hw: tuple[int, int],
        img_embeddings: list[torch.Tensor],
        prompt_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        multimask_output: bool = True,
    ) -> SAM2Result:

        low_res_img_embeddings = img_embeddings[-1]
        high_res_img_embeddings = img_embeddings[:-1]

        if len(high_res_img_embeddings) == 0:
            high_res_img_embeddings = None

        B, C, H, W = low_res_img_embeddings.shape

        assert C == self.sam_prompt_embed_dim
        assert H == self.sam_image_embedding_size
        assert W == self.sam_image_embedding_size

        if high_res_img_embeddings is not None:
            assert len(high_res_img_embeddings) == 2
            assert high_res_img_embeddings[0].shape == (B, C // 8, 4 * H, 4 * W)
            assert high_res_img_embeddings[1].shape == (B, C // 4, 2 * H, 2 * W)

        if prompt_embeddings is None:
            sparse_prompt_embeddings, dense_prompt_embeddings = (
                self.empty_prompt_embeddings
            )
            sparse_prompt_embeddings = sparse_prompt_embeddings.to(self.device)
            dense_prompt_embeddings = dense_prompt_embeddings.to(self.device)
            sparse_prompt_embeddings = sparse_prompt_embeddings.expand(B, -1, -1)
            dense_prompt_embeddings = dense_prompt_embeddings.expand(B, -1, -1, -1)
        else:
            sparse_prompt_embeddings, dense_prompt_embeddings = prompt_embeddings

        prompt_positional_encoding = self.sam_prompt_encoder.get_dense_pe()

        (
            masks_logits,
            ious,
            sam_output_tokens,
            obj_scores_logits,
        ) = self.sam_mask_decoder.forward(
            image_embeddings=low_res_img_embeddings,
            image_pe=prompt_positional_encoding,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,  # TODO
            high_res_features=high_res_img_embeddings,
        )

        # Upscale the masks to the image_size
        masks_logits = self._transforms.postprocess_masks(
            masks_logits, (self.image_size, self.image_size)
        )
        masks_logits = torch.clamp(masks_logits, -32.0, 32.0)

        # Apply non-overlapping constraints if specified
        if self.non_overlap_masks:
            masks_logits = self._apply_non_overlapping_constraints(masks_logits)

        masks_logits = self._transforms.upscale_masks_logits(masks_logits, orig_hw)

        # Extract object pointer from the SAM output token (with occlusion handling)
        sam_output_token = sam_output_tokens[:, 0]
        obj_ptr = self.obj_ptr_proj.forward(sam_output_token)

        # TODO: review this part. I'm not sure if this is correct.
        # Allow *soft* no obj ptr, unlike masks
        if self.soft_no_obj_ptr:
            obj_visibility = torch.sigmoid(obj_scores_logits)
        else:
            obj_visibility = (obj_scores_logits > 0).float()

        if self.fixed_no_obj_ptr:
            obj_ptr = obj_visibility * obj_ptr
        obj_ptr = obj_ptr + (1 - obj_visibility) * self.no_obj_ptr

        return SAM2Result(
            masks_logits=masks_logits,
            ious=ious,
            obj_ptrs=obj_ptr,
            obj_scores_logits=obj_scores_logits,
        )
