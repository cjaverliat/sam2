from dataclasses import dataclass

import numpy as np
import torch

from sam2.modeling.sam2_base import NO_OBJ_SCORE

from sam2.modeling.sam2_generic import SAM2Generic
from sam2.modeling.memory import ObjectMemoryBank
from sam2.modeling.sam2_prompt import SAM2Prompt
from sam2.modeling.sam2_result import SAM2Result


class SAM2GenericVideoPredictor(SAM2Generic):
    """
    SAM2GenericVideoPredictor provides a handy video prediction interface.

    Note: works in a forward-only manner.
    """

    def __init__(
        self,
        memory_bank: ObjectMemoryBank,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._video_hw: tuple[int, int] | None = None
        self.memory_bank = memory_bank

    @torch.inference_mode()
    def forward(
        self,
        frame_idx: int,
        frame: torch.Tensor,
        prompts: list[SAM2Prompt] = [],
        multimask_output: bool = True,
        reverse_tracking: bool = False,
    ) -> dict[int, SAM2Result]:
        # First frame, initialize video_hw
        if self._video_hw is None:
            self._video_hw = frame.shape[-2:]

        assert frame.shape in [
            (1, *self._video_hw),
            (3, *self._video_hw),
        ], f"Expected frame to be of shape (C, H, W) or (1, C, H, W) with H and W equal to {self._video_hw}, got {frame.shape}"

        img_embeddings, img_pos_embeddings = self.encode_image(frame)

        assert prompts is None or np.unique([p.obj_id for p in prompts]).size == len(
            prompts
        ), "Only one prompt per object should be provided"

        results: dict[int, SAM2Result] = {}

        objs_with_prompts_id = set([p.obj_id for p in prompts])
        objs_without_prompts_id = self.memory_bank.known_objs_id - objs_with_prompts_id
        objs_id = objs_with_prompts_id | objs_without_prompts_id
        n_objs = len(objs_id)

        # 1. Handle objects with prompts
        prompts_dicts: dict[int, SAM2Prompt] = {
            prompt.obj_id: prompt for prompt in prompts
        }

        for obj_id in objs_with_prompts_id:
            prompt = prompts_dicts[obj_id]

            prompt_embeddings = self.encode_prompts(
                orig_hw=self._video_hw,
                points_coords=prompt.points_coords,
                points_labels=prompt.points_labels,
                boxes=prompt.boxes,
                masks_logits=prompt.masks_logits,
            )

            results[obj_id] = self.generate_masks(
                orig_hw=self._video_hw,
                img_embeddings=img_embeddings,
                prompt_embeddings=prompt_embeddings,
                multimask_output=multimask_output,
            )

        # 2. Handle objects without prompts (i.e. find objects by using the memories)
        if len(objs_without_prompts_id) > 0:
            B = len(objs_without_prompts_id)

            objs_memories = self.memory_bank.select_memories(
                objs_id=objs_without_prompts_id,
                current_frame_idx=frame_idx,
                max_conditional_memories=self.max_cond_frames_in_attn,
                max_non_conditional_memories=self.num_maskmem - 1,
                max_ptr_memories=self.max_obj_ptrs_in_encoder,
                only_include_pointers_in_past=self.only_obj_ptrs_in_the_past_for_eval,
                reverse_tracking=reverse_tracking,
            )

            max_cond_memories = 0
            max_obj_ptrs_memories = 0
            max_non_cond_memories = self.num_maskmem - 1

            for obj_id in objs_without_prompts_id:
                max_cond_memories = max(
                    max_cond_memories,
                    len(objs_memories[obj_id].conditional_memories),
                )

                max_obj_ptrs_memories = max(
                    max_obj_ptrs_memories,
                    len(objs_memories[obj_id].ptr_memories),
                )

            cond_mem_embeddings = torch.zeros(
                (B, max_cond_memories, self.mem_dim, self.mem_dim, self.mem_dim),
                device=self.device,
            )
            cond_mem_pos_embeddings = torch.zeros(
                (B, max_cond_memories, self.mem_dim, self.mem_dim, self.mem_dim),
                device=self.device,
            )

            non_cond_mem_embeddings = torch.zeros(
                (B, max_non_cond_memories, self.mem_dim, self.mem_dim, self.mem_dim),
                device=self.device,
            )
            non_cond_mem_pos_embeddings = torch.zeros(
                (B, max_non_cond_memories, self.mem_dim, self.mem_dim, self.mem_dim),
                device=self.device,
            )
            non_cond_mem_frame_idx = torch.zeros(
                (B, max_non_cond_memories),
                device=self.device,
            )

            obj_ptrs_mem = torch.zeros(
                (B, max_obj_ptrs_memories, self.hidden_dim),
                device=self.device,
            )
            obj_ptrs_mem_frame_idx = torch.zeros(
                (B, max_obj_ptrs_memories),
                device=self.device,
            )

            cond_mem_mask = torch.zeros(
                (B, max_cond_memories),
                dtype=torch.bool,
                device=self.device,
            )
            non_cond_mem_mask = torch.zeros(
                (B, max_non_cond_memories),
                dtype=torch.bool,
                device=self.device,
            )
            obj_ptrs_mem_mask = torch.zeros(
                (B, max_obj_ptrs_memories),
                dtype=torch.bool,
                device=self.device,
            )

            # Now we apply the actual memories
            for obj_id in objs_without_prompts_id:
                obj_memories = objs_memories[obj_id]
                obj_memories = obj_memories.to(self.device)

                for i, cond_mem in enumerate(obj_memories.conditional_memories):
                    cond_mem_embeddings[obj_id, i] = cond_mem.memory_embeddings
                    cond_mem_pos_embeddings[obj_id, i] = cond_mem.memory_pos_embeddings
                    cond_mem_mask[obj_id, i] = True

                for i, non_cond_mem in enumerate(obj_memories.non_conditional_memories):
                    non_cond_mem_embeddings[obj_id, i] = non_cond_mem.memory_embeddings
                    non_cond_mem_pos_embeddings[obj_id, i] = (
                        non_cond_mem.memory_pos_embeddings
                    )
                    non_cond_mem_frame_idx[obj_id, i] = non_cond_mem.frame_idx
                    non_cond_mem_mask[obj_id, i] = True

                for i, ptr_mem in enumerate(obj_memories.ptr_memories):
                    obj_ptrs_mem[obj_id, i] = ptr_mem.ptr
                    obj_ptrs_mem_frame_idx[obj_id, i] = ptr_mem.frame_idx
                    obj_ptrs_mem_mask[obj_id, i] = True

            low_res_img_embeddings = img_embeddings[-1].expand(B, -1, -1, -1)
            low_res_img_pos_embeddings = img_pos_embeddings[-1].expand(B, -1, -1, -1)
            high_res_img_embeddings = [
                x.expand(B, -1, -1, -1) for x in img_embeddings[:-1]
            ]

            conditioned_low_res_img_embeddings = (
                self.condition_image_embeddings_on_memories(
                    frame_idx=frame_idx,
                    low_res_img_embeddings=low_res_img_embeddings,
                    low_res_img_pos_embeddings=low_res_img_pos_embeddings,
                    non_conditional_memories_mask=non_cond_mem_mask,
                    non_conditional_memories_embeddings=non_cond_mem_embeddings,
                    non_conditional_memories_pos_embeddings=non_cond_mem_pos_embeddings,
                    non_conditional_memories_frame_idx=non_cond_mem_frame_idx,
                    conditional_memories_mask=cond_mem_mask,
                    conditional_memories_embeddings=cond_mem_embeddings,
                    conditional_memories_pos_embeddings=cond_mem_pos_embeddings,
                    obj_ptrs_memories_mask=obj_ptrs_mem_mask,
                    obj_ptrs_memories=obj_ptrs_mem,
                    obj_ptrs_memories_frame_idx=obj_ptrs_mem_frame_idx,
                    reverse_tracking=reverse_tracking,
                )
            )

            conditioned_img_embeddings = high_res_img_embeddings + [
                conditioned_low_res_img_embeddings
            ]

            result = self.generate_masks(
                orig_hw=self._video_hw,
                img_embeddings=conditioned_img_embeddings,
                multimask_output=multimask_output,
            )

            for obj_id in objs_without_prompts_id:
                results[obj_id] = result[obj_id]

        objs_id = list(results.keys())
        batched_results = SAM2Result.cat(list(results.values()))

        is_prompt = torch.tensor(
            [obj_id in prompts_dicts for obj_id in objs_id],
            dtype=torch.bool,
            device=self.device,
        )

        low_res_img_embeddings = img_embeddings[-1]

        memory_embeddings, memory_pos_embeddings = self.encode_memory(
            low_res_img_embeddings=low_res_img_embeddings.expand((n_objs, -1, -1, -1)),
            masks_logits=batched_results.best_mask_logits,
            obj_score_logits=batched_results.obj_score_logits,
            is_prompt=is_prompt,
        )

        self.memory_bank.try_add_memories(
            frame_idx=frame_idx,
            objs_id=objs_id,
            memory_embeddings=memory_embeddings,
            memory_pos_embeddings=memory_pos_embeddings,
            results=batched_results,
            prompts=prompts,
        )

        self.memory_bank.prune_memories(
            objs_id=objs_id,
            current_frame_idx=frame_idx,
        )

        return {obj_id: result for obj_id, result in zip(objs_id, batched_results)}
