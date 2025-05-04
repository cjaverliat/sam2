from dataclasses import dataclass

import numpy as np
import torch

from sam2.modeling.sam2_generic import SAM2Generic
from sam2.modeling.memory import ObjectMemoryBank, ObjectMemory
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

        # Unique list of all objects to propagate the masks for (includes previous objects and new prompts).
        all_obj_ids = self.memory_bank.known_obj_ids | set([p.obj_id for p in prompts])
        n_objs = len(all_obj_ids)

        prompts_dicts: dict[int, SAM2Prompt] = {
            prompt.obj_id: prompt for prompt in prompts
        }

        objects_selected_memories = self.memory_bank.select_memories(
            obj_ids=self.memory_bank.known_obj_ids,
            current_frame_idx=frame_idx,
            max_conditional_memories=self.max_cond_frames_in_attn,
            max_non_conditional_memories=self.num_maskmem - 1,
            max_ptr_memories=self.max_obj_ptrs_in_encoder,
            only_include_pointers_in_past=self.only_obj_ptrs_in_the_past_for_eval,
            reverse_tracking=reverse_tracking,
        )

        results: list[SAM2Result] = []

        for obj_id in all_obj_ids:

            prompt = prompts_dicts.get(obj_id, None)
            has_prompt = prompt is not None

            if has_prompt:

                prompt_embeddings = self.encode_prompts(
                    orig_hw=self._video_hw,
                    points_coords=prompt.points_coords,
                    points_labels=prompt.points_labels,
                    boxes=prompt.boxes,
                    masks_logits=prompt.masks_logits,
                )

                result = self.generate_masks(
                    orig_hw=self._video_hw,
                    img_embeddings=img_embeddings,
                    prompt_embeddings=prompt_embeddings,
                    multimask_output=multimask_output,
                )

            else:

                assert (
                    obj_id in objects_selected_memories
                ), f"Expected memory bank to have a memory for object {obj_id} but it does not."

                object_selected_memories = objects_selected_memories[obj_id]
                # Transfer the memories to the correct device
                object_selected_memories = object_selected_memories.to(self.device)

                conditioned_img_embeddings = self.condition_image_embeddings_on_memories(
                    frame_idx=frame_idx,
                    img_embeddings=img_embeddings,
                    img_pos_embeddings=img_pos_embeddings,
                    non_conditional_memories=object_selected_memories.non_conditional_memories,
                    conditional_memories=object_selected_memories.conditional_memories,
                    ptr_memories=object_selected_memories.ptr_memories,
                )

                result = self.generate_masks(
                    orig_hw=self._video_hw,
                    img_embeddings=conditioned_img_embeddings,
                    multimask_output=True,
                )

            results.append(result)

        batched_results = SAM2Result.cat(results)

        is_prompt = torch.tensor(
            [obj_id in prompts_dicts for obj_id in all_obj_ids],
            dtype=torch.bool,
            device=batched_results.device,
        )

        memory_embeddings, memory_pos_embeddings = self.encode_memory(
            img_embeddings=[m.expand((n_objs, -1, -1, -1)) for m in img_embeddings],
            masks_logits=batched_results.best_mask_logits,
            obj_score_logits=batched_results.obj_score_logits,
            is_prompt=is_prompt,
        )

        self.memory_bank.try_add_memories(
            frame_idx=frame_idx,
            obj_ids=all_obj_ids,
            memory_embeddings=memory_embeddings,
            memory_pos_embeddings=memory_pos_embeddings,
            results=batched_results,
            prompts=prompts,
        )

        self.memory_bank.prune_memories(
            obj_ids=all_obj_ids,
            current_frame_idx=frame_idx,
        )
        
        return {obj_id: result for obj_id, result in zip(all_obj_ids, batched_results)}
