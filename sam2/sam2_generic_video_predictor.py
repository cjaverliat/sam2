from dataclasses import dataclass

import numpy as np
import torch

from sam2.modeling.sam2_generic import SAM2Generic
from sam2.sam2_generic_video_memory import (
    MemoryMemorizationStrategy,
    MemorySelectionStrategy,
    ObjectMemory,
    ObjectMemoryBank,
)


@dataclass
class Prompt:
    obj_id: int
    points_coords: torch.Tensor | None = None
    points_labels: torch.Tensor | None = None
    boxes: torch.Tensor | None = None
    masks_logits: torch.Tensor | None = None

    def __post_init__(self):
        assert (
            self.points_coords is not None
            or self.boxes is not None
            or self.masks_logits is not None
        ), "At least one of points_coords, boxes, or masks_logits must be provided"


class SAM2GenericVideoPredictor(SAM2Generic):
    """
    SAM2GenericVideoPredictor provides a handy video prediction interface.

    Note: works in a forward-only manner.
    """

    def __init__(
        self,
        memory_selection_strategy: MemorySelectionStrategy,
        memory_memorization_strategy: MemoryMemorizationStrategy,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._video_hw: tuple[int, int] | None = None
        self.current_frame_idx = 0

        # Number of selectable memories is num_maskmem - 1 because the conditioning memory label (0) is included in num_maskmem
        self.n_max_selectable_volatile_memories = self.num_maskmem - 1
        self.n_max_selectable_prompt_memories = self.max_cond_frames_in_attn
        self.n_max_selectable_volatile_object_memories = self.max_obj_ptrs_in_encoder

        self.memory_selection_strategy = memory_selection_strategy
        self.memory_memorization_strategy = memory_memorization_strategy

        self.object_memory_bank: dict[int, ObjectMemoryBank] = {}

    def get_or_create_object_memory_bank(self, obj_id: int) -> ObjectMemoryBank:
        if obj_id not in self.object_memory_bank:
            self.object_memory_bank[obj_id] = ObjectMemoryBank(obj_id=obj_id)
        return self.object_memory_bank[obj_id]

    @torch.inference_mode()
    def forward(
        self, frame: torch.Tensor, object_prompts: list[Prompt] = []
    ) -> dict[int, ObjectMemory]:
        # First frame, initialize video_hw
        if self._video_hw is None:
            self._video_hw = frame.shape[-2:]

        assert frame.shape in [
            (1, *self._video_hw),
            (3, *self._video_hw),
        ], f"Expected frame to be of shape (C, H, W) or (1, C, H, W) with H and W equal to {self._video_hw}, got {frame.shape}"

        img_embeddings, img_pos_embeddings = self.encode_image(frame)

        assert object_prompts is None or np.unique(
            [p.obj_id for p in object_prompts]
        ).size == len(object_prompts), "Only one prompt per object should be provided"

        prompts_dicts: dict[int, Prompt | None] = {
            obj_id: None for obj_id in self.object_memory_bank.keys()
        }
        prompts_dicts.update({p.obj_id: p for p in object_prompts})

        results: dict[int, ObjectMemory] = {}

        for obj_id, prompt in prompts_dicts.items():

            object_memory_bank = self.get_or_create_object_memory_bank(obj_id)

            has_prompt = prompt is not None

            if has_prompt:

                prompt_embeddings = self.encode_prompts(
                    orig_hw=self._video_hw,
                    points_coords=prompt.points_coords,
                    points_labels=prompt.points_labels,
                    boxes=prompt.boxes,
                    masks_logits=prompt.masks_logits,
                )

                masks_logits, ious, obj_ptrs, object_score_logits = self.generate_masks(
                    orig_hw=self._video_hw,
                    img_embeddings=img_embeddings,
                    prompt_embeddings=prompt_embeddings,
                    multimask_output=True,
                )

            else:
                # No prompt, so we condition the image embeddings on the memory to find the object
                prompt_memories = self.memory_selection_strategy.select_prompt_memories(
                    memory_bank=object_memory_bank,
                    n_max_prompt_memories=self.n_max_selectable_prompt_memories,
                )

                volatile_memories = (
                    self.memory_selection_strategy.select_volatile_memories(
                        memory_bank=object_memory_bank,
                        n_max_volatile_memories=self.n_max_selectable_volatile_memories,
                    )
                )

                object_memories = self.memory_selection_strategy.select_object_memories(
                    memory_bank=object_memory_bank,
                    n_max_volatile_object_memories=self.n_max_selectable_volatile_object_memories,
                )

                conditioned_img_embeddings = (
                    self.condition_image_embeddings_on_memories(
                        frame_idx=self.current_frame_idx,
                        img_embeddings=img_embeddings,
                        img_pos_embeddings=img_pos_embeddings,
                        non_conditional_memory_embeddings=[
                            memory.memory_embeddings for memory in volatile_memories
                        ],
                        non_conditional_memory_pos_embeddings=[
                            memory.memory_pos_embeddings for memory in volatile_memories
                        ],
                        conditional_memory_embeddings=[
                            memory.memory_embeddings for memory in prompt_memories
                        ],
                        conditional_memory_pos_embeddings=[
                            memory.memory_pos_embeddings for memory in prompt_memories
                        ],
                        obj_ptrs_seq=torch.stack(
                            [memory.obj_ptrs for memory in object_memories]
                        ),
                        obj_ptrs_frame_indices=[
                            memory.frame_idx for memory in object_memories
                        ],
                    )
                )

                masks_logits, ious, obj_ptrs, object_score_logits = self.generate_masks(
                    orig_hw=self._video_hw,
                    img_embeddings=conditioned_img_embeddings,
                    multimask_output=True,
                )

            # Select the best mask based on IoU scores
            best_mask_idx = torch.argmax(ious, dim=1, keepdim=True)
            batch_indices = torch.arange(
                masks_logits.shape[0], device=masks_logits.device
            )

            # Extract the best mask for each item in the batch
            best_masks_logits = masks_logits[batch_indices, best_mask_idx]

            memory_embeddings, memory_pos_embeddings = self.encode_memory(
                img_embeddings=img_embeddings,
                masks_logits=best_masks_logits,
                object_score_logits=object_score_logits,
                is_prompt=has_prompt,
            )

            memory = ObjectMemory(
                obj_id=obj_id,
                frame_idx=self.current_frame_idx,
                memory_embeddings=memory_embeddings,
                memory_pos_embeddings=memory_pos_embeddings,
                best_mask_idx=best_mask_idx,
                masks_logits=masks_logits,
                ious=ious,
                obj_ptrs=obj_ptrs,
                object_score_logits=object_score_logits,
                is_prompt=has_prompt,
            )

            self.memory_memorization_strategy.try_add_memory_to_memory_bank(
                memory=memory, memory_bank=object_memory_bank
            )

            self.memory_memorization_strategy.prune_memories_from_memory_bank(
                memory_bank=object_memory_bank
            )

            results[obj_id] = memory

        self.current_frame_idx += 1

        return results
