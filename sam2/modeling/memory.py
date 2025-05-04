from __future__ import annotations
import torch

from abc import ABC, abstractmethod
from sam2.modeling.sam2_prompt import SAM2Prompt
from sam2.modeling.sam2_result import SAM2Result

class ObjectMemory:

    def __init__(
        self,
        obj_id: int,
        frame_idx: int,
        memory_embeddings: torch.Tensor,
        memory_pos_embeddings: torch.Tensor,
        ptr: torch.Tensor,
        is_conditional: bool = False,
    ):
        self.obj_id = obj_id
        self.frame_idx = frame_idx
        self.memory_embeddings = memory_embeddings
        self.memory_pos_embeddings = memory_pos_embeddings
        self.ptr = ptr
        self.is_conditional = is_conditional
    
    @abstractmethod
    def to(self, device: torch.device) -> ObjectMemory:
        return ObjectMemory(
            obj_id=self.obj_id,
            frame_idx=self.frame_idx,
            memory_embeddings=self.memory_embeddings.to(device),
            memory_pos_embeddings=self.memory_pos_embeddings.to(device),
            ptr=self.ptr.to(device),
        )

class ObjectMemorySelection:

    def __init__(
        self,
        conditional_memories: list[ObjectMemory],
        non_conditional_memories: list[ObjectMemory],
        ptr_memories: list[ObjectMemory],
    ):
        self.conditional_memories = conditional_memories
        self.non_conditional_memories = non_conditional_memories
        self.ptr_memories = ptr_memories

    def to(self, device: torch.device) -> ObjectMemorySelection:
        return ObjectMemorySelection(
            conditional_memories=[
                memory.to(device) for memory in self.conditional_memories
            ],
            non_conditional_memories=[
                memory.to(device) for memory in self.non_conditional_memories
            ],
            ptr_memories=[memory.to(device) for memory in self.ptr_memories],
        )


class ObjectMemoryBank(ABC):

    def __init__(self):
        self.known_obj_ids = set()

    def clear_known_obj_ids(self):
        self.known_obj_ids = set()

    @abstractmethod
    def try_add_memories(
        self,
        frame_idx: int,
        obj_ids: list[int],
        memory_embeddings: list[torch.Tensor],
        memory_pos_embeddings: list[torch.Tensor],
        results: list[SAM2Result],
        prompts: list[SAM2Prompt | None],
    ) -> list[tuple[bool, ObjectMemory]]:
        """
        Try to add memories to the memory bank.

        Args:
            frame_idx: The frame index.
            obj_ids: The object IDs.
            memory_embeddings: The memory embeddings.
            memory_pos_embeddings: The memory positional embeddings.
            results: The results from SAM.
            prompts: The prompts.

        Returns:
            A list of tuples containing a boolean indicating whether the memory was added and the memory itself.
        """
        raise NotImplementedError

    @abstractmethod
    def prune_memories(self, obj_ids: list[int], current_frame_idx: int) -> dict[int, list[ObjectMemory]]:
        """
        Remove memories that are no longer needed for a list of objects and return the list of pruned memories.

        Args:
            obj_ids: The object IDs.
            current_frame_idx: The current frame index.

        Returns:
            A dictionary mapping object IDs to lists of pruned memories.
        """
        raise NotImplementedError

    @abstractmethod
    def select_memories(
        self,
        obj_ids: list[int],
        current_frame_idx: int,
        max_conditional_memories: int,
        max_non_conditional_memories: int,
        max_ptr_memories: int,
        only_include_pointers_in_past: bool = False,
        reverse_tracking: bool = False,
    ) -> dict[int, ObjectMemorySelection]:
        """
        Select the memories for each object for conditioning the frame at current_frame_idx.

        Args:
            obj_ids: The object IDs to select memories for.
            max_conditional_memories: The maximum number of conditional memories to select.
            max_non_conditional_memories: The maximum number of non-conditional memories to select.
            max_object_memories: The maximum number of object memories (obj_ptrs) to select.
            current_frame_idx: The current frame index.
            reverse_tracking: Whether the tracking direction is reversed.

        Returns:
            A dictionary mapping object IDs to memory selections.
        """
        raise NotImplementedError

    @abstractmethod
    def clear_object_non_conditional_memories_in_frame_range(
        self, obj_id: int, frame_idx_start: int, frame_idx_end: int
    ) -> list[ObjectMemory]:
        """
        Clear the non-conditional memories for an object in a given frame range (inclusive).

        Args:
            obj_id: The object ID.
            frame_idx_start: The start frame index (inclusive).
            frame_idx_end: The end frame index (inclusive).

        Returns:
            A list of removed memories.
        """
        raise NotImplementedError
