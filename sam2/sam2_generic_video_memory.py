import torch
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ObjectMemory:
    obj_id: int
    frame_idx: int
    memory_embeddings: torch.Tensor
    memory_pos_embeddings: torch.Tensor
    masks_logits: torch.Tensor
    best_mask_idx: torch.Tensor
    ious: torch.Tensor
    obj_ptrs: torch.Tensor
    object_score_logits: torch.Tensor
    is_prompt: bool

    @property
    def best_mask_logits(self) -> torch.Tensor:
        # Select the best mask based on IoU scores
        best_mask_idx = torch.argmax(self.ious, dim=1, keepdim=True)
        batch_indices = torch.arange(self.masks_logits.shape[0], device=self.masks_logits.device)
        
        # Extract the best mask for each item in the batch
        best_masks_logits = self.masks_logits[batch_indices, best_mask_idx]
        return best_masks_logits
    
    @property
    def best_iou(self) -> torch.Tensor:
        best_mask_idx = torch.argmax(self.ious, dim=1, keepdim=True)
        batch_indices = torch.arange(self.ious.shape[0], device=self.ious.device)
        return self.ious[batch_indices, best_mask_idx]


@dataclass
class ObjectMemoryBank:
    obj_id: int

    def __post_init__(self):
        self.volatile_memories: list[ObjectMemory] = []
        self.prompt_memories: list[ObjectMemory] = []

    def add(
        self,
        memory: ObjectMemory,
    ):
        if memory.is_prompt:
            self.prompt_memories.append(memory)
        else:
            self.volatile_memories.append(memory)

    def remove(
        self,
        memory: ObjectMemory,
    ):
        if memory.is_prompt:
            self.prompt_memories.remove(memory)
        else:
            self.volatile_memories.remove(memory)

    def clear_memories(self):
        self.volatile_memories = []
        self.prompt_memories = []


class MemorySelectionStrategy(ABC):
    @abstractmethod
    def select_prompt_memories(
        self, memory_bank: ObjectMemoryBank, n_max_prompt_memories: int = -1
    ) -> list[ObjectMemory]:
        """
        Select prompt memories from the memory bank.

        Args:
            memory_bank (ObjectMemoryBank): The memory bank to select from.
            n_max_prompt_memories (int): The maximum number of prompt memories to select. If -1, no limit is applied.

        Returns:
            list[ObjectMemory]: A list of the selected prompt memories.
        """
        raise NotImplementedError

    @abstractmethod
    def select_volatile_memories(
        self, memory_bank: ObjectMemoryBank, n_max_volatile_memories: int = -1
    ) -> list[ObjectMemory]:
        """
        Select volatile memories from the memory bank.

        Args:
            memory_bank (ObjectMemoryBank): The memory bank to select from.
            n_max_volatile_memories (int): The maximum number of volatile memories to select. If -1, no limit is applied.

        Returns:
            list[ObjectMemory]: A list of the selected volatile memories.
        """
        raise NotImplementedError

    @abstractmethod
    def select_object_memories(
        self, memory_bank: ObjectMemoryBank, n_max_volatile_object_memories: int = -1
    ) -> list[ObjectMemory]:
        """
        Select object memories from the memory bank.

        Args:
            memory_bank (ObjectMemoryBank): The memory bank to select from.
            n_max_object_memories (int): The maximum number of object memories to select. If -1, no limit is applied.

        Returns:
            list[ObjectMemory]: A list of the selected object memories.
        """
        raise NotImplementedError


class MemoryMemorizationStrategy(ABC):

    @abstractmethod
    def try_add_memory_to_memory_bank(
        self, memory: ObjectMemory, memory_bank: ObjectMemoryBank
    ) -> bool:
        """
        Try to add a memory to the memory bank.

        Args:
            memory (ObjectMemory): The memory to add.
            memory_bank (ObjectMemoryBank): The memory bank to add the memory to.

        Returns:
            bool: True if the memory was added, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def prune_memories_from_memory_bank(
        self, memory_bank: ObjectMemoryBank
    ) -> list[ObjectMemory]:
        """
        Prune memories from the memory bank.

        Args:
            memory_bank (ObjectMemoryBank): The memory bank to prune.

        Returns:
            list[ObjectMemory]: The memories that were pruned.
        """
        raise NotImplementedError


class DefaultMemoryMemorizationStrategy(MemoryMemorizationStrategy):

    def __init__(
        self,
        volatile_memory_bank_max_size: int = 10,
        prompt_memory_bank_max_size: int = -1,
    ):
        self.volatile_memory_bank_max_size = volatile_memory_bank_max_size
        self.prompt_memory_bank_max_size = prompt_memory_bank_max_size

    def try_add_memory_to_memory_bank(
        self, memory: ObjectMemory, memory_bank: ObjectMemoryBank
    ) -> bool:
        memory_bank.add(memory)
        return True

    def prune_memories_from_memory_bank(self, memory_bank: ObjectMemoryBank) -> list[ObjectMemory]:
        pruned_memories: list[ObjectMemory] = []

        # Prune memories that are too old
        if (
            self.volatile_memory_bank_max_size > 0
            and len(memory_bank.volatile_memories) > self.volatile_memory_bank_max_size
        ):
            # Sort by frame_idx in descending order
            sorted_volatile_memories = sorted(
                memory_bank.volatile_memories, key=lambda x: x.frame_idx, reverse=True
            )
            kept_volatile_memories = sorted_volatile_memories[
                : self.volatile_memory_bank_max_size
            ]
            pruned_volatile_memories = sorted_volatile_memories[
                self.volatile_memory_bank_max_size :
            ]
            pruned_memories.extend(pruned_volatile_memories)

            # Only keep the last N memories (closest to the current frame)
            memory_bank.volatile_memories = kept_volatile_memories

        if (
            self.prompt_memory_bank_max_size > 0
            and len(memory_bank.prompt_memories) > self.prompt_memory_bank_max_size
        ):
            # Sort by frame_idx in descending order
            sorted_prompt_memories = sorted(
                memory_bank.prompt_memories, key=lambda x: x.frame_idx, reverse=True
            )
            kept_prompt_memories = sorted_prompt_memories[
                : self.prompt_memory_bank_max_size
            ]
            pruned_prompt_memories = sorted_prompt_memories[
                self.prompt_memory_bank_max_size :
            ]
            pruned_memories.extend(pruned_prompt_memories)

            # Only keep the last N memories (closest to the current frame)
            memory_bank.prompt_memories = kept_prompt_memories

        return pruned_memories


class DefaultMemorySelectionStrategy(MemorySelectionStrategy):

    def select_volatile_memories(
        self, memory_bank: ObjectMemoryBank, n_max_volatile_memories: int = -1
    ) -> list[ObjectMemory]:
        """
        Select memories from the volatile memory bank to condition on.
        """
        # Only keep the last N memories closest to the current frame. This is similar to SAM2 initial memory selection.
        selected_memories = sorted(
            memory_bank.volatile_memories, key=lambda x: x.frame_idx
        )

        if n_max_volatile_memories > 0:
            selected_memories = selected_memories[-n_max_volatile_memories:]

        # Conditioning expects the most important memories to be first
        selected_memories = selected_memories[::-1]
        return selected_memories

    def select_prompt_memories(
        self, memory_bank: ObjectMemoryBank, n_max_prompt_memories: int = -1
    ) -> list[ObjectMemory]:
        """
        Select memories from the prompt memory bank to condition on.
        """
        selected_memories = sorted(
            memory_bank.prompt_memories, key=lambda x: x.frame_idx
        )

        # If n_max_prompt_memories == -1, no limit on number of prompt memories
        if n_max_prompt_memories > 0:
            selected_memories = selected_memories[-n_max_prompt_memories:]

        # Conditioning expects the most important memories to be first
        selected_memories = selected_memories[::-1]
        return selected_memories

    def select_object_memories(
        self, memory_bank: ObjectMemoryBank, n_max_volatile_object_memories: int = -1
    ) -> list[ObjectMemory]:
        """
        Select memories from the object memory bank to condition on.
        """
        selected_object_memories: list[ObjectMemory] = []
        # Add all object memories from the prompt memory bank
        selected_object_memories.extend(memory_bank.prompt_memories)

        # Add object memories from the volatile memory bank
        selected_volatile_obj_memories = sorted(
            memory_bank.volatile_memories, key=lambda x: x.frame_idx
        )
        if n_max_volatile_object_memories > 0:
            # Only keep the last N object memories
            selected_volatile_obj_memories = selected_volatile_obj_memories[
                -n_max_volatile_object_memories:
            ]
        selected_object_memories.extend(selected_volatile_obj_memories)
        return selected_object_memories
