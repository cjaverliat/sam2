from __future__ import annotations

import bisect

import torch

from sam2.modeling.memory import ObjectMemory
from sam2.modeling.sam2_memory import SAM2ObjectMemoryBank
from sam2.modeling.sam2_result import SAM2Result
from sam2.modeling.sam2_prompt import SAM2Prompt


class SAM2ScoredPeriodicObjectMemoryBank(SAM2ObjectMemoryBank):
    """
    Memory bank that keeps non-conditional memories by recency AND quality.

    Non-conditional memories are stored only when BOTH gates pass:
      - Period gate: at least ``storage_period`` frames have elapsed since the last
        stored non-conditional memory for that object.
      - Score gate: the frame's object-presence probability is ``>= score_threshold``.
        The score is ``sigmoid(obj_score_logits)`` in ``[0, 1]`` (0 = surely absent,
        1 = certainly present); ``0.5`` matches SAM2's own present/absent boundary
        (``is_obj_appearing = obj_score_logits > 0``). The probability is stored on
        ``ObjectMemory.score``.

    "Keep probing" semantics: once the period has elapsed, every subsequent frame is
    checked and the first one that also passes the score gate is stored (which resets
    the period). Conditional (prompt) memories are always stored, unchanged.

    Because stored non-conditional frames are sparse and irregular, selection uses
    recency (the N most-recent stored memories) instead of the base class's exact
    frame-index / stride probing.

    Old non-conditional memories are forgotten like ``SAM2ForgetfulObjectMemoryBank``:
    if ``memory_window_size`` is set, any non-conditional memory whose frame index is
    outside ``[current_frame_idx - memory_window_size, current_frame_idx +
    memory_window_size]`` is pruned each frame. ``None`` keeps them indefinitely.
    Conditional memories are never pruned.

    Note: gating is symmetric for forward and reverse tracking (uses absolute frame
    distance). Selection is direction-aware.
    """

    def __init__(
            self,
            score_threshold: float = 0.5,
            storage_period: int = 1,
            memory_window_size: int | None = None,
            memory_temporal_stride: int = 1,
            storage_device: torch.device = torch.device("cpu"),
    ):
        super().__init__(
            memory_temporal_stride=memory_temporal_stride,
            storage_device=storage_device,
        )
        assert storage_period >= 1, f"storage_period must be >= 1, got {storage_period}"
        assert (
            0.0 <= score_threshold <= 1.0
        ), f"score_threshold is a probability in [0, 1], got {score_threshold}"
        assert (
            memory_window_size is None or memory_window_size >= 0
        ), f"memory_window_size must be >= 0 or None, got {memory_window_size}"
        self.score_threshold = score_threshold
        self.storage_period = storage_period
        self.memory_window_size = memory_window_size
        # Per-object frame index of the last stored non-conditional memory.
        self._last_stored_non_cond_frame: dict[int, int] = {}

    def try_add_memories(
            self,
            frame_idx: int,
            obj_ids: list[int],
            memory_embeddings: torch.Tensor,
            memory_pos_embeddings: torch.Tensor,
            results: SAM2Result,
            prompts: list[SAM2Prompt],
    ) -> list[tuple[bool, ObjectMemory]]:
        n_objs = len(obj_ids)
        assert len(set(obj_ids)) == len(
            obj_ids
        ), f"obj_ids must be unique, got {obj_ids}"
        assert (
                memory_embeddings.ndim == 4
        ), f"Expected memory_embeddings to be of shape (B, N, H, W), got {memory_embeddings.shape}"
        assert (
                memory_pos_embeddings.ndim == 4
        ), f"Expected memory_pos_embeddings to be of shape (B, N, H, W), got {memory_pos_embeddings.shape}"
        assert (
                memory_embeddings.shape[0] == n_objs
        ), f"Expected memory_embeddings to have batch size {n_objs}, got {memory_embeddings.shape[0]}"
        assert (
                memory_pos_embeddings.shape[0] == n_objs
        ), f"Expected memory_pos_embeddings to have batch size {n_objs}, got {memory_pos_embeddings.shape[0]}"
        assert (
                results.batch_size == n_objs
        ), f"Expected {n_objs} results, got {results.batch_size}"

        prompts_dict = {p.obj_id: p for p in prompts}
        prompts = [prompts_dict.get(obj_id, None) for obj_id in obj_ids]

        ret = []

        for i, obj_id in enumerate(obj_ids):
            memory_embedding = memory_embeddings[[i]]
            memory_pos_embedding = memory_pos_embeddings[[i]]
            result = results[i]
            prompt = prompts[i]
            is_conditional = prompt is not None
            # Object-presence probability in [0, 1]. obj_score_logits is [1, 1] here
            # (one presence logit per object), so squeeze to a scalar.
            score = torch.sigmoid(result.obj_score_logits).squeeze().item()

            self.known_obj_ids.add(obj_id)

            memory = ObjectMemory(
                obj_id=obj_id,
                frame_idx=frame_idx,
                memory_embeddings=memory_embedding,
                memory_pos_embeddings=memory_pos_embedding,
                ptr=result.obj_ptrs,
                is_conditional=is_conditional,
                score=score,
            )
            memory = memory.to(self.storage_device)

            if is_conditional:
                # Conditional memories are always stored (bypass both gates).
                self._insert_memory(self.conditional_memories, obj_id, memory)
                ret.append((True, memory))
                continue

            # Non-conditional: apply the period + score gates.
            last = self._last_stored_non_cond_frame.get(obj_id, None)
            period_ok = last is None or abs(frame_idx - last) >= self.storage_period
            if period_ok and score >= self.score_threshold:
                self._insert_memory(self.non_conditional_memories, obj_id, memory)
                self._last_stored_non_cond_frame[obj_id] = frame_idx
                ret.append((True, memory))
            else:
                ret.append((False, memory))

        return ret

    def prune_memories(
            self, obj_ids: list[int], current_frame_idx: int
    ) -> dict[int, list[ObjectMemory]]:
        """Forget non-conditional memories outside the window
        ``[current - memory_window_size, current + memory_window_size]``.
        No-op when ``memory_window_size`` is None. Conditional memories are kept."""
        if self.memory_window_size is None:
            return {}

        removed_memories: dict[int, list[ObjectMemory]] = {}
        lo = current_frame_idx - self.memory_window_size
        hi = current_frame_idx + self.memory_window_size

        for obj_id in obj_ids:
            non_cond_obj_memories = self.non_conditional_memories.get(obj_id, [])
            kept, removed = [], []
            for m in non_cond_obj_memories:
                if lo <= m.frame_idx <= hi:
                    kept.append(m)
                else:
                    removed.append(m)
            if removed:
                removed_memories[obj_id] = removed
            self.non_conditional_memories[obj_id] = kept

        return removed_memories

    @staticmethod
    def _insert_memory(
            store: dict[int, list[ObjectMemory]], obj_id: int, memory: ObjectMemory
    ) -> None:
        """Insert a memory into a per-object frame-sorted list, replacing any memory
        for the same frame_idx."""
        obj_memories = store.setdefault(obj_id, [])
        pos = bisect.bisect_left([m.frame_idx for m in obj_memories], memory.frame_idx)
        if pos < len(obj_memories) and obj_memories[pos].frame_idx == memory.frame_idx:
            obj_memories[pos] = memory
        else:
            obj_memories.insert(pos, memory)

    def _select_non_conditional_memories(
            self,
            non_conditional_memories: list[ObjectMemory],
            N: int,
            current_frame_idx: int,
            reverse_tracking: bool,
    ) -> list[ObjectMemory]:
        """Select the N most-recent stored non-conditional memories strictly in the
        past (or future when reverse tracking), by recency rather than exact frame
        index. Memories are assumed sorted by frame_idx ascending.

        Returns closest-first (most-recent first), matching the ordering of the base
        class's stride selector so downstream temporal-position-encoding assignment
        (t_pos = num_maskmem - 1 - i) is consistent."""
        if N <= 0:
            return []

        if reverse_tracking:
            # Future frames, closest first -> the N closest (smallest frame_idx).
            candidates = [m for m in non_conditional_memories if m.frame_idx > current_frame_idx]
            return candidates[:N]
        else:
            # Past frames, closest = largest frame_idx -> the last N, closest first.
            candidates = [m for m in non_conditional_memories if m.frame_idx < current_frame_idx]
            return list(reversed(candidates[-N:]))
