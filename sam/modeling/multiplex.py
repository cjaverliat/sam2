# SPDX-License-Identifier: LicenseRef-SAM
# Vendored from facebookresearch/sam3 @ 5dd401d (sam3/model/multiplex_utils.py): the SAM 3.1
# MULTIPLEX state + controller. ``MultiplexState`` converts tensors between the *data* space
# ``(num_objects, ...)`` and the *multiplex* space ``(num_buckets, multiplex_count, ...)`` via a
# pair of precomputed partial-permutation matrices (``mux`` / ``demux``); ``MultiplexController``
# builds a per-frame state from the active-object count (``num_buckets = ceil(N / K)``). This is
# the mux/demux machinery the ``Sam3MultiplexTracker`` wraps around its bucket-space decode (spec
# §8); the bank / streaming loop still see per-object tensors because demux precedes storage.
# Pure tensor/Python logic -- nothing stripped (no Triton / flash / einops / training deps).
"""SAM 3.1 multiplex state + controller (mux/demux precomputed permutations)."""

import logging
import math
from typing import Optional

import torch
from torch import nn

# Special values for object tracking
_PADDING_NUM = -1  # Marks empty slots in buckets
_REMOVED_NUM = -1116  # Marks objects that have been removed


logger = logging.getLogger(__name__)


class MultiplexState:
    """Records the multiplexing state for one or more buckets.

    Converts tensors between the data space ``(num_objects, ...)`` and the multiplex space
    ``(num_buckets, multiplex_count, ...)``. Each bucket has ``multiplex_count`` slots; not all
    need be filled. The batch size equals ``total_valid_entries`` (sum of valid entries over
    buckets). ``mux`` scatters data-space rows into their assigned slots (padding slots = 0);
    ``demux`` is its inverse (padding slots dropped).
    """

    def __init__(
        self,
        assignments: list,
        device: torch.device,
        dtype: torch.dtype,
        allowed_bucket_capacity: int,
        *,
        object_ids: Optional[list] = None,
    ):
        self.device = device
        self.dtype = dtype
        self.allowed_bucket_capacity = allowed_bucket_capacity
        self._initialize_assignments(assignments, object_ids=object_ids)

    def _initialize_assignments(self, assignments: list, *, object_ids: Optional[list] = None):
        self.assignments = assignments
        self.num_buckets = len(self.assignments)
        if self.num_buckets == 0:
            raise ValueError("No buckets found in the state")

        self.multiplex_count = len(self.assignments[0])
        assert all(
            len(self.assignments[i]) == self.multiplex_count
            for i in range(self.num_buckets)
        )

        self.total_valid_entries = sum(
            sum(1 for x in bucket if x >= 0) for bucket in self.assignments
        )
        self.total_non_padding_entries = sum(
            sum(1 for x in bucket if x != _PADDING_NUM) for bucket in self.assignments
        )

        self.object_ids = object_ids
        if self.object_ids is not None:
            assert len(self.object_ids) == self.total_valid_entries, (
                "object_ids should map 1:1 to the valid entries"
            )

        all_object_idxs = set()
        for bucket in self.assignments:
            valid_entries_in_bucket = sum(1 for x in bucket if x != _PADDING_NUM)
            assert valid_entries_in_bucket <= self.allowed_bucket_capacity, (
                f"{valid_entries_in_bucket=} > {self.allowed_bucket_capacity=}"
            )
            for obj_idx in bucket:
                if obj_idx >= 0:
                    assert obj_idx < self.total_non_padding_entries, (
                        f"object ID {obj_idx} >= {self.total_non_padding_entries}"
                    )
                    assert obj_idx not in all_object_idxs, "object IDs must be unique"
                    all_object_idxs.add(obj_idx)

        self._precompute_transition_matrices(self.device, self.dtype)

    @property
    def available_slots(self) -> int:
        return (
            self.num_buckets * self.allowed_bucket_capacity
            - self.total_non_padding_entries
        )

    def find_next_batch_of_available_indices(
        self, num_objects: int, *, allow_new_buckets: bool = False,
        prefer_new_buckets: bool = False,
    ) -> list:
        assert num_objects > 0, f"{num_objects=} must be positive"
        if not allow_new_buckets:
            assert self.available_slots >= num_objects, (
                f"not enough available slots {self.available_slots} < {num_objects}"
            )
        return list(
            range(self.total_valid_entries, self.total_valid_entries + num_objects)
        )

    def add_objects(
        self, object_indices: list, *, object_ids: Optional[list] = None,
        allow_new_buckets: bool = False, prefer_new_buckets: bool = False,
    ):
        """Add new objects by filling empty slots and creating new buckets if necessary."""
        if len(object_indices) == 0:
            return
        object_indices = object_indices.copy()
        assert (object_ids is None) == (self.object_ids is None), (
            "object_ids must either be always given or always omitted"
        )
        if object_ids is not None:
            assert len(object_ids) == len(object_indices)
            object_ids = object_ids.copy()

        num_new_objects = len(object_indices)
        assert object_indices == sorted(object_indices), "object_indices must be sorted"
        object_indices.reverse()
        if object_ids is not None:
            object_ids.reverse()
        if prefer_new_buckets:
            assert allow_new_buckets, "prefer_new_buckets requires allow_new_buckets"

        def _pop_next():
            idx = object_indices.pop()
            if object_ids is not None and self.object_ids is not None:
                self.object_ids.append(object_ids.pop())
            return idx

        if not prefer_new_buckets:
            for bucket in self.assignments:
                for i in range(self.allowed_bucket_capacity):
                    if bucket[i] == _PADDING_NUM:
                        bucket[i] = _pop_next()
                        if len(object_indices) == 0:
                            break
                if len(object_indices) == 0:
                    break

        if len(object_indices) > 0 and not allow_new_buckets:
            raise ValueError(
                f"Cannot place objects {list(reversed(object_indices))} without new buckets"
            )

        while len(object_indices) > 0:
            new_bucket = [_PADDING_NUM] * self.multiplex_count
            for i in range(self.allowed_bucket_capacity):
                if len(object_indices) == 0:
                    break
                new_bucket[i] = _pop_next()
            self.assignments.append(new_bucket)

        original_num_entries = self.total_valid_entries
        self._initialize_assignments(self.assignments, object_ids=self.object_ids)
        assert self.total_valid_entries == original_num_entries + num_new_objects

    def remove_objects(self, object_indices: list, strict: bool = True):
        """Remove objects (mark as removed); drop a bucket if all its slots are removed/padding."""
        object_indices = object_indices.copy()
        for bucket_idx, bucket in enumerate(self.assignments):
            for slot_idx, obj_id in enumerate(bucket):
                if obj_id in object_indices:
                    self.assignments[bucket_idx][slot_idx] = _REMOVED_NUM
                    object_indices.remove(obj_id)
        if strict:
            assert len(object_indices) == 0, f"Failed to remove objects: {object_indices}"

        buckets_to_remove, buckets_to_keep = [], []
        for bucket_idx, bucket in enumerate(self.assignments):
            if all(obj_id in [_PADDING_NUM, _REMOVED_NUM] for obj_id in bucket):
                buckets_to_remove.append(bucket_idx)
            else:
                buckets_to_keep.append(bucket_idx)
        for bucket_idx in reversed(buckets_to_remove):
            del self.assignments[bucket_idx]

        if len(buckets_to_keep) == 0:
            self.assignments = None
            if self.object_ids is not None:
                self.object_ids = []
            return buckets_to_keep

        all_positive_ids = set()
        for bucket in self.assignments:
            for obj_id in bucket:
                if obj_id >= 0:
                    all_positive_ids.add(obj_id)
        sorted_ids = sorted(all_positive_ids)
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted_ids)}
        for bucket in self.assignments:
            for i, obj_id in enumerate(bucket):
                if obj_id >= 0:
                    bucket[i] = id_mapping[obj_id]

        if self.object_ids is not None:
            new_object_ids = [None] * len(sorted_ids)
            for old_idx, new_idx in id_mapping.items():
                new_object_ids[new_idx] = self.object_ids[old_idx]
            assert not any(obj_id is None for obj_id in new_object_ids)
            self.object_ids = new_object_ids

        self._initialize_assignments(self.assignments, object_ids=self.object_ids)
        return buckets_to_keep

    def _precompute_transition_matrices(self, device: torch.device, dtype: torch.dtype):
        """Precompute the (partial-permutation) mux / demux matrices."""
        self.mux_matrix = torch.zeros(
            self.num_buckets * self.multiplex_count,
            self.total_valid_entries,
            device=device, dtype=dtype,
        )
        self.demux_matrix = torch.zeros(
            self.total_valid_entries,
            self.num_buckets * self.multiplex_count,
            device=device, dtype=dtype,
        )
        for i in range(self.num_buckets):
            for j in range(self.multiplex_count):
                bucket_idx = i * self.multiplex_count + j
                object_idx = self.assignments[i][j]
                if object_idx >= 0:
                    self.mux_matrix[bucket_idx, object_idx] = 1.0
                    self.demux_matrix[object_idx, bucket_idx] = 1.0

    def mux(self, x: torch.Tensor) -> torch.Tensor:
        """``(total_valid_entries, ...)`` -> ``(num_buckets, multiplex_count, ...)`` (pad = 0)."""
        num_valid_entries = x.shape[0]
        assert num_valid_entries == self.total_valid_entries, (
            f"{num_valid_entries=} != {self.total_valid_entries=}"
        )
        output_shape = (self.num_buckets, self.multiplex_count) + x.shape[1:]
        x_flat = x.reshape(num_valid_entries, -1)
        result_flat = self.mux_matrix.to(x_flat.dtype) @ x_flat
        return result_flat.view(output_shape)

    def demux(self, x: torch.Tensor) -> torch.Tensor:
        """``(num_buckets, multiplex_count, ...)`` -> ``(total_valid_entries, ...)`` (drop pad)."""
        num_buckets, multiplex_count = x.shape[:2]
        assert num_buckets == self.num_buckets, f"{num_buckets=} != {self.num_buckets=}"
        assert multiplex_count == self.multiplex_count, (
            f"{multiplex_count=} != {self.multiplex_count=}"
        )
        output_shape = (self.total_valid_entries,) + x.shape[2:]
        x_flat = x.reshape(num_buckets * multiplex_count, -1)
        result_flat = self.demux_matrix.to(x_flat.dtype) @ x_flat
        return result_flat.view(output_shape)

    def get_valid_object_mask(self) -> torch.Tensor:
        """``(num_buckets, multiplex_count)`` -- 1 for valid slots, 0 for padding."""
        valid_mask = self.mux_matrix.sum(dim=1) > 0
        return valid_mask.reshape(self.num_buckets, self.multiplex_count)

    def get_all_valid_object_idx(self) -> set:
        """The set of valid internal object indices (not the arbitrary input object IDs)."""
        return {
            obj_idx for bucket in self.assignments for obj_idx in bucket if obj_idx >= 0
        }


class MultiplexController(nn.Module):
    def __init__(
        self, multiplex_count: int, full_shuffle: bool = False,
        eval_multiplex_count: int = -1,
    ):
        super().__init__()
        self.multiplex_count = multiplex_count
        self.full_shuffle = full_shuffle
        if eval_multiplex_count < 0:
            self.eval_multiplex_count = multiplex_count
        else:
            self.eval_multiplex_count = eval_multiplex_count
        assert self.multiplex_count >= 1

    @property
    def allowed_bucket_capacity(self) -> int:
        if self.training:
            return self.multiplex_count
        return self.eval_multiplex_count

    def get_state(
        self, num_valid_entries: int, device: torch.device, dtype: torch.dtype,
        random: bool = True, *, object_ids: Optional[list] = None,
    ) -> MultiplexState:
        """Map ``num_valid_entries`` objects into buckets of ``multiplex_count`` slots.

        In eval (``random=False``) objects fill slots in order (identity embedding into the
        first ``num_valid_entries`` slots); the remaining slots are ``_PADDING_NUM``.
        """
        allowed_bucket_capacity = self.allowed_bucket_capacity
        true_bucket_capacity = self.multiplex_count
        num_buckets = math.ceil(num_valid_entries / allowed_bucket_capacity)

        if self.full_shuffle:
            ids = torch.cat([
                torch.arange(num_valid_entries, dtype=torch.long),
                torch.tensor(
                    [_PADDING_NUM] * (num_buckets * true_bucket_capacity - num_valid_entries),
                    dtype=torch.long,
                ),
            ], dim=0)
            if random:
                ids = ids[torch.randperm(ids.shape[0], dtype=torch.long)]
            assignments = [
                ids[i * true_bucket_capacity : (i + 1) * true_bucket_capacity].tolist()
                for i in range(num_buckets)
            ]
        else:
            if random:
                ids = torch.randperm(num_valid_entries, dtype=torch.int64)
            else:
                ids = torch.arange(num_valid_entries)
            total_elements = num_buckets * allowed_bucket_capacity
            if ids.shape[0] < total_elements:
                ids = torch.cat([
                    ids,
                    torch.tensor([_PADDING_NUM] * (total_elements - ids.shape[0])),
                ])
            assignments = [
                ids[i * allowed_bucket_capacity : (i + 1) * allowed_bucket_capacity].tolist()
                + [_PADDING_NUM] * (true_bucket_capacity - allowed_bucket_capacity)
                for i in range(num_buckets)
            ]

        return MultiplexState(
            assignments, device, dtype,
            allowed_bucket_capacity=allowed_bucket_capacity, object_ids=object_ids,
        )
