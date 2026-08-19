# SPDX-License-Identifier: LicenseRef-SAM
"""The session wrapper over a video predictor's ``forward`` + state.

A :class:`VideoSession` binds a predictor to one video: it owns the state, counts
frames, and (SAM 3 concept sessions) carries the concept declared at birth. It adds
no model logic -- ``session.process(frame, ...)`` is exactly one ``forward`` call, so
anything measured against a golden can use either form interchangeably.

Sessions are independent: hold several over one loaded model and interleave them
freely (each counts its own frames). The underlying state stays reachable as
``session.state`` for callers that need the explicit form.
"""
from __future__ import annotations

import operator
from typing import Any, Callable


def validate_video_hw(video_hw) -> tuple[int, int]:
    """Normalize a state's ``video_hw`` to an ``(int, int)`` tuple, or raise.

    A predictor state otherwise only learns its frame size is wrong when the first
    frame's shape assert fires, several calls later and pointing at the frame rather
    than at the caller. Checking here keeps the error where the mistake was made --
    e.g. a phrase passed positionally, which ``tuple()`` would happily spread into
    six one-character "dimensions".

    Integer-like values (``numpy`` ints, ``torch.Size`` entries) are accepted and
    narrowed to ``int``; ``bool`` is not an image dimension.

    Raises:
        ValueError: if ``video_hw`` is None, is not a pair, or holds a non-integer
            or non-positive dimension.
    """
    if video_hw is None:
        raise ValueError("Video height and width cannot be None")
    bad = ValueError(
        f"video_hw must be a pair of positive ints (height, width), got {video_hw!r}"
    )
    try:
        hw = tuple(video_hw)
    except TypeError:
        raise bad from None
    if len(hw) != 2:
        raise bad
    dims = []
    for v in hw:
        if isinstance(v, bool):
            raise bad
        try:
            v = operator.index(v)  # np.int64 / torch.Size entries pass; floats do not
        except TypeError:
            raise bad from None
        if v <= 0:
            raise bad
        dims.append(v)
    return (dims[0], dims[1])


class VideoSession:
    """One video's worth of streaming inference over a shared predictor.

    Built by a predictor's ``start_session`` / ``start_concept_session`` -- not
    directly. The size is never passed in: the state is built from the FIRST frame's
    shape, so ``state`` is None until then. Every frame that a predictor can process
    carries its own size unambiguously (``(H, W, C)`` arrays, ``(C, H, W)`` tensors),
    and a caller who wants the size pinned up front builds the state explicitly and
    calls ``forward`` -- which is where every other state option lives anyway.

    Args:
        predictor: the predictor whose ``forward`` this session drives.
        make_state: builds the predictor's state class for a given ``(H, W)``.
        on_state: applied once to the freshly built state (e.g. set the concept).
    """

    def __init__(
        self,
        predictor,
        make_state: Callable[[tuple[int, int]], Any],
        on_state: Callable[[Any], None] | None = None,
    ) -> None:
        self._predictor = predictor
        self._make_state = make_state
        self._on_state = on_state
        self._next_frame_idx = 0
        self._state = None

    def _build_state(self, video_hw: tuple[int, int]) -> None:
        self._state = self._make_state(tuple(video_hw))
        if self._on_state is not None:
            self._on_state(self._state)

    @staticmethod
    def _frame_hw(frame) -> tuple[int, int]:
        """``(H, W)`` of a frame: ``(H, W, C)`` arrays or ``(C, H, W)`` tensors."""
        import torch

        if isinstance(frame, torch.Tensor):
            return tuple(frame.shape[-2:])
        return tuple(frame.shape[:2])

    @property
    def state(self):
        """The underlying predictor state (None until the first frame arrives)."""
        return self._state

    def process(self, frame, prompts=None, frame_idx: int | None = None, **kwargs):
        """Process one frame; returns the predictor's ``dict[obj_id, MaskletResult]``.

        Args:
            frame: the frame, in whatever form the predictor's ``forward`` takes.
            prompts: optional :class:`~sam.prompts.GeometryPrompt` list for this frame.
            frame_idx: override the internal counter (e.g. to re-run a frame); the
                counter resumes from ``frame_idx + 1``.
            **kwargs: passed through to the predictor's ``forward``.

        In a ``start_concept_session`` session, tracker prompts (click / box / mask)
        are for REFINING ids the session already returned. Seeding a new object with
        them is an id minefield: detection spawns its own ids in the same call, before
        tracker prompts apply, so a fresh id can collide and silently refine a detector
        object instead. Seed new objects in a ``start_session`` session.
        """
        if self._state is None:
            self._build_state(self._frame_hw(frame))
        idx = self._next_frame_idx if frame_idx is None else frame_idx
        self._next_frame_idx = idx + 1
        return self._predictor.forward(
            self._state, idx, frame, prompts=prompts or [], **kwargs
        )
