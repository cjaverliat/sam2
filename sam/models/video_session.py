# SPDX-License-Identifier: LicenseRef-SAM
"""The session wrapper over a video predictor's ``forward`` + state.

A :class:`VideoSession` binds a predictor to one video: it owns the state, counts
frames, and (SAM 3) carries the concept declared at ``start_session``. It adds no
model logic -- ``session(frame, ...)`` is exactly one ``forward`` call, so anything
measured against a golden can use either form interchangeably.

Sessions are independent: hold several over one loaded model and interleave them
freely (each counts its own frames). The underlying state stays reachable as
``session.state`` for callers that need the explicit form.
"""
from __future__ import annotations

from typing import Any, Callable


class VideoSession:
    """One video's worth of streaming inference over a shared predictor.

    Built by a predictor's ``start_session`` -- not directly. ``video_hw`` may be
    deferred: the state is then created from the first frame's shape.

    Args:
        predictor: the predictor whose ``forward`` this session drives.
        make_state: builds the predictor's state class for a given ``(H, W)``.
        video_hw: the video size, or None to infer it from the first frame.
        on_state: applied once to the freshly built state (e.g. set the concept).
    """

    def __init__(
        self,
        predictor,
        make_state: Callable[[tuple[int, int]], Any],
        video_hw: tuple[int, int] | None = None,
        on_state: Callable[[Any], None] | None = None,
    ) -> None:
        self._predictor = predictor
        self._make_state = make_state
        self._on_state = on_state
        self._next_frame_idx = 0
        self._state = None
        if video_hw is not None:
            self._build_state(video_hw)

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

        In a concept session, tracker prompts (click / box / mask) are for REFINING
        ids the session already returned. Seeding a new object with them is an id
        minefield: detection spawns its own ids in the same call, before tracker
        prompts apply, so a fresh id can collide and silently refine a detector
        object instead. Seed new objects in a concept-free session.
        """
        if self._state is None:
            self._build_state(self._frame_hw(frame))
        idx = self._next_frame_idx if frame_idx is None else frame_idx
        self._next_frame_idx = idx + 1
        return self._predictor.forward(
            self._state, idx, frame, prompts=prompts or [], **kwargs
        )
