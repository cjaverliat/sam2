# SPDX-License-Identifier: LicenseRef-SAM
"""Tracklet lifecycle state machine: PENDING → CONFIRMED → DEAD.

Extracted from ``Sam3VideoBase.update_masklet_confirmation_status`` and
``_process_hotstart`` in upstream ``sam3/model/sam3_video_base.py``
(@ 5dd401d), simplified into a single, testable class.

State transitions
-----------------
- ``spawn(obj_id)``                : register a new PENDING tracklet.
- ``step(matched_track_ids, new_det_ids)`` : advance all alive tracklets.
    * matched for ``confirmation_thresh`` consecutive frames → CONFIRMED.
    * unmatched for ``kill_thresh`` consecutive frames       → DEAD.
- ``is_dead(obj_id)``              : True when the tracklet has been killed.
- ``get_state(obj_id)``            : return the current ``TrackletState``.
- ``active_ids()``                 : set of non-DEAD tracklet IDs.
- ``confirmed_ids()``              : set of CONFIRMED tracklet IDs.

Gating note
-----------
Upstream gates confirmation on *detection match* (``consecutive_det_num``
incremented only when the tracklet is matched by a detection in the current
frame), not a bare frame counter.  The kill is similarly gated on frames
where no detection is matched (``unmatched_count``).  Presence / object-score
signal from the tracker is the caller's responsibility: pass non-present
tracks *outside* ``matched_track_ids`` so the unmatched counter increments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Set


class TrackletState(Enum):
    PENDING = "pending"      # newly spawned; awaiting confirmation
    CONFIRMED = "confirmed"  # matched by detector for ≥ confirmation_thresh consecutive frames
    DEAD = "dead"            # unmatched for ≥ kill_thresh frames; no longer tracked


@dataclass
class _TrackletInfo:
    state: TrackletState = TrackletState.PENDING
    consecutive_det_count: int = 0   # consecutive frames with a det match
    unmatched_count: int = 0         # consecutive frames without a det match


class TrackletManager:
    """Explicit PENDING → CONFIRMED → DEAD state machine for concept-driven tracklets.

    Holds per-tracklet state keyed by ``obj_id`` (integer).  Designed to live
    inside ``Sam3VideoPredictorState`` (Task 6 dataclass) as a ``tracklet_mgr``
    field so the whole session state is in one place.

    Parameters
    ----------
    confirmation_thresh:
        Number of consecutive detection-matched frames before a PENDING
        tracklet is promoted to CONFIRMED.  Mirrors upstream
        ``masklet_confirmation_consecutive_det_thresh`` (default 3).
    kill_thresh:
        Number of consecutive unmatched frames before any alive tracklet is
        marked DEAD.  Mirrors upstream ``hotstart_unmatch_thresh`` (default 3).
    """

    def __init__(
        self,
        confirmation_thresh: int = 3,
        kill_thresh: int = 3,
    ) -> None:
        self.confirmation_thresh = confirmation_thresh
        self.kill_thresh = kill_thresh
        self._tracks: Dict[int, _TrackletInfo] = {}

    # ------------------------------------------------------------------
    # Mutation API
    # ------------------------------------------------------------------

    def spawn(self, obj_id: int) -> None:
        """Register a new PENDING tracklet with the given object ID."""
        if obj_id in self._tracks:
            raise ValueError(
                f"tracklet {obj_id!r} already registered; "
                "call remove() first or use a fresh obj_id"
            )
        self._tracks[obj_id] = _TrackletInfo()

    def step(
        self,
        matched_track_ids: Set[int],
        new_det_ids: Set[int],
    ) -> None:
        """Advance all alive tracklets by one frame.

        Args:
            matched_track_ids:
                Set of existing tracklet IDs that were matched to a detection
                on this frame (from ``associate_det_trk`` det2track values).
            new_det_ids:
                Set of tracklet IDs that were just spawned from new detections
                on this frame.  They count as matched on their first frame so
                the confirmation clock starts from the spawn frame, matching
                upstream behaviour where new-det objects have
                ``consecutive_det_num = 1`` on first appearance.
        """
        all_matched = matched_track_ids | new_det_ids
        for obj_id, info in self._tracks.items():
            if info.state is TrackletState.DEAD:
                continue  # no further transitions once dead
            if obj_id in all_matched:
                info.consecutive_det_count += 1
                info.unmatched_count = 0
                if (
                    info.state is TrackletState.PENDING
                    and info.consecutive_det_count >= self.confirmation_thresh
                ):
                    info.state = TrackletState.CONFIRMED
            else:
                info.consecutive_det_count = 0
                info.unmatched_count += 1
                if info.unmatched_count >= self.kill_thresh:
                    info.state = TrackletState.DEAD

    def remove(self, obj_id: int) -> None:
        """Immediately remove a tracklet (e.g. user-requested removal)."""
        self._tracks.pop(obj_id, None)

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def is_dead(self, obj_id: int) -> bool:
        """Return True if the tracklet has transitioned to DEAD."""
        return self._tracks[obj_id].state is TrackletState.DEAD

    def get_state(self, obj_id: int) -> TrackletState:
        """Return the current ``TrackletState`` for the given obj_id."""
        return self._tracks[obj_id].state

    def active_ids(self) -> Set[int]:
        """Return the set of non-DEAD tracklet IDs."""
        return {
            oid
            for oid, info in self._tracks.items()
            if info.state is not TrackletState.DEAD
        }

    def confirmed_ids(self) -> Set[int]:
        """Return the set of CONFIRMED tracklet IDs."""
        return {
            oid
            for oid, info in self._tracks.items()
            if info.state is TrackletState.CONFIRMED
        }

    def __len__(self) -> int:
        return len(self._tracks)

    def __repr__(self) -> str:
        counts: Dict[str, int] = {}
        for info in self._tracks.values():
            counts[info.state.value] = counts.get(info.state.value, 0) + 1
        return f"TrackletManager({counts})"
