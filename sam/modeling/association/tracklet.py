# SPDX-License-Identifier: LicenseRef-SAM
"""Tracklet lifecycle: hotstart-gated kill + keep-alive suppress (upstream parity).

Mirrors upstream ``sam3/model/sam3_multiplex_base.py::_process_hotstart_gpu``
(@ 5dd401d): two INDEPENDENT counters per object.

- **Kill** (``unmatched_count`` -> ``removed``) fires ONLY inside an object's
  hotstart window (its first ``hotstart_delay`` frames) once it has been unmatched
  for ``hotstart_unmatch_thresh`` frames. It is a new-object-quality gate, NOT an
  absence policy -- an established object is *never* removed.
- **Suppress** (``keep_alive`` hysteresis, +1 matched / -1 not, clamped) only HIDES
  an object from output while it is absent; the object stays alive with its memory
  retained, so when it re-appears it un-hides under the SAME id (re-ID).
- **Confirmation** (``consecutive_det_count`` -> CONFIRMED) is an orthogonal display
  gate with no kill (upstream ``masklet_confirmation``).

The predictor reads three id-sets each frame: ``removed_ids`` (purge bank + free mux
slot), ``alive_ids`` (propagate -- includes suppressed, so memory is rewritten), and
``visible_ids`` (emit in results).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Set


class TrackletState(Enum):
    PENDING = "pending"      # newly spawned; awaiting confirmation
    CONFIRMED = "confirmed"  # matched for >= confirmation_thresh consecutive frames


@dataclass
class _TrackletInfo:
    first_frame: int
    state: TrackletState = TrackletState.PENDING
    consecutive_det_count: int = 0   # consecutive frames with a det match
    unmatched_count: int = 0         # consecutive frames without a det match
    keep_alive: int = 0              # suppress hysteresis (visible when > 0)
    removed: bool = False            # purged (within-hotstart failure / user removal)


class TrackletManager:
    """PENDING -> CONFIRMED display gate + hotstart-gated kill + keep-alive suppress.

    Holds per-tracklet state keyed by ``obj_id``. Lives inside
    ``Sam3VideoPredictorState`` as ``tracklet_mgr``.

    Parameters
    ----------
    confirmation_thresh:
        Consecutive detection-matched frames before PENDING -> CONFIRMED
        (upstream ``masklet_confirmation_consecutive_det_thresh``, default 3).
    hotstart_delay:
        Width of the removable window from an object's first frame; after this an
        object can never be killed (upstream ``hotstart_delay``, default 15).
    hotstart_unmatch_thresh:
        Unmatched frames before a *within-hotstart* object is removed (upstream
        ``hotstart_unmatch_thresh``, default 8).
    init_keep_alive / max_keep_alive / min_keep_alive:
        Suppress-hysteresis counter bounds (upstream ``init/max/min_trk_keep_alive``,
        defaults 0 / 8 / -4). Visible while ``keep_alive > 0``.
    """

    def __init__(
        self,
        confirmation_thresh: int = 3,
        hotstart_delay: int = 15,
        hotstart_unmatch_thresh: int = 8,
        init_keep_alive: int = 0,
        max_keep_alive: int = 8,
        min_keep_alive: int = -4,
    ) -> None:
        self.confirmation_thresh = confirmation_thresh
        self.hotstart_delay = hotstart_delay
        self.hotstart_unmatch_thresh = hotstart_unmatch_thresh
        self.init_keep_alive = init_keep_alive
        self.max_keep_alive = max_keep_alive
        self.min_keep_alive = min_keep_alive
        self._tracks: Dict[int, _TrackletInfo] = {}

    # ------------------------------------------------------------------
    # Mutation API
    # ------------------------------------------------------------------

    def spawn(self, obj_id: int, frame_idx: int) -> None:
        """Register a new PENDING tracklet born on ``frame_idx``."""
        if obj_id in self._tracks:
            raise ValueError(
                f"tracklet {obj_id!r} already registered; "
                "call remove() first or use a fresh obj_id"
            )
        self._tracks[obj_id] = _TrackletInfo(
            first_frame=frame_idx, keep_alive=self.init_keep_alive
        )

    def step(
        self,
        matched_track_ids: Set[int],
        new_det_ids: Set[int],
        frame_idx: int,
    ) -> None:
        """Advance all live tracklets one frame.

        Args:
            matched_track_ids: existing tracklet ids matched to a detection now.
            new_det_ids: tracklet ids just spawned from new detections (count as
                matched on their first frame).
            frame_idx: the current frame index (for the hotstart gate).
        """
        all_matched = matched_track_ids | new_det_ids
        for obj_id, info in self._tracks.items():
            if info.removed:
                continue
            matched = obj_id in all_matched
            if matched:
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
            info.keep_alive = max(
                self.min_keep_alive,
                min(self.max_keep_alive, info.keep_alive + (1 if matched else -1)),
            )
            within_hotstart = info.first_frame > frame_idx - self.hotstart_delay
            if within_hotstart and info.unmatched_count >= self.hotstart_unmatch_thresh:
                info.removed = True

    def remove(self, obj_id: int) -> None:
        """Immediately drop a tracklet (e.g. user-requested removal)."""
        self._tracks.pop(obj_id, None)

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def removed_ids(self) -> Set[int]:
        """Tracklets purged this session (within-hotstart failures)."""
        return {oid for oid, info in self._tracks.items() if info.removed}

    def alive_ids(self) -> Set[int]:
        """Tracklets still tracked (not removed) -- propagated, memory retained."""
        return {oid for oid, info in self._tracks.items() if not info.removed}

    def visible_ids(self) -> Set[int]:
        """Tracklets shown in output (alive and not suppressed)."""
        return {
            oid for oid, info in self._tracks.items()
            if not info.removed and info.keep_alive > 0
        }

    def confirmed_ids(self) -> Set[int]:
        """CONFIRMED (past the confirmation gate), not removed."""
        return {
            oid for oid, info in self._tracks.items()
            if not info.removed and info.state is TrackletState.CONFIRMED
        }

    def __len__(self) -> int:
        return len(self._tracks)

    def __repr__(self) -> str:
        n_removed = sum(1 for i in self._tracks.values() if i.removed)
        n_vis = len(self.visible_ids())
        return f"TrackletManager(n={len(self._tracks)}, visible={n_vis}, removed={n_removed})"
