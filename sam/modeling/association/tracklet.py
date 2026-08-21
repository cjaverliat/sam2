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

A user-clicked (``interactive``) tracklet sits outside ALL THREE: upstream registers
only detector-spawned ids in the hotstart bookkeeping, so a click neither decays nor
dies -- see :meth:`TrackletManager.force_confirm`.

The predictor reads three id-sets each frame: ``removed_ids`` (purge bank + free mux
slot), ``alive_ids`` (propagate -- includes suppressed, so memory is rewritten), and
``visible_ids`` (emit in results).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Set

from sam.results import Emit


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
    interactive: bool = False        # user-clicked: exempt from the hotstart kill


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
    confirmation_enable:
        Whether the PENDING -> CONFIRMED gate applies at all (upstream
        ``masklet_confirmation_enable``). When False every tracklet is CONFIRMED from
        birth -- the base-lineage setting.

    The defaults are the MULTIPLEX lineage's (``Sam3MultiplexBase`` class defaults +
    the demo builder's ``masklet_confirmation_enable=True``). The base video lineage
    differs (``build_sam3_video_model``: keep-alive 30/30/-1, confirmation disabled);
    each predictor applies its own via :meth:`configure`.
    """

    def __init__(
        self,
        confirmation_thresh: int = 3,
        hotstart_delay: int = 15,
        hotstart_unmatch_thresh: int = 8,
        init_keep_alive: int = 0,
        max_keep_alive: int = 8,
        min_keep_alive: int = -4,
        confirmation_enable: bool = True,
    ) -> None:
        self.confirmation_enable = confirmation_enable
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

    def configure(
        self,
        *,
        confirmation_enable: bool,
        confirmation_thresh: int,
        hotstart_delay: int,
        hotstart_unmatch_thresh: int,
        init_keep_alive: int,
        max_keep_alive: int,
        min_keep_alive: int,
    ) -> None:
        """Apply a lineage's upstream lifecycle constants to a not-yet-used manager.

        The predictor calls this on its first frame, since a
        ``Sam3VideoPredictorState`` is built by the caller and cannot know which
        lineage (base / multiplex) will drive it.

        Raises:
            ValueError: if any tracklet has already been spawned.
        """
        if self._tracks:
            raise ValueError("configure() must run before the first tracklet is spawned")
        self.confirmation_enable = confirmation_enable
        self.confirmation_thresh = confirmation_thresh
        self.hotstart_delay = hotstart_delay
        self.hotstart_unmatch_thresh = hotstart_unmatch_thresh
        self.init_keep_alive = init_keep_alive
        self.max_keep_alive = max_keep_alive
        self.min_keep_alive = min_keep_alive

    def spawn(self, obj_id: int, frame_idx: int, interactive: bool = False) -> None:
        """Register a new tracklet born on ``frame_idx`` (PENDING, or CONFIRMED when
        the confirmation gate is disabled).

        Args:
            obj_id: id of the new tracklet.
            frame_idx: birth frame (starts the hotstart window).
            interactive: the tracklet was seeded by a user click, which upstream
                force-confirms and never treats as a purge candidate -- see
                :meth:`force_confirm`.

        Raises:
            ValueError: if ``obj_id`` is already registered.
        """
        if obj_id in self._tracks:
            raise ValueError(
                f"tracklet {obj_id!r} already registered; "
                "call remove() first or use a fresh obj_id"
            )
        self._tracks[obj_id] = _TrackletInfo(
            first_frame=frame_idx,
            state=(
                TrackletState.PENDING if self.confirmation_enable
                else TrackletState.CONFIRMED
            ),
            keep_alive=self.init_keep_alive,
        )
        if interactive:
            self.force_confirm(obj_id)

    def force_confirm(self, obj_id: int) -> None:
        """Confirm a tracklet on a user click and exempt it from the hotstart kill.

        Upstream's ``add_tracker_new_points`` sets the object's
        ``masklet_confirmation`` status to 1 and its ``consecutive_det_num`` to the
        threshold (``sam3_video_inference.py:1522-1531``), and its hotstart purge only
        ever considers ids the DETECTOR registered (``_process_hotstart`` walks
        ``unmatched_frame_inds``, keyed from ``new_det_obj_ids``). A clicked object is
        therefore confirmed at once and never removed for going unmatched -- which
        matters most in a click-only session, where detection never runs and so
        nothing can ever match it.

        The same registration gate covers ``trk_keep_alive``: upstream only ever
        initialises it for ``new_det_obj_ids`` (``sam3_multiplex_base.py:2344`` and the
        GPU-metadata concat at :1320), so a clicked object is outside the suppression
        hysteresis too. Pinning ``keep_alive`` at its maximum reproduces that here --
        the default ``init_keep_alive`` is 0, i.e. already at the visibility boundary,
        so a click would otherwise be hidden by the first unmatched frame.
        """
        info = self._tracks[obj_id]
        info.interactive = True
        info.state = TrackletState.CONFIRMED
        info.consecutive_det_count = max(
            info.consecutive_det_count, self.confirmation_thresh
        )
        info.unmatched_count = 0
        info.keep_alive = self.max_keep_alive

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
            if info.interactive:
                # Upstream registers a clicked object in NONE of the hotstart
                # bookkeeping (see :meth:`force_confirm`): it neither decays nor dies.
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
            if (
                not info.interactive
                and within_hotstart
                and info.unmatched_count >= self.hotstart_unmatch_thresh
            ):
                info.removed = True

    def remove(self, obj_id: int) -> None:
        """Immediately drop a tracklet (e.g. user-requested removal)."""
        self._tracks.pop(obj_id, None)

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def managed_ids(self) -> Set[int]:
        """All obj_ids this manager tracks (regardless of state)."""
        return set(self._tracks)

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

    def emitted_ids(self, emit: Emit) -> Set[int]:
        """The managed ids an ``Emit`` policy puts in the predictor's output."""
        if emit is Emit.CONFIRMED:
            return self.confirmed_ids()
        if emit is Emit.VISIBLE:
            return self.visible_ids()
        return self.alive_ids()

    def state_of(self, obj_id: int) -> TrackletState | None:
        """Lifecycle state of ``obj_id``, or None if this manager does not track it."""
        info = self._tracks.get(obj_id)
        return None if info is None else info.state

    def __len__(self) -> int:
        return len(self._tracks)

    def __repr__(self) -> str:
        n_removed = sum(1 for i in self._tracks.values() if i.removed)
        n_vis = len(self.visible_ids())
        return f"TrackletManager(n={len(self._tracks)}, visible={n_vis}, removed={n_removed})"
