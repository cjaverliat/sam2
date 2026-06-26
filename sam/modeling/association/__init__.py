# SPDX-License-Identifier: LicenseRef-SAM
"""Association utilities for SAM 3 streaming video.

Provides:
    associate_det_trk  — stateless det↔track Hungarian matcher.
    TrackletManager    — PENDING → CONFIRMED → DEAD lifecycle state machine.
    TrackletState      — state enum consumed by TrackletManager.
"""
from sam.modeling.association.associate import associate_det_trk
from sam.modeling.association.tracklet import TrackletManager, TrackletState

__all__ = ["associate_det_trk", "TrackletManager", "TrackletState"]
