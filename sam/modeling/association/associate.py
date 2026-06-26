# SPDX-License-Identifier: LicenseRef-SAM
"""Stateless detection-to-track association function.

Ported from ``sam3.perflib.associate_det_trk`` (upstream @ 5dd401d), keeping
the torch/scipy reference math and dropping the Triton/compiled fast-paths.

Reference semantics
-------------------
- Hungarian assignment on cost = 1 - IoU (scipy.optimize.linear_sum_assignment).
- ``iou_threshold_trk``: a track is "matched" only if its best-Hungarian-paired
  det has IoU >= this threshold (one-to-one, strict).
- ``iou_threshold``: a detection is considered to overlap a track if IoU >= this
  threshold (many-to-one gate used for both det2track and new-det decision).
- A detection is "new" (eligible to spawn a fresh tracklet) when:
    (1) it does not overlap any track above ``iou_threshold``, AND
    (2) its score >= ``new_det_thresh``.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def _mask_iou(pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between two sets of binary masks.

    Args:
        pred_masks: (N, H, W) bool tensor.
        gt_masks:   (M, H, W) bool tensor.

    Returns:
        (N, M) float IoU matrix.
    """
    assert pred_masks.dtype == gt_masks.dtype == torch.bool
    assert pred_masks.shape[1:] == gt_masks.shape[1:]

    m1 = pred_masks.flatten(1).float()  # (N, H*W)
    m2 = gt_masks.flatten(1).float()    # (M, H*W)
    intersection = torch.mm(m1, m2.t())  # (N, M)
    area1 = m1.sum(dim=1)                # (N,)
    area2 = m2.sum(dim=1)                # (M,)
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / union.clamp(min=1)


def associate_det_trk(
    det_masks: torch.Tensor,
    track_masks: torch.Tensor,
    iou_threshold: float = 0.5,
    iou_threshold_trk: float = 0.5,
    det_scores: Optional[torch.Tensor] = None,
    new_det_thresh: float = 0.0,
) -> Tuple[List[int], List[int], Dict[int, List[int]], Dict[int, List[float]]]:
    """Match detections to existing tracks via Hungarian on 1 - IoU.

    Args:
        det_masks:       (N, H, W) float or bool — predicted detection masks.
        track_masks:     (M, H, W) float or bool — current tracklet masks.
        iou_threshold:   IoU gate used for the det→track many-to-one map and
                         to determine whether a detection overlaps any track.
        iou_threshold_trk: Stricter IoU gate used to decide whether a track is
                         "matched" (unmatched tracks = those whose best Hungarian
                         pair is below this threshold).
        det_scores:      (N,) float detection confidence scores.
        new_det_thresh:  Minimum score for an unmatched det to be spawned as a
                         new tracklet.

    Returns:
        new_dets:         List[int] — det indices with no track overlap
                          (IoU < iou_threshold for all tracks) and score
                          >= new_det_thresh.
        unmatched_tracks: List[int] — track indices not matched by Hungarian
                          at iou_threshold_trk.
        det2track:        Dict[int, List[int]] — det_idx → list of track
                          indices with IoU >= iou_threshold (many-to-one).
        matched_scores:   Dict[int, List[float]] — track_idx → [det_score,
                          det_score * iou] for the Hungarian-assigned pair.
    """
    assert isinstance(det_masks, torch.Tensor), "det_masks must be a tensor"
    assert isinstance(track_masks, torch.Tensor), "track_masks must be a tensor"

    # Empty-input fast paths (mirrors upstream).
    if det_masks.size(0) == 0 or track_masks.size(0) == 0:
        new_dets: List[int] = []
        if track_masks.size(0) == 0:
            # all detections that meet the score gate are new
            if det_scores is not None:
                scores_list = det_scores.cpu().float().tolist()
                new_dets = [
                    d for d, s in enumerate(scores_list) if s >= new_det_thresh
                ]
            else:
                new_dets = list(range(det_masks.size(0)))
        return new_dets, list(range(track_masks.size(0))), {}, {}

    # Align spatial dims (resize to the smaller grid to save memory).
    if det_masks.shape[-2:] != track_masks.shape[-2:]:
        det_area = det_masks.shape[-2] * det_masks.shape[-1]
        trk_area = track_masks.shape[-2] * track_masks.shape[-1]
        if det_area < trk_area:
            track_masks = (
                F.interpolate(
                    track_masks.unsqueeze(1).float(),
                    size=det_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                > 0
            ).float()
        else:
            det_masks = (
                F.interpolate(
                    det_masks.unsqueeze(1).float(),
                    size=track_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                > 0
            ).float()

    det_masks_bin = det_masks > 0    # (N, H, W) bool
    track_masks_bin = track_masks > 0  # (M, H, W) bool

    iou = _mask_iou(det_masks_bin, track_masks_bin)  # (N, M)

    # Boolean gate matrices (CPU lists for the branchy per-pair loop below)
    igeit = iou >= iou_threshold          # (N, M) — many-to-one gate
    igeit_trk = iou >= iou_threshold_trk  # (N, M) — strict Hungarian gate
    igeit_any = igeit.any(dim=1)           # (N,)  — any track overlap per det?

    iou_np = iou.cpu().float().numpy()
    igeit_list = igeit.cpu().numpy().tolist()
    igeit_trk_list = igeit_trk.cpu().numpy().tolist()
    igeit_any_list = igeit_any.cpu().numpy().tolist()

    det_scores_list = (
        det_scores.cpu().float().tolist() if det_scores is not None else None
    )

    # Hungarian: minimise cost = 1 - IoU (maximise IoU).
    cost_matrix = 1.0 - iou_np
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_trk: set = set()
    matched_scores: Dict[int, List[float]] = {}
    for d, t in zip(row_ind.tolist(), col_ind.tolist()):
        if det_scores_list is not None:
            matched_scores[t] = [
                det_scores_list[d],
                det_scores_list[d] * float(iou_np[d][t]),
            ]
        if igeit_trk_list[d][t]:
            matched_trk.add(t)

    # Tracks not matched at iou_threshold_trk are "unmatched".
    unmatched_tracks = [
        t for t in range(track_masks.size(0)) if t not in matched_trk
    ]

    # A detection is "new" if it overlaps no track above iou_threshold AND
    # its score is high enough.
    new_dets = [
        d
        for d in range(det_masks.size(0))
        if not igeit_any_list[d]
        and (det_scores_list is None or det_scores_list[d] >= new_det_thresh)
    ]

    # det2track: det_idx → list of track indices overlapping above iou_threshold.
    det2track: Dict[int, List[int]] = {}
    for d in range(det_masks.size(0)):
        matched = [t for t in range(track_masks.size(0)) if igeit_list[d][t]]
        if matched:
            det2track[d] = matched

    return new_dets, unmatched_tracks, det2track, matched_scores
