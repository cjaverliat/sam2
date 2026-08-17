# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import enum
import typing

import torch

if typing.TYPE_CHECKING:
    from sam.modeling.association.tracklet import TrackletState


class MaskletResult:
    def __init__(
        self,
        masks_logits: torch.Tensor,
        ious: torch.Tensor,
        obj_ptrs: torch.Tensor,
        obj_scores_logits: torch.Tensor,
        tracklet_state: "TrackletState | None" = None,
    ):
        assert (
            masks_logits.ndim == 4
        ), f"Expected masks_logits to be of shape (B, N, H, W), got {masks_logits.shape}"
        assert ious.ndim == 2, f"Expected ious to be of shape (B, N), got {ious.shape}"
        assert (
            obj_ptrs.ndim == 2
        ), f"Expected obj_ptrs to be of shape (B, N), got {obj_ptrs.shape}"
        assert (
            obj_scores_logits.ndim == 2
        ), f"Expected obj_score_logits to be of shape (B, N), got {obj_scores_logits.shape}"

        self.batch_size = masks_logits.shape[0]

        assert (
            ious.shape[0] == self.batch_size
        ), f"Expected ious to have batch size {self.batch_size}, got {ious.shape[0]}"
        assert (
            obj_ptrs.shape[0] == self.batch_size
        ), f"Expected obj_ptrs to have batch size {self.batch_size}, got {obj_ptrs.shape[0]}"
        assert (
            obj_scores_logits.shape[0] == self.batch_size
        ), f"Expected obj_score_logits to have batch size {self.batch_size}, got {obj_scores_logits.shape[0]}"

        self.masks_logits = masks_logits
        self.ious = ious
        self.obj_ptrs = obj_ptrs
        self.obj_score_logits = obj_scores_logits
        # Lifecycle state of the tracklet this masklet belongs to; None for objects the
        # TrackletManager does not manage (click-seeded) and for non-SAM-3 producers.
        # Stamped by ``select_emitted`` so a caller can apply its own display policy on
        # top of whatever ``Emit`` mode the predictor ran with.
        self.tracklet_state = tracklet_state

    def to(self, device: torch.device) -> MaskletResult:
        return MaskletResult(
            masks_logits=self.masks_logits.to(device),
            ious=self.ious.to(device),
            obj_ptrs=self.obj_ptrs.to(device),
            obj_scores_logits=self.obj_score_logits.to(device),
            tracklet_state=self.tracklet_state,
        )

    def clone(self) -> MaskletResult:
        return MaskletResult(
            masks_logits=self.masks_logits.clone(),
            ious=self.ious.clone(),
            obj_ptrs=self.obj_ptrs.clone(),
            obj_scores_logits=self.obj_score_logits.clone(),
            tracklet_state=self.tracklet_state,
        )

    @staticmethod
    def cat(results: list[MaskletResult]) -> MaskletResult:
        # tracklet_state is per-object, so it survives the concat only when every
        # input agrees; a mixed batch has no single state to report.
        states = {r.tracklet_state for r in results}
        return MaskletResult(
            masks_logits=torch.cat([r.masks_logits for r in results], dim=0),
            ious=torch.cat([r.ious for r in results], dim=0),
            obj_ptrs=torch.cat([r.obj_ptrs for r in results], dim=0),
            obj_scores_logits=torch.cat([r.obj_score_logits for r in results], dim=0),
            tracklet_state=states.pop() if len(states) == 1 else None,
        )

    def __getitem__(self, idx: int) -> MaskletResult:
        return MaskletResult(
            masks_logits=self.masks_logits[idx].unsqueeze(0),
            ious=self.ious[idx].unsqueeze(0),
            obj_ptrs=self.obj_ptrs[idx].unsqueeze(0),
            obj_scores_logits=self.obj_score_logits[idx].unsqueeze(0),
            tracklet_state=self.tracklet_state,
        )

    @property
    def device(self) -> torch.device:
        return self.masks_logits.device

    @property
    def best_mask_logits(self) -> torch.Tensor:
        best_mask_idx = torch.argmax(self.ious, dim=1, keepdim=True)
        batch_indices = torch.arange(
            self.masks_logits.shape[0], device=self.masks_logits.device
        ).unsqueeze(1)
        return self.masks_logits[batch_indices, best_mask_idx]


# SPDX-License-Identifier: LicenseRef-SAM
# --- SAM 3 detection result (Phase 1, Task 4) ----------------------------------------
# Output of ``Sam3DetrDetector.detect`` (the base, per-object DETR detector): the kept
# detections for one text phrase after presence-weighted thresholding. Mirrors the style
# of ``MaskletResult`` above (Apache-2.0; untouched).
class Sam3DetectionResult:
    """Per-object detections grounded from a text phrase.

    Args:
        masks_logits: ``(N, H, W)`` mask logits at the original image resolution
            (binarise at 0, i.e. ``sigmoid > 0.5``, to recover the boolean masks).
        boxes: ``(N, 4)`` boxes in ``xyxy`` format, in original-image pixels.
        scores: ``(N,)`` presence-weighted confidences (``sigmoid(class) * sigmoid(presence)``).
        presence: scalar phrase-level presence probability (``sigmoid(presence_logit_dec)``).
        instance_ids: ``(N,)`` integer ids, one per kept detection.
    """

    def __init__(
        self,
        masks_logits: torch.Tensor,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        presence: float,
        instance_ids: torch.Tensor,
    ):
        assert (
            masks_logits.ndim == 3
        ), f"Expected masks_logits to be of shape (N, H, W), got {masks_logits.shape}"
        assert boxes.ndim == 2 and boxes.shape[-1] == 4, (
            f"Expected boxes to be of shape (N, 4), got {boxes.shape}"
        )
        assert scores.ndim == 1, f"Expected scores to be of shape (N,), got {scores.shape}"
        n = boxes.shape[0]
        assert masks_logits.shape[0] == n, (
            f"Expected {n} masks, got {masks_logits.shape[0]}"
        )
        assert scores.shape[0] == n, f"Expected {n} scores, got {scores.shape[0]}"
        assert instance_ids.shape[0] == n, (
            f"Expected {n} instance ids, got {instance_ids.shape[0]}"
        )

        self.masks_logits = masks_logits
        self.boxes = boxes
        self.scores = scores
        self.presence = float(presence)
        self.instance_ids = instance_ids

    @property
    def num_detections(self) -> int:
        return self.boxes.shape[0]

    @property
    def device(self) -> torch.device:
        return self.boxes.device

    def to(self, device: torch.device) -> "Sam3DetectionResult":
        return Sam3DetectionResult(
            masks_logits=self.masks_logits.to(device),
            boxes=self.boxes.to(device),
            scores=self.scores.to(device),
            presence=self.presence,
            instance_ids=self.instance_ids.to(device),
        )


class Emit(enum.Enum):
    """Which tracked objects a streaming video predictor puts in its output.

    A display policy only: every alive tracklet is propagated and keeps its memory
    whatever the mode, so switching modes never changes the tracking. An empty mask
    is dropped in all modes (upstream ``_postprocess_output`` keeps ``mask.any()``
    unconditionally).

    Attributes:
        CONFIRMED: matched by a detection for ``confirmation_thresh`` consecutive
            frames. The default: an object is never shown before it is corroborated,
            so a one-frame detector false positive never reaches the caller. Costs
            visibility of an object's first frames (upstream shows them by deciding
            non-causally, which a streaming predictor cannot do).
        VISIBLE: alive and not suppressed (``keep_alive > 0``). Reproduces what the
            upstream goldens captured, birth frames included -- what the parity
            tests measure.
        ALIVE: every tracklet that has not been removed, suppressed ones included.
            Raw feed for a caller applying its own policy; pays the full-resolution
            mask build for dormant objects.
    """

    CONFIRMED = "confirmed"
    VISIBLE = "visible"
    ALIVE = "alive"

