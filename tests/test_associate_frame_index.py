# SPDX-License-Identifier: LicenseRef-SAM
"""``_associate_and_update`` uses the frame index it was given.

``VideoSession.process`` documents a ``frame_idx`` override ("e.g. to re-run a
frame"), so the forward count and the frame index are independent. The tracklet
birth frame drives the hotstart gate, which must live on the same scale as the
``frame_idx`` every later ``step()`` is passed.

CPU-only: association is pure tensor work, so the test drives the method on an
uninitialised predictor carrying only the one threshold it reads.
"""
import types

import torch

from sam.models.sam3_predictor import Sam3VideoPredictor, Sam3VideoPredictorState


def _predictor():
    pred = object.__new__(Sam3VideoPredictor)
    pred.new_det_thresh = 0.7
    return pred


def _detection():
    """One high-scoring detection, matching no track."""
    masks = torch.zeros(1, 8, 8)
    masks[0, :4, :4] = 1.0
    return types.SimpleNamespace(masks_logits=masks, scores=torch.tensor([0.9]))


def test_spawned_tracklet_is_born_on_the_frame_forward_was_given():
    state = Sam3VideoPredictorState(video_hw=(8, 8))
    state.num_frames_processed = 5           # five forwards so far...
    new_objects = _predictor()._associate_and_update(
        state, 2, _detection(), [], {}, {}   # ...but this call re-runs frame 2
    )
    assert len(new_objects) == 1
    (obj_id, _det_idx), = new_objects
    assert state.tracklet_mgr._tracks[obj_id].first_frame == 2
