# SPDX-License-Identifier: Apache-2.0
"""Named constructors on GeometryPrompt: click / box / mask / concept_box.

Each builds the same prompt the verbose ``__init__`` form would, from plain Python
values (tuples, lists, ndarrays) -- no torch ceremony at the call site. CPU-only.
"""
import numpy as np
import pytest
import torch

from sam.prompts import PromptRoute, GeometryPrompt


def test_click_positive_by_default():
    prompt = GeometryPrompt.click(1, (385, 230))
    assert prompt.obj_id == 1
    assert prompt.points_coords.tolist() == [[385.0, 230.0]]
    assert prompt.points_labels.tolist() == [1]
    assert prompt.boxes is None and prompt.masks_logits is None


def test_click_negative_label():
    prompt = GeometryPrompt.click(1, (10, 20), label=0)
    assert prompt.points_labels.tolist() == [0]


def test_click_accepts_list_and_ndarray():
    assert GeometryPrompt.click(1, [5, 6]).points_coords.tolist() == [[5.0, 6.0]]
    arr = np.array([7.0, 8.0])
    assert GeometryPrompt.click(1, arr).points_coords.tolist() == [[7.0, 8.0]]


def test_box_is_tracker_route():
    prompt = GeometryPrompt.box(2, (285, 0, 535, 430))
    assert prompt.boxes.tolist() == [[285.0, 0.0, 535.0, 430.0]]
    assert prompt.route is PromptRoute.TRACKER
    coords, labels = prompt.tracker_points()
    assert coords.tolist() == [[285.0, 0.0], [535.0, 430.0]]
    assert labels.tolist() == [2, 3]


def test_concept_box_is_detector_route():
    prompt = GeometryPrompt.concept_box((285, 0, 535, 430))
    assert prompt.route is PromptRoute.DETECTOR
    assert prompt.to_detector
    assert prompt.boxes_labels is None  # positive by default (packer defaults to 1)


def test_concept_box_negative_label():
    prompt = GeometryPrompt.concept_box((10, 20, 30, 40), label=0)
    assert prompt.boxes_labels.tolist() == [0]


def test_mask_from_bool_and_logits():
    mask_bool = np.zeros((540, 960), dtype=bool)
    mask_bool[100:200, 300:400] = True
    prompt = GeometryPrompt.mask(4, mask_bool)
    assert prompt.masks_logits.shape == (1, 540, 960)
    # boolean masks become +/-10 logits (binarise at 0 recovers the input)
    recovered = (prompt.masks_logits[0] > 0).numpy()
    assert (recovered == mask_bool).all()

    logits = torch.randn(540, 960)
    prompt = GeometryPrompt.mask(4, logits)
    assert torch.equal(prompt.masks_logits[0], logits)


def test_constructors_reject_bad_shapes():
    with pytest.raises((ValueError, RuntimeError)):
        GeometryPrompt.click(1, (1.0, 2.0, 3.0))     # not an (x, y)
    with pytest.raises((ValueError, RuntimeError)):
        GeometryPrompt.box(1, (1.0, 2.0, 3.0))       # not an xyxy
