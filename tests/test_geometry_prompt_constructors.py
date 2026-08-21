# SPDX-License-Identifier: Apache-2.0
"""Named constructors on GeometryPrompt.

Two families, and the constructor is what says which: ``click`` / ``clicks`` / ``box``
/ ``mask`` mark ONE object for the tracker, ``exemplar_point(s)`` / ``exemplar_box(es)``
describe what a concept search should look for. Each builds the same prompt the verbose
``__init__`` form would, from plain Python values (tuples, lists, ndarrays) -- no torch
ceremony at the call site. CPU-only.
"""
import numpy as np
import pytest
import torch

from sam.prompts import GeometryPrompt, PromptRoute, merge_object_prompts


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


def test_exemplar_box_is_detector_route():
    prompt = GeometryPrompt.exemplar_box((285, 0, 535, 430))
    assert prompt.route is PromptRoute.DETECTOR
    assert prompt.to_detector
    assert prompt.boxes_labels is None  # positive by default (packer defaults to 1)


def test_exemplar_box_negative_label():
    prompt = GeometryPrompt.exemplar_box((10, 20, 30, 40), label=0)
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


# ---------------------------------------------------------------------------
# Plural forms: one prompt carries all of one object's / one search's geometry.
# ---------------------------------------------------------------------------
def test_clicks_carries_every_point_for_one_object():
    p = GeometryPrompt.clicks(2, [(315, 310), (330, 200), (250, 250)], labels=[1, 1, 0])
    assert p.obj_id == 2
    assert p.points_coords.tolist() == [[315.0, 310.0], [330.0, 200.0], [250.0, 250.0]]
    assert p.points_labels.tolist() == [1, 1, 0]
    assert p.route is PromptRoute.TRACKER


def test_clicks_defaults_every_point_to_positive():
    assert GeometryPrompt.clicks(1, [(1, 2), (3, 4)]).points_labels.tolist() == [1, 1]


def test_exemplar_plurals_are_detector_route():
    points = GeometryPrompt.exemplar_points([(1, 2), (3, 4)], labels=[1, 0])
    boxes = GeometryPrompt.exemplar_boxes([(1, 2, 3, 4), (5, 6, 7, 8)], labels=[1, 0])
    assert points.route is PromptRoute.DETECTOR and boxes.route is PromptRoute.DETECTOR
    assert points.points_labels.tolist() == [1, 0]
    assert boxes.boxes_labels.tolist() == [1, 0]
    assert boxes.boxes.shape == (2, 4)


@pytest.mark.parametrize("build", [
    lambda: GeometryPrompt.clicks(1, [(1, 2)], labels=[1, 0]),
    lambda: GeometryPrompt.exemplar_points([(1, 2)], labels=[1, 0]),
    lambda: GeometryPrompt.exemplar_boxes([(1, 2, 3, 4)], labels=[1, 0]),
])
def test_plurals_reject_a_label_count_mismatch(build):
    with pytest.raises(ValueError, match="label"):
        build()


def test_tracker_takes_several_boxes_for_one_object():
    """The prompt encoder repeats labels 2/3 per box, so N boxes are N corner pairs.

    Not a hypothetical: ``sam2_predictor`` encodes ``n_boxes = boxes.shape[1]`` and
    repeats the corner labels, which is how an object that one rectangle describes
    badly gets described by two.
    """
    prompt = GeometryPrompt.boxes(1, [(285, 0, 535, 430), (300, 150, 470, 420)])
    assert prompt.boxes.shape == (2, 4)
    assert prompt.route is PromptRoute.TRACKER

    coords, labels = prompt.tracker_points()
    assert coords.shape == (4, 2), "two corners per box"
    assert labels.tolist() == [2, 3, 2, 3]


def test_box_corners_match_the_encoders_native_box_path():
    """`box()` is not a workaround: labels 2/3 ARE the box embedding.

    ``PromptEncoder._embed_boxes`` adds ``point_embeddings[2]`` to the first corner and
    ``[3]`` to the second, which is exactly what ``_embed_points`` does for those
    labels -- the same weights after the same +0.5 shift.
    """
    corners, labels = GeometryPrompt.box(1, (285, 0, 535, 430)).tracker_points()
    assert corners.tolist() == [[285.0, 0.0], [535.0, 430.0]]
    assert labels.tolist() == [2, 3]


# ---------------------------------------------------------------------------
# Merging: prompts for one object are evidence, and evidence adds up.
# ---------------------------------------------------------------------------
def test_prompts_for_one_object_merge_by_type():
    merged = merge_object_prompts([
        GeometryPrompt.clicks(2, [(315, 310), (330, 200)], labels=[1, 0]),
        GeometryPrompt.boxes(2, [(255, 95, 400, 470)]),
        GeometryPrompt.click(1, (210, 310)),
    ])
    by_id = {p.obj_id: p for p in merged}
    assert sorted(by_id) == [1, 2]
    assert by_id[2].points_coords.shape == (2, 2)
    assert by_id[2].points_labels.tolist() == [1, 0]
    assert by_id[2].boxes.shape == (1, 4)
    assert by_id[1].boxes is None


def test_merging_matches_the_verbose_prompt():
    """The point of merging: listing prompts == building one by hand."""
    listed = merge_object_prompts([
        GeometryPrompt.click(2, (315, 310)),
        GeometryPrompt.boxes(2, [(255, 95, 400, 470)]),
    ])[0]
    verbose = GeometryPrompt(
        obj_id=2,
        points_coords=torch.tensor([[315.0, 310.0]]),
        points_labels=torch.tensor([1], dtype=torch.int32),
        boxes=torch.tensor([[255.0, 95.0, 400.0, 470.0]]),
    )
    assert torch.equal(listed.points_coords, verbose.points_coords)
    assert torch.equal(listed.points_labels, verbose.points_labels)
    assert torch.equal(listed.boxes, verbose.boxes)


def test_exemplars_merge_into_one_search():
    merged = merge_object_prompts([
        GeometryPrompt.exemplar_box((1, 2, 3, 4)),
        GeometryPrompt.exemplar_point((5, 6), label=0),
    ])
    assert len(merged) == 1, "exemplars describe one search, not one object each"
    assert merged[0].route is PromptRoute.DETECTOR
    assert merged[0].boxes.shape == (1, 4) and merged[0].points_coords.shape == (1, 2)


def test_unsigned_boxes_stay_positive_when_another_prompt_signs_its_own():
    merged = merge_object_prompts([
        GeometryPrompt.exemplar_boxes([(1, 2, 3, 4)]),                 # unsigned
        GeometryPrompt.exemplar_boxes([(5, 6, 7, 8)], labels=[0]),     # a counter-example
    ])[0]
    assert merged.boxes_labels.tolist() == [1, 0]


def test_merge_reconciles_coordinate_spaces():
    """Pixels are divided through -- the same division the predictor does later."""
    pixels = GeometryPrompt.clicks(2, [(480, 270)])
    normalized = GeometryPrompt(
        obj_id=2,
        points_coords=torch.tensor([[0.25, 0.5]]),
        points_labels=torch.tensor([1], dtype=torch.int32),
        is_normalized=True,
    )
    merged = merge_object_prompts([pixels, normalized], image_hw=(540, 960))[0]
    assert merged.is_normalized
    assert merged.points_coords.tolist() == [[0.5, 0.5], [0.25, 0.5]]

    with pytest.raises(ValueError, match="no image size"):
        merge_object_prompts([pixels, normalized])


def test_merge_refuses_contradictions():
    with pytest.raises(ValueError, match="two masks|2 masks"):
        merge_object_prompts([
            GeometryPrompt.mask(1, torch.zeros(4, 4, dtype=torch.bool)),
            GeometryPrompt.mask(1, torch.ones(4, 4, dtype=torch.bool)),
        ])

    both_routes = GeometryPrompt.exemplar_box((1, 2, 3, 4))
    both_routes.obj_id = 1  # same id as a tracker prompt, which cannot be reconciled
    with pytest.raises(ValueError, match="both routes"):
        merge_object_prompts([GeometryPrompt.click(1, (1, 2)), both_routes])


def test_merge_passes_single_prompts_through_untouched():
    only = GeometryPrompt.click(1, (1, 2))
    assert merge_object_prompts([only])[0] is only
    assert merge_object_prompts([]) == []
    assert merge_object_prompts(None) == []
