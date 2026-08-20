# SPDX-License-Identifier: LicenseRef-SAM
"""One verb, two paths: ``process(image, concept=...)`` vs ``process(image, geometry=...)``.

Which path runs is decided by what the caller passed, never guessed, and the two never
mix in one result -- their ids come from different authorities.

The object path is the video predictor's frame-0 tracker call without a memory bank, so
the test that matters is the one comparing the two. They agree to within the difference
their preprocessing regimes make (image: GPU resize; video: the image-folder loader),
which is IoU ~0.997 rather than bit-equality -- see ``test_object_path_matches_video``.
"""
import os

import numpy as np
import pytest
import torch
from PIL import Image

from sam.prompts import ConceptPrompt, GeometryPrompt

CKPT = "checkpoints/sam3.pt"
MUX_CKPT = "checkpoints/sam3.1_multiplex.pt"
IMAGE = "notebooks/images/truck.jpg"

needs_model = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.isfile(CKPT),
    reason="needs CUDA + sam3.pt",
)
needs_mux = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.isfile(MUX_CKPT),
    reason="needs CUDA + sam3.1_multiplex.pt",
)

REAR_WHEEL = (450.0, 620.0, 670.0, 840.0)
CLICK = (560.0, 730.0)


@pytest.fixture(scope="module")
def predictor():
    from sam.build_sam import build_sam3

    return build_sam3(config_file="configs/sam3/sam3.yaml", ckpt_path=CKPT, device="cuda")


@pytest.fixture(scope="module")
def image():
    return np.asarray(Image.open(IMAGE).convert("RGB"))


def assert_same(got, expected):
    assert got.presence == expected.presence
    assert torch.equal(got.scores, expected.scores)
    assert torch.equal(got.boxes, expected.boxes)
    assert torch.equal(got.masks_logits, expected.masks_logits)


# ---------------------------------------------------------------- concept path
@needs_model
def test_concept_path_returns_every_match(predictor, image):
    r = predictor.process(image, concept="wheel")
    assert int(r.boxes.shape[0]) == 4, "both road wheels and both far-side wheels"
    assert r.presence > 0.9
    assert r.instance_ids.tolist() == [0, 1, 2, 3], "ids are the detector's"


@needs_model
def test_concept_accepts_phrase_prompt_and_placeholder(predictor, image):
    """Three spellings of "what to look for", one of them lineage-specific."""
    assert_same(predictor.process(image, concept="wheel"),
                predictor.process(image, concept=ConceptPrompt(text="wheel")))
    assert_same(
        predictor.process(image, concept=predictor.PLACEHOLDER,
                          geometry=GeometryPrompt.exemplar_box(REAR_WHEEL)),
        predictor.process(image, concept=ConceptPrompt(text=predictor.BOX_ONLY_CAPTION),
                          geometry=GeometryPrompt.exemplar_box(REAR_WHEEL)))


@needs_model
@pytest.mark.parametrize("geometry", [
    GeometryPrompt.exemplar_box(REAR_WHEEL),
    GeometryPrompt.exemplar_box(REAR_WHEEL, label=0),
    GeometryPrompt.exemplar_point(CLICK),
])
def test_exemplars_bias_without_selecting(predictor, image, geometry):
    """An exemplar moves scores; it never narrows the result to the marked instance."""
    plain = predictor.process(image, concept="wheel")
    biased = predictor.process(image, concept="wheel", geometry=geometry.clone())
    assert int(biased.boxes.shape[0]) == int(plain.boxes.shape[0])
    assert not torch.equal(biased.scores, plain.scores), "the exemplar did nothing"


# ---------------------------------------------------------------- object path
@needs_model
def test_object_path_returns_only_what_you_marked(predictor, image):
    r = predictor.process(image, geometry=GeometryPrompt.box(1, REAR_WHEEL))
    assert r.instance_ids.tolist() == [1], "the id is the caller's, not the detector's"
    assert r.presence is None, "no concept was asked for, so nothing is present"
    assert int((r.masks_logits > 0).sum()) > 10_000, "the wheel, not an empty mask"


@needs_model
def test_object_path_takes_several_prompts(predictor, image):
    r = predictor.process(image, geometry=[
        GeometryPrompt.click(1, CLICK),
        GeometryPrompt.box(2, REAR_WHEEL),
    ])
    assert r.instance_ids.tolist() == [1, 2]
    assert r.masks_logits.shape[0] == 2
    assert (r.masks_logits > 0).sum(dim=(1, 2)).min() > 0


@needs_model
def test_object_path_matches_video(predictor, image):
    """The same tracker call the video predictor makes on frame 0.

    Not bit-equality: the image path preprocesses with the GPU resize while the video
    path uses the image-folder loader, and that regime difference is worth ~0.3% of the
    mask. Everything downstream of the encode is the same code.
    """
    from sam.build_sam import build_sam3_video_predictor

    ours = (predictor.process(image, geometry=GeometryPrompt.box(1, REAR_WHEEL))
            .masks_logits[0] > 0)

    video = build_sam3_video_predictor(
        config_file="configs/sam3/sam3.yaml", ckpt_path=CKPT, device="cuda")
    out = video.start_session().process(image, prompts=[GeometryPrompt.box(1, REAR_WHEEL)])
    theirs = out[1].best_mask_logits.reshape(ours.shape) > 0

    iou = float((ours & theirs).sum()) / float((ours | theirs).sum())
    assert iou > 0.99, f"image and video selection disagree (IoU {iou:.4f})"


# ---------------------------------------------------------------- encode reuse
@needs_model
def test_encode_is_reusable_and_exact(predictor, image, monkeypatch):
    enc = predictor.encode(image)
    assert enc.image_hw == (image.shape[0], image.shape[1])

    calls = []
    original = type(predictor)._encode_views
    monkeypatch.setattr(type(predictor), "_encode_views",
                        lambda self, x: calls.append(1) or original(self, x))

    assert_same(predictor.process(enc, concept="wheel"),
                predictor.process(image, concept="wheel"))
    assert len(calls) == 1, "only the array form should re-encode"

    reused = predictor.process(enc, geometry=GeometryPrompt.box(1, REAR_WHEEL))
    assert reused.instance_ids.tolist() == [1]
    assert len(calls) == 1, "the object path must reuse the same encode"


# ---------------------------------------------------------------- the edges
@needs_model
def test_process_refuses_the_ambiguous_calls(predictor, image):
    with pytest.raises(ValueError, match="needs something to do"):
        predictor.process(image)
    with pytest.raises(ValueError, match="different ids"):
        predictor.process(image, concept="wheel",
                          geometry=GeometryPrompt.click(1, CLICK))
    with pytest.raises(ValueError, match="need a concept"):
        predictor.process(image, geometry=GeometryPrompt.exemplar_box(REAR_WHEEL))
    with pytest.raises(ValueError, match="one exemplar prompt"):
        predictor.process(image, concept="wheel", geometry=[
            GeometryPrompt.exemplar_box(REAR_WHEEL),
            GeometryPrompt.exemplar_point(CLICK),
        ])


@pytest.fixture(scope="module")
def mux():
    from sam.build_sam import build_sam3_multiplex

    return build_sam3_multiplex(
        config_file="configs/sam3/sam3.1.yaml", ckpt_path=MUX_CKPT, device="cuda")


@needs_mux
def test_multiplex_answers_both_paths(mux, image):
    """SAM 3.1 carries its own tracker, so the same two paths work on that lineage."""
    detected = mux.process(image, concept="wheel")
    assert int(detected.boxes.shape[0]) == 4
    assert detected.presence > 0.9

    picked = mux.process(image, geometry=GeometryPrompt.box(1, REAR_WHEEL))
    assert picked.instance_ids.tolist() == [1]
    assert picked.presence is None
    assert int((picked.masks_logits > 0).sum()) > 10_000


@needs_mux
def test_multiplex_decodes_several_prompts_at_once(mux, image):
    """One batched call through the interactive head, ids preserved in order."""
    r = mux.process(image, geometry=[
        GeometryPrompt.click(1, CLICK),
        GeometryPrompt.box(2, (1396.0, 560.0, 1627.0, 776.0)),
    ])
    assert r.instance_ids.tolist() == [1, 2]
    assert (r.masks_logits > 0).sum(dim=(1, 2)).min() > 0


@needs_mux
def test_multiplex_rejects_mask_prompts(mux, image):
    """No multiplex path for a mask -- said plainly rather than guessed around."""
    with pytest.raises(NotImplementedError, match="mask prompt"):
        mux.process(image, geometry=GeometryPrompt.mask(1, np.zeros(image.shape[:2], bool)))


@needs_mux
def test_multiplex_placeholder_is_its_own_caption(mux, image):
    """The 3.1 caption differs from base, and PLACEHOLDER is how you avoid typing it."""
    assert mux.BOX_ONLY_CAPTION == "<text placeholder>"
    r = mux.process(image, concept=mux.PLACEHOLDER,
                    geometry=GeometryPrompt.exemplar_box(REAR_WHEEL))
    assert r.presence is not None


def test_geometry_split_is_by_route():
    """The split the whole design rests on, without needing a model."""
    from sam.models.sam3_predictor import _split_image_geometry

    exemplar, objects = _split_image_geometry(GeometryPrompt.exemplar_box(REAR_WHEEL))
    assert exemplar is not None and objects == []

    exemplar, objects = _split_image_geometry(
        [GeometryPrompt.click(1, CLICK), GeometryPrompt.box(2, REAR_WHEEL)])
    assert exemplar is None and len(objects) == 2

    assert _split_image_geometry(None) == (None, [])
