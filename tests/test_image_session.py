# SPDX-License-Identifier: LicenseRef-SAM
"""Image sessions: the wrapper over one encode + many detects.

A session binds a predictor to one image. It adds NO model logic -- every test here
asserts a session run is bit-identical to the equivalent ``predict`` call, which is
what makes it safe to reach for whenever an image takes more than one prompt.

The per-call verb matches the video sessions': ``process``.
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


@pytest.fixture(scope="module")
def predictor():
    from sam.build_sam import build_sam3

    return build_sam3(config_file="configs/sam3/sam3.yaml", ckpt_path=CKPT, device="cuda")


@pytest.fixture(scope="module")
def image():
    return np.asarray(Image.open(IMAGE).convert("RGB"))


def assert_same(got, expected):
    """Two ``Sam3DetectionResult``s agree exactly."""
    assert got.presence == expected.presence
    assert torch.equal(got.scores, expected.scores)
    assert torch.equal(got.boxes, expected.boxes)
    assert torch.equal(got.masks_logits, expected.masks_logits)


@needs_model
def test_process_matches_predict(predictor, image):
    expected = predictor.predict(image, ConceptPrompt(text="wheel"))
    got = predictor.start_image_session(image).process(ConceptPrompt(text="wheel"))
    assert_same(got, expected)


@needs_model
@pytest.mark.parametrize("threshold", [0.1, 0.5, 0.9])
def test_process_matches_predict_per_threshold(predictor, image, threshold):
    expected = predictor.predict(
        image, ConceptPrompt(text="wheel"), confidence_threshold=threshold)
    got = predictor.start_image_session(image).process(
        ConceptPrompt(text="wheel"), confidence_threshold=threshold)
    assert_same(got, expected)


@needs_model
@pytest.mark.parametrize("geometry", [
    GeometryPrompt.concept_box(REAR_WHEEL),
    GeometryPrompt.concept_box(REAR_WHEEL, label=0),
    GeometryPrompt.concept_point((560.0, 730.0)),
])
def test_process_matches_predict_with_geometry(predictor, image, geometry):
    expected = predictor.predict(
        image, ConceptPrompt(text="wheel"), geometry=geometry.clone())
    got = predictor.start_image_session(image).process(
        ConceptPrompt(text="wheel"), geometry=geometry.clone())
    assert_same(got, expected)


@needs_model
def test_one_session_serves_many_prompts(predictor, image):
    """The whole point: several prompts, one encode, each still exact."""
    session = predictor.start_image_session(image)
    for concept, threshold in (("wheel", 0.5), ("truck", 0.5), ("wheel", 0.9)):
        expected = predictor.predict(
            image, ConceptPrompt(text=concept), confidence_threshold=threshold)
        assert_same(session.process(ConceptPrompt(text=concept),
                                    confidence_threshold=threshold), expected)


@needs_model
def test_session_encodes_once(predictor, image, monkeypatch):
    session = predictor.start_image_session(image)

    calls = []
    original = type(predictor).encode_image
    monkeypatch.setattr(type(predictor), "encode_image",
                        lambda self, x: calls.append(1) or original(self, x))

    for _ in range(3):
        session.process(ConceptPrompt(text="wheel"))
    assert calls == [], "process() must not re-run the vision encoder"

    predictor.predict(image, ConceptPrompt(text="wheel"))
    assert len(calls) == 1, "predict() still encodes per call"


@needs_model
def test_session_reports_image_size(predictor, image):
    session = predictor.start_image_session(image)
    assert session.image_hw == (image.shape[0], image.shape[1])


class _RecordingPredictor:
    """Enough of a predictor to see what a session forwards (no weights, no CUDA)."""

    def __init__(self):
        self.calls = []

    def _detect_encoded(self, feats, pos, image_hw, concept,
                        confidence_threshold, geometry, dtype):
        self.calls.append({
            "feats": feats, "pos": pos, "image_hw": image_hw, "concept": concept,
            "confidence_threshold": confidence_threshold, "geometry": geometry,
            "dtype": dtype,
        })
        return "detection"


def test_session_forwards_its_state_and_arguments():
    """The session holds the encode and the dtype; everything else is the caller's."""
    from sam.models.image_session import ImageSession

    predictor = _RecordingPredictor()
    feats, pos = ["feats"], ["pos"]
    session = ImageSession(predictor, feats, pos, (120, 80), torch.float32)

    concept = ConceptPrompt(text="wheel")
    geometry = GeometryPrompt.concept_box(REAR_WHEEL)
    assert session.process(concept, confidence_threshold=0.7, geometry=geometry) == "detection"

    (call,) = predictor.calls
    assert call["feats"] is feats and call["pos"] is pos
    assert call["image_hw"] == (120, 80)
    assert call["concept"] is concept
    assert call["confidence_threshold"] == 0.7
    assert call["geometry"] is geometry
    assert call["dtype"] is torch.float32, "the session's dtype drives every detect"


def test_session_defaults_match_predict_defaults():
    from sam.models.image_session import ImageSession

    predictor = _RecordingPredictor()
    ImageSession(predictor, [], [], (10, 10)).process(ConceptPrompt(text="wheel"))

    (call,) = predictor.calls
    assert call["confidence_threshold"] == 0.5
    assert call["geometry"] is None
    assert call["dtype"] is torch.bfloat16


@needs_mux
def test_multiplex_process_matches_predict():
    """SAM 3.1 keeps its own post-processing outside autocast -- both paths, same split."""
    from sam.build_sam import build_sam3_multiplex

    mux = build_sam3_multiplex(
        config_file="configs/sam3/sam3.1.yaml", ckpt_path=MUX_CKPT, device="cuda")
    img = np.asarray(Image.open(IMAGE).convert("RGB"))

    expected = mux.predict(img, ConceptPrompt(text="wheel"))
    got = mux.start_image_session(img).process(ConceptPrompt(text="wheel"))
    assert_same(got, expected)


@needs_model
def test_placeholder_uses_this_lineage_box_only_caption(predictor, image):
    """PLACEHOLDER == upstream's no-text caption for the base lineage: "visual"."""
    from sam.prompts import ConceptPrompt

    expected = predictor.predict(
        image, ConceptPrompt(text=predictor.BOX_ONLY_CAPTION),
        geometry=GeometryPrompt.concept_box(REAR_WHEEL))
    got = predictor.predict(
        image, predictor.PLACEHOLDER, geometry=GeometryPrompt.concept_box(REAR_WHEEL))
    assert_same(got, expected)
    assert predictor.BOX_ONLY_CAPTION == "visual"


@needs_model
def test_predict_takes_a_bare_phrase(predictor, image):
    """`predict(img, "wheel")` matches ConceptPrompt("wheel"), as sessions already allow."""
    assert_same(predictor.predict(image, "wheel"),
                predictor.predict(image, ConceptPrompt(text="wheel")))


@needs_model
def test_image_path_refuses_tracker_prompts(predictor, image):
    """The SAM 2 gesture names the alternative instead of quietly meaning something else."""
    with pytest.raises(ValueError, match="concept_box"):
        predictor.predict(image, "wheel", geometry=GeometryPrompt.box(1, REAR_WHEEL))
    with pytest.raises(ValueError, match="concept_point"):
        predictor.predict(image, "wheel", geometry=GeometryPrompt.click(1, (560.0, 730.0)))


def test_placeholder_caption_differs_by_lineage():
    """Base and multiplex encode different captions -- passing the wrong one finds nothing."""
    from sam.models.sam3_predictor import Sam3MultiplexPredictor, Sam3Predictor

    assert Sam3Predictor.BOX_ONLY_CAPTION == "visual"
    assert Sam3MultiplexPredictor.BOX_ONLY_CAPTION == "<text placeholder>"
    assert Sam3Predictor.PLACEHOLDER is Sam3MultiplexPredictor.PLACEHOLDER
