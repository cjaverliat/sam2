# SPDX-License-Identifier: LicenseRef-SAM
import os

import pytest
import torch

CKPT = "checkpoints/sam3.1_multiplex.pt"
needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.isfile(CKPT),
    reason="needs CUDA + sam3.1_multiplex.pt",
)


def _detector():
    from sam.build_sam import build_sam3_multiplex_video_predictor
    return build_sam3_multiplex_video_predictor(
        config_file="configs/sam3/sam3.1.yaml", ckpt_path=CKPT, device="cuda").detector


@needs_gpu
def test_geometry_encoder_box_and_point_token_counts():
    ge = _detector().geometry_encoder
    B, C = 1, ge.d_model
    hw = 72
    img_feats = [torch.randn(hw * hw, B, C, device="cuda")]
    img_pos = [torch.randn(hw * hw, B, C, device="cuda")]
    # text-only (null) -> CLS-only: 1 token
    f0, m0 = ge(img_feats, img_pos, B)
    assert f0.shape[0] == 1 and m0.shape == (B, 1)
    # 2 boxes -> 2 + 1(cls) tokens
    box = torch.tensor([[[0.5, 0.5, 0.2, 0.2]], [[0.3, 0.3, 0.1, 0.1]]], device="cuda")
    lbl = torch.ones(2, B, device="cuda")
    fb, mb = ge(img_feats, img_pos, B, img_sizes=[(hw, hw)],
                box_coords=box, box_labels=lbl)
    assert fb.shape[0] == 3 and torch.isfinite(fb).all()
    # 1 point -> 1 + 1(cls) tokens
    pt = torch.tensor([[[0.4, 0.6]]], device="cuda")
    pl = torch.ones(1, B, device="cuda")
    fp, mp = ge(img_feats, img_pos, B, img_sizes=[(hw, hw)],
                point_coords=pt, point_labels=pl)
    assert fp.shape[0] == 2 and torch.isfinite(fp).all()


def test_pack_geometry_box_and_point():
    import numpy as np
    from sam.models.sam3_predictor import _pack_geometry
    from sam.prompts import GeometryPrompt
    dev = torch.device("cpu")
    # box xyxy pixel -> cxcywh normalized
    p = GeometryPrompt.exemplar_box((100.0, 200.0, 300.0, 400.0))
    geo = _pack_geometry(p, (540, 960), dev)
    assert geo["box_coords"].shape == (1, 1, 4)
    cx, cy, bw, bh = geo["box_coords"][0, 0].tolist()
    assert abs(cx - 200 / 960) < 1e-6 and abs(cy - 300 / 540) < 1e-6
    assert abs(bw - 200 / 960) < 1e-6 and abs(bh - 200 / 540) < 1e-6
    # point
    pp = GeometryPrompt.exemplar_point((480.0, 270.0))
    g2 = _pack_geometry(pp, (540, 960), dev)
    assert g2["point_coords"].shape == (1, 1, 2)
    assert abs(g2["point_coords"][0, 0, 0].item() - 0.5) < 1e-6


def test_pack_geometry_defaults_box_labels_to_positive():
    from sam.models.sam3_predictor import _pack_geometry
    from sam.prompts import GeometryPrompt
    p = GeometryPrompt.exemplar_box((100.0, 200.0, 300.0, 400.0))
    geo = _pack_geometry(p, (540, 960), torch.device("cpu"))
    assert geo["box_labels"].shape == (1, 1)
    assert geo["box_labels"].flatten().tolist() == [1]


def test_pack_geometry_forwards_negative_box_labels():
    from sam.models.sam3_predictor import _pack_geometry
    from sam.prompts import GeometryPrompt, PromptRoute
    p = GeometryPrompt(
        obj_id=-1,
        boxes=torch.tensor([[100.0, 200.0, 300.0, 400.0], [10.0, 20.0, 30.0, 40.0]]),
        boxes_labels=torch.tensor([1, 0]),
        route=PromptRoute.DETECTOR,
    )
    geo = _pack_geometry(p, (540, 960), torch.device("cpu"))
    assert geo["box_labels"].shape == (2, 1)
    assert geo["box_labels"].flatten().tolist() == [1, 0]


def test_geometry_prompt_rejects_boxes_labels_length_mismatch():
    from sam.prompts import GeometryPrompt
    with pytest.raises(ValueError):
        GeometryPrompt(obj_id=1, boxes=torch.zeros(2, 4),
                       boxes_labels=torch.tensor([1]))


def test_geometry_prompt_rejects_boxes_labels_without_boxes():
    from sam.prompts import GeometryPrompt
    with pytest.raises(ValueError):
        GeometryPrompt(obj_id=1, boxes_labels=torch.tensor([1]))


def test_geometry_prompt_clone_and_to_preserve_boxes_labels():
    from sam.prompts import GeometryPrompt
    p = GeometryPrompt(obj_id=1, boxes=torch.zeros(1, 4),
                       boxes_labels=torch.tensor([0]))
    assert p.clone().boxes_labels.flatten().tolist() == [0]
    assert p.to(torch.device("cpu")).boxes_labels.flatten().tolist() == [0]


def test_pack_geometry_rejects_tracker_route_prompts():
    """An image predictor has no tracker, so a SAM 2-style prompt is a mistake, not a hint."""
    from sam.models.sam3_predictor import _pack_geometry
    from sam.prompts import GeometryPrompt

    dev = torch.device("cpu")
    with pytest.raises(ValueError, match="exemplar_box"):
        _pack_geometry(GeometryPrompt.box(1, (10.0, 20.0, 30.0, 40.0)), (540, 960), dev)
    with pytest.raises(ValueError, match="exemplar_point"):
        _pack_geometry(GeometryPrompt.click(1, (10.0, 20.0)), (540, 960), dev)


def test_exemplar_point_and_click_pack_the_same_geometry():
    """Only the intent differs: the numbers the detector sees are identical."""
    from sam.models.sam3_predictor import _pack_geometry
    from sam.prompts import GeometryPrompt, PromptRoute

    dev = torch.device("cpu")
    concept = _pack_geometry(GeometryPrompt.exemplar_point((480.0, 270.0)), (540, 960), dev)

    as_click = GeometryPrompt.click(1, (480.0, 270.0))
    as_click.route = PromptRoute.DETECTOR  # what the old code did silently
    assert torch.equal(concept["point_coords"], _pack_geometry(as_click, (540, 960), dev)["point_coords"])


def test_exemplar_point_rejects_bad_input():
    from sam.prompts import GeometryPrompt

    with pytest.raises(ValueError, match="x, y"):
        GeometryPrompt.exemplar_point((1.0, 2.0, 3.0))
    with pytest.raises(ValueError, match="1 or 0"):
        GeometryPrompt.exemplar_point((1.0, 2.0), label=2)
