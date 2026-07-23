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
    from sam.models.sam3_predictor import Sam3Predictor
    from sam.prompts import GeometryPrompt
    dev = torch.device("cpu")
    # box xyxy pixel -> cxcywh normalized
    p = GeometryPrompt(obj_id=1, boxes=torch.tensor([[100.0, 200.0, 300.0, 400.0]]))
    geo = Sam3Predictor._pack_geometry(p, (540, 960), dev)
    assert geo["box_coords"].shape == (1, 1, 4)
    cx, cy, bw, bh = geo["box_coords"][0, 0].tolist()
    assert abs(cx - 200 / 960) < 1e-6 and abs(cy - 300 / 540) < 1e-6
    assert abs(bw - 200 / 960) < 1e-6 and abs(bh - 200 / 540) < 1e-6
    # point
    pp = GeometryPrompt(obj_id=1, points_coords=torch.tensor([[480.0, 270.0]]),
                        points_labels=torch.tensor([1]))
    g2 = Sam3Predictor._pack_geometry(pp, (540, 960), dev)
    assert g2["point_coords"].shape == (1, 1, 2)
    assert abs(g2["point_coords"][0, 0, 0].item() - 0.5) < 1e-6
