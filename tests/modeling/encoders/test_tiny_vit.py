import torch
import pytest
from sam.modeling.encoders.tiny_vit import tiny_vit_11m_224
from sam.modeling.encoders.efficientsam3_trunk import EfficientSam3Trunk


def test_tiny_vit_11m_builds_and_channel():
    """tiny_vit_11m_224(img_size=1008) builds; norm_head exposes final embed_dim=448."""
    model = tiny_vit_11m_224(img_size=1008).eval()
    # embed_dims[-1] = 448 for 11m variant
    assert model.norm_head.normalized_shape[0] == 448


def test_tinyvit_trunk_channel_list():
    trunk = EfficientSam3Trunk(backbone_type="tinyvit", model_name="11m").eval()
    assert trunk.channel_list == [1024]


def test_tinyvit_trunk_forward_shape():
    trunk = EfficientSam3Trunk(backbone_type="tinyvit", model_name="11m").eval()
    with torch.no_grad():
        out = trunk(torch.randn(1, 3, 1008, 1008))
    assert isinstance(out, list) and len(out) == 1
    assert out[0].shape == (1, 1024, 72, 72)
