import pytest
import torch
from sam.modeling.encoders.efficientsam3_trunk import EfficientSam3Trunk


# Shared trunk contract across all three distilled backbones: channel_list == [1024] and a
# single stride-14 feature map of shape (1, 1024, 72, 72) for a 1008x1008 input. Backbone-
# specific checks (b1 width_list, triton flag, tinyvit norm_head dim) live in the respective
# test_efficientvit.py / test_tiny_vit.py modules.
@pytest.mark.parametrize(
    "backbone_type,model_name",
    [("repvit", "m1_1"), ("efficientvit", "b1"), ("tinyvit", "11m")],
)
def test_trunk_channel_list_and_forward_shape(backbone_type, model_name):
    trunk = EfficientSam3Trunk(backbone_type=backbone_type, model_name=model_name).eval()
    assert trunk.channel_list == [1024]
    with torch.no_grad():
        out = trunk(torch.randn(1, 3, 1008, 1008))
    assert isinstance(out, list) and len(out) == 1
    assert out[0].shape == (1, 1024, 72, 72)
