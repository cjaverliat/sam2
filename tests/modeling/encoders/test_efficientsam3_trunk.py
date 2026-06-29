import torch
from sam.modeling.encoders.efficientsam3_trunk import EfficientSam3Trunk


def test_trunk_outputs_1024_at_72():
    trunk = EfficientSam3Trunk(backbone_type="repvit", model_name="m1_1").eval()
    assert trunk.channel_list == [1024]
    out = trunk(torch.randn(1, 3, 1008, 1008))
    assert isinstance(out, list) and len(out) == 1
    assert out[0].shape == (1, 1024, 72, 72)
