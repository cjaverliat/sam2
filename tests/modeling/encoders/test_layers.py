import torch
from sam.modeling.encoders._layers import SqueezeExcite, to_2tuple

def test_to_2tuple():
    assert to_2tuple(3) == (3, 3)
    assert to_2tuple((4, 5)) == (4, 5)

def test_squeeze_excite_shape_preserving():
    se = SqueezeExcite(16, rd_ratio=0.25).eval()
    x = torch.randn(2, 16, 8, 8)
    y = se(x)
    assert y.shape == x.shape
