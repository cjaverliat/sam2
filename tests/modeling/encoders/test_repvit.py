import torch
from sam.modeling.encoders.repvit import repvit_m1_1

def test_repvit_m1_1_feature_channels():
    m = repvit_m1_1(distillation=False).eval()
    x = torch.randn(1, 3, 224, 224)
    feats = x
    for f in m.features:
        feats = f(feats)
    # RV-M (m1.1) final stage = 512 channels (validated from checkpoint)
    assert feats.shape[1] == 512
