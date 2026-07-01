from sam.modeling.encoders.tiny_vit import tiny_vit_11m_224


def test_tiny_vit_11m_builds_and_channel():
    """tiny_vit_11m_224(img_size=1008) builds; norm_head exposes final embed_dim=448."""
    model = tiny_vit_11m_224(img_size=1008).eval()
    # embed_dims[-1] = 448 for 11m variant
    assert model.norm_head.normalized_shape[0] == 448


# NOTE: the EfficientSam3Trunk channel_list == [1024] and (1,1024,72,72) forward-shape checks
# for tinyvit/11m are covered by the parametrized test in test_efficientsam3_trunk.py
# (shared across repvit / efficientvit / tinyvit).
