# SPDX-License-Identifier: LicenseRef-SAM
"""TDD tests for EfficientViT backbone vendor + EfficientSam3Trunk branch (C1).

RED phase: these fail before the vendor package and trunk branch are implemented.
GREEN phase: pass after implementation.
"""
import torch


def test_efficientvit_backbone_b1_builds():
    """The vendor package is importable and b1 factory builds correctly."""
    from sam.modeling.encoders.efficientvit.efficientvit.backbone import (
        efficientvit_backbone_b1,
    )

    backbone = efficientvit_backbone_b1()
    # b1 width_list = [16, 32, 64, 128, 256]; final stage = 256 channels
    assert backbone.width_list[-1] == 256


def test_triton_fallback_flag():
    """On this CPU-only Windows env, triton is absent → _TRITON_AVAILABLE is False.

    If triton were installed, the CPU forward still exercises the pure-torch path
    because TritonRMSNorm2d.forward checks x.is_cuda before calling the kernel.
    """
    from sam.modeling.encoders.efficientvit.nn.norm import _TRITON_AVAILABLE

    # The flag must exist and be a bool; its value is env-dependent (False without
    # triton, True with it). The pure-torch fallback is exercised regardless by the
    # CPU forward test below (TritonRMSNorm2d.forward checks x.is_cuda first).
    assert isinstance(_TRITON_AVAILABLE, bool)


def test_efficientvit_trunk_channel_list():
    from sam.modeling.encoders.efficientsam3_trunk import EfficientSam3Trunk

    trunk = EfficientSam3Trunk(backbone_type="efficientvit", model_name="b1")
    assert trunk.channel_list == [1024]


def test_efficientvit_trunk_forward_shape():
    """CPU forward: exercises pure-torch RMSNorm fallback (no triton needed)."""
    from sam.modeling.encoders.efficientsam3_trunk import EfficientSam3Trunk

    trunk = EfficientSam3Trunk(backbone_type="efficientvit", model_name="b1")
    trunk.eval()
    with torch.no_grad():
        out = trunk(torch.randn(1, 3, 1008, 1008))
    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0].shape == (1, 1024, 72, 72), f"expected (1,1024,72,72), got {out[0].shape}"
