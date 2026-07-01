# SPDX-License-Identifier: LicenseRef-SAM
"""TDD tests for EfficientViT backbone vendor + EfficientSam3Trunk branch (C1).

RED phase: these fail before the vendor package and trunk branch are implemented.
GREEN phase: pass after implementation.
"""


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


# NOTE: the EfficientSam3Trunk channel_list == [1024] and (1,1024,72,72) forward-shape checks
# for efficientvit/b1 are covered by the parametrized test in test_efficientsam3_trunk.py
# (shared across repvit / efficientvit / tinyvit). The pure-torch RMSNorm CPU fallback is
# exercised by that forward pass.
