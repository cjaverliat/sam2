# SPDX-License-Identifier: LicenseRef-SAM
"""EfficientSAM3 vision trunk: a lightweight backbone + a projection to the PE-trunk-compatible
feature map (1024-ch @ 72x72), exposing the VisionTrunk contract (.channel_list + forward->list)
so it drops into the existing Sam3DualViTDetNeck unchanged. Submodule names mirror the upstream
student encoder so EfficientSAM3 checkpoints load strict (see spec §7).

Upstream cross-check (efficientsam3_reference/stage1/model.py):
  - ImageStudentEncoder.head[0] uses bias=False — applied here (brief omitted it).
  - All other layers, attribute names, and nesting match the brief exactly.
"""
import torch
import torch.nn as nn
from sam.modeling.encoders.repvit import repvit_m0_9, repvit_m1_1, repvit_m2_3
from sam.modeling.encoders.tiny_vit import tiny_vit_5m_224, tiny_vit_11m_224, tiny_vit_21m_224

_REPVIT = {
    "m0_9": repvit_m0_9, "m0.9": repvit_m0_9,
    "m1_1": repvit_m1_1, "m1.1": repvit_m1_1,
    "m2_3": repvit_m2_3, "m2.3": repvit_m2_3,
}

_TINYVIT = {
    "5m": tiny_vit_5m_224,
    "11m": tiny_vit_11m_224,
    "21m": tiny_vit_21m_224,
}


class _RepViTTrunk(nn.Module):
    """Runs RepViT.features; exposes channel_list. (Upstream: RepViTAdapter.)"""

    def __init__(self, model_name: str):
        super().__init__()
        # num_classes=0 -> the unused ImageNet head is an nn.Identity (zero params), so the
        # trunk state_dict carries ONLY the features (matching the EfficientSAM3 checkpoint,
        # which has no classifier). forward() runs model.features directly, never the head.
        self.model = _REPVIT[model_name](num_classes=0, distillation=False)
        dummy = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            for f in self.model.features:
                dummy = f(dummy)
        self.channel_list = [dummy.shape[1]]

    def forward(self, x):
        for f in self.model.features:
            x = f(x)
        return x


class _TinyViTTrunk(nn.Module):
    """Runs TinyViT.patch_embed then layers; reshapes (B,L,C) → (B,C,H,W).
    Exposes channel_list = [embed_dims[-1]]. (Upstream: TinyViTTrunkWrapper.)"""

    def __init__(self, model_name: str, img_size: int = 1008):
        super().__init__()
        self.model = _TINYVIT[model_name](img_size=img_size)
        self.channel_list = [self.model.norm_head.normalized_shape[0]]

    def forward(self, x):
        x = self.model.patch_embed(x)
        for layer in self.model.layers:
            x = layer(x)
        B, L, C = x.shape
        side = int(L ** 0.5)
        assert side * side == L, f"TinyViT token count {L} is not a perfect square (img_size must be square)"
        return x.view(B, side, side, C).permute(0, 3, 1, 2).contiguous()


class _ImageStudentEncoder(nn.Module):
    """Project backbone features to embed_dim @ embed_size (upstream: ImageStudentEncoder).
    Submodules: .backbone (the trunk wrapper), .head (1x1 conv + BN + GELU + 3x3 conv).

    Note: head[0] uses bias=False matching upstream checkpoint layout where head.0 stores
    only the weight tensor with shape (embed_dim, in_channels, 1, 1)."""

    def __init__(self, backbone: nn.Module, in_channels: int, embed_dim=1024, embed_size=72):
        super().__init__()
        self.backbone = backbone
        self.embed_size = embed_size
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        if x.shape[-1] != self.embed_size:
            x = nn.functional.interpolate(
                x,
                size=(self.embed_size, self.embed_size),
                mode="bilinear",
                align_corners=False,
            )
        return x


class EfficientSam3Trunk(nn.Module):
    """VisionTrunk: .channel_list=[embed_dim]; forward(x)->[feat].

    Submodule nesting for strict checkpoint load:
      self.model               → trunk.model  (ImageStudentEncoder)
      self.model.backbone      → trunk.model.backbone  (RepViTAdapter wrapper)
      self.model.backbone.model→ trunk.model.backbone.model  (RepViT .features / TinyViT patch_embed+layers)
      self.model.head          → trunk.model.head  (projection Sequential)
    """

    def __init__(
        self,
        backbone_type: str = "repvit",
        model_name: str = "m1_1",
        embed_dim: int = 1024,
        embed_size: int = 72,
        img_size: int = 1008,
    ):
        super().__init__()
        if backbone_type == "repvit":
            bk = _RepViTTrunk(model_name)
        elif backbone_type == "tinyvit":
            bk = _TinyViTTrunk(model_name, img_size=img_size)
        else:
            raise NotImplementedError(
                f"backbone_type={backbone_type!r} is not supported; expected 'repvit' or 'tinyvit'"
            )
        self.model = _ImageStudentEncoder(bk, bk.channel_list[0], embed_dim, embed_size)
        self.channel_list = [embed_dim]

    def forward(self, x):
        if isinstance(x, list):
            x = x[0]
        return [self.model(x)]
