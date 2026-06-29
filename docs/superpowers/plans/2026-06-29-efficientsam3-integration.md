# EfficientSAM3 / EfficientSAM3.1 Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the four EfficientSAM3 families (EfficientSAM3, SAM3-LiteText, SAM3.1-LiteText, EfficientSAM3.1) to the `sam/` package as PyTorch inference, by swapping only the vision trunk and text encoder and reusing every SAM 3 internal.

**Architecture:** EfficientSAM3 = SAM 3 with two encoders swapped. A new `EfficientSam3Trunk` (vendored lightweight backbone + a projection to 1024-ch @ 72×72) plugs into the **existing** `Sam3DualViTDetNeck`; a new `MobileClipTextEncoder` (vendored MobileCLIP + projector, reusing the existing `Sam3Tokenizer`) is a drop-in for `Sam3TextEncoder`. The detector, multiplex tracker, memory bank, mask decoder, predictors, association and streaming are reused unchanged. Variant = {distilled trunk | PE trunk} × {base `Sam3Predictor` | multiplex `Sam3MultiplexVideoPredictor`}.

**Tech Stack:** PyTorch 2.11 (cu128), Hydra configs, `huggingface_hub` (public download), pytest. No new third-party deps (timm/einops avoided by vendoring two small utilities).

**Spec:** [`docs/superpowers/specs/2026-06-29-efficientsam3-integration-design.md`](../specs/2026-06-29-efficientsam3-integration-design.md)

## Global Constraints

- **No new third-party deps.** Avoid `timm` and `einops`. Vendor `SqueezeExcite` + `to_2tuple` into `sam/modeling/encoders/_layers.py`; reuse the existing local `DropPath` (`sam/modeling/encoders/pe_vitdet.py:34`) and `torch.nn.init.trunc_normal_`; drop `@register_model` / `build_model_with_cfg` (call factories directly).
- **Strict checkpoint loading.** `load_state_dict(..., strict=True)` after remap. No silent `strict=False`. Base EfficientSAM3 image ckpt is detector-root → prepend `detector.`; SAM 3.1 ckpts are full-model-root (`detector.*` + `tracker.model.*`) → load directly.
- **Per-file SPDX headers.** Vendored backbone files: each backbone's own license (`repvit.py`, `tiny_vit.py`, `efficientvit/*`); `mobile_clip.py` → `LicenseRef-AppleML`; EfficientSAM3 glue / anything importing SAM 3 → `LicenseRef-SAM`.
- **Weights never vendored.** Public download only, from HF `Simon7108528/EfficientSAM3`. `HF_HUB_DISABLE_XET=1`.
- **Naming:** version visible in class names — `EfficientSam3*`. Configs under `sam/configs/efficientsam3/`.
- **Preserve `state_dict` keys / attribute names** so checkpoints load strict (or add an explicit documented remap).
- **Text encoder built at the checkpoint's context length directly** (16/32) — never "build-at-77-then-truncate".
- **Parity tolerance:** mask IoU ≥ 0.99 vs upstream golden on matched instances; instance count exact; FPS within ~15%/phase on the same GPU/input/precision (regression guard).

---

## File structure

**New files:**
- `sam/modeling/encoders/_layers.py` — vendored `SqueezeExcite`, `to_2tuple` (de-timm helpers).
- `sam/modeling/encoders/repvit.py` — vendored RepViT (de-timm'd). `[Phase A]`
- `sam/modeling/encoders/tiny_vit.py` — vendored TinyViT (de-timm'd). `[Phase B]`
- `sam/modeling/encoders/efficientvit/` — vendored EfficientViT package (+ Triton fallback). `[Phase C]`
- `sam/modeling/encoders/efficientsam3_trunk.py` — `EfficientSam3Trunk` (+ `ImageStudentProjection`).
- `sam/modeling/text/mobile_clip.py` — vendored MobileCLIP text transformer (de-timm'd).
- `sam/modeling/text/mobileclip_text_encoder.py` — `MobileClipTextEncoder`.
- `sam/modeling/text/text_encoder_base.py` — `TextEncoder` `Protocol`.
- `sam/configs/efficientsam3/*.yaml` — one per variant.
- `tools/download_efficientsam3.py` — public HF download.
- `tools/benchmark_efficientsam3.py` — FPS benchmark (mirrors `efficientsam3_reference/_bench.py`).
- `tests/parity/reference_efficientsam3/` — committed golden fixtures + parity tests.
- `LICENSE_mobileclip`, `LICENSE_repvit`, `LICENSE_tinyvit`, `LICENSE_efficientvit` — license texts.

**Modified files:**
- `sam/build_sam.py` — add `build_efficientsam3*` builders.
- `sam/modeling/text/__init__.py`, `sam/modeling/encoders/__init__.py` — exports.
- `pyproject.toml` — pixi `download-efficientsam3-*` tasks.
- `README.md` — EfficientSAM3 section + license-table rows.
- `NOTICE` — add vendored components.

**Reused unchanged:** `Sam3DualViTDetNeck`, `Sam3VisionEncoder`, `Sam3Tokenizer`, `Sam3DetrDetector`, `Sam3Tracker`/multiplex, memory bank, mask decoder, `Sam3Predictor`/`Sam3MultiplexVideoPredictor`, association, tracklet.

---

## Phase A — EfficientSAM3 RepViT image (slice + shared scaffolding)

### Task A1: Commit the upstream golden fixtures

**Files:**
- Create: `tests/parity/reference_efficientsam3/golden/efficientsam3_repvit_summary.json`
- Create: `tests/parity/reference_efficientsam3/golden/efficientsam3_repvit_masks_dog.npz`
- Create: `tests/parity/reference_efficientsam3/golden/efficientsam3_repvit_masks_person.npz`
- Create: `tests/parity/reference_efficientsam3/README.md` (provenance: upstream commit, ckpt, command)

**Interfaces:**
- Produces: golden fixtures consumed by A6's parity test (`summary.json` keys: `prompts.{dog,person}.{masks.sha1, num_instances, scores}`).

- [ ] **Step 1: Copy the captured golden** from `efficientsam3_reference/_golden/` into the fixture dir (renamed with the `efficientsam3_repvit_` prefix).

```bash
mkdir -p tests/parity/reference_efficientsam3/golden
cp ../efficientsam3_reference/_golden/summary.json tests/parity/reference_efficientsam3/golden/efficientsam3_repvit_summary.json
cp ../efficientsam3_reference/_golden/masks_dog.npz tests/parity/reference_efficientsam3/golden/efficientsam3_repvit_masks_dog.npz
cp ../efficientsam3_reference/_golden/masks_person.npz tests/parity/reference_efficientsam3/golden/efficientsam3_repvit_masks_person.npz
```

- [ ] **Step 2: Write the provenance README** documenting upstream repo URL + commit, checkpoint (`efficientsam3_ft/efficientsam3_repvit.pt`), image (`dog_person.jpeg`), prompts, threshold 0.1, fp32, and that masks are float (sigmoid) at original resolution.

- [ ] **Step 3: Confirm the npz fixtures are not gigantic** (compressed ~0.5 MB total — acceptable to commit). Run: `ls -la tests/parity/reference_efficientsam3/golden/`. Expected: 3 files, < 1 MB total.

- [ ] **Step 4: Commit**

```bash
git add tests/parity/reference_efficientsam3/
git commit -m "test(efficientsam3): commit upstream RepViT golden fixtures"
```

### Task A2: Vendor de-timm layer helpers

**Files:**
- Create: `sam/modeling/encoders/_layers.py`
- Test: `tests/modeling/encoders/test_layers.py`

**Interfaces:**
- Produces: `SqueezeExcite(channels: int, rd_ratio: float = 0.25)` (nn.Module), `to_2tuple(x) -> tuple`.

- [ ] **Step 1: Write the failing test**

```python
# tests/modeling/encoders/test_layers.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/modeling/encoders/test_layers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sam.modeling.encoders._layers'`

- [ ] **Step 3: Implement `_layers.py`**

```python
# SPDX-License-Identifier: Apache-2.0
"""Small layer utilities vendored to avoid a `timm` dependency (see spec E9).
`SqueezeExcite` mirrors `timm.layers.SqueezeExcite` (Apache-2.0); `to_2tuple`
mirrors `timm.layers.to_2tuple`. Pair with the local `DropPath` in pe_vitdet.py
and `torch.nn.init.trunc_normal_`."""
from collections.abc import Iterable
import torch
import torch.nn as nn


def to_2tuple(x):
    if isinstance(x, Iterable) and not isinstance(x, str):
        return tuple(x)
    return (x, x)


def _make_divisible(v, divisor=8):
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation block (timm-compatible: gate after fc2, hard-sigmoid)."""
    def __init__(self, channels: int, rd_ratio: float = 0.25):
        super().__init__()
        rd = _make_divisible(channels * rd_ratio)
        self.fc1 = nn.Conv2d(channels, rd, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(rd, channels, kernel_size=1, bias=True)
        self.gate = nn.Hardsigmoid()

    def forward(self, x):
        s = x.mean((2, 3), keepdim=True)
        s = self.fc2(self.act(self.fc1(s)))
        return x * self.gate(s)
```

> **Note for implementer:** the exact SE gate (hard-sigmoid vs sigmoid) and reduction rounding must match what `timm.layers.SqueezeExcite` produced for RepViT, or the checkpoint values won't reproduce. Verify against the upstream RepViT forward in Task A5's strict-load + A6's parity (a wrong SE shape fails strict load; a wrong gate fails parity). If parity fails, diff against `efficientsam3_reference/sam3/sam3/backbones/repvit.py` imports.

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run pytest tests/modeling/encoders/test_layers.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sam/modeling/encoders/_layers.py tests/modeling/encoders/test_layers.py
git commit -m "feat(efficientsam3): vendor SqueezeExcite + to_2tuple (de-timm)"
```

### Task A3: Vendor RepViT (de-timm'd)

**Files:**
- Create: `sam/modeling/encoders/repvit.py`
- Test: `tests/modeling/encoders/test_repvit.py`

**Interfaces:**
- Consumes: `sam.modeling.encoders._layers.SqueezeExcite`, `sam.modeling.encoders.pe_vitdet.DropPath`.
- Produces: `repvit_m0_9(distillation=False)`, `repvit_m1_1(distillation=False)`, `repvit_m2_3(distillation=False)` → `RepViT` (nn.Module) with `.features` (nn.ModuleList).

- [ ] **Step 1: Write the failing test**

```python
# tests/modeling/encoders/test_repvit.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/modeling/encoders/test_repvit.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Vendor `repvit.py`** by copying `efficientsam3_reference/sam3/sam3/backbones/repvit.py`, then de-timm:

  1. Add header: `# SPDX-License-Identifier: <RepViT license — check upstream repvit.py header>` + `# Vendored from SimonZeng7108/efficientsam3 sam3/backbones/repvit.py`.
  2. Replace `from timm.layers import SqueezeExcite` → `from sam.modeling.encoders._layers import SqueezeExcite`.
  3. Replace `from timm.models.vision_transformer import trunc_normal_` → `from torch.nn.init import trunc_normal_`.
  4. Remove `from timm.models import register_model` and delete every `@register_model` decorator line.
  5. Keep the `repvit_m0_9 / m1_1 / m2_3` factory functions (they `return RepViT(cfgs, ...)` directly — no timm needed).

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run pytest tests/modeling/encoders/test_repvit.py -v`
Expected: PASS (final channel 512)

- [ ] **Step 5: Commit**

```bash
git add sam/modeling/encoders/repvit.py tests/modeling/encoders/test_repvit.py
git commit -m "feat(efficientsam3): vendor RepViT backbone (de-timm)"
```

### Task A4: `EfficientSam3Trunk` (backbone + projection → trunk contract)

**Files:**
- Create: `sam/modeling/encoders/efficientsam3_trunk.py`
- Test: `tests/modeling/encoders/test_efficientsam3_trunk.py`

**Interfaces:**
- Consumes: `repvit_m1_1` (A3).
- Produces: `EfficientSam3Trunk(backbone_type: str, model_name: str, embed_dim: int = 1024, embed_size: int = 72, img_size: int = 1008)` with attribute `.channel_list = [embed_dim]` and `forward(x) -> list[Tensor]` (single-element list, `[B, embed_dim, embed_size, embed_size]`). Internal submodule names reproduce the upstream nesting (`trunk`-side) so checkpoints load strict: `self.model.backbone.model` is the RepViT, `self.model.head` is the projection. (See the remap note in A5.)

- [ ] **Step 1: Write the failing test**

```python
# tests/modeling/encoders/test_efficientsam3_trunk.py
import torch
from sam.modeling.encoders.efficientsam3_trunk import EfficientSam3Trunk

def test_trunk_outputs_1024_at_72():
    trunk = EfficientSam3Trunk(backbone_type="repvit", model_name="m1_1").eval()
    assert trunk.channel_list == [1024]
    out = trunk(torch.randn(1, 3, 1008, 1008))
    assert isinstance(out, list) and len(out) == 1
    assert out[0].shape == (1, 1024, 72, 72)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/modeling/encoders/test_efficientsam3_trunk.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `efficientsam3_trunk.py`**

```python
# SPDX-License-Identifier: LicenseRef-SAM
"""EfficientSAM3 vision trunk: a lightweight backbone + a projection to the PE-trunk-compatible
feature map (1024-ch @ 72x72), exposing the VisionTrunk contract (.channel_list + forward->list)
so it drops into the existing Sam3DualViTDetNeck unchanged. Submodule names mirror the upstream
student encoder so EfficientSAM3 checkpoints load strict (see spec §7)."""
import torch
import torch.nn as nn
from sam.modeling.encoders.repvit import repvit_m0_9, repvit_m1_1, repvit_m2_3

_REPVIT = {"m0_9": repvit_m0_9, "m0.9": repvit_m0_9, "m1_1": repvit_m1_1,
           "m1.1": repvit_m1_1, "m2_3": repvit_m2_3, "m2.3": repvit_m2_3}


class _RepViTTrunk(nn.Module):
    """Runs RepViT.features; exposes channel_list. (Upstream: RepViTTrunkWrapper.)"""
    def __init__(self, model_name: str):
        super().__init__()
        self.model = _REPVIT[model_name](distillation=False)
        dummy = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            for f in self.model.features:
                dummy = f(dummy)
        self.channel_list = [dummy.shape[1]]

    def forward(self, x):
        for f in self.model.features:
            x = f(x)
        return x


class _ImageStudentEncoder(nn.Module):
    """Project backbone features to embed_dim @ embed_size (upstream: ImageStudentEncoder).
    Submodules: .backbone (the trunk wrapper), .head (1x1 conv + BN + GELU + 3x3 conv)."""
    def __init__(self, backbone: nn.Module, in_channels: int, embed_dim=1024, embed_size=72):
        super().__init__()
        self.backbone = backbone
        self.embed_size = embed_size
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        if x.shape[-1] != self.embed_size:
            x = nn.functional.interpolate(x, size=(self.embed_size, self.embed_size),
                                          mode="bilinear", align_corners=False)
        return x


class EfficientSam3Trunk(nn.Module):
    """VisionTrunk: .channel_list=[embed_dim]; forward(x)->[feat]."""
    def __init__(self, backbone_type="repvit", model_name="m1_1",
                 embed_dim=1024, embed_size=72, img_size=1008):
        super().__init__()
        if backbone_type == "repvit":
            bk = _RepViTTrunk(model_name)
        else:
            raise NotImplementedError(f"backbone_type={backbone_type} added in a later phase")
        self.model = _ImageStudentEncoder(bk, bk.channel_list[0], embed_dim, embed_size)
        self.channel_list = [embed_dim]

    def forward(self, x):
        if isinstance(x, list):
            x = x[0]
        return [self.model(x)]
```

> **Note for implementer:** the projection head architecture (`trunk.model.head.*`) must match the checkpoint exactly — validated shapes are `head.0 (1024,512,1,1)`, `head.1` BN(1024), `head.3 (1024,1024,3,3)` (so the head is `Conv1x1 → BN → GELU → Conv3x3`, indices 0/1/2/3). Confirm against the checkpoint in A5; adjust the `nn.Sequential` if upstream `ImageStudentEncoder` differs (check `efficientsam3_reference/sam3/sam3/model/encoder.py`).

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run pytest tests/modeling/encoders/test_efficientsam3_trunk.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sam/modeling/encoders/efficientsam3_trunk.py tests/modeling/encoders/test_efficientsam3_trunk.py
git commit -m "feat(efficientsam3): EfficientSam3Trunk (backbone + projection)"
```

### Task A5: `TextEncoder` Protocol + vendor MobileCLIP + `MobileClipTextEncoder`

**Files:**
- Create: `sam/modeling/text/text_encoder_base.py`
- Create: `sam/modeling/text/mobile_clip.py`
- Create: `sam/modeling/text/mobileclip_text_encoder.py`
- Modify: `sam/modeling/text/__init__.py`
- Test: `tests/modeling/text/test_mobileclip_text_encoder.py`

**Interfaces:**
- Consumes: existing `sam.modeling.text.tokenizer.Sam3Tokenizer`.
- Produces: `TextEncoder` `Protocol` (`forward(text, input_boxes=None, device=None) -> tuple[Tensor, Tensor, Tensor]`, `encode(phrases) -> Tensor`); `MobileClipTextEncoder(tokenizer, variant="MobileCLIP-S0", context_length=16, output_dim=256)` satisfying it, with submodules `.encoder` (MobileCLIP transformer) + `.projector` (`nn.Linear`) matching `language_backbone.{encoder,projector}.*`.

- [ ] **Step 1: Write the failing test**

```python
# tests/modeling/text/test_mobileclip_text_encoder.py
import torch
from sam.modeling.text.tokenizer import Sam3Tokenizer
from sam.modeling.text.mobileclip_text_encoder import MobileClipTextEncoder

def test_mobileclip_text_encoder_contract():
    tok = Sam3Tokenizer()
    enc = MobileClipTextEncoder(tokenizer=tok, variant="MobileCLIP-S0",
                                context_length=16, output_dim=256).eval()
    mask, memory, embeds = enc(["dog", "a red car"], device=torch.device("cpu"))
    # contract matches Sam3TextEncoder: (seq, batch, ...) layout, projected to output_dim
    assert memory.shape[-1] == 256
    assert memory.shape[1] == 2  # batch
    assert mask.dtype == torch.bool

def test_encode_returns_language_features():
    tok = Sam3Tokenizer()
    enc = MobileClipTextEncoder(tokenizer=tok, variant="MobileCLIP-S0",
                                context_length=16, output_dim=256).eval()
    feats = enc.encode(["dog"])
    assert feats.shape[-1] == 256
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/modeling/text/test_mobileclip_text_encoder.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3a: Write `text_encoder_base.py`**

```python
# SPDX-License-Identifier: LicenseRef-SAM
"""Structural contract shared by Sam3TextEncoder and MobileClipTextEncoder (spec §6).
No forced inheritance — both classes satisfy it via duck typing."""
from typing import Optional, Protocol, runtime_checkable
import torch


@runtime_checkable
class TextEncoder(Protocol):
    def forward(self, text, input_boxes: Optional[list] = None,
                device: Optional[torch.device] = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def encode(self, phrases: list[str]) -> torch.Tensor: ...
```

- [ ] **Step 3b: Vendor `mobile_clip.py`** from `efficientsam3_reference/sam3/sam3/backbones/mobile_clip.py`: add SPDX `LicenseRef-AppleML` + provenance header; replace `from timm.models.layers import DropPath, trunc_normal_` with `from sam.modeling.encoders.pe_vitdet import DropPath` and `from torch.nn.init import trunc_normal_`. Keep the `MobileCLIPTextTransformer` class and the per-variant config (S0/S1/2-L dims).

- [ ] **Step 3c: Write `mobileclip_text_encoder.py`** wrapping the vendored transformer + a `nn.Linear` projector, reusing `Sam3Tokenizer`. Port the forward from `efficientsam3_reference/sam3/sam3/model/text_encoder_student.py::TextStudentEncoder` (tokenize → `forward_embedding` → transformer with `input_is_embeddings=True` → project → build padding mask). Output `(text_attention_mask [seq,batch] bool, text_memory [seq,batch,output_dim], inputs_embeds [seq,batch,student_dim])`. `encode(phrases)` returns `text_memory`.

> **Note for implementer:** match the variant dims to the checkpoint — MobileCLIP-S0 width = 512, `transformer` = (count from ckpt: `language_backbone.encoder.transformer.*` = 104 keys), pos-embed table built at `context_length`. The forward and tuple layout must byte-match `Sam3TextEncoder.forward` (verify the existing `sam/modeling/text/text_encoder.py:310`).

- [ ] **Step 3d: Export** from `sam/modeling/text/__init__.py`: add `MobileClipTextEncoder`, `TextEncoder`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run pytest tests/modeling/text/test_mobileclip_text_encoder.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sam/modeling/text/ tests/modeling/text/test_mobileclip_text_encoder.py
git commit -m "feat(efficientsam3): MobileClipTextEncoder + TextEncoder protocol (de-timm)"
```

### Task A6: Config + `build_efficientsam3` (image) with detector-prefix remap

**Files:**
- Create: `sam/configs/efficientsam3/efficientsam3_repvit.yaml`
- Modify: `sam/build_sam.py` (add `build_efficientsam3`, `build_efficientsam3_hf`, `_load_efficientsam3_image_checkpoint`)
- Test: `tests/test_build_efficientsam3.py`

**Interfaces:**
- Consumes: `EfficientSam3Trunk` (A4), `MobileClipTextEncoder` (A5), existing `Sam3DualViTDetNeck`, `Sam3VisionEncoder`, `Sam3DetrDetector`, `Sam3Predictor`, `Sam3Tokenizer`.
- Produces: `build_efficientsam3(config_file="configs/efficientsam3/efficientsam3_repvit.yaml", ckpt_path=None, device="cuda", mode="eval", **kw) -> Sam3Predictor`; `build_efficientsam3_hf(model_id="repvit", **kw)`.

- [ ] **Step 1: Write the failing test** (requires the checkpoint; skip if absent)

```python
# tests/test_build_efficientsam3.py
import os, pytest, torch
from sam.build_sam import build_efficientsam3

CKPT = "checkpoints/_esam3_validate/efficientsam3_ft/efficientsam3_repvit.pt"

@pytest.mark.skipif(not os.path.exists(CKPT), reason="EfficientSAM3 RepViT ckpt absent")
def test_build_efficientsam3_strict_load():
    model = build_efficientsam3(ckpt_path=CKPT, device="cpu")
    from sam.models.sam3_predictor import Sam3Predictor
    assert isinstance(model, Sam3Predictor)
    # vision trunk + text encoder are the swapped ones
    assert model.vision_encoder.vision_backbone.trunk.channel_list == [1024]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_build_efficientsam3.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_efficientsam3'`

- [ ] **Step 3a: Write the config** `sam/configs/efficientsam3/efficientsam3_repvit.yaml` — clone `sam/configs/sam3/sam3.yaml`, then change only the trunk and text targets:

```yaml
# @package _global_
# SPDX-License-Identifier: LicenseRef-SAM
# EfficientSAM3 (RepViT-M1.1 + MobileCLIP-S0 ctx16), base SAM 3 lineage (image).
# Identical to configs/sam3/sam3.yaml EXCEPT vision_backbone.trunk and text_encoder.
# ... (copy the full sam3.yaml body) ...
# vision encoder neck trunk:
#   _target_: sam.modeling.encoders.efficientsam3_trunk.EfficientSam3Trunk
#   backbone_type: repvit
#   model_name: m1_1
# text_encoder:
#   _target_: sam.modeling.text.mobileclip_text_encoder.MobileClipTextEncoder
#   variant: MobileCLIP-S0
#   context_length: 16
#   output_dim: 256
#   tokenizer: { _target_: sam.modeling.text.tokenizer.Sam3Tokenizer }
```

> **Note for implementer:** open `sam/configs/sam3/sam3.yaml`, copy it verbatim, and replace exactly two nodes — the neck's `trunk` block (was `pe_vitdet.ViT`) with the `EfficientSam3Trunk` block above, and the `text_encoder` block (was `Sam3TextEncoder`) with the `MobileClipTextEncoder` block. Keep `scalp`, neck `scale_factors`, `d_model`, detector, everything else identical. The base image config uses a single detection neck (`add_sam2_neck: false`).

- [ ] **Step 3b: Add the builder** to `sam/build_sam.py` mirroring `build_sam3` (around line 840). The only new logic is the checkpoint remap:

```python
def _load_efficientsam3_image_checkpoint(model, ckpt_path):
    """EfficientSAM3 image ckpt is detector-root ({'model': sd}, keys begin
    'backbone.'/'transformer.'/...). Our model wraps the detector under 'detector.',
    so prepend it, then load strict."""
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = ck["model"] if "model" in ck else ck
    remapped = {f"detector.{k}": v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    # strict=True would raise; we assert nothing is missing/unexpected ourselves for a clear message
    assert not missing, f"missing keys: {missing[:5]} ... ({len(missing)})"
    assert not unexpected, f"unexpected keys: {unexpected[:5]} ... ({len(unexpected)})"


def build_efficientsam3(config_file="configs/efficientsam3/efficientsam3_repvit.yaml",
                        ckpt_path=None, device="cuda", mode="eval",
                        hydra_overrides_extra=[], **kwargs):
    # ... identical hydra compose+instantiate as build_sam3 ...
    # then: if ckpt_path is not None: _load_efficientsam3_image_checkpoint(model, ckpt_path)
    ...
```

> **Note for implementer:** copy `build_sam3`'s hydra compose/instantiate body exactly; swap its `_load_sam3_image_checkpoint(...)` call for `_load_efficientsam3_image_checkpoint(...)`. Add `build_efficientsam3_hf(model_id, **kw)` mapping `{"repvit": ("configs/efficientsam3/efficientsam3_repvit.yaml", "efficientsam3_ft/efficientsam3_repvit.pt")}` and `hf_hub_download("Simon7108528/EfficientSAM3", filename, ...)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run pytest tests/test_build_efficientsam3.py -v`
Expected: PASS (strict load, 1107/1107)

- [ ] **Step 5: Commit**

```bash
git add sam/configs/efficientsam3/ sam/build_sam.py tests/test_build_efficientsam3.py
git commit -m "feat(efficientsam3): config + build_efficientsam3 (image, strict load)"
```

### Task A7: Image parity test vs upstream golden

**Files:**
- Create: `tests/parity/reference_efficientsam3/test_efficientsam3_repvit_parity.py`

**Interfaces:**
- Consumes: `build_efficientsam3` (A6), golden fixtures (A1).

- [ ] **Step 1: Write the parity test** (skip if ckpt absent)

```python
import os, json, numpy as np, pytest, torch
from sam.build_sam import build_efficientsam3
from sam.models.sam3_predictor import Sam3Predictor  # for ConceptPrompt path
from sam.prompts import ConceptPrompt

CKPT = "checkpoints/_esam3_validate/efficientsam3_ft/efficientsam3_repvit.pt"
GOLD = "tests/parity/reference_efficientsam3/golden"
IMG = "../efficientsam3_reference/sam3/assets/dog_person.jpeg"

def _iou(a, b):
    a, b = a > 0.5, b > 0.5
    return (a & b).sum() / max((a | b).sum(), 1)

@pytest.mark.skipif(not (os.path.exists(CKPT) and os.path.exists(IMG)), reason="ckpt/img absent")
@pytest.mark.parametrize("prompt", ["dog", "person"])
def test_parity(prompt):
    summ = json.load(open(f"{GOLD}/efficientsam3_repvit_summary.json"))
    g = summ["prompts"][prompt]
    model = build_efficientsam3(ckpt_path=CKPT, device="cuda")
    from PIL import Image
    res = model.predict(Image.open(IMG).convert("RGB"), ConceptPrompt(text=prompt),
                        confidence_threshold=summ["threshold"])
    assert res.masks_logits.shape[0] == g["num_instances"]
    gold = np.load(f"{GOLD}/efficientsam3_repvit_masks_{prompt}.npz")["masks"]
    ours = (res.masks_logits.sigmoid() if res.masks_logits.is_floating_point() else res.masks_logits).cpu().numpy()
    ious = sorted(_iou(ours[i, 0], gold[j, 0]) for i in range(len(ours)) for j in [i])
    assert min(ious) >= 0.99
```

> **Note for implementer:** the exact `predict()` return type/fields come from `sam/models/sam3_predictor.py::Sam3Predictor.predict` (returns `Sam3DetectionResult` with `masks_logits`, `boxes`, `scores`). Align field names and the instance ordering with how the golden was captured (`_run_golden.py`). If ordering differs, match instances by box IoU before comparing masks.

- [ ] **Step 2: Run** `pixi run pytest tests/parity/reference_efficientsam3/test_efficientsam3_repvit_parity.py -v`
Expected: PASS (IoU ≥ 0.99 vs upstream) — **this is the Phase A acceptance gate.**

- [ ] **Step 3: Commit**

```bash
git add tests/parity/reference_efficientsam3/test_efficientsam3_repvit_parity.py
git commit -m "test(efficientsam3): RepViT image parity vs upstream golden"
```

### Task A8: Public download tool + pixi task

**Files:**
- Create: `tools/download_efficientsam3.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write `download_efficientsam3.py`** — mirror `tools/download_sam3.py` but **no login** (public repo). CLI `--variant repvit --out-dir checkpoints`. Map `repvit → efficientsam3_ft/efficientsam3_repvit.pt`. Set `os.environ["HF_HUB_DISABLE_XET"]="1"`; `hf_hub_download("Simon7108528/EfficientSAM3", filename)`; copy into `checkpoints/`.

- [ ] **Step 2: Add pixi task** in `pyproject.toml`:

```toml
[tool.pixi.tasks.download-efficientsam3-repvit]
cmd = "python tools/download_efficientsam3.py --variant repvit --out-dir checkpoints"
outputs = ["checkpoints/efficientsam3_repvit.pt"]
```

- [ ] **Step 3: Smoke-run** `pixi run download-efficientsam3-repvit` (or assert the file already exists). Expected: checkpoint present.

- [ ] **Step 4: Commit**

```bash
git add tools/download_efficientsam3.py pyproject.toml
git commit -m "feat(efficientsam3): public HF download tool + pixi task"
```

### Task A9: Build smoke (meta-build, no checkpoint)

**Files:**
- Create: `tests/test_efficientsam3_build_smoke.py`

- [ ] **Step 1: Write smoke test** — instantiate the config via hydra **without** a checkpoint (random weights) and run a forward on a dummy `torch.zeros(1,3,1008,1008)` through `vision_encoder`, asserting the pyramid shapes (detector level 72×72×256). This guards config/dim correctness with no weights (CI-safe).

```python
import torch
from sam.build_sam import build_efficientsam3

def test_meta_build_and_forward():
    model = build_efficientsam3(ckpt_path=None, device="cpu")
    feats, pos = model.vision_encoder(torch.zeros(1, 3, 1008, 1008))
    assert feats[-1].shape[-2:] == (72, 72)
    assert feats[-1].shape[1] == 256
```

- [ ] **Step 2: Run** `pixi run pytest tests/test_efficientsam3_build_smoke.py -v`. Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_efficientsam3_build_smoke.py
git commit -m "test(efficientsam3): build smoke (meta-build + vision forward)"
```

### Task A10: Benchmark tool + FPS gate

**Files:**
- Create: `tools/benchmark_efficientsam3.py`

**Interfaces:**
- Consumes: `build_efficientsam3`.

- [ ] **Step 1: Port `efficientsam3_reference/_bench.py`** to use `build_efficientsam3` + `Sam3Predictor.predict`. Keep the hygiene (warmup, `cuda.synchronize()`, median/std, vision/prompt/e2e breakdown, fp32+autocast). CLI `--ckpt --image --prompt --iters`.

- [ ] **Step 2: Run** `pixi run python tools/benchmark_efficientsam3.py --ckpt checkpoints/efficientsam3_repvit.pt`. Expected: vision/prompt/e2e medians within ~15% of the spec §10 reference (fp32: vision ~31.5 ms, e2e ~136.6 ms on RTX 3080 Ti). Record the run.

- [ ] **Step 3: Commit**

```bash
git add tools/benchmark_efficientsam3.py
git commit -m "feat(efficientsam3): FPS benchmark tool"
```

### Task A11: Docs + licensing

**Files:**
- Modify: `README.md`, `NOTICE`
- Create: `LICENSE_repvit`, `LICENSE_mobileclip`

- [ ] **Step 1: Add README "EfficientSAM3" section** — build/download/predict example mirroring the SAM 3 section; the 2×2 variant matrix (mark which phases shipped); the FPS reference table.
- [ ] **Step 2: Add README license-table rows** — RepViT, MobileCLIP (Apple ML, research terms) and a note that SAM 3-derived glue is `LicenseRef-SAM`.
- [ ] **Step 3: Add `LICENSE_repvit`, `LICENSE_mobileclip`** texts (copy from upstream backbone repos) + reference them in `NOTICE`.
- [ ] **Step 4: Verify SPDX headers** present on every file created in A2–A6 (`grep -L SPDX sam/modeling/encoders/repvit.py ...`).
- [ ] **Step 5: Commit**

```bash
git add README.md NOTICE LICENSE_repvit LICENSE_mobileclip
git commit -m "docs(efficientsam3): README section + per-component licensing"
```

---

## Phase B — EfficientSAM3 TinyViT image

### Task B1: Vendor TinyViT (de-timm'd) + extend trunk

**Files:** Create `sam/modeling/encoders/tiny_vit.py`; Modify `sam/modeling/encoders/efficientsam3_trunk.py`; Test `tests/modeling/encoders/test_tiny_vit.py`.

- [ ] **Step 1: Failing test** — `tiny_vit_11m_224(img_size=1008)` builds; a `_TinyViTTrunk` forward returns `(B,C,H,W)` (reshape from `(B,L,C)`). Assert final channel and that `EfficientSam3Trunk(backbone_type="tinyvit", model_name="11m")` yields `channel_list==[1024]` and output `(1,1024,72,72)`.
- [ ] **Step 2: Run → fails.**
- [ ] **Step 3: Vendor `tiny_vit.py`** from upstream; de-timm: replace `from timm.layers import DropPath as TimmDropPath, to_2tuple, trunc_normal_` with the local `DropPath` (alias it), `to_2tuple` from `_layers`, `trunc_normal_` from `torch.nn.init`; remove `register_model`; replace the `build_model_with_cfg` factory bodies with direct `TinyViT(**cfg)` construction (drop pretrained/cfg path). Add `_TinyViTTrunk` (patch_embed → layers → reshape `(B,L,C)→(B,C,H,W)`) to `efficientsam3_trunk.py` and a `backbone_type=="tinyvit"` branch.
- [ ] **Step 4: Run → passes.**
- [ ] **Step 5: Commit** `feat(efficientsam3): vendor TinyViT + trunk branch (de-timm)`.

### Task B2: Config + download + parity for TinyViT

**Files:** Create `sam/configs/efficientsam3/efficientsam3_tinyvit.yaml`; Modify `tools/download_efficientsam3.py`, `pyproject.toml`, `build_sam.py` (extend `build_efficientsam3_hf` map); Create `tests/parity/.../test_efficientsam3_tinyvit_parity.py` + golden.

- [ ] **Step 1:** Capture the TinyViT golden via the upstream repo (`_run_golden.py --checkpoint efficientsam3_tinyvit.pt`, backbone_type tinyvit, model_name 11m); pin `model_name` from the trunk channel signature; commit fixtures.
- [ ] **Step 2: Failing parity test** (mirror A7 with the tinyvit ckpt/golden).
- [ ] **Step 3:** Add `efficientsam3_tinyvit.yaml` (copy A6's, set `backbone_type: tinyvit, model_name: 11m`); add `tinyvit` to the download map + pixi task.
- [ ] **Step 4: Run → strict load + parity pass.**
- [ ] **Step 5: Commit** `feat(efficientsam3): TinyViT variant (config + download + parity)`.

---

## Phase C — EfficientSAM3 EfficientViT image

### Task C1: Vendor EfficientViT package (+ Triton fallback) + trunk branch

**Files:** Create `sam/modeling/encoders/efficientvit/` (package: `backbone.py`, `nn/`, `utils/`); Modify `efficientsam3_trunk.py`; Test `tests/modeling/encoders/test_efficientvit.py`.

- [ ] **Step 1: Failing test** — `efficientvit_backbone_b1()` builds; `EfficientSam3Trunk(backbone_type="efficientvit", model_name="b1")` → `channel_list==[1024]`, output `(1,1024,72,72)`; **and** a test that forces the pure-torch RMSNorm path (monkeypatch the triton import to raise) still produces identical output.
- [ ] **Step 2: Run → fails.**
- [ ] **Step 3: Vendor `efficientvit/`** from upstream (no timm). In `nn/triton_rms_norm.py`, wrap the triton import: `try: import triton ... except Exception: triton = None`, and in the RMSNorm module use a pure-torch `x * rsqrt(mean(x^2)+eps) * weight` path when `triton is None` or input is on CPU. Add `EfficientViTTrunkWrapper` (returns `out['stage_final']`) + `backbone_type=="efficientvit"` branch.
- [ ] **Step 4: Run → passes (both triton-present and fallback paths).**
- [ ] **Step 5: Commit** `feat(efficientsam3): vendor EfficientViT (+ pure-torch RMSNorm fallback)`.

### Task C2: Config + download + parity for EfficientViT

**Files:** as B2, for efficientvit. Pin `model_name` (`b1/b2/b3`) from the channel signature; capture golden; strict load + parity.

- [ ] **Steps 1-5:** mirror B2 with `efficientsam3_efficientvit.yaml` (`backbone_type: efficientvit, model_name: <pinned>`), download `efficientsam3_ft/efficientsam3_efficientvit.pt`, parity test. Commit `feat(efficientsam3): EfficientViT variant`.

---

## Phase D — SAM3-LiteText image (base lineage, PE vision)

### Task D1: LiteText configs (PE vision + MobileCLIP) + builder + parity

**Files:** Create `sam/configs/efficientsam3/sam3_litetext_{s0_ctx16,s0_ctx32,s1_ctx16,s1_ctx32,l_ctx16,l_ctx32}.yaml`; Modify `build_sam.py`, `tools/download_efficientsam3.py`, `pyproject.toml`; Create parity tests + goldens.

**Interfaces:** Consumes existing `Sam3VisionEncoder` (PE, unchanged) + `MobileClipTextEncoder` (A5).

- [ ] **Step 1:** For each (S0/S1/2-L × ctx16/32) capture the upstream golden (`build_sam3_image_model` LiteText path) and commit fixtures.
- [ ] **Step 2: Failing parity test** per variant.
- [ ] **Step 3:** Write each yaml = copy `configs/sam3/sam3.yaml` (keep PE `pe_vitdet.ViT` trunk **unchanged**) and replace only `text_encoder` with `MobileClipTextEncoder(variant=<S0/S1/2-L>, context_length=<16/32>, output_dim=256)`. Add a `build_efficientsam3` overload/param that accepts these configs (no new builder needed — reuses A6 with detector-root remap). Add download entries (`sam3_litetext/...`).
- [ ] **Step 4: Run → strict load + parity per variant.**
- [ ] **Step 5: Commit** `feat(efficientsam3): SAM3-LiteText variants (PE vision + MobileCLIP)`.

---

## Phase E — SAM3.1-LiteText streaming video (multiplex; introduces video)

### Task E1: Multiplex video builder (full-model-root remap) + LiteText video config

**Files:** Modify `sam/build_sam.py` (add `build_efficientsam3_litetext_video_predictor` + `_load_efficientsam3_video_checkpoint`); Create `sam/configs/efficientsam3/sam3p1_litetext_{s0,s1,l}_ctx{16,32}.yaml`; Modify download tool + pixi.

**Interfaces:** Consumes existing `Sam3MultiplexVideoPredictor`, tri-neck `Sam3DualViTDetNeck` (`add_interactive_neck=True`), 457-key multiplex tracker, `MobileClipTextEncoder`, existing PE vision.

- [ ] **Step 1: Failing strict-load test** — build the multiplex video predictor and load `sam3p1_litetext/efficient_sam3p1_litetext_mobileclip_s0_ctx16.pt`; assert `tracker.model` loads 457/457 and detector loads fully.

```python
def _load_efficientsam3_video_checkpoint(model, ckpt_path):
    """SAM 3.1 ckpt is full-model-root: keys already 'detector.*' and 'tracker.model.*'.
    Load directly (no prefix surgery)."""
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = ck["model"] if "model" in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
```

- [ ] **Step 2: Run → fails** (builder missing).
- [ ] **Step 3:** Write `build_efficientsam3_litetext_video_predictor` mirroring `build_sam3_multiplex_video_predictor` (tri-neck vision encoder, multiplex tracker) but with `text_encoder` = `MobileClipTextEncoder`. Write the configs = copy `configs/sam3/sam3.1.yaml`, replace only `text_encoder`. Add download entries.
- [ ] **Step 4: Run → strict load passes (incl. tracker 457/457).**
- [ ] **Step 5: Commit** `feat(efficientsam3): SAM3.1-LiteText multiplex video builder + configs`.

### Task E2: Streaming parity + VRAM-flat + video FPS reference

**Files:** Create `tests/parity/.../test_sam3p1_litetext_video_parity.py`; Modify `tools/benchmark_efficientsam3.py` (video mode); capture upstream video golden + video FPS reference.

- [ ] **Step 1:** Capture the upstream **video** golden (run upstream `build_sam3_video_model` LiteText on the `sam3/assets/videos/0001` frames + a concept) — per-frame masklets + a video FPS reference (text cached once + per-frame vision+detect+track). Commit fixtures + record the FPS reference into the spec §10 (video row).
- [ ] **Step 2: Failing streaming parity test** — stream the same frames through `Sam3MultiplexVideoPredictor` (our build); assert per-frame masklet IoU ≥ 0.99, instance ids stable.
- [ ] **Step 3:** Implement/verify the streaming loop wiring (encoders injected; tracker reused). Add a VRAM-flat assertion (peak allocated bytes after frame N ≈ after frame 2N, within slack) with the forgetful bank.
- [ ] **Step 4: Run → parity + VRAM-flat pass; record video FPS.**
- [ ] **Step 5: Commit** `test(efficientsam3): SAM3.1-LiteText streaming parity + VRAM + FPS`.

---

## Phase F — EfficientSAM3.1 streaming video (multiplex, distilled trunk; stage1)

### Task F1: EfficientSAM3.1 configs + builder + parity

**Files:** Create `sam/configs/efficientsam3/efficientsam3p1_{repvit,tinyvit,efficientvit}_{s,m,l}_mobileclip_s0_ctx16.yaml`; Modify `build_sam.py` (add `build_efficientsam3p1_video_predictor`), download tool, pixi; Create parity tests + goldens.

**Interfaces:** Consumes `EfficientSam3Trunk` (A4/B1/C1), `Sam3MultiplexVideoPredictor`, multiplex tracker, `MobileClipTextEncoder`.

- [ ] **Step 1:** Capture the upstream golden for one variant (`efficient_sam3p1_repvit_m_mobileclip_s0_ctx16`) — note stage1, so parity is vs upstream stage1 output. Commit fixtures.
- [ ] **Step 2: Failing strict-load + streaming parity test** (tri-neck distilled trunk + tracker 457/457).
- [ ] **Step 3:** Configs = copy E1's `sam3p1_litetext_*.yaml` and replace the PE `trunk` with `EfficientSam3Trunk(backbone_type, model_name)` (the only diff vs LiteText video). `build_efficientsam3p1_video_predictor` = E1's builder with the distilled trunk. Pin each `model_name` from the trunk channel signature. Add download entries (`stage1_sam3p1/...`).
- [ ] **Step 4: Run → strict load + streaming parity (vs stage1) pass.**
- [ ] **Step 5: Commit** `feat(efficientsam3): EfficientSAM3.1 multiplex video variants (stage1)`.

### Task F2: Finalize docs + variant matrix

- [ ] **Step 1:** Update the README EfficientSAM3 section to mark all four families shipped; complete the variant/checkpoint table + license rows for every vendored backbone; note EfficientSAM3.1 stage1 maturity.
- [ ] **Step 2:** Run the **full** test suite: `pixi run pytest tests/ -v` (skips weight-dependent tests when ckpts absent). Expected: all pass/skip, none fail.
- [ ] **Step 3: Commit** `docs(efficientsam3): finalize all-variant docs + license table`.

---

## Self-review notes (coverage vs spec)

- Spec §2 in-scope (4 families): A–C (EfficientSAM3 ×3 backbones), D (SAM3-LiteText), E (SAM3.1-LiteText video), F (EfficientSAM3.1 video). ✓
- E1 trunk swap / E2 text swap / E3 tokenizer reuse / E11 multiplex reuse / E12 two remaps: Tasks A4, A5, E1, F1. ✓
- E7 ctx-at-build: A5 (`context_length` param) + config nodes. ✓
- E9 no-deps / de-timm: A2, A3, B1, C1. ✓
- E8 licensing: A11, F2. ✓
- FPS gate (§10): A10, E2. ✓ · Parity gate: A7, B2, C2, D1, E2, F1. ✓ · VRAM-flat: E2. ✓
- Validation already retired the key-remap risk (1107/1107 base; 457/457 tracker sam3.1).
