# EfficientSAM3 Integration — Design Spec

- **Date:** 2026-06-29
- **Status:** Draft (awaiting user review)
- **Builds on:** [`2026-06-26-sam3-integration-design.md`](2026-06-26-sam3-integration-design.md) §16
  (EfficientSAM3 readiness). That spec predicted this port "validates the design"; this spec
  makes it concrete and records empirical validation against the upstream repo.
- **Scope:** Add EfficientSAM3 (distilled SAM 3, ~848M → ~90M) to the `sam/` package by swapping
  **only the two encoders** (vision trunk + text tower), reusing SAM 3's detector, tracker, memory,
  multiplex, mask decoder, predictors, association and streaming **unchanged**. Same integration
  *treatment* as SAM 3.1: PyTorch inference (image + streaming video), configs, builders, public
  checkpoint download, reference parity, build smoke, docs, per-component licensing. **No ONNX, no
  training/distillation** (load pretrained distilled checkpoints only).

---

## 1. Goal

EfficientSAM3 (https://github.com/SimonZeng7108/efficientsam3) is the "EfficientTAM-of-SAM 3": a
multi-stage **distillation** of SAM 3 that keeps the detector, tracker and memory and replaces the two
heavy encoders with lightweight ones — vision (PE ViT-H → EfficientViT / RepViT / TinyViT) and text
(SAM 3 CLIP tower → MobileCLIP). Integrate **inference** so that:

1. EfficientSAM3 variants run through the existing `Sam3Predictor` / `Sam3VideoPredictor` with the
   version visible in class names (`EfficientSam3*`), exactly as EfficientTAM coexists with SAM 2.
2. The swap is **config-only** at the predictor level — the predictor already injects the vision and
   text encoders as `nn.Module`s (composition), so no predictor edits are needed.
3. Readability and reuse first: the only net-new modules are the lightweight backbones (vendored),
   one vision-trunk adapter, and one text-encoder adapter.

## 2. Scope

**In scope (this plan) — all four published families** (the 2×2 matrix of §4a):
- Vendor the lightweight backbones: `repvit`, `tiny_vit`, `efficientvit` (package), `mobile_clip`.
- `EfficientSam3Trunk` — adapts a lightweight backbone + projection into the existing
  `Sam3DualViTDetNeck` (trunk contract: `.channel_list` + `forward(x) -> List[Tensor]`).
- `MobileClipTextEncoder` — wraps MobileCLIP, reusing the existing `Sam3Tokenizer`, with the same
  `(mask, memory, embeds)` output contract as `Sam3TextEncoder`.
- **EfficientSAM3** (base, image) · **SAM3-LiteText** (base, image) · **SAM3.1-LiteText** (multiplex,
  video) · **EfficientSAM3.1** (multiplex, video) — each = a config assembly over
  {distilled trunk | PE trunk} × {base `Sam3Predictor` | multiplex `Sam3MultiplexVideoPredictor`} ×
  shared `MobileClipTextEncoder`. All map onto **predictors that already exist in the repo** (§4, §4a).
- `configs/efficientsam3/*.yaml`, `build_efficientsam3_*` builders, `tools/download_efficientsam3.py`
  (public HF, **not gated**), pixi tasks.
- Image predictor (base lineage) + streaming video predictor (multiplex lineage); reference parity;
  build smoke; docs; D11 license extension (SPDX + README table + license texts; weights never vendored).

**Deferred (follow-up specs):**
- ONNX / TRT export of the EfficientSAM3 encoders (SAM 3 ONNX itself is still deferred).
- Training / distillation (stage1/stage3) — we consume pretrained distilled checkpoints only.
- A base-lineage *distilled video* model — **not published** upstream (base EfficientSAM3 ships
  image-only; the distilled video model is the multiplex EfficientSAM3.1). No checkpoint → out of scope.

## 3. Decision log

| # | Decision | Rationale |
|---|----------|-----------|
| E1 | **Swap encoders only; reuse all SAM 3 internals.** New code = vendored backbones + one trunk adapter + one text adapter | Validated: every EfficientSAM3 weight maps onto SAM 3's detector with the two encoders swapped (§4) |
| E2 | **Reuse `Sam3DualViTDetNeck` unchanged** — `EfficientSam3Trunk` plugs into the existing neck as its `trunk` | The neck is trunk-agnostic (`dim = trunk.channel_list[-1]`, `xs = trunk(x); x = xs[-1]`); the upstream student encoder projects to **1024-ch @ 72×72**, byte-identical to the PE trunk output the neck expects |
| E3 | **Reuse `Sam3Tokenizer` unchanged** | EfficientSAM3's student text encoder uses the same CLIP-BPE `bpe_simple_vocab_16e6.txt.gz` (vocab 49408) we already vendor |
| E4 | **Base SAM 3 lineage, image-only** for the headline EV-M/RV-M/TV-M checkpoints (→ `Sam3Predictor`, **not** multiplex; no base video model published) | Validated: the RV-M checkpoint has no `multiplex`/`mux`/`bucket`/`tracker` keys; multiplex + all video models are separate `*_sam3p1_*` files (§4a) |
| E5 | **Vertical slice first (RepViT), spec covers all variants** | Lowest risk: prove the full image+video+parity pipeline on one pure-torch backbone before fanning out to EfficientViT (Triton) / TinyViT / LiteText |
| E6 | **Public, ungated checkpoint download** via HF `Simon7108528/EfficientSAM3` | Differs from SAM 3 (gated). No login/token; simpler than `download_sam3.py` |
| E7 | **Builder builds the text encoder at the checkpoint's context length directly** (16/32), not "build-at-77-then-truncate" | The upstream `build_efficientsam3_image_model` builds text at ctx 77 then truncates *after* load — but the strict `load_state_dict` raises on the pos-embed size mismatch first, so the truncation is dead code (upstream bug for ctx16 checkpoints). We build at the right ctx up front |
| E8 | **Extend D11 per-component licensing** | MobileCLIP (Apple ML), EfficientViT / RepViT / TinyViT each carry their own terms; per-file SPDX + README rows + shipped license texts; weights never vendored |
| E9 | **No new third-party deps. Avoid `timm`** (and `einops`) by vendoring two tiny utilities; EfficientViT Triton RMSNorm kernel behind a **pure-torch fallback** | timm is used only for small layer helpers — vendor `SqueezeExcite` + `to_2tuple`, reuse the **existing local `DropPath`** (`pe_vitdet.py:34`) and `torch.nn.init.trunc_normal_`, drop `@register_model` / `build_model_with_cfg` (call factories directly, no pretrained-cfg path). `einops` is not used by any vendored file. `iopath`, `ftfy`, `regex`, `torchvision` (CUDA `roi_align`) already present. This matches the repo's existing no-timm discipline (PE ViT already vendors `DropPath`) |
| E10 | **Cover all four families** (EfficientSAM3, SAM3-LiteText, SAM3.1-LiteText, EfficientSAM3.1) in this spec, sequenced base-first | They are a clean 2×2 (trunk × lineage) over a shared text encoder; the repo already has both lineages' predictors, so each family is config assembly, not new architecture (§4a) |
| E11 | **SAM 3.1 families reuse the existing multiplex video stack unchanged** — `Sam3MultiplexVideoPredictor` + tri-neck (`Sam3DualViTDetNeck` `add_interactive_neck` / `forward_all`) + the 457-key multiplex tracker | Validated: both `efficient_sam3p1_repvit_m` and `efficient_sam3p1_litetext` carry `tracker.model.*` = **457 keys** (byte-matching `build_sam3_multiplex_video_predictor`) and a `convs`/`interactive_convs`/`propagation_convs` tri-neck; only `trunk` differs (RepViT vs PE ViT) |
| E12 | **Two checkpoint roots → two remaps.** Base EfficientSAM3 image ckpt is **detector-root** (prepend `detector.`); SAM 3.1 ckpts are **full-model-root** (`detector.*` + `tracker.model.*`, load directly via the multiplex video builder) | Validated by key inspection (§4); both load `strict=True` after their respective trivial remap |

### Naming map
```
EfficientSam3Trunk         [new]   modeling/encoders/efficientsam3_trunk.py  (backbone + projection; trunk contract)
RepViT / TinyViT / EfficientViT  [vendored]  modeling/encoders/{repvit,tiny_vit,efficientvit/}
MobileClipTextEncoder      [new]   modeling/text/mobileclip_text_encoder.py
MobileCLIPTextTransformer  [vendored]  modeling/text/mobile_clip.py
build_efficientsam3_*      [new]   build_sam.py  (mirror build_sam3_*)
configs/efficientsam3/*    [new]   one yaml per variant (start: efficientsam3_repvit.yaml)
```
Reused unchanged: `Sam3VisionEncoder`, `Sam3DualViTDetNeck`, `Sam3Tokenizer`, `Sam3DetrDetector`,
`Sam3Tracker`, multiplex, memory bank, mask decoder, `Sam3Predictor` / `Sam3VideoPredictor`,
association, tracklet.

## 4. Validation evidence (performed before finalizing this spec)

Ran the **upstream** EfficientSAM3 RepViT model end-to-end and inspected its checkpoint, to confirm the
reuse story before committing to it. Upstream cloned to a throwaway sibling (`efficientsam3_reference/`,
uv venv, Python 3.12); checkpoint `efficientsam3_ft/efficientsam3_repvit.pt` downloaded from the public
HF repo.

**Checkpoint structure (`efficientsam3_repvit.pt`, RV-M):**
- Wrapper `{'model': state_dict, 'optimizer', 'epoch', 'preserved_base_*', ...}` (a training
  checkpoint). 1107 tensors under `model`.
- Subtrees: `backbone.vision_backbone.{trunk,convs}.*` (675; RepViT + the SimpleFPN neck),
  `backbone.language_backbone.{encoder,projector}.*` (111; MobileCLIP), and the **base SAM 3 detector
  body** (`transformer.encoder/decoder`, `segmentation_head`, `dot_prod_scoring`) — identical to SAM 3.
- **No `multiplex` / `mux` / `bucket` / `tracker` keys** → base SAM 3 lineage, image (detector) model.
- The neck weights are byte-structurally identical to our `Sam3DualViTDetNeck` (`convs.0.dconv_2x2_0`,
  `dconv_2x2_1`, `conv_1x1`, `conv_3x3`), with `dconv_2x2_0.weight (1024,512,2,2)` confirming the trunk
  emits **1024 channels** like the PE trunk.
- The vision backbone is **RepViT-M1.1** (`model_name="m1_1"`): channel stages `64×3, 128×4, 256×14,
  512×3`, final 512, projected to 1024 by `trunk.model.head` — an exact match to the `repvit_m1_1` cfg.
- The text encoder is **MobileCLIP-S0, context length 16** (`positional_embedding (1,1,16,512)`,
  embedding `(49408,512)` = CLIP-BPE).

**End-to-end run** (`Sam3Processor.set_image` + `set_text_prompt`, fp32, RTX 3080 Ti):
- **Load integrity: 1107 / 1107 keys matched by shape**; probes for trunk feature, neck conv, text
  embedding and text projector all loaded (allclose `True`).
- `"dog"` → 13 instances (top scores 0.69 / 0.44 + a low-confidence tail at threshold 0.1); largest
  mask 2.37 Mpx. `"person"` → 19 instances (top 0.81 / 0.81 / 0.36); largest masks ~2.3 / 1.2 / 1.1 Mpx.
- Output state exposes `masks_logits, masks, boxes, scores`. These signatures (sha1 + sums + scores)
  are the **golden reference** for the parity gate.

**Conclusion:** the design holds with no surprises — EfficientSAM3 is SAM 3 with the two encoders
swapped, and the checkpoint maps cleanly onto the reused detector via a `detector.`-prefix remap.

**Incidental findings folded into decisions:** upstream ctx16 builder bug (→ E7); upstream
`model_builder.py` eagerly imports the training/video stack (timm, einops, opencv/torchcodec,
pycocotools, psutil, …), which is why the throwaway upstream venv needed them — but our clean port
imports only the inference path. The files we actually vendor (`repvit`, `tiny_vit`, `efficientvit`,
`mobile_clip`, `text_encoder_student`) use **no `einops`** and only small `timm` layer helpers, so the
integration adds **no new third-party deps** (→ E9); the geometry/box-exemplar path uses
`torchvision.ops.roi_align` (CUDA build required — already satisfied here).

**SAM 3.1 lineage validation (key inspection).** Downloaded and inspected
`stage1_sam3p1/efficient_sam3p1_repvit_m_mobileclip_s0_ctx16.pt` (EfficientSAM3.1) and
`sam3p1_litetext/efficient_sam3p1_litetext_mobileclip_s0_ctx16.pt` (SAM3.1-LiteText). Both are
**full multiplex video models** (full-model-root: `detector.*` + `tracker.model.*`):
- `detector.backbone.vision_backbone.trunk.*` — EfficientSAM3.1 = RepViT (`trunk.model.…features.0.0.c.weight
  (32,3,3,3)`, identical to base RV-M); SAM3.1-LiteText = PE ViT-H (`trunk.{blocks,patch_embed,
  pos_embed (1,577,1024),ln_pre}`). **Only the trunk differs.**
- `detector.backbone.vision_backbone.{convs,interactive_convs,propagation_convs}` — the SAM 3.1
  **tri-neck** (detection + interactive + propagation), already supported by `Sam3DualViTDetNeck`
  (`add_interactive_neck` / `forward_all`).
- `detector.backbone.language_backbone.{encoder,projector}` — MobileCLIP, identical across both.
- `tracker.model.*` = **457 keys** (`maskmem_tpos_enc`, …) — **byte-matches our existing
  `build_sam3_multiplex_video_predictor` tracker**, reused unchanged.

So both SAM 3.1 families are a **pure trunk swap** into the existing tri-neck + multiplex tracker +
`Sam3MultiplexVideoPredictor`. EfficientSAM3.1 ships **stage1-only** (no `_ft`); SAM3.1-LiteText ships
full.

## 4a. Variant matrix & checkpoint inventory

Every variant = **(vision trunk) × (lineage)**, all with a MobileCLIP text encoder. All four cells map
onto predictors that already exist in the repo — the integration adds the distilled `EfficientSam3Trunk`
and the shared `MobileClipTextEncoder`, nothing else architectural.

| vision ↓ / lineage → | **base SAM 3** (`Sam3Predictor` / `Sam3VideoPredictor`) | **SAM 3.1 multiplex** (`Sam3MultiplexPredictor` / `Sam3MultiplexVideoPredictor`) |
|---|---|---|
| **distilled** trunk (`EfficientSam3Trunk`) | **EfficientSAM3** — `efficientsam3_ft/efficientsam3_{efficientvit,repvit,tinyvit}.pt` · image (detector-root) · MobileCLIP-S0 ctx16 · **✅ validated (RV-M, 1107/1107)** | **EfficientSAM3.1** — `stage1_sam3p1/efficient_sam3p1_{efficientvit,repvit,tinyvit}_{s,m,l}_mobileclip_s0_ctx16.pt` · video · **stage1-only** · **✅ key-validated (repvit_m)** |
| **PE ViT-H** trunk (existing `Sam3VisionEncoder`) | **SAM3-LiteText** — `sam3_litetext/sam3_litetext_mobileclip_{s0,s1,2_l}_ctx{16,32}.pt` · MobileCLIP-{S0,S1,2-L} | **SAM3.1-LiteText** — `sam3p1_litetext/efficient_sam3p1_litetext_mobileclip_{s0,s1,l}_ctx{16,32}.pt` · video · full · **✅ key-validated (s0)** |

**Backbone `model_name` mapping** (distilled cells): EfficientViT `{b1,b2,b3}` ≈ `{s,m,l}`, RepViT
`{m0_9,m1_1,m2_3}` ≈ `{s,m,l}`, TinyViT `{5m,11m,21m}` ≈ `{s,m,l}`; RV-M validated as `m1_1`. Exact
per-checkpoint mapping is pinned from the trunk channel signature at vendor time (the method used for RV-M).

**Per-variant work** once the slice (Phase A) lands: a new config yaml + a download entry + a parity/build
test. No new modules (the distilled trunk handles all three backbones; LiteText reuses the PE trunk; all
reuse `MobileClipTextEncoder` and the existing predictors).

## 5. Architecture — new vs reused

```
ConceptPrompt(text) ─► MobileClipTextEncoder ──► (mask, memory, embeds) ─┐
                         (reuses Sam3Tokenizer)                          │
image ─► EfficientSam3Trunk ─► Sam3DualViTDetNeck ─► Sam3VisionEncoder ──┼─► Sam3DetrDetector ─► masks
         (vendored backbone     (REUSED, unchanged)   (REUSED)           │   (REUSED, unchanged)
          + 1024@72×72 proj)                                             │
                                                                         └─► (video) Sam3Tracker + memory bank (REUSED)
```

| Component | Action | Notes |
|---|---|---|
| `EfficientSam3Trunk` | **new** | backbone + `ImageStudentEncoder`-style projection to 1024-ch @ 72×72; exposes `.channel_list=[1024]` and `forward(x) -> [feat]` |
| `repvit.py`, `tiny_vit.py`, `efficientvit/` | **vendor** | distilled trunks; **de-timm'd** (vendor `SqueezeExcite`/`to_2tuple`, reuse local `DropPath` + torch `trunc_normal_`); EfficientViT carries an optional Triton RMSNorm kernel (pure-torch fallback) |
| `MobileClipTextEncoder` + `mobile_clip.py` | **new + vendor** | `.encoder` (MobileCLIP transformer) + `.projector` (Linear → d_model); built at the checkpoint's ctx (E7) |
| `Sam3DualViTDetNeck`, `Sam3VisionEncoder` | **reuse** | trunk-agnostic; the trunk is injected (E2) |
| `Sam3Tokenizer` | **reuse** | same CLIP-BPE asset (E3) |
| detector, tracker (+multiplex), memory bank, mask decoder, predictors, association, tracklet | **reuse** | unchanged; predictor injects encoders, so swap is config-only |

## 6. Seams (per the SAM 3 spec §16)

- **Vision trunk** — already a structural contract (`.channel_list` + `forward -> List[Tensor]`, the
  last element fed to the SimpleFPN). Document it as `VisionTrunk`; `EfficientSam3Trunk` is the second
  implementer alongside the PE `ViT`. No new ABC.
- **Text encoder** — now two real implementers (`Sam3TextEncoder`, `MobileClipTextEncoder`) → introduce
  a light **`TextEncoder` Protocol** (structural; no forced inheritance on vendored code) documenting
  the `(text, input_boxes, device) -> (text_attention_mask, text_memory, inputs_embeds)` forward plus
  `encode(phrases) -> Tensor`. Both already satisfy it; the predictor keeps taking `nn.Module`.

## 7. Checkpoint mapping / remap

The upstream checkpoint root **is** the detector (keys begin `backbone.…`, `transformer.…`,
`segmentation_head.…`, `dot_prod_scoring.…`); our build wraps the detector under `detector.`. The
loader therefore:
1. unwraps `ck["model"]`,
2. prepends `detector.` to every key,
3. maps the trunk subtree to `EfficientSam3Trunk`'s attribute names (the upstream nesting is
   `trunk.model.backbone.model.features.*` + `trunk.model.head.*`; we either reproduce that nesting or
   apply an explicit remap — decided in the plan, both load strictly),
4. loads `strict=True` after remap (the project's discipline; **no silent `strict=False`** — the
   validation already proved 1107/1107 match, so strict load is the correct guard).

## 8. Configs / builders / download

- **Configs:** `configs/efficientsam3/efficientsam3_repvit.yaml` first — clone `configs/sam3/sam3.yaml`,
  repoint `vision_encoder.vision_backbone.trunk._target_` → `EfficientSam3Trunk` (backbone=repvit,
  model_name=m1_1) and `text_encoder._target_` → `MobileClipTextEncoder` (MobileCLIP-S0, ctx 16). Later:
  `_tinyvit.yaml`, `_efficientvit.yaml`, LiteText yamls.
- **Builders:** `build_efficientsam3`, `build_efficientsam3_video_predictor`, `build_efficientsam3_hf`
  (+ video hf) mirror `build_sam3_*` (hydra compose/instantiate; checkpoint remap §7).
- **Download:** `tools/download_efficientsam3.py` — public `hf_hub_download` from
  `Simon7108528/EfficientSAM3` (paths like `efficientsam3_ft/efficientsam3_repvit.pt`); no auth.
  Pixi tasks `download-efficientsam3-repvit` (+ tinyvit / efficientvit / litetext) with `outputs=[…]`
  cache-skip. `HF_HUB_DISABLE_XET=1`.
- **Deps (E9):** **none added.** Vendor `SqueezeExcite` + `to_2tuple` into a small shared
  `modeling/encoders/_layers.py`; reuse the existing local `DropPath` and `torch.nn.init.trunc_normal_`;
  drop the timm registry decorators / cfg-builder. EfficientViT's Triton kernel is imported lazily with
  a pure-torch RMSNorm fallback so RepViT/TinyViT and Windows/CPU paths never touch Triton.

## 9. Data flow

Unchanged from the SAM 3 spec §10 (image: `encode_image → encode_text → detect → masks`; streaming
video: per-frame `encode_image`, gated detection, tracker step, forgetful memory bank, per-object
`MaskletResult`). EfficientSAM3 only changes which `encode_image` / `encode_text` modules are
instantiated.

## 10. Phasing + verification gates

Sequenced base-image first (lowest risk, already validated), then LiteText, then the multiplex video
families. Phase A carries the shared scaffolding (trunk, text encoder, `TextEncoder` Protocol, builders,
download tool, licensing) that every later phase reuses. **Streaming video first appears in Phase E**
(the multiplex lineage owns the only published video/tracker checkpoints).

**Phase A — EfficientSAM3 RepViT image (slice; the scaffolding phase)**
- Vendor `repvit.py` + `mobile_clip.py` (de-timm'd: vendor `SqueezeExcite`/`to_2tuple`, reuse local
  `DropPath` + torch `trunc_normal_`); add `EfficientSam3Trunk`, `MobileClipTextEncoder`, the
  `TextEncoder` Protocol, `configs/efficientsam3/efficientsam3_repvit.yaml`, `build_efficientsam3*`
  (image; detector-root remap = prepend `detector.`), the download tool, SPDX headers + README rows. No
  new pixi deps.
- **Verify:** checkpoint loads **strict (1107/1107)**; image masks match the captured upstream golden
  (`dog`/`person` on `dog_person.jpeg`) within tolerance (mask IoU + instance count + score ordering);
  build smoke (meta-build + forward); FPS within ~15%/phase of the §10 reference.

**Phase B — EfficientSAM3 TinyViT image**
- Vendor `tiny_vit.py` (de-timm'd); extend `EfficientSam3Trunk` to TinyViT; add
  `efficientsam3_tinyvit.yaml` + download entry.
- **Verify:** strict load (pin `model_name` from the trunk channel signature) + golden parity + build smoke.

**Phase C — EfficientSAM3 EfficientViT image**
- Vendor `efficientvit/` (its own `nn/` package; **Triton RMSNorm behind a lazy pure-torch fallback**);
  extend the trunk; add `efficientsam3_efficientvit.yaml` + download entry.
- **Verify:** strict load + golden parity + build smoke; fallback path exercised on CPU/Windows.

**Phase D — SAM3-LiteText image (base lineage, PE vision)**
- No new vision code — pair the **existing PE `Sam3VisionEncoder`** with `MobileClipTextEncoder`; add
  `sam3_litetext_*.yaml` per {S0,S1,2-L}×{ctx16,ctx32} + download entries.
- **Verify:** strict load + golden parity per text variant/context length.

**Phase E — SAM3.1-LiteText streaming video (multiplex lineage; introduces video)**
- Wire the **existing PE vision encoder** + `MobileClipTextEncoder` into the **existing
  `Sam3MultiplexVideoPredictor`** (tri-neck + 457-key multiplex tracker, both reused unchanged);
  `build_efficientsam3_litetext_video_predictor*` (full-model-root remap = load `detector.*`/`tracker.model.*`
  directly); add `sam3p1_litetext_*.yaml` + download entries.
- **Verify:** strict load (incl. `tracker.model` 457/457); streaming masklets reproduce the upstream video
  example within tolerance; **peak VRAM flat vs video length** with the forgetful bank; capture the
  Phase-B-style **video FPS reference** (text cached once + per-frame vision+detect+track).

**Phase F — EfficientSAM3.1 streaming video (multiplex lineage, distilled trunk; stage1)**
- Reuse the Phase A–C `EfficientSam3Trunk` in the multiplex video predictor (the only new thing vs Phase
  E is the distilled trunk in place of PE); add `efficientsam3p1_{repvit,tinyvit,efficientvit}_*.yaml` +
  download entries.
- **Verify:** strict load + streaming parity + VRAM-flat + video FPS. **Note:** stage1-only checkpoints
  (no `_ft`) — parity tolerance vs upstream stage1, not vs a finetuned model.

**Reference parity (acceptance gate).** Each phase compares against the **upstream** EfficientSAM3 (not
only internal tests): same image/video + prompt through upstream and our predictor; compare masks (IoU),
instance counts, scores. Fixtures captured once (the §4 golden) so CI runs without the upstream repo.

### Performance reference (image path)

Captured from the **upstream** EfficientSAM3 RepViT model on this machine (RTX 3080 Ti,
torch 2.11+cu128), `dog_person.jpeg` 2048×1365, prompt "dog", threshold 0.1, 50 iters + 10 warmup,
`torch.cuda.synchronize()` per timed call, TF32 on, `cudnn.benchmark` on. Script:
`efficientsam3_reference/_bench.py`; data: `efficientsam3_reference/_golden/bench.json`.

| Phase | fp32 (TF32) | autocast bf16 |
|---|---|---|
| vision (`set_image`: RepViT trunk + neck + backbone feats) | **31.5 ms · 31.7 fps** | 45.6 ms · 21.9 fps |
| prompt (text + DETR detect + segmentation, 13 instances) | 97.4 ms · 10.3 fps | 119 ms · 8.4 fps |
| end-to-end (image + 1 prompt) | **136.6 ms · 7.3 fps** | 188 ms · 5.3 fps |

- The **distilled vision encoder (~31 ms)** is the EfficientSAM3 win; the **reused SAM 3 DETR
  detector + per-instance mask head (~97 ms) dominates** end-to-end. Our integration reuses both, so
  it should match these numbers.
- autocast bf16 is **slower than fp32+TF32** on this Ampere GPU (RepViT is conv-heavy; TF32 already
  saturates the tensor cores, so bf16 adds cast overhead). **Canonical reference = fp32+TF32.**
- **Image path only** — the upstream image checkpoint has no tracker. Per-frame **video FPS** (text
  cached once per concept + tracker step) is a separate reference captured in Phase E.
- Prompt-phase cost scales with instance count (fixed prompt+threshold for comparability); absolute
  ms shifts ±10–15% with GPU thermal/clock — the breakdown **ratio** (vision ⅓ / detector ⅔) is the
  robust takeaway.

**FPS parity gate.** The integrated `EfficientSam3` image predictor must reproduce this breakdown
within ~15% per phase on the same GPU / input / precision (a regression guard, not a hard SLA), benched
with identical methodology (`tools/benchmark_efficientsam3.py`, to be added). Video FPS gate set in
Phase E against its own reference.

## 11. Licensing (extends D11)

No single umbrella — extend the existing per-component scheme:
- **SPDX per vendored file:** `mobile_clip.py` → Apple ML Research license (`LicenseRef-AppleML`);
  `efficientvit/*`, `repvit.py`, `tiny_vit.py` → each backbone's own license; EfficientSAM3 glue and
  anything importing SAM 3 → `LicenseRef-SAM`.
- **Ship each license text** (`LICENSE_*`) and add a **per-model row** to the README license table
  (model → license → key restrictions), noting MobileCLIP's research-oriented terms.
- **Weights never vendored** (public download only).
- **Legal review before any public release** (carried over from D11's open item; MobileCLIP especially).

## 12. Risks & mitigations

1. **Checkpoint key remap** — **retired** by §4 (1107/1107 strict match proven); plan keeps `strict=True`.
2. **Trunk output resolution** — backbones are not natively 72×72; the projection must land exactly.
   Mitigation: cross-check against the upstream build + the golden (already passing for RepViT).
3. **Triton RMSNorm (EfficientViT)** on Windows → pure-torch fallback, lazy import (Phase C only).
4. **ctx16/77 pos-embed (E7)** — build text at the checkpoint's ctx; assert pos-embed shape pre-load.
5. **MobileCLIP licensing** — SPDX + README table + legal-review note.
6. **No-weights CI** — skip-if-absent tests + the captured golden fixtures (mirror SAM 3.1).
7. **Video checkpoint availability** — **resolved** by §4/§4a: video/tracker checkpoints exist only in
   the SAM 3.1 lineage (`sam3p1_litetext/*`, `stage1_sam3p1/*`), validated to carry the 457-key multiplex
   tracker. Base EfficientSAM3 is image-only → streaming video is delivered via Phases E/F.
8. **EfficientSAM3.1 maturity** — distilled-on-multiplex ships **stage1-only** (no `_ft`); parity is
   measured against upstream stage1 output, and the README notes the lower maturity vs the base `_ft` models.

## 13. References

- EfficientSAM3 repo: https://github.com/SimonZeng7108/efficientsam3
- Checkpoints (public): https://huggingface.co/Simon7108528/EfficientSAM3
  (`efficientsam3_ft/`, `sam3_litetext/`, `sam3p1_litetext/`, `stage1_*`)
- Upstream builder / student encoders: `sam3/sam3/model_builder.py`
  (`build_efficientsam3_image_model`, `_create_student_vision_backbone`,
  `_create_student_text_encoder`), `sam3/sam3/backbones/{repvit,tiny_vit,efficientvit,mobile_clip}.py`,
  `sam3/sam3/model/text_encoder_student.py`
- SAM 3 base spec: [`2026-06-26-sam3-integration-design.md`](2026-06-26-sam3-integration-design.md)
- Validation golden (this machine): `efficientsam3_reference/_golden/summary.json` (+ `masks_*.npz`)
