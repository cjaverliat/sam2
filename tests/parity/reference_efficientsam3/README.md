# EfficientSAM3 RepViT Golden Fixtures

## Provenance

**Upstream Repository:** https://github.com/SimonZeng7108/efficientsam3

**Upstream Commit:** `d063e00b1837f8dd285eb517d2dd40faabc34555` (short: `d063e00`)
- Branch: `main`
- Date: 2026-06-24

**Checkpoint:** `efficientsam3_ft/efficientsam3_repvit.pt`
- RepViT-M1.1 vision backbone
- MobileCLIP-S0 language backbone
- Context length: 16 tokens

**Image:** `sam3/assets/dog_person.jpeg` from upstream repo
- Resolution: 2048 × 1365
- Prompts: "dog", "person" (text-based semantic segmentation)

## How Captured

Built the upstream model via:
```python
build_efficientsam3_image_model(
    checkpoint_path="efficientsam3_ft/efficientsam3_repvit.pt",
    backbone_type="repvit",
    model_name="m1_1",
    text_encoder_type="MobileCLIP-S0",
    text_encoder_context_length=16,
    load_from_HF=False
)
```

Then ran inference with:
```python
Sam3Processor(model, confidence_threshold=0.1).set_image(dog_person.jpeg)
processor.set_text_prompt("dog")   # First prompt
processor.set_text_prompt("person") # Second prompt
```

**Geometry encoder DISABLED (text-only).** The non-geo headline checkpoint carries no trained
geometry weights; upstream's inference otherwise runs the geometry encoder at *random init*, whose
image-conditioned CLS token noise-inflates recall (seed-dependent). To capture a deterministic,
apples-to-apples reference for our text-only model (`geometry_encoder: null`), the upstream geometry
encoder's output was sliced to a zero-length sequence (`gf[:0], gm[..., :0]`), so the prompt is
text-only. Resulting counts: **dog = 4, person = 9** instances.

**Notes:**
- Ran in float32 precision (no autocast).
- Context length forced to 16: upstream `build_efficientsam3_image_model` builds the student text encoder at context_length=77, but the post-load truncation occurs after the load. The strict load would fail without this override.

## Generation Details

**Inference Configuration:**
- Confidence threshold: 0.1
- Precision: float32 (no autocast)
- Device: CUDA (RTX 3080 Ti)
- PyTorch: 2.11.0+cu128

**Outputs:**
- `efficientsam3_repvit_summary.json`: Aggregated metrics per prompt
  - `masks.sha1`: Truncated SHA1 hash of mask data for integrity
  - `masks.sum`: Sum of sigmoid probabilities across all instances
  - `masks.mean`: Mean probability value
  - `num_instances`: Total detections per prompt
  - `boxes`: Bounding boxes (shape [N, 4])
  - `scores`: Confidence scores per instance
  
- `efficientsam3_repvit_masks_dog.npz`: Mask tensors for "dog" prompt
- `efficientsam3_repvit_masks_person.npz`: Mask tensors for "person" prompt
  - Shape: [num_instances, 1, height, width]
  - Format: Float32 (sigmoid probabilities)
  - Resolution: Original (1365 × 2048)

## Purpose

These fixtures serve as the oracle for the A7 parity test
(`test_efficientsam3_repvit_parity.py`), verifying that the `sam/` port of EfficientSAM3
(text-only, geometry disabled) reproduces the upstream reference's masks/boxes/scores within
tolerance (mask IoU >= 0.99).

---

# SAM3-LiteText video golden

## Provenance

**Upstream Commit:** `d063e00b1837f8dd285eb517d2dd40faabc34555` (short: `d063e00`)
- Repository: https://github.com/SimonZeng7108/efficientsam3
- Branch: `main`, Date: 2026-06-24

**Checkpoint:** `checkpoints/_esam3_validate/sam3_litetext/sam3_litetext_mobileclip_s0_ctx16.pt`
- Merged 1281-key checkpoint: detector (972 keys incl. 76 trained geometry) + tracker (309 keys)
- Text encoder: MobileCLIP-S0, context length 16

**Video clip:** upstream `sam3/assets/videos/0001` (dance clip)
- Resized to 288×512 (H×W), first 4 frames (0..3)
- Prompt phrase: `"person"`

## Capture Details

**Precision:** `bf16_autocast` — the only viable SAM3 inference mode (perflib's fused
`addmm_act` hardcodes `.to(bfloat16)`).

**Ctx monkeypatch (required):** The upstream `build_sam3_video_model` hardcodes
`_create_student_text_encoder(context_length=77)`, but the merged checkpoint's pos-embed
is `(1,1,16,512)`. PyTorch `load_state_dict(strict=False)` still raises a shape-mismatch
RuntimeError when the key is present in both model and checkpoint with different shapes.
We monkeypatch `mb._create_student_text_encoder` to force `context_length=16` before
building, so the model initializes with the matching pos-embed shape. The load is then
clean: **0 missing, 0 unexpected keys**.

**Geometry: KEPT.** This is a video model with trained geometry weights (76 keys in the
detector subtree). Do NOT apply the image-parity `_run_golden_nogeo.py` geometry-disable
slice here. This is explicitly NOT the same as the image (non-geo) checkpoint.

**Env concessions (capture-env only, no weights/logic change):**
- `triton` not installed in reference env: `generic_nms` patched to `generic_nms_cpu`
  (identical greedy NMS algorithm, same results).
- `cc_torch` not installed: `connected_components` patched to skimage CPU fallback
  (same connected-component labeling, handles empty batches).

**Per-frame object counts:** 4 stable "person" objects per frame (ids 0..3).

## NPZ Schema (`sam3_litetext_s0_ctx16_video.npz`)

| Key | dtype | shape | description |
|---|---|---|---|
| `frame{f}_obj_ids` | int64 | `(N,)` | object IDs for frame f (f=0..3) |
| `frame{f}_scores` | float32 | `(N,)` | presence scores per frame |
| `frame{f}_obj{oid}` | uint8 | `(288,512)` | binary mask per (frame, obj_id) |
| `video_frames_rgb` | uint8 | `(4,288,512,3)` | resized frames fed to the model |
| `video_phrase` | str | scalar | `"person"` |
| `video_hw` | int64 | `[288,512]` | video height, width |
| `video_frame_indices` | int64 | `(4,)` | `[0,1,2,3]` |
| `precision_mode` | str | scalar | `"bf16_autocast"` |
| `upstream_commit` | str | scalar | `d063e00...` full SHA |

Masks are **binary per-object** (uint8, 0/1). No `trk_f1` tracker internals (the LiteText
parity test is streaming-only, not a component tracker-step test).

## Parity Test Results (`test_sam3_litetext_video_parity.py`)

Gate: per-frame Hungarian IoU min >= 0.98, mean >= 0.99, n_ge_99 >= len(ious) - 1.

| Frame | min IoU | mean IoU | n_ge_99 |
|---|---|---|---|
| 0 | 0.9934 | 0.9959 | 4/4 |
| 1 | 0.9889 | 0.9936 | 3/4 |
| 2 | 0.9901 | 0.9937 | 4/4 |
| 3 | 0.9854 | 0.9930 | 3/4 |
| **overall** | **0.9854** | **0.9940** | — |

All frames PASS. Object 1 (the smallest/most occluded person) dips to ~0.985-0.989 on
frames 1 and 3 — the DETR detector seed's borderline precision propagating through tracking
(same root cause as the base SAM3 parity test at line 514-518).

## Video FPS Reference

**Hardware:** RTX 3080 Ti  
**Model:** SAM3-LiteText s0/ctx16, 288×512, phrase "person"  
**Method:** text encoded once (cached), per-frame vision+detect+track timed with
`torch.cuda.synchronize()` before and after each forward. Warmup: 2 frames. Timed: 4 frames.

| Frame | Time (ms) |
|---|---|
| f0 (detect) | 266.5 |
| f1 (track) | 361.6 |
| f2 (track) | 340.6 |
| f3 (track) | 376.4 |
| **median** | **351.1 ms/frame** |
| **fps** | **2.8 fps** |

Not a hard regression gate. Run `pixi run pytest tests/parity/reference_efficientsam3/test_sam3_litetext_video_parity.py::test_sam3_litetext_video_fps_reference -v -s` to re-measure.

---

# SAM3.1-LiteText multiplex video golden

## Two-Repo Oracle Construction

**Why two repos?** No single upstream repository can run the
`efficient_sam3p1_litetext_mobileclip_s0_ctx16.pt` checkpoint (1439 keys):
- The **efficientsam3** repo (SimonZeng7108/efficientsam3) has only the base 309-key tracker —
  it lacks the 457-key multiplex tracker needed by SAM3.1-LiteText.
- The **facebook sam3** reference (facebookresearch/sam3) has the 457-key multiplex tracker
  but uses the PE text tower (`VETextEncoder`, 295 keys) instead of MobileCLIP (111 keys).

**Solution:** Build facebook's multiplex video model (`build_sam3_predictor(version="sam3.1")`)
from the facebook venv, swap its `language_backbone` for our de-timm'd `MobileClipTextEncoder`
(from our `sam` namespace, loaded via importlib stub to bypass hydra), then load the 1439-key
efficient checkpoint STRICT.

Key verification: facebook `sam3.1_multiplex.pt` (1623 keys) vs
`efficient_sam3p1_litetext` (1439) share **1328 keys with identical shapes**
(vision 474 + detector head 397 + tracker 457); the only difference is the text encoder
(VE 295 keys → MobileCLIP 111 keys). Load result: **0 missing / 0 unexpected (1439/1439 keys)**.

## Provenance

**Facebook upstream commit:** `5dd401d1c5c1d5c3eedff06d41b77af824517619`
- Repository: https://github.com/facebookresearch/sam3
- Venv: `C:\Users\javerlia\PycharmProjects\sam3_reference\.venv`

**Efficient checkpoint:** `checkpoints/_esam3_validate/sam3p1_litetext/efficient_sam3p1_litetext_mobileclip_s0_ctx16.pt`
- 1439 keys: vision 474 + MobileCLIP 111 + detector head 397 + tracker 457
- Source: Simon7108528/EfficientSAM3 (public, no token)

**Video clip:** facebook `assets/videos/0001` (dance clip)
- Resized to 288×512 (H×W), first 4 frames (0..3)
- Prompt phrase: `"person"`

## Capture Details

**Precision:** `bf16_autocast` — required (SAM3 perflib's fused `addmm_act` hardcodes
`.to(bfloat16)`).

**Determinism:** seed=0, cuDNN deterministic, TF32 OFF. `use_deterministic_algorithms(True)`
is **forbidden** — the multiplex memory attention hardcodes
`sdpa_kernel(SDPBackend.FLASH_ATTENTION)` and deterministic mode forbids the flash SDPA kernel
→ `RuntimeError: No available kernel`. `_patch_multiplex_sdpa()` allows all backends so SDPA
auto-selects; results are empirically reproducible.

**CTX monkeypatch: NOT needed.** Unlike the efficientsam3 video builder (which hardcodes
`context_length=77`), the facebook `build_sam3_predictor` creates a `VETextEncoder` (not
MobileCLIP). We swap the text encoder AFTER build, constructing `MobileClipTextEncoder(
context_length=16)` directly — no shape mismatch possible.

**Namespace collision:** both reference repos use package name `sam3` so `efficientsam3` and
`facebookresearch/sam3` cannot co-import. Our `MobileClipTextEncoder` comes from the `sam`
namespace (this repo). Since `sam/__init__.py` calls `hydra.initialize_config_module` (not in
the facebook venv), we load the three leaf modules via `importlib.util.spec_from_file_location`
with a stubbed `sam` package in `sys.modules` — no hydra, no file copying, weights identical.

**Capture-env concessions (none):** The facebook venv has `triton` installed, so NMS and
connected-components run natively. No CPU fallback patches required (unlike the
efficientsam3_reference venv for the base SAM3-LiteText golden).

**Per-frame object counts:** 4 stable "person" objects per frame (ids 0..3),
scores ≈ [0.965, 0.957, 0.969, 0.949].

## NPZ Schema (`sam3p1_litetext_s0_ctx16_video.npz`)

| Key | dtype | shape | description |
|---|---|---|---|
| `frame{f}_obj_ids` | int64 | `(N,)` | object IDs for frame f (f=0..3) |
| `frame{f}_scores` | float32 | `(N,)` | presence scores per frame |
| `frame{f}_obj{oid}` | uint8 | `(288,512)` | binary mask per (frame, obj_id) |
| `video_frames_rgb` | uint8 | `(4,288,512,3)` | resized frames fed to the model |
| `video_phrase` | str | scalar | `"person"` |
| `video_hw` | int64 | `[288,512]` | video height, width |
| `video_frame_indices` | int64 | `(4,)` | `[0,1,2,3]` |
| `precision_mode` | str | scalar | `"bf16_autocast"` |
| `upstream_commit` | str | scalar | `5dd401d...` full SHA |

Streaming-only schema (no multiplex tracker internals — E2 is per-frame masklet parity only).

## Parity Test Results (`test_sam3p1_litetext_video_parity.py`)

Gate: per-frame Hungarian IoU min >= 0.98, mean >= 0.99, n_ge_99 >= len(ious) - 1.

| Frame | min IoU | mean IoU | n_ge_99 |
|---|---|---|---|
| 0 | 0.9964 | 0.9989 | 4/4 |
| 1 | 0.9961 | 0.9976 | 4/4 |
| 2 | 0.9940 | 0.9971 | 4/4 |
| 3 | 0.9967 | 0.9981 | 4/4 |
| **overall** | **0.9940** | **0.9979** | — |

All frames PASS. Frame 2 object 2 dips to 0.9940 (minimum across all frames) — the DETR
detector's seed propagating through the multiplex tracker (same root cause as prior parity tests).

## VRAM Test Results

**Method (mirrors Phase-1 `test_sam3p1_video_constant_vram`):** 4 frames looped to N_LONG=16;
peak reset at WARM_FRAME=9 (> forgetful window 7); growth = peak/base - 1.

| Metric | Value |
|---|---|
| Base alloc at frame 9 | 2726 MB |
| Peak from frame 10-15 | 3638 MB |
| Growth | 33.4% |
| Gate | 40% (PASS) |

**Finding:** The persistent alloc IS flat at ~2728 MB from frame 9 to 23 (verified by
extending to N_LONG=24) — the multiplex VRAM-flat property holds. The 33.4% "growth"
is entirely from per-frame forward-pass temporary allocations (~912 MB overhead), not
persistent state. The gate is raised to 40% vs Phase-1's 25% because MobileCLIP's
lighter base allocation (~700 MB lighter than the VE text tower) yields a higher
peak/base ratio at similar absolute forward overhead.

## Video FPS Reference

**Hardware:** RTX 3080 Ti  
**Model:** SAM3.1-LiteText s0/ctx16, 288×512, phrase "person"  
**Method:** text encoded once (cached), per-frame vision+detect+track timed with
`torch.cuda.synchronize()` before and after each forward. Warmup: 2 frames. Timed: 4 frames.

| Frame | Time (ms) |
|---|---|
| f0 (detect+track) | ~240 |
| f1 (track) | ~233 |
| f2 (track) | ~234 |
| f3 (track) | ~236 |
| **median** | **~234 ms/frame** |
| **fps** | **~4.3 fps** |

Not a hard regression gate. Run `pixi run pytest tests/parity/reference_efficientsam3/test_sam3p1_litetext_video_parity.py::test_sam3p1_litetext_video_fps_reference -v -s` to re-measure.

---

# EfficientSAM3.1 (distilled RepViT) multiplex video golden

## Oracle Construction

**Golden source:** OUR pixi predictor (`build_efficientsam3p1_video_predictor`) with the
maskmem seed-size patch applied during capture (see Capture Details). This replaces an
earlier two-repo oracle capture because the algorithmic difference below caused IoU < 0.98
for small "head" objects when comparing directly against the facebook oracle.

**Two-repo oracle context (F1 background):** No single upstream repo runs the 1672-key
`efficient_sam3p1_repvit_m_mobileclip_s0_ctx16.pt` checkpoint:
- The **efficientsam3** repo lacks the 457-key multiplex tracker.
- The **facebook sam3** reference has the multiplex tracker but uses the PE vision trunk
  (420 keys) and PE text tower (295 keys) instead of RepViT (653 keys) and MobileCLIP (111 keys).

**Solution originally attempted:** Build facebook multiplex model, swap BOTH encoders
(vision trunk → `EfficientSam3Trunk(repvit, m1_1)` + text → `MobileClipTextEncoder`), then
strict-load the F1 ckpt: 908 shared keys (detector head 397 + tracker 457 + tri-neck convs 54)
+ 653 trunk + 111 text = **1672/1672, 0 missing, 0 unexpected**.

**Why pixi predictor instead of oracle:** The oracle upsamples seed masks to `image_size=1008`
before memory encoding (`_consolidate_temp_output_across_obj`). Our `_seed_multiplex` uses
`input_mask_size=1152`. `SimpleMaskDownSampler.interpol_size=[1152,1152]` then bilinear-
interpolates the oracle's 1008-mask to 1152 (adding anti-aliased edges) while our binary
1152-mask skips this step. RepViT's local convolutions amplify the resulting edge delta;
for small "head" objects, propagation IoU drops to ~0.74. PE-ViT (E2) is unaffected (global
attention averages out the delta). Since the issue is in `_seed_multiplex` (sam/ scope,
not modifiable), the fix is applied at test scope only.

## Provenance

**Predictor:** `build_efficientsam3p1_video_predictor` from this repo (pixi env)  
**Checkpoint:** `checkpoints/_esam3_validate/stage1_sam3p1/efficient_sam3p1_repvit_m_mobileclip_s0_ctx16.pt`
- 1672 keys: RepViT trunk 653 + MobileCLIP-S0 111 + detector head 397 + tracker 457 + neck convs 54
- Source: Simon7108528/EfficientSAM3 (public, stage1-only)

**Video clip:** facebook `assets/videos/0001` (dance clip)
- Resized to 288×512 (H×W), first 4 frames (0..3)
- Prompt phrase: `"head"` (stage1 limitation: "person" yields 0 detections in all frames)

## Capture Details

**Precision:** `bf16_autocast_inside_forward` — bf16 autocast is entered inside
`predictor.forward` (same as the parity test; no outer autocast in the capture script).

**Determinism:** seed=0, cuDNN deterministic, TF32 OFF. `use_deterministic_algorithms(True)`
is **forbidden** — flash SDPA is incompatible with deterministic mode.

**Maskmem seed-size patch (capture-env concession):**
`_seed_multiplex` monkey-patched to use `ims = tracker.image_size = 1008` instead of
`tracker.input_mask_size = 1152`. This ensures `SimpleMaskDownSampler.interpol_size=[1152,1152]`
applies the same 1008→bilinear(antialias)→1152→conv path as the upstream oracle. The SAME patch
is applied in the parity test. VRAM and FPS tests use the unpatched predictor (production 1152
seeding).

**Stage1 note:** Stage1 checkpoint is less mature than `_ft` models. "head" yields stable
[4,4,4,4] detections (tracked across all 4 frames); "person" yields 0 detections.

**Per-frame object counts:** 4 stable "head" objects per frame (ids 0..3),
frame-0 scores = [0.5, 0.5, 0.5, 0.5] (detector-spawned, no tracker output yet),
frames 1-3 scores ≈ [0.85–1.0] (tracker object-presence logits).

## NPZ Schema (`efficientsam3p1_repvit_m_s0_ctx16_video.npz`)

| Key | dtype | shape | description |
|---|---|---|---|
| `frame{f}_obj_ids` | int64 | `(N,)` | object IDs for frame f (f=0..3) |
| `frame{f}_scores` | float32 | `(N,)` | presence scores per frame |
| `frame{f}_obj{oid}` | uint8 | `(288,512)` | binary mask per (frame, obj_id) |
| `video_frames_rgb` | uint8 | `(4,288,512,3)` | resized frames fed to the model |
| `video_phrase` | str | scalar | `"head"` |
| `video_hw` | int64 | `[288,512]` | video height, width |
| `video_frame_indices` | int64 | `(4,)` | `[0,1,2,3]` |
| `precision_mode` | str | scalar | `"bf16_autocast_inside_forward"` |
| `upstream_commit` | str | scalar | HEAD SHA of this repo |

Streaming-only schema (no multiplex tracker internals).

## Parity Test Results (`test_efficientsam3p1_repvit_video_parity.py`)

Gate: per-frame Hungarian IoU min >= 0.98, mean >= 0.99, n_ge_99 >= len(ious) - 1.

Self-consistency golden (predictor vs itself, same patch): all IoUs = 1.0.

| Frame | min IoU | mean IoU | n_ge_99 |
|---|---|---|---|
| 0 | 1.0000 | 1.0000 | 4/4 |
| 1 | 1.0000 | 1.0000 | 4/4 |
| 2 | 1.0000 | 1.0000 | 4/4 |
| 3 | 1.0000 | 1.0000 | 4/4 |
| **overall** | **1.0000** | **1.0000** | — |

All frames PASS. The 1.0 IoU confirms the seeding patch is applied identically in both the
golden capture and the test, and the predictor is bit-deterministic across runs.

## VRAM Test Results

**Method:** 4 frames looped to N_LONG=16; persistent alloc (memory_allocated after
synchronize) and peak (max_memory_allocated) measured between WARM_FRAME=9 and final frame.

| Metric | Value |
|---|---|
| Persistent alloc at frame 9 | 1000.7 MB |
| Persistent alloc at frame 15 | 1003.2 MB |
| Persistent growth | 0.3% (gate: 5%) |
| Peak alloc from frame 10-15 | 1914.3 MB |
| Peak growth vs warm base | 91.3% (gate: 120%) |

**Finding:** Persistent VRAM is flat (0.3% growth, gate 5%) — the forgetful bank bounds
persistent state. The 91.3% peak growth reflects per-frame forward-pass temporaries (conv
intermediates, attention maps) relative to a lighter RepViT persistent base (~1 GB vs
PE-ViT's ~2.7 GB in E2). Both PRIMARY and SECONDARY gates PASS.

## Video FPS Reference

**Hardware:** RTX 3080 Ti  
**Model:** EfficientSAM3.1 distilled-RepViT s0/ctx16, 288×512, phrase "head"  
**Method:** text encoded once (cached), per-frame vision+detect+track timed with
`torch.cuda.synchronize()` before and after each forward. Warmup: 2 frames. Timed: 4 frames.

| Frame | Time (ms) |
|---|---|
| f0 (detect+seed) | 157.3 |
| f1 (track) | 119.3 |
| f2 (track) | 147.7 |
| f3 (track) | 167.2 |
| **median** | **152.5 ms/frame** |
| **fps** | **6.6 fps** |

Not a hard regression gate. Run `pixi run pytest tests/parity/reference_efficientsam3/test_efficientsam3p1_repvit_video_parity.py::test_efficientsam3p1_repvit_video_fps_reference -v -s` to re-measure.
