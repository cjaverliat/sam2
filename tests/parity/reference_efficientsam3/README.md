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
