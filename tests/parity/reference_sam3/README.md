<!-- SPDX-License-Identifier: LicenseRef-SAM -->
# SAM 3 reference golden (parity oracle)

`capture_sam3_golden.py` captures golden activations from the **official** SAM 3
(`facebook/sam3`) and writes the fixtures in `tests/parity/fixtures/sam3/`. These are the
**parity oracle** for Phase 1: every vendored `Sam3*` re-implementation (Tasks 2–10)
validates against them (mask IoU ≥ 0.99, score/logit `atol`). The capture runs **once**, in
an **isolated reference env** (not this repo's pixi env); the fixtures let CI run the parity
tests without the upstream repo or the gated weights present.

## Upstream commit

`facebookresearch/sam3` @ **`5dd401d1c5c1d5c3eedff06d41b77af824517619`** (cloned to the sibling
`../sam3_reference`). The SAM License text from that clone is vendored here as `LICENSE_sam`.

## Reference env recipe (isolated — do NOT use this repo's pixi env)

The upstream model is **not fp32-runnable** with `perflib` enabled (the fused
`perflib/fused.py::addmm_act` in the ViT MLP hardcodes `.to(torch.bfloat16)`), and the
official demo/notebook always run under bf16 autocast. The capture therefore runs under
**bf16 autocast** while keeping every other determinism lever from
`tests/parity/run_pipelines.py::_determinism` (seed 0, deterministic algorithms, cuDNN
deterministic, TF32 off). Determinism was verified: a re-run reproduces boxes/scores
exactly and masks at IoU = 1.0. **Downstream parity must use the same bf16-autocast +
deterministic regime.**

```bash
# from the repo parent dir; upstream cloned at ../sam3_reference
uv venv ../sam3_reference/.venv --python 3.12
PY=../sam3_reference/.venv/Scripts/python.exe          # Windows; .../bin/python on Linux
uv pip install --python $PY torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python $PY -e ../sam3_reference        # installs sam3 (timm, numpy<2, …)
uv pip install --python $PY triton-windows              # 'triton' on Linux; edt.py imports it
uv pip install --python $PY pycocotools psutil opencv-python einops
uv pip install --python $PY "numpy<2" "scipy<1.14" "scikit-image<0.25"   # sam3 pins numpy<2

# run the capture (loads local checkpoints/sam3.pt; no HF download needed)
$PY tests/parity/reference_sam3/capture_sam3_golden.py
```

Notes:
- **triton** only needs to *import* (via `sam3.model.edt`); its kernel (`edt_triton`) is only
  *called* during interactive point-correction, which the text→propagate scenario never hits.
  `nms` / `connected_components` fall back to CPU (no `torch_generic_nms` / `cc_torch` needed).
- HF auth is not required when loading the **local** `checkpoints/sam3.pt`
  (`load_from_HF=False`); the weights are never committed.
- GPU used: RTX 3080 Ti (12 GB). The full model loads in bf16-autocast within budget.

## Scenario

| | Image | Video |
|---|---|---|
| Source | this repo's `notebooks/images/truck.jpg` | upstream `assets/videos/0001` (dance clip) |
| Resize (H×W) | 384×512 | 288×512, first 4 frames |
| Phrase | `"truck"` | `"person"` |
| Flow | `Sam3Processor`: set_image → set_text_prompt | `build_sam3_video_model`: init_state → add_prompt(text @ f0) → propagate |
| Result | 1 detection | 4 objects (ids 0–3) tracked across 4 frames |

The exact preprocessed inputs are saved **inside** the fixtures (`image_input_rgb`,
`video_frames_rgb`) so downstream tests feed byte-identical tensors. `scenario.json` records
the inputs, per-frame object ids/scores, precision mode, and commit.

## Fixture keys

**`image.npz`** (committed, ~2.5 MB) — detector + encoder + text golden:
`boxes` (N,4 xyxy px), `scores` (N,), `presence`/`presence_logit` (1,1), `masks` (N,H,W
uint8), `masks_logits` (N,H,W f16), `pred_boxes_cxcywh` (1,200,4), `pred_logits` (1,200,1),
`text_emb` (32,1,256 f16), `text_embeds_pre` (32,1,1024 f16), `enc_feat_lastlevel`
(1,256,72,72 f16 — the single stride-14 level the DETR detector consumes,
`num_feature_levels=1`), `image_input_rgb` (384,512,3 uint8), plus `image_hw`,
`image_phrase`, `precision_mode`, `upstream_commit`, `confidence_threshold`.

**`image_encoder_pyramid.npz`** (NOT committed — git-ignored, ~38 MB, regenerable) — the full
high-res encoder pyramid at the model's fixed 1008 internal resolution:
`enc_feat_l0` (1,256,288,288), `enc_feat_l1` (1,256,144,144), `enc_feat_l2` (1,256,72,72) and
`enc_pos_l0..l2`. Too large to commit; positional encodings are a computed sine function. Use
for finer Task-2 encoder checks by re-running the capture.

**`video.npz`** (committed, ~2.5 MB) — streaming masklets + one tracker step:
`frame{i}_obj{id}` (H,W uint8) for i∈0..3, id∈0..3; `frame{i}_obj_ids` (4,), `frame{i}_scores`
(4,); `trk_f1` (4,3,288,288 f16 — raw `sam_mask_decoder` low-res mask logits at frame 1,
per-object × 3 multimask tokens; for Task-5 tracker-step parity); `trk_f1_num_calls`;
`video_frames_rgb` (4,288,512,3 uint8), plus `video_hw`, `video_phrase`,
`video_frame_indices`, `precision_mode`, `upstream_commit`.

## Upstream layout (actual paths @ 5dd401d — corrects the plan's guesses)

The plan's guessed `sam3/model/encoder.py` for the **vision** encoder is wrong: `encoder.py`
is the DETR early-fusion *transformer* encoder; the PE **vision trunk** is `vitdet.py` +
`necks.py`. Builders live in `sam3/model_builder.py` (not hydra configs).

- **Vision encoder (PE):** `sam3/model/vitdet.py` (`ViT` trunk, patch 14 / img 1008 / depth 32,
  RoPE+abs-pos) · `sam3/model/necks.py` (`Sam3DualViTDetNeck`, `Sam3TriViTDetNeck` — FPN,
  scale_factors `[4,2,1,0.5]`, `scalp=1`) · `sam3/model/position_encoding.py`
  (`PositionEmbeddingSine`) · `sam3/model/vl_combiner.py` (`SAM3VLBackbone` /
  `SAM3VLBackboneTri` — wraps vision+text, `forward_image`/`forward_text`).
- **Text encoder + tokenizer:** `sam3/model/text_encoder_ve.py` (`VETextEncoder`, width 1024 /
  24 layers) · `sam3/model/tokenizer_ve.py` (`SimpleTokenizer`, BPE
  `sam3/assets/bpe_simple_vocab_16e6.txt.gz`).
- **DETR detector + geometry + mask head + multiplex:** `sam3/model/decoder.py`
  (`TransformerDecoder`/`…Layer`, `SimpleRoPEAttention`, `DecoupledTransformerDecoderLayerv2`,
  presence token) · `sam3/model/encoder.py` (`TransformerEncoderFusion`, early VL fusion) ·
  `sam3/model/geometry_encoders.py` (`SequenceGeometryEncoder`, `Prompt`) ·
  `sam3/model/maskformer_segmentation.py` (`PixelDecoder`, `UniversalSegmentationHead`) ·
  `sam3/model/model_misc.py` (`DotProductScoring` = presence/box scoring, `MLP`,
  `TransformerWrapper`, `MultiheadAttentionWrapper`) · `sam3/model/multiplex_mask_decoder.py` ·
  `sam3/model/multiplex_utils.py` (`MultiplexController`). Detector model:
  `sam3/model/sam3_image.py` (`Sam3Image`, `Sam3ImageOnVideoMultiGPU`); image-predict API
  `sam3/model/sam3_image_processor.py` (`Sam3Processor`).
- **Tracker (SAM2-lineage):** `sam3/model/sam3_tracker_base.py` (`Sam3TrackerBase`,
  `sam_mask_decoder`, `sam_prompt_encoder`, `_forward_sam_heads`) ·
  `sam3/model/sam3_tracking_predictor.py` (`Sam3TrackerPredictor`) ·
  `sam3/model/sam3_tracker_utils.py` · `sam3/sam/rope.py` · `sam3/sam/transformer.py`
  (`RoPEAttention`) · `sam3/sam/mask_decoder.py` (`MaskDecoder`) · `sam3/sam/prompt_encoder.py`
  · `sam3/sam/common.py` · memory encoder `sam3/model/memory.py` (`SimpleMaskEncoder`,
  `SimpleFuser`, `SimpleMaskDownSampler`, `CXBlock`). (Memory is held in the tracker inference
  state — there is no separate pluggable "bank" file upstream.)
- **Association / tracklet lifecycle:** `sam3/perflib/associate_det_trk.py`
  (`_associate_det_trk_compilable`) + `sam3/model/sam3_video_base.py`
  (`Sam3VideoBase._associate_det_trk`, module-level `_associate_det_trk_compilable`,
  hotstart / masklet-confirmation lifecycle). Video orchestration:
  `sam3/model/sam3_video_inference.py` (`Sam3VideoInference[WithInstanceInteractivity]`) ·
  `sam3/model/sam3_video_predictor.py` (`Sam3VideoPredictor[MultiGPU]` — session/`handle_request`
  API; world_size 1 runs single-process, so forward hooks work) ·
  `sam3/model/sam3_base_predictor.py` (`Sam3BasePredictor` request dispatch).
- **Checkpoint loader / builders:** `sam3/model_builder.py` — `build_sam3_image_model`,
  `build_sam3_video_model`/`build_sam3_video_predictor` (base `sam3.pt`),
  `build_sam3_multiplex_video_*` (SAM 3.1 `sam3.1_multiplex.pt`), `download_ckpt_from_hf`.
  `sam3.pt` keys split under `detector.*` (1156) and `tracker.*` (309); the loader strips the
  `detector.` prefix for the image model.
