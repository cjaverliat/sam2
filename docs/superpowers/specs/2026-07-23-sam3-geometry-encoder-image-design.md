# SAM 3 — geometry-prompt encoder + image box/point prompts (Feature 2a)

**Date:** 2026-07-23
**Status:** design (fast-track; user approved "image+video", decomposed 2a-then-2b)
**Reference:** upstream `facebookresearch/sam3` @ `5dd401d` at `../sam3_reference/`
**Blueprint:** upstream `geometry_encoders.py::SequenceGeometryEncoder` + `sam3_image.py::_encode_prompt`.

## Context

`ConceptPrompt.exemplars` and box geometry prompts are deferred: our
`Sam3GeometryEncoder.forward` (`detr_decoder.py:929`) runs only the **CLS-token**
path (ignores geometry), and `forward_grounding` asserts `exemplar_emb is None`. But
all 76 geometry-encoder weights **already load** in both `sam3.pt` and
`sam3.1_multiplex.pt` — the box/point projection submodules are built and dormant.
Neither checkpoint ships a `mask_encoder` (0 keys), so **mask geometry is
unsupported** (raises); this feature activates **box + point** encoding.

2a is the shared foundation (the encoder + image path); 2b (video box routing) builds
on it.

## Key mechanism (from the blueprint)

`num_labels==2` ⇒ `encode_boxes_as_points=False` ⇒ a plain box goes the **ROI path**,
and every active sub-encoder is **summed**:

- **Point** (`_encode_points`, normalized xy `(N,B,2)`): `points_direct_project` (Linear
  2→C) + `points_pool_project` (grid_sample the NCHW image feature at the point, Linear
  C→C) + `points_pos_enc_project` (sine `_encode_xy`, Linear C→C) + `label_embed(label)`.
- **Box** (`_encode_boxes`, normalized cxcywh `(N,B,4)`): `boxes_direct_project` (Linear
  4→C) + `boxes_pool_project` (roi_align the NCHW feature at the box → 7×7 → Conv2d
  collapse) + `boxes_pos_enc_project` (sine `encode_boxes`, Linear 258→C) +
  `label_embed(label)`.
- Tokens are right-padded-concatenated (points ‖ boxes ‖ cls), `final_proj`+`norm`, then
  3 `encode` cross-attn layers over the image + `encode_norm` → `(geo_feats, geo_mask)`.
  Token count = `Npoints + Nboxes + 1(cls)` (was always 1 for text-only).

`forward_grounding` concatenates along the sequence axis: `prompt =
cat([text_emb, geo_feats], 0)`, `prompt_mask = cat([text_mask, geo_mask], 1)`.
Everything downstream (VL encoder, 200-query decoder, dot-product scorer, mask head) is
prompt-length-agnostic — no change. The pooled prompt (text+geo) shifts the class/
presence scores, which is exactly how a box/point biases the detection.

Pure box/point (this feature) uses the **GEOMETRIC slot only**; the VISUAL/exemplar
slot stays empty (unchanged default). The image `pos_enc` (`PositionEmbeddingSine`,
`position_encoding.py:54,74`) API matches upstream; `forward_grounding` already exposes
the feature `(h, w)`.

## Scope

**In:**
- `Sam3GeometryEncoder.forward` rewrite: accept an optional geometry prompt
  (`box_coords` cxcywh-norm `(N,B,4)` + `box_labels`, `point_coords` xy-norm `(N,B,2)`
  + `point_labels`) and `img_sizes=[(h,w)]`; port `_encode_points`, `_encode_boxes`
  (ROI), `concat_padded_sequences`, cls append, `final_proj`/`norm`/`encode`/`encode_norm`.
  Wire a `PositionEmbeddingSine` into `__init__`. **Null prompt ⇒ CLS-only, bit-identical
  to today** (protects the existing image parity gate).
- `Sam3DetrDetector.forward_grounding` / `detect`: drop the `exemplar_emb is None`
  assert; accept an optional `geo_prompt`; concat text+geo.
- `Sam3Predictor.predict` / `Sam3MultiplexPredictor.predict`: accept a `GeometryPrompt`
  (boxes and/or points) → pack into the encoder's geometry inputs → `forward_grounding`.
- A box-format helper: `GeometryPrompt.boxes` xyxy (pixel or `is_normalized`) →
  cxcywh-normalized `(N,B,4)`; points `(N,2)`+labels → xy-norm `(N,B,2)`+`(N,B)`.

**Out (raise/defer):**
- Mask geometry / mask exemplars — no weights in either checkpoint.
- Exemplar (VISUAL slot) supplementing text — deferred; only the GEOMETRIC slot here.
- Video box routing — Feature 2b.

## Architecture

The encoder becomes a real geometry encoder; `forward_grounding` gains one optional
argument; `predict`/`detect` gain a `geometry` argument and a packing step. The
text-only call site (`geo_prompt=None`) reduces to today's CLS-only path — the
regression guard.

Data flow (image, pure box):
```
GeometryPrompt(boxes xyxy) --pack--> box_coords cxcywh-norm (N,1,4), labels (N,1)
Sam3Predictor.predict(image, ConceptPrompt(text), geometry=prompt)
  encode_image -> feats,pos ; encode_text -> text_emb,text_mask
  detector.forward_grounding(feats, pos, text_emb, text_mask, geo_prompt=packed)
    geometry_encoder(box_coords,..., img_sizes=[(h,w)]) -> geo_feats,geo_mask
    prompt = cat([text_emb, geo_feats]) -> encoder -> decoder -> scores/boxes/masks
  -> Sam3DetectionResult
```

## Parity gate

Golden captured in `../sam3_reference/.venv` (reuse the interactive/model-find capture
harness + patches), committed as fixtures:

- **Image box-prompt detect**: a fixed image (a bedroom/dance frame), a text phrase +
  one box prompt (normalized), compared to our `predict`/`detect`:
  boxes `atol=2px`, scores `atol=1e-2`, top-mask IoU ≥ 0.99, presence `atol=1e-2`
  (phase1 detector tolerances). A point-prompt variant if cheap.
- **Text-only regression**: the existing image parity fixture still passes bit-for-
  tolerance (null geo path unchanged).

## Testing (TDD)

1. **Geometry-encoder unit** (GPU) — box-only and point-only `forward` produce
   `token count == N + 1` and finite features; optionally compare `(geo_feats, geo_mask)`
   to an upstream capture. A **null-prompt** case asserts the output equals the current
   CLS-only path (bit-identical regression).
2. **Box-format helper unit** (CPU) — xyxy→cxcywh-norm and point normalization exact.
3. **Image box-prompt parity** — `test_sam3_box_prompt_parity`: our `predict` vs the
   golden (boxes/scores/mask IoU/presence).
4. **Regression** — existing `tests/parity/test_sam3_parity.py` image detect (text-only)
   still passes; sam3.1 mux text/interactive/model-find unaffected.

## Files

- `sam/modeling/decoders/detr_decoder.py` — `Sam3GeometryEncoder.__init__/forward`
  (pos_enc + box/point encode + concat), `forward_grounding` (geo_prompt), `detect`.
- `sam/models/sam3_predictor.py` — `predict`/`detect` accept `geometry`; a
  `_pack_geometry(prompt, image_hw, device)` helper (xyxy→cxcywh-norm, point norm).
- `tests/parity/reference_sam3/capture_sam3_box_golden.py` + fixtures;
  `tests/parity/test_sam3_box_prompt_parity.py`; `tests/test_geometry_encoder.py`.

## Risks

1. **Encoder numerics under bf16** — roi_align (`.float()` cast) + grid_sample + the
   summed sub-encoders must match upstream within the detector tolerances; the ROI
   denormalization (cxcywh→xyxy→×[W,H,W,H] feature px) is easy to get wrong. Highest;
   covered by the parity gate + a direct `(geo_feats)` unit if needed.
2. **Null-prompt regression** — the text-only path must stay bit-identical; guarded by a
   unit + the existing image parity fixture.
3. **Box format** — `GeometryPrompt.boxes` is xyxy; upstream wants cxcywh-normalized.
   Centralize the conversion at the packing boundary; unit-tested.

## CLAUDE.md fit

Activates already-loaded weights (no new checkpoint deps); the encoder gains one clear
responsibility (encode a geometry prompt) and stays pure-tensor. Text-only reduces to
the existing path (no behavior change for current callers). Mask geometry and the
exemplar slot fail loud / stay out of scope rather than half-working.
