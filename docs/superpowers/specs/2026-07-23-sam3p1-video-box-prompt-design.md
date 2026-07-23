# SAM 3.1 multiplex — video box prompt (Feature 2b)

**Date:** 2026-07-23
**Status:** design (fast-track; user: max upstream parity, keep our streaming/memory arch)
**Reference:** upstream `facebookresearch/sam3` @ `5dd401d` at `../sam3_reference/`

## Context

2a activated the box/point geometry encoder and the image `predict`/`detect` box
path. 2b extends box prompts to the **streaming video** predictor (multiplex, sam3.1).

Upstream behavior (verified, `sam3_multiplex_tracking.py:1706-1728`): a box
`add_prompt` is **not** obj_id-addressed. It records a **per-frame geometric prompt**
for that frame; the detector runs that frame with the box in the GEOMETRIC slot
(caption `"geometric"`, `find_text_batch=["<text placeholder>","visual","geometric"]`
`:115`); the resulting detections seed tracklets through **normal association**
(mask-IoU, fresh ids). It is a per-frame detection bias, not a distinct add-object
mechanism. Per the user directive: **replicate this behavior with max parity, but
integrate it into OUR streaming/forgetful-bank architecture** (reuse
`_associate_and_update` → `_seed_multiplex`/`_grow_mux_state`), not upstream's
session envelope.

## Goal

Support a box `GeometryPrompt` in the multiplex video `forward`: on the prompt frame,
the box biases the detector; detections seed/track via the existing pipeline.

## Scope

**In:**
- `_detect(det_feats, det_pos, concept, geo=None)` (mux) — pass `geo` to
  `forward_grounding` (2a threading).
- A placeholder `"geometric"` concept (`encode_text("geometric")`, cached on the
  predictor) used when a box prompt is present but no text concept is set.
- `forward` (mux): when the frame's `geometry_prompts` include **boxes**, pack them
  (2a `_pack_geometry`) and run `_detect(det_f, det_p, concept or placeholder, geo=box)`
  even if `concept is None`; the biased detection flows into the unchanged
  `_associate_and_update` → seed/grow pipeline (fresh ids, forgetful bank).
- Routing: **boxes → detector-geometry** (this path); **points → interactive click**
  (1a/1b, unchanged). A prompt carrying both routes each to its path.
- `_check_mux_geometry`: box no longer raises; mask still raises (no weights).

**Out:**
- Base (sam3.pt) video box prompt — parallel follow-on; 2b targets the sam3.1 mux
  path (where the geometry goldens live).
- Exemplar (VISUAL slot), negative_phrases, multi-concept — separate features.
- obj_id-addressed box add (not upstream's behavior).

## Architecture (our streaming, max behavioral parity)

No new architecture. `forward` gains a box branch that feeds the frame's detection;
downstream is the existing pipeline:

```
box GeometryPrompt on frame k
  -> _pack_geometry -> {box_coords cxcywh-norm, box_labels}
  -> det = _detect(det_f, det_p, concept or _placeholder_concept(), geo=box)
  -> _associate_and_update(det, ...)      # unchanged (mask-IoU, spawn/confirm/kill)
  -> _seed_multiplex / _grow_mux_state     # unchanged (our forgetful-bank sinks)
  -> propagate forward                     # constant-VRAM
```

Text + box on one frame: concept text in the TEXT slot + box in the GEOMETRIC slot,
one `_detect` pass (upstream semantics). Box only: placeholder `"geometric"` text +
box geo.

## Parity gate

Golden captured in `../sam3_reference/.venv` (reuse the harness + `--patches`),
committed as a fixture:

- **Video box add**: a bedroom (or dance) clip, one box prompt on frame 0 (no text),
  propagate forward N frames. Golden = per-frame `{obj_ids, masks}`.
- Our port streams the same; **id-agnostic** per-frame mask matching (mean IoU ≥ 0.95,
  count within 1) — matching the model-find gate style (association/id/timing nuances,
  and the 2a geometry bf16 drift, make strict bit-exact inappropriate).

## Testing (TDD)

1. **`_detect` geo passthrough unit** (GPU) — `_detect(..., geo=box)` returns detections
   biased toward the box region (non-empty, differs from text-only).
2. **Video box-add smoke** (GPU) — a box on frame 0 (no concept) spawns and tracks an
   object forward; a box prompt with `masks_logits` raises.
3. **Video box parity** — `test_sam3p1_video_box_parity` vs the golden (id-agnostic
   mean IoU ≥ 0.95, count within 1).
4. **Regression** — 1a/1b/re-ID/model-find/mux-text and the 2a image box parity still pass.
5. **Notebook** — a mux video box-prompt demo cell.

## Files

- `sam/models/sam3_predictor.py` — mux `_detect` (geo); a cached `_placeholder_concept()`;
  `forward` box branch + routing; `_check_mux_geometry` (drop box raise).
- `tests/parity/reference_sam3/capture_sam3p1_video_box_golden.py` + fixtures;
  `tests/parity/test_sam3p1_video_box_parity.py`; a `_detect`-geo unit + smoke in
  `tests/test_sam3p1_interactive_smoke.py`.
- `notebooks/sam3_video_predictor_example.ipynb` — box-prompt demo cell.
- Ledger.

## Risks

1. **Placeholder caption fidelity** — the `"geometric"` caption must tokenize/embed as
   upstream's; verified against the golden. If off, detections shift. Medium.
2. **Text+box coexistence** — combining concept text (TEXT slot) + box (GEOMETRIC slot)
   in one pass must match upstream's slot order; covered by a text+box golden variant
   if the box-only gate is insufficient.
3. **bf16 geometry drift** (inherited from 2a) — the id-agnostic IoU gate absorbs it.

## CLAUDE.md fit

Reuses the 2a encoder + the 1a/1b seed/grow sinks + the existing association pipeline;
the only new code is a placeholder concept + a `forward` box branch. Behavior matches
upstream; the architecture stays our streaming/forgetful-bank design (per the user
directive). Mask geometry and base-path video box fail loud / stay out of scope.
