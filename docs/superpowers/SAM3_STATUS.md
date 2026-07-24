# SAM 3 integration — status & resume guide

**Branch:** `feat/sam3-integration` (125 commits ahead of `main`). Tree clean.
**Last session:** 2026-07-23/24. **Read this first to resume.**

Upstream reference for parity is `../sam3_reference` (facebook `sam3` @ `5dd401d`).
See the memories `sam3-reference-envs` and `sam3-parity-architecture-preference`
(the directive: **max behavioral/numerical parity with upstream, but keep OUR
streaming / forgetful-bank / memory-efficient architecture** — don't copy upstream's
session model).

## Done (this line of work)

Ledger detail: `docs/superpowers/plans/2026-06-26-phase1-sam3-torch-inference.md`
(the "Post-phase1" section). Each feature has a spec + plan under
`docs/superpowers/{specs,plans}/2026-07-2*`.

- **1a — mux seed-frame point-click** add-object (interactive VOS, no text). Parity golden.
- **1b — dynamic mux-state growth** (`add_new_masks_to_existing_state`): detector
  mid-stream spawn (fixed a latent text-tracking crash) + interactive mid-stream
  click + text/click co-seed. Model-find parity on `dance`.
- **Tracklet re-ID** — re-entering object reuses its id (hotstart-gated kill +
  keep-alive suppress; upstream has NO re-association). Fixes base + mux.
- **2a — box/point geometry encoder (image)** — activated the dormant box/point
  encoders (`Sam3GeometryEncoder`, roi_align + grid_sample + pos-enc), threaded a
  `geo` dict through `forward_grounding`/`detect`/`predict`. Image box-prompt parity.
- **2b — video box prompt** — a box `GeometryPrompt` biases the prompt frame's
  detection (GEOMETRIC slot, `"<text placeholder>"` concept) and seeds via the
  existing association pipeline. Video box parity.
- **Quality pass** — dedup (shared `_seed_mux_state` / `_masklets_from_demux` /
  tracker `masks_from_points` + `_interactive_high_res_features`), decomposed the
  mux `forward` god-method (`_split_and_pack_geometry` / `_detector_add` /
  `_clicks_add`), base→mux `_purge_removed` override hook, `_pack_geometry` →
  module fn, and skipped the full-res masklet build for suppressed tracklets.

## Open (recommended order)

1. **`negative_phrases`** (smallest) — currently embedded then dropped in
   `Sam3Predictor.encode_text` / video `encode_text`; feed them into the detector
   presence/score head so they suppress matches.
2. **Hotstart visibility for box-only tracking** — upstream HIDES a box-seeded
   object during its `hotstart_delay=15` warm-up; ours SHOWS it (correct masks,
   wrong show/hide timing — see `test_sam3p1_video_box_parity` docstring, frames 1-4
   ungated). Fix: step the tracklet lifecycle every frame using the tracker
   object-score as the match signal, not only when a detection ran.
3. **Exemplar (VISUAL slot)** — reference box/mask supplementing a text concept.
   Reuses the 2a encoder; wire the VISUAL slot in `forward_grounding`. NOTE: mask
   exemplars are permanently out — 0 `mask_encoder` keys in BOTH checkpoints.
4. **Multi-concept (`MAX_CONCEPTS>1`)** — loop the detector over concepts + merge.
5. **Geometry-prompt bit-exact parity** — image box path matches upstream at
   box-IoU ≥ 0.8 / score atol 0.06, looser than text-only's 2px/1e-2 (geometry
   tokens lengthen the decoder → bf16 drift). Investigate if bit-exact is wanted.
6. **Pre-existing:** `tests/parity/test_sam3_parity.py::test_encoder_parity` (vision
   encoder pyramid) fails vs golden — fails on `HEAD~` too (NOT this work); stale
   golden / env drift. Recapture or retolerance.
7. **Base (`sam3.pt`) video box prompt** — 2b targeted the mux path; mirror on base.
8. **Minor cleanups** (low value): point-normalization dup across `_pack_geometry` /
   `_build_mux_point_inputs`; dead `TrackletManager.confirmed_ids()` + the CONFIRMED
   machinery if truly unused; small scale-tensor rebuilds on prompt frames.

## How to verify (regression gate)

GPU (RTX 3090) + pixi `notebooks` env (has matplotlib):
```
MPLBACKEND=Agg pixi run -e notebooks pytest \
  tests/parity/test_sam3p1_interactive_parity.py \
  tests/parity/test_sam3p1_modelfind_parity.py \
  tests/parity/test_sam3_box_prompt_parity.py \
  tests/parity/test_sam3p1_video_box_parity.py \
  tests/test_sam3p1_interactive_smoke.py tests/test_sam3p1_mux_growth.py \
  tests/test_geometry_encoder.py -q
# then revert the pixi.lock churn: git checkout pixi.lock
```
CPU-fast (no GPU): `pixi run pytest tests/test_tracklet_reid.py
tests/test_sam3p1_point_inputs.py tests/characterization/test_sam3_build.py -q`.

## Key code map

- `sam/models/sam3_predictor.py` — predictors. Mux video `forward` (orchestration),
  `_seed_mux_state` / `_grow_mux_state` / `_seed_multiplex` / `_seed_points_multiplex`,
  `_detector_add` / `_clicks_add`, `_detect(concept, geo=)`, `_placeholder_concept`,
  `_filter_visible`, `_purge_removed`, module fns `_build_mux_point_inputs` / `_pack_geometry`.
- `sam/modeling/tracking/sam3_multiplex_tracker.py` — `track_step`,
  `add_new_masks_to_existing_state`, `masks_from_points`, `_interactive_high_res_features`.
- `sam/modeling/association/tracklet.py` — `TrackletManager` (hotstart kill +
  keep-alive suppress; `removed/alive/visible/managed_ids`).
- `sam/modeling/decoders/detr_decoder.py` — `Sam3GeometryEncoder` (box/point),
  `_concat_padded_sequences`, `Sam3DetrDetector.forward_grounding(..., geo=)`.

## Gotchas

- **Never `sed`/`awk`/`python -c` for edits on this host** — BOM/escape corruption
  (repo rule; I slipped once on a doc — no harm that time, but don't).
- Reference-env captures: build the predictor then drive `predictor.model` directly
  (`init_state`/`add_prompt`/`propagate_in_video`) — `handle_request(start_session)`
  is broken at `5dd401d`. Needs `--patches` (edt stub + all-backend SDPA + CPU
  NMS/CC; triton/FA3 absent). Box-only prompt keeps the TEXT slot at the literal
  `"<text placeholder>"` (NOT `"geometric"`).
- Goldens/fixtures under `tests/parity/fixtures/sam3{,p1}/`; capture scripts under
  `tests/parity/reference_sam3/`.
