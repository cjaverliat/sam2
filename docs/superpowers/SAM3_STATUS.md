# SAM 3 integration — status & resume guide

**Branch:** `feat/sam3-integration` (128 commits ahead of `main`, head `2e25c0b`).
Tree clean, nothing pushed.
**Last session:** 2026-08-17. **Read this first to resume.**

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
- **Hotstart visibility for box-only tracking (2026-08-17)** — the tracklet lifecycle
  now steps on every frame (`_advance_lifecycle`), not only when a detection ran, so a
  concept-less box session decays and can be killed like upstream. Click-seeded objects
  stay unmanaged and exempt (upstream runs click-only sessions through SAM 2 partial
  propagation, bypassing hotstart). Ledger has the measured trace.
- **Quality pass** — dedup (shared `_seed_mux_state` / `_masklets_from_demux` /
  tracker `masks_from_points` + `_interactive_high_res_features`), decomposed the
  mux `forward` god-method (`_split_and_pack_geometry` / `_detector_add` /
  `_clicks_add`), base→mux `_purge_removed` override hook, `_pack_geometry` →
  module fn, and skipped the full-res masklet build for suppressed tracklets.

- **`negative_phrases` — removed, not implemented.** Upstream has NO inference-time
  semantics for negatives: `SAM3VLBackbone.forward_text` encodes an optional
  `additional_text` and exports `additional_text_features`, but **no caller passes it
  and nothing reads it** (all 16 hits live in `vl_combiner.py`; `VisionOnly.forward_text`
  ignores the arg outright). Upstream's negative supervision is a different mechanism —
  dataset-level `include_negatives` (a caption with zero GT instances trains the presence
  head to say "absent"), not a per-concept negative list. No head accepts a
  negative-caption input, so honouring them would mean inventing untrained behaviour.
  Dropped the field from `ConceptPrompt`; both `encode_text` are now plain single-phrase
  encodes (bit-identical — the old batch was already `[text]` with `n_pos=1`).
  Re-verified against `origin/main` `8f0b7f4` (2026-08-13) on **2026-08-17**: unchanged.

## Open (recommended order)

1. **Buffered confirmation gate** — upstream's multiplex hide-set is
   `unconfirmed(min(f + thresh-1, last)) ∪ empty-mask`, nothing else: keep-alive
   suppression is dead code there (`to_suppress_mask` never consumed;
   `suppressed_obj_ids` only written by the CPU `_process_hotstart`, never called
   in the multiplex path). Our `keep_alive > 0` rule agrees with it on the frames
   we test, but by coincidence, not mechanism. Matching it means gating on
   CONFIRMED with a `thresh-1` lookahead, i.e. buffering `forward()` output by 2
   frames — an output-contract change. Would also close the frames 5-7 gap in
   `test_sam3p1_video_box_parity` (upstream reveals a hotstart-killed object for
   the frames preceding its death; we keep it hidden).

   Resume notes: upstream's rule is assembled in `sam3_multiplex_tracking.py`
   `_postprocess_output` (~704-714, the `obj_ids_to_hide` list) fed by the
   `propagate_in_video` hotstart buffer (~336-390, `unconfirmed_status_delay =
   thresh - 1`, clamped to `num_frames - 1`); status itself is updated in
   `sam3_multiplex_base.py` ~2786-2802 (sticky once CONFIRMED, counter resets on a
   miss). `masklet_confirmation_enable=True` + `thresh=3` come from the demo builder
   (`model_builder.py` ~1184). Our side already tracks the counter
   (`TrackletManager.consecutive_det_count` -> `TrackletState.CONFIRMED`,
   `confirmed_ids()`) — it is computed and currently unused, so the work is the
   output path, not the state machine. **This decision is open, not settled** —
   whether to take the 2-frame latency at all is a judgement call for the owner.
   Re-verify any hypothesis with
   `tests/parity/reference_sam3/debug_sam3p1_video_box_hotstart.py` before coding;
   two plausible stories (empty masks vs. lifecycle gate) were indistinguishable
   from the golden alone last time, and only the instrumented rerun separated them.
2. **Exemplar (VISUAL slot)** — reference box/mask supplementing a text concept.
   Reuses the 2a encoder; wire the VISUAL slot in `forward_grounding`. NOTE: mask
   exemplars are permanently out — 0 `mask_encoder` keys in BOTH checkpoints.
3. **Multi-concept (`MAX_CONCEPTS>1`)** — loop the detector over concepts + merge.
4. **Geometry-prompt bit-exact parity** — image box path matches upstream at
   box-IoU ≥ 0.8 / score atol 0.06, looser than text-only's 2px/1e-2 (geometry
   tokens lengthen the decoder → bf16 drift). Investigate if bit-exact is wanted.
5. **Pre-existing:** `tests/parity/test_sam3_parity.py::test_encoder_parity` (vision
   encoder pyramid) fails vs golden — fails on `HEAD~` too (NOT this work); stale
   golden / env drift. Recapture or retolerance.
6. **Base (`sam3.pt`) video box prompt** — 2b targeted the mux path; mirror on base.
7. **Minor cleanups** (low value): point-normalization dup across `_pack_geometry` /
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
  keep-alive suppress; `removed/alive/visible/managed_ids`). `step()` is driven by
  `_advance_lifecycle` in the predictor, which runs on EVERY frame — including
  frames with no detection pass (all managed tracklets count as unmatched then).
  Only *managed* ids step: click-seeded objects are intentionally never registered,
  which is what keeps them out of hotstart (upstream parity — see the ledger).
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
  `tests/parity/reference_sam3/`. That dir also holds
  `debug_sam3p1_video_box_hotstart.py`, an instrumented (write-nothing) rerun that
  prints upstream's per-frame hide sets, keep-alive / unmatch counters and
  confirmation status — use it before theorising about visibility behaviour.
- A golden's empty frame does NOT tell you *why* upstream hid an object (empty mask?
  suppressed? unconfirmed? removed?). Instrument before fixing — the box-only
  visibility item had two equally plausible explanations that only the rerun split.
