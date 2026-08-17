# SAM 3 integration — status & resume guide

**Branch:** `feat/sam3-integration` (135 commits ahead of `main`). Tree clean,
nothing pushed (44 ahead of `origin/feat/sam3-integration`). Latest: per-box
labels on `GeometryPrompt` (`54ef29d`) + the exemplar/VISUAL-slot retirement
(`8dd1364`).
**Last session:** 2026-08-17. **Read this first to resume.**

**Next up:** open item 1, multi-concept (`MAX_CONCEPTS>1`).

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
- **Output policy `Emit` (2026-08-17)** — closes the "buffered confirmation gate" item
  WITHOUT buffering. `forward()` keeps its synchronous contract (feed frame `f`, get
  frame `f`); the predictor gained `emit: Emit` (`CONFIRMED` default / `VISIBLE` /
  `ALIVE`, `sam/results.py`) and every `MaskletResult` now carries its
  `tracklet_state`, so a caller can layer its own display policy. `select_emitted`
  (module fn, `sam3_predictor.py`) applies the policy, always drops empty masks
  (upstream's unconditional `mask.any()`), and lets unmanaged (click-seeded) ids pass.
  `TrackletManager.confirmed_ids()` is live now via `emitted_ids()`.

  Parity reasoning: upstream's causal half ("show nothing until 3 consecutive
  detections") is now reproducible exactly, at zero latency, by running the default.
  Its non-causal half (the retroactive reveal of frames preceding a hotstart kill)
  stays out of reach for any streaming design — and out of reach for a streaming
  *consumer* too, which cannot un-draw a frame it already displayed. The goldens
  captured the observable of a non-causal pipeline (objects visible from their birth
  frame), so every golden-measuring test now sets `pred.emit = Emit.VISIBLE`; the
  `CONFIRMED` default is covered by `tests/test_emit_modes.py` (CPU).

- **Negative box labels (2026-08-17)** — `GeometryPrompt` gained `boxes_labels`
  (per-box sign, mirroring the existing `points_labels`); `_pack_geometry` forwards it
  instead of hardcoding `torch.ones`, so `Sam3GeometryEncoder.label_embed`
  (`nn.Embedding(2, d)`) row 0 is reachable on the box path. No `boxes_labels` -> the
  old all-positive default, byte-identical. Golden captured from upstream
  `add_geometric_prompt(box, label=0)` via `capture_sam3_box_golden.py --label 0`
  (the script's `BOX_LABEL` constant became a `--label {0,1}` arg; stem
  `box_prompt` / `box_prompt_neg`). `test_sam3_box_prompt_parity` is now
  parametrized over both stems.

  Strongly discriminative, and the semantics are exactly "not this one": the same box
  at `[300,150,470,420]` on "person" gives 3 dets / presence 0.99999 when positive
  (incl. the boxed person at `[302.5,158.9,468.7,412.4]`, score 0.914) and 2 dets /
  presence 0.8606 when negative, with that detection gone. Falsified the test by
  re-hardcoding `ones`: `box_prompt_neg` fails `3 detections vs golden 2` while
  `box_prompt` stays green.

- **Exemplar / VISUAL slot — removed, not implemented (2026-08-17).** Same disposition
  as `negative_phrases`, for a sharper reason: the slot's *consumer* is live but it has
  NO producer. `sam3_image.py:205` concatenates
  `[txt_feats, geo_feats, visual_prompt_embed]`, and that prompt feeds both the VL
  encoder and `DotProductScoring.mean_pool_text` (`model_misc.py:734-741`, which pools
  EVERY valid token) — a tensor placed there really does move `pred_logits`. But nothing
  builds one: the sole live `_encode_prompt` call (`sam3_image.py:449`) passes 3
  positional args, the `sam3_video_base.py:1982` wrapper has zero callers, and all 4
  assignments to `inference_state["visual_prompt_embed"]` are `= None`. No encoder module
  exists for it. Training doesn't fill it either — `TextQueryToVisual`
  (`train/transforms/filter_query_transforms.py:532-567`) implements "image exemplar" as
  `input_bbox` + caption `"visual"`, i.e. the GEOMETRIC slot, which is 2a/2b. The
  `sam3_video_inference.py:177` comment ("a single visual prompt embedding is shared for
  all frames") reads as scaffolding for CROSS-IMAGE exemplars, unreleased. Implementing it
  would feed untrained input to a live scorer — worse than a no-op.
  Dropped `ConceptPrompt.exemplars`, both `encode_exemplars` stubs, `ConceptState
  .exemplar_emb`, and the `exemplar_emb` parameter from `detect` / `forward_grounding`
  (with its deferral assert). Reference geometry is a `GeometryPrompt` box/point.
  Mask exemplars were separately, permanently out — 0 `mask_encoder` keys in BOTH
  checkpoints.

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

1. **Multi-concept (`MAX_CONCEPTS>1`)** — loop the detector over concepts + merge.
2. **Geometry-prompt bit-exact parity** — image box path matches upstream at
   box-IoU ≥ 0.8 / score atol 0.06, looser than text-only's 2px/1e-2 (geometry
   tokens lengthen the decoder → bf16 drift). Investigate if bit-exact is wanted.
3. **Pre-existing:** `tests/parity/test_sam3_parity.py::test_encoder_parity` (vision
   encoder pyramid) fails vs golden — fails on `HEAD~` too (NOT this work); stale
   golden / env drift. Recapture or retolerance. Re-confirmed by stash-and-rerun on
   **2026-08-17** (identical failure with a clean tree).
4. **Base (`sam3.pt`) video box prompt** — 2b targeted the mux path; mirror on base.
5. **Minor cleanups** (low value): point-normalization dup across `_pack_geometry` /
   `_build_mux_point_inputs`; small scale-tensor rebuilds on prompt frames.
6. **Stale doc:** `notebooks/sam3_video_predictor_example.ipynb` cell 0 still claims
   text+click co-seed, mid-stream add and box prompts raise `NotImplementedError`.
   All three ship now (1a/1b/2a/2b). Pre-existing; not touched.

### Closed by the `Emit` policy — the retroactive-reveal residue

Upstream's multiplex hide-set is `unconfirmed(min(f + thresh-1, last)) ∪ empty-mask`,
nothing else: keep-alive suppression is dead code there (`to_suppress_mask` never
consumed; `suppressed_obj_ids` only written by the CPU `_process_hotstart`, never
called in the multiplex path). `Emit.CONFIRMED` + the unconditional empty-mask drop
now reproduce that rule *causally*. What remains unreproducible is the LOOKAHEAD:
upstream buffers `propagate_in_video` output by `hotstart_delay` (15) frames and
snapshots the removed set on the way out, so a hotstart kill retroactively reveals the
frames preceding the death (the frames 5-7 gap in `test_sam3p1_video_box_parity`).
Deliberately not implemented: it would cost a 15-frame output lag, and a streaming
consumer cannot act on it anyway (a frame already shown cannot be un-shown).

Upstream references if this is ever revisited: `sam3_multiplex_tracking.py`
`_postprocess_output` (~703-714, `keep = masks.any()` then `obj_ids_to_hide`) fed by
the `propagate_in_video` hotstart buffer (~336-390, `unconfirmed_status_delay =
thresh - 1`, clamped to `num_frames - 1`); status updated in `sam3_multiplex_base.py`
~2786-2802 (sticky once CONFIRMED, counter resets on a miss);
`masklet_confirmation_enable=True` + `thresh=3` from the demo builder
(`model_builder.py` ~1184). Re-verify any hypothesis with
`tests/parity/reference_sam3/debug_sam3p1_video_box_hotstart.py` before coding — two
plausible stories (empty masks vs. lifecycle gate) were indistinguishable from the
golden alone, and only the instrumented rerun separated them.

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
tests/test_sam3p1_point_inputs.py tests/characterization/test_sam3_build.py
tests/test_emit_modes.py -q`.

The 7-file GPU gate does NOT cover `tests/parity/conftest.py::run_streaming_parity`
(5 video-parity tests live outside it). After touching that fixture also run
`pytest tests/parity/test_sam3_parity.py -k video_parity
tests/parity/reference_efficientsam3/ -q`.

The efficientsam3 goldens are all committed under
`tests/parity/reference_efficientsam3/golden/`; what gates those tests is the
CHECKPOINT, fetchable from the public HF repo `Simon7108528/EfficientSAM3`:
`pixi run python tools/download_efficientsam3.py --variant <v>` (variants
`video-{repvit,tinyvit,efficientvit}-m`, `sam3p1-repvit-m-s0-ctx16`,
`litetext-s0-ctx16`, `sam3p1-litetext-s0-ctx16`). Only `video-repvit-m` (1.6 GB) is
present on this host.

**EfficientSAM3 is an image/detection model, not a video tracker.** Upstream released
the Stage 1 encoder distillation but NEVER Stage 2 (memory-bank alignment on SA-V —
unchecked on its roadmap), and every video checkpoint here is Stage 1 lineage, so the
tracker propagates features it was never trained on. Frame-0 detection matches ≥0.99
everywhere; propagation drifts (base lineage 1-4%, SAM3.1 RepViT-M down to min 0.7412)
against efficientsam3's OWN native reference with strict-loaded identical weights. So
the `xfail(strict=True)` markers are expected-by-construction, NOT triage items — no
change in `sam/` can close them. Full write-up:
`tests/parity/reference_efficientsam3/README.md` (top section). The LiteText fixtures
are the exception — they swap only the TEXT tower and keep PE vision, so they pass.

Reading those xfails: `strict=True` means ANY exception reads as XFAIL — an import
error in your own change looks identical to the documented drift. Use `--runxfail` to
see the real assertion. Baselined on 2026-08-17 (stash the working tree, rerun):
`test_efficientsam3_video_parity[repvit]` fails at `frame 3: count 0 != golden 2` after
clean frames 0-2, identically with and without the `Emit` work. Under `Emit.ALIVE` all
4 frames track (frame 3 IoU 0.9775/0.991) and the failure becomes the documented IoU
gate (min 0.9575 / mean 0.9842) — i.e. keep-alive suppression hides the objects once
detection stops matching them; tracking itself is not lost.

## Key code map

- `sam/models/sam3_predictor.py` — predictors. Mux video `forward` (orchestration),
  `_seed_mux_state` / `_grow_mux_state` / `_seed_multiplex` / `_seed_points_multiplex`,
  `_detector_add` / `_clicks_add`, `_detect(concept, geo=)`, `_placeholder_concept`,
  `_purge_removed`, module fns `select_emitted` (output policy + empty-mask drop +
  `tracklet_state` stamp) / `_build_mux_point_inputs` / `_pack_geometry`.
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
