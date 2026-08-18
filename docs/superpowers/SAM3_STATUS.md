# SAM 3 integration — status & resume guide

**Branch:** `feat/sam3-integration`, pushed. Latest: the base interactive-click
lifecycle fix and its 30-frame golden, after the notebook box-prompt/dark-figure pass.
**Last session:** 2026-08-18. **Read this first to resume.**

**Next up:** open item 1, the box-seeded object's propagation drift — the only known
divergence left. Everything else is green: `tests/parity` 26 passed, 16 skipped,
1 xfailed; the rest of `tests/` 72 passed, 8 skipped.

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
  are exempt from that kill — on the mux path because they are never registered, on the
  base path via the force-confirm below. Ledger has the measured trace.
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

- **Prompt API aligned with SAM 2 + mask prompts (2026-08-18)** — `forward`'s prompt
  argument is now `prompts` on both SAM 3 lineages, the name `Sam2VideoPredictor.forward`
  already uses, so the interactive surface is one interface across SAM 2 and SAM 3
  (`GeometryPrompt` was already shared). Call sites updated in tests, README and the
  notebook; the dated specs/plans keep the old name as historical record.

  Mask prompts work on the base lineage now. They always could have:
  `_apply_geometry_prompt` builds `mask_inputs` and the tracker's
  `sam_prompt_encoder.mask_downscaling` weights ship in `sam3.pt` (10 tensors) — the same
  path a detector-seeded tracklet uses. But `_split_and_pack_geometry` filtered prompts
  into "has boxes" / "has points", so a mask-only prompt matched neither list and was
  dropped silently, returning `{}` with no error. It now splits into `box_geo` +
  `tracker_prompts` (points OR mask). Unchanged: a mask paired with a box raises, since
  that is the DETECTOR's mask slot (`mask_encoder`, 0 keys in both checkpoints), and the
  multiplex still rejects masks — its `_seed_mux_state` / `_grow_mux_state` take masks
  only from the detector, which is what its error message now says.
  Tests: `tests/test_sam3_mask_prompt.py`.

- **Base interactive-click lifecycle (2026-08-18)** — a click-seeded tracklet on the
  BASE lineage was registered like a detected one, so in a click-only session (detection
  gated off, nothing can ever re-match it) the hotstart kill purged it at frame 8: the
  notebook's click demo showed a mask on frame 0 and nothing on frames 15/29. Upstream
  keeps it for the whole clip — `add_tracker_new_points` force-confirms the object
  (`masklet_confirmation` status 1, `consecutive_det_num` at threshold,
  `sam3_video_inference.py:1522-1531`) and `_process_hotstart` only ever considers ids
  the DETECTOR registered. `TrackletManager` gained an `interactive` flag
  (`spawn(..., interactive=True)`, `force_confirm()`) that confirms the tracklet and
  exempts it from the kill; refining an existing object with a click force-confirms it
  too. New 30-frame golden `fixtures/sam3/interactive_noconcept.npz` +
  `reference_sam3/capture_sam3_interactive_golden.py`, gated by
  `tests/parity/test_sam3_interactive_parity.py` (30/30 frames, min IoU ≥ 0.90,
  mean ≥ 0.95).

  Capture gotcha worth keeping: the base `add_prompt` has no `clear_old_points`, so
  driving the second (cache-seeding) pass with a repeated click hits
  `use_stateless_refinement`, which removes and re-adds the object — the re-seeded click
  then resolves to the girl's skirt (8736 px vs 31889 px for the identical click) and
  that fragment propagates. The capture re-propagates with NO second click instead.

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

- **Multi-concept — removed, not implemented (2026-08-18).** Upstream has NO concurrent
  multi-concept. Its only multi-concept path (`sam3_multiplex_tracking.py` `forward`,
  ~1756-1815, labelled "only used for benchmark eval (not used in the demo)") loops the
  phrases and for each one calls `add_prompt(frame_idx=0, text_str=prompt)`, propagates the
  WHOLE video, offsets the obj ids by `max(seen)+1`, then `reset_state`s. Concepts therefore
  never interact: no shared association, no cross-concept dedup/NMS, separate id spaces, and
  an object matching two phrases is emitted twice with two ids.
  That IS N independent sessions — already supported today with no new code: one
  `Sam3VideoPredictorState` per concept, feed the video once per state, merge with an id
  offset. Same cost as upstream, which also re-encodes the video per phrase.
  Sharing ONE session across concepts would NOT be upstream-equivalent: `associate_det_trk`
  matches all dets × all tracks, so concept A's detection could capture concept B's tracklet
  (upstream can never do that), and multiplex bucket memory is a joint K-object encoding, so
  co-bucketing two concepts' objects perturbs their masks.
  Retired the scaffolding that existed only for the concurrent variant: `MAX_CONCEPTS`,
  `ConceptState.concept_id` (always 0), and `Sam3VideoPredictorState.concepts: list` ->
  `concept: ConceptState | None` — which kills the `concepts[0] if concepts else None` dance
  in both `forward`s. `set_concept` keeps both guards (pre-roll-only + already-set) and still
  returns 0; the N-sessions recipe lives in its docstring. Behaviour unchanged (a pure field
  rename behind `set_concept`): CPU-fast set green (28 passed) and the full 7-file GPU gate
  green (20 passed).

- **Geometry-prompt bit-exact parity (2026-08-18)** — it already WAS bit-exact. The loose
  gates (box-IoU ≥ 0.8 / score atol 0.06 / mask IoU ≥ 0.85) paid for two convention
  mismatches in the test, and the old docstring's story ("geometry tokens lengthen the
  decoder → bf16 drift") was wrong.
  1. **Preprocessing.** The golden was captured through `model.init_state(resource_path=...)`
     — upstream's image-FOLDER video loader (`io_utils._load_img_as_tensor`: PIL CPU resize
     → float16), which `preprocess_to_1008_video` mirrors — while the test drives `predict()`,
     whose `preprocess_to_1008` mirrors the IMAGE api (`Sam3Processor`: uint8 → GPU →
     `v2.Resize(1008)` → float32). Running our weights on the golden's own regime made every
     score **bit-identical** (dp = 0.0, all 5 dets, both stems), isolating the resize as the
     entire cause. Attribution of that resize delta (960x540 → 1008², measured): CPU-vs-GPU
     uint8 rounding dominates (mean 7.7e-4, 9.8% of pixels > 1e-3, max 7.8e-3 = 2/255);
     antialias is nearly a no-op because this is an UPSCALE (mean 1e-5) — it would matter for
     a source larger than 1008; PIL ≈ torchvision-CPU (mean 4e-5); fp16 storage adds 4.9e-4.
  2. **Box convention.** The npz `boxes` are raw DETR `pred_boxes_xyxy`; `predict()` returns
     `masks_to_boxes` of the output mask (multiplex demo semantics). Up to 15.7px apart on the
     same detection — the whole reason a 0.80 box-IoU gate was needed.
  Fix: recaptured both goldens in the image regime (`capture_sam3_box_golden.py` now overwrites
  `input_batch.img_batch` with the `Sam3Processor` tensor; two reference-env runs, `--label 1`
  and `--label 0`), and the test re-derives the golden box from the golden MASKS. Residual vs
  upstream: scores 0.0 on all 3 `box_prompt` dets / 3.8e-3 on one `box_prompt_neg` det,
  presence ≤ 9.3e-4, mask-derived boxes **0.00px**, mask IoU ≥ 0.9898. Gates are now the
  text-only image bar — 2px / 1e-2 / 1e-2, mask IoU ≥ 0.98 — and the test adopts the capture's
  regime via the `determinism_no_det_algos` fixture (it previously set none).
  Ruled out with measurements, not argument: SDPA-kernel choice (|Δlogit| ≤ 0.04 against a
  0.23 gap) and structural mask disagreement (100% of differing pixels had |logit| < 0.28,
  median 0.03 — pure sign-flip at the decision boundary). Falsified by pointing the tightened
  test at the OLD goldens: `box_prompt` fails `mask IoU 0.9299 < 0.98`.

- **Base video predictor now preprocesses in the VIDEO regime (2026-08-18)** — the same
  class of mismatch as the box item, but in shipped code: `Sam3VideoPredictor.forward`
  used `preprocess_to_1008` (the image/`Sam3Processor` regime) while upstream's video path
  only ever loads frames through the image-folder loader, which is what the mux `forward`
  already mirrored via `preprocess_to_1008_video`. `capture_sam3_golden.capture_video`
  writes PNGs and drives `init_state`, so the base video golden is in the loader's regime
  too. Switched `forward` (and the two explicit preprocessing sites in
  `test_sam3_parity.py`, `_sam2_pyramid` / `_pyramids`, both of which measure video
  goldens). Effect on the base video golden: per-object IoU mean 0.9938 -> 0.9944, n>=0.99
  13/16 -> 15/16, min ~unchanged (0.9854 -> 0.9846) — small, and it does NOT explain the
  one hard object at ~0.985, which sits at 0.9846 in the golden's own regime (confirming
  the existing "detector-seed limit, not a streaming defect" note). Wider gate green:
  `test_sam3_parity.py` + `reference_efficientsam3/` = 10 passed, 1 xfailed, 16 skipped
  (checkpoints absent), no XPASS, with only the pre-existing `test_encoder_parity` failing.

- **`test_encoder_parity` fixed — the golden WAS stale (2026-08-18)** — and it was stale,
  not wrong: re-running TODAY's upstream encoder on the same input in the reference env
  (`Sam3Processor.set_image`, bf16) reproduced OUR output **bit-exactly**
  (`np.array_equal` True), while both differed from the committed golden by the identical
  amount (max 0.7363, p99.9 0.1172, median 0.0039, 16.8% of elements > 1e-2). So the
  fixture predated an env/kernel change on this host. Corroborating measurement: flipping
  our own SDPA backend to `MATH` moves the encoder output by max 0.785 / median 1.1%
  relative — the same magnitude as the gap — so no fixed golden survives `atol=1e-2` on a
  deep bf16 ViT across kernel selection, and re-tolerancing would have hidden this instead
  of fixing it. NOT fp16 storage: |golden| tops out at 6.0, whose fp16 ulp is 0.004.
  Recaptured all four goldens from one env at the same upstream commit `5dd401d`
  (`image.npz`, `video.npz`, `image_sam31.npz`, `video_sam31.npz` + `scenario.json`);
  the encoder is now bit-exact (max delta 0.0000) and the whole suite is green:
  `test_sam3_parity.py` + `reference_efficientsam3/` = **11 passed, 1 xfailed, 16 skipped,
  0 failed**. Fixture deltas from the refresh were small (boxes 0.024px, scores 3.9e-3,
  presence 2.1e-3; `text_emb` unchanged — the text tower is env-stable).

- **Base video box prompt (2026-08-18)** — closes the "mirror 2b on base" item, and turned up
  two lineage differences we had been applying the multiplex's answer to.
  Boxes previously went to the SAM 2 tracker as corner points (labels 2/3); upstream never
  does that — `sam3_video_inference._get_visual_prompt` (181-222) stores the first box on a
  fresh frame as that frame's GEOMETRIC prompt, so it biases DETECTION and the boxed instance
  seeds a tracklet through association, exactly like the mux path. (`visual_prompt_embed`
  stays None there — the helper's "visual prompt" name is about UI provenance, not the VISUAL
  slot, so it does not disturb the exemplar finding.)
  1. **Caption.** A box-only `add_prompt` takes the `else` branch and selects
     `TEXT_ID_FOR_VISUAL` (`sam3_video_inference.py:868-876`), i.e. the encoded caption is
     `find_text_batch[1]` = the literal **`"visual"`** — NOT the multiplex's
     `"<text placeholder>"` (`sam3_multiplex_tracking.py:1698-1705` has no else branch, so its
     `text_ids` stay 0). With the wrong caption frame 0 finds 1 detection instead of 2 and the
     seed mask sits at IoU 0.64; with the right one, 0.9854 and scores 0.906/0.523 against
     upstream's 0.906/0.524. Now a per-lineage `BOX_ONLY_CAPTION`.
  2. **Every-frame detection.** `add_prompt` writes that text id into EVERY frame's
     `find_inputs`, so a box-only session keeps detecting after the prompt frame instead of
     propagating blind. `_concept_for_detection` adopts the placeholder into the state.
  3. **Lifecycle.** `TrackletManager`'s defaults are the MULTIPLEX class defaults
     (`sam3_multiplex_base.py:228-230`, keep-alive 0/8/-4) plus the demo builder's
     `masklet_confirmation_enable=True`; the base video builder uses **30/30/-1 with
     confirmation disabled** (`model_builder.py:746-762`). We were applying the mux constants
     to both lineages, which hid both objects from frame 1. `TrackletManager` gained
     `confirmation_enable` + `configure()`, and each predictor declares its own `LIFECYCLE`,
     applied on the state's first frame (the caller builds the state and cannot know the
     lineage).
  Golden: `capture_sam3_video_box_golden.py` (base `build_sam3_video_model`, bedroom, 8
  frames) -> `fixtures/sam3/video_box.npz`; gate `tests/parity/test_sam3_video_box_parity.py`
  asserts the object count every frame, the frame-0 seed mask (>= 0.95) and the static
  object (>= 0.99). The moving object's propagation drift is open item 1.

## Open (recommended order)

1. **Box-seeded object drifts while propagating (base lineage)** — the ONE ungated part of
   the base video box work. Per-frame IoU of the moving box-seeded object vs the golden:
   0.985, 0.943, 0.744, 0.670, 0.842, 0.741, 0.370, 0.844 (mean 0.88). Bounded diagnosis:
   NOT base propagation in general (same predictor, same 8 bedroom frames, TEXT concept
   "person" -> min IoU **0.9960** / mean 0.9980 vs an upstream control capture), NOT
   detection (frame-0 scores 0.906/0.523 vs upstream 0.906/0.524, seed mask IoU 0.9854),
   NOT the lifecycle (object counts match every frame). Enabling the tracker's
   `use_memory_selection` — upstream's `apply_temporal_disambiguation=True`, which our
   forgetful bank otherwise supersedes (`build_sam.py:1124`) — recovers only part of it
   (mean 0.88 -> 0.91), so it is not the whole story either. Next suspects: upstream's
   `clear_non_cond_mem_around_input=True` (we have no equivalent) and the base model's
   `fill_hole_area=16`. The control capture script is
   `scratchpad/probe_text_bedroom.py`-style: `capture_sam3_video_box_golden.py` with
   `text_str="person"` instead of the box args.
2. **Minor cleanups** (low value): small scale-tensor rebuilds on prompt frames. (The
   point-normalization dup is done — `_normalized_points`.)

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
  tests/parity/test_sam3_video_box_parity.py \
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
  `_detector_add` / `_clicks_add`, `_detect(concept, geo=)`, `_placeholder_concept`
  (per-lineage `BOX_ONLY_CAPTION`), `_concept_for_detection` (box-only sessions adopt that
  caption so detection keeps running), `_split_and_pack_geometry` + `LIFECYCLE` (both on the
  BASE class, shared with the mux), `_purge_removed`, module fns `select_emitted` (output
  policy + empty-mask drop + `tracklet_state` stamp) / `_build_mux_point_inputs` /
  `_pack_geometry` / `_normalized_points`.
- `sam/modeling/tracking/sam3_multiplex_tracker.py` — `track_step`,
  `add_new_masks_to_existing_state`, `masks_from_points`, `_interactive_high_res_features`.
- `sam/modeling/association/tracklet.py` — `TrackletManager` (hotstart kill +
  keep-alive suppress; `removed/alive/visible/managed_ids`). Its constructor defaults are
  the MULTIPLEX lineage's; each predictor applies its own constants via `configure()` from
  its `LIFECYCLE` dict on the state's first frame (the base lineage disables confirmation
  and starts keep-alive at 30). `step()` is driven by
  `_advance_lifecycle` in the predictor, which runs on EVERY frame — including
  frames with no detection pass (all managed tracklets count as unmatched then).
  Only *managed* ids step. Click-seeded objects are kept out of the hotstart kill two
  different ways: the mux path never registers them at all, while the base path
  registers them `interactive=True` (`spawn(..., interactive=True)` / `force_confirm`),
  which confirms them at once and skips the kill — mirroring upstream's
  `add_tracker_new_points`.
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
