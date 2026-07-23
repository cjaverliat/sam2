# SAM 3.1 multiplex — interactive seed-frame click (Feature 1a)

**Date:** 2026-07-23
**Status:** design (awaiting review)
**Reference:** upstream `facebookresearch/sam3` @ `5dd401d` at `../sam3_reference/`

## Context & decomposition

Closing the SAM 3.1 (multiplex) video predictor's parity gap with upstream. The
multiplex `forward` currently `raise`s `NotImplementedError` for **all** geometry
prompts (`sam/models/sam3_predictor.py:897-901`); upstream supports interactive
point-click add-object.

Investigation of `../sam3_reference/` established:

- A **point click** flows through the tracker's per-object interactive path
  (`interactive_sam_prompt_encoder` → `interactive_sam_mask_decoder` →
  `interactive_obj_ptr_proj`), distinct from the detector. A **box** flows through
  `detector.geometry_encoder` — the *same* encoder that consumes
  `ConceptPrompt.exemplars`, so box prompts belong with the future **exemplars**
  feature, not here.
- Our mux tracker (`sam/modeling/tracking/sam3_multiplex_tracker.py`) **already
  instantiates and strict-loads** the full interactive stack (157 checkpoint keys),
  and `_forward_sam_heads` already runs the `is_interactive` branch. The gap is at
  the **predictor** level only.
- Adding an object **mid-stream** (frame > seed) is blocked by a hard guard in
  `_seed_multiplex` (`sam3_predictor.py:1010-1016`): the `MultiplexState` is built
  once on the seed frame and frozen. Lifting that requires porting upstream's
  `add_new_masks_to_existing_state` (growable mux state) — a separate foundation.

This line of work is therefore decomposed:

- **1a (this spec):** seed-frame interactive click — click-only VOS on the seed
  frame, no text, forward-track. Pure predictor wiring; no architecture change.
- **1b (later):** dynamic mux-state growth (`add_new_masks_to_existing_state` port),
  unlocking mid-stream add for both interactive clicks and detector spawn.
- Alongside-add = 1a + 1b. Box/exemplar prompts, `negative_phrases`, and
  multi-concept are separate parity features tracked outside this spec.

## Goal

Support interactive point-click object segmentation + tracking on the SAM 3.1
multiplex video predictor for the **seed-frame, text-free** case, validated for
numerical parity against upstream.

## Scope

**In:**

- One or more `GeometryPrompt`s carrying **points only** (`points_coords` (N,2) +
  `points_labels` (N,), 1=positive / 0=negative), applied on the **seed frame** of a
  fresh `Sam3VideoPredictorState` (no existing `mux_state`, no text concept).
- One or more objects seeded together on that frame (each its own `obj_id`).
- Forward propagation of the seeded objects across subsequent frames (existing
  `_propagate_multiplex` path, unchanged).
- The click frame is written as a conditioning memory (existing `_encode_new_memory`).

**Out (explicit — each `raise`s a clear message pointing elsewhere):**

- Box or mask prompts in the multiplex predictor → route to the exemplars feature.
- Mid-stream add (prompt when `state.mux_state` already exists) → 1b.
- Concurrent text concept + click on the same seed frame → 1b flavor (co-seed).
- Cross-frame refinement of an existing tracklet → later.
- Backward re-propagation (retroactive appearance) → out of the forward-streaming
  model by decision.

## Architecture & data flow

No architecture change. Reuses existing tracker helpers; adds predictor-level
routing + one seeding method that mirrors `_seed_multiplex` with `point_inputs`
substituted for the detector `mask_inputs`.

`Sam3MultiplexVideoPredictor.forward(state, frame_idx, frame, geometry_prompts)`:

1. Partition `geometry_prompts`: reject (raise) any with `boxes`/`masks_logits`
   set → exemplars feature. Keep point-only prompts.
2. If point prompts present:
   - If `state.mux_state is not None` → raise (mid-stream → 1b).
   - If a text `concept` is set → raise (co-seed → 1b).
   - Else → `_seed_points_multiplex(state, frame_idx, point_prompts, bf_int, bf_prop, num_frames)`.
3. Otherwise the existing text/detection flow runs unchanged.

`_seed_points_multiplex` (new; structural twin of `_seed_multiplex`):

1. `mux_state = self.tracker.multiplex_controller.get_state(len(prompts), ...)`.
2. Build batched `point_inputs` from each prompt: scale coords to the tracker input
   space, stack `(n, num_pts, 2)` coords + `(n, num_pts)` labels (design assumption:
   the interactive branch batches over objects like the mask path — **verified in
   plan step 1**).
3. `out = self.tracker.track_step(frame_idx, is_init_cond_frame=True,
   backbone_features_interactive=bf_int, backbone_features_propagation=bf_prop,
   point_inputs=point_inputs, mask_inputs=None, output_dict={...}, num_frames,
   multiplex_state=mux_state)` — the interactive head turns clicks into masks +
   pointers, then the joint memory encoder writes the cond frame.
4. Persist `state.mux_state`, `state.mux_obj_ids`, `state.mux_output_dict`,
   register ids on `state.bank`.
5. Build `MaskletResult`s via the existing `_masklet_from_lowres` / `_demux_outputs`.

## Parity gate

Golden captured once in `../sam3_reference/.venv`, committed as fixtures so CI runs
without the upstream repo/weights (mirrors phase1 Task 1).

- **Capture script:** `tests/parity/reference_sam3/capture_sam3p1_interactive_golden.py`
  reproducing the upstream interactive add-object flow:
  `build_sam3_multiplex_video_predictor` → `handle_request(add_prompt, points=[[x,y]],
  point_labels=[1], obj_id, frame_index=0)` (no text) → `propagate_in_video` forward
  over the first N bedroom frames. Saves per-frame masklets + `scenario.json`
  (coords, labels, obj_id, N, resolution) to `tests/parity/fixtures/sam3p1/`.
- **Tolerances (phase1):** per-frame masklet IoU ≥ 0.99, matching object ids.

## Testing (TDD)

1. **Strict-load / attribute smoke** — confirm the built mux tracker exposes and
   loaded the interactive submodules (verifies the plan-step-1 assumption before
   wiring).
2. **Parity** `test_sam3p1_interactive_seed_parity` — stream our predictor through
   the golden scenario; assert per-frame IoU ≥ 0.99, ids stable. (Fails first.)
3. **Unit guards** — box/mask prompt in mux `forward` raises with the exemplars
   message; a click when `mux_state` already exists raises the 1b message; a click
   with a text concept set raises the co-seed message.
4. **Constant-VRAM** — the existing forgetful-bank property still holds over the
   click-seeded clip.
5. **Notebook** — flip the "geometry not supported" cell in
   `notebooks/sam3_video_predictor_example.ipynb` to a live seed-frame click
   add-object demo on 3.1; confirm it runs.

## Files

- `sam/models/sam3_predictor.py` — `Sam3MultiplexVideoPredictor.forward` routing +
  `_seed_points_multiplex`; the blanket `NotImplementedError` narrows to the
  out-of-scope cases with precise messages.
- `tests/parity/reference_sam3/capture_sam3p1_interactive_golden.py` + fixtures
  under `tests/parity/fixtures/sam3p1/`.
- `tests/parity/test_sam3p1_interactive_parity.py`.
- `notebooks/sam3_video_predictor_example.ipynb` — enable the demo cell.
- Ledger note (the phase1/efficientsam3 plans) recording 1a done + 1b/others open.

## Risks & assumptions

- **Batched interactive `point_inputs`** — assumed the tracker's `_forward_sam_heads`
  interactive branch + `track_step` accept per-object batched points exactly as they
  accept batched `mask_inputs` in `_seed_multiplex`. Verified as plan step 1; if the
  interactive head is single-object, seed objects sequentially into the mux state.
- **Coordinate space** — upstream scales relative `[0,1]` coords by `image_size`
  inside the demo tracker; our `GeometryPrompt` carries pixel coords + an
  `is_normalized` flag. The seeding method must map to the tracker input space
  consistently with `_seed_multiplex`'s mask resolution (`input_mask_size`).
- **Determinism** — upstream seeds RNG 42 and runs bf16 autocast; the capture must
  mirror this (as the existing sam3 capture scripts do) for a stable golden.

## CLAUDE.md fit

Reuses existing helpers (`_forward_sam_heads` interactive branch, `_use_mask_as_output`,
`_encode_new_memory`, `track_step`, `multiplex_controller.get_state`) — no speculative
abstraction. New code is one routing block + one seeding method, structurally mirroring
the established `_seed_multiplex`. Parity/coordinate mapping stays pure; state lives in
the tracker/predictor that owns it. Out-of-scope cases fail loud with specific messages
rather than silently degrading.
