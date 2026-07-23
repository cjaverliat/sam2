# SAM 3.1 multiplex — dynamic mid-stream add-object (Feature 1b)

**Date:** 2026-07-23
**Status:** design (fast-track; user approved scope "all three", review-later)
**Reference:** upstream `facebookresearch/sam3` @ `5dd401d` at `../sam3_reference/`
**Blueprint:** captured from `video_tracking_multiplex.py::add_new_masks_to_existing_state`

## Context

Feature 1a shipped seed-frame interactive clicks. The multiplex predictor still
cannot add an object **after** the seed frame: `_seed_multiplex`
(`sam3_predictor.py`) hard-raises when `state.mux_state` already exists, and
`_check_mux_geometry` blocks mid-stream clicks + text co-seed. This also means a
**latent text-tracking bug**: if the detector finds a NEW instance mid-video
(`_associate_and_update` returns `new_objects` at frame > 0), `forward` calls
`_seed_multiplex`, which raises — text tracking crashes on a mid-video entrance.

Feature 1b ports upstream's dynamic add so all three add-object paths work
mid-stream: **detector spawn**, **interactive click**, **co-seed**.

## Key mechanism (from the blueprint)

Upstream keeps the multiplex **bucket grid invariant** across frames. A mid-stream
add fills a `_PADDING_NUM` slot inside the existing buckets
(`add_objects(allow_new_buckets=False)`); `num_buckets` / `multiplex_count` never
change. Because the joint bucket-space memory (`maskmem_features`
`(num_buckets, C, H, W)`, muxed `obj_ptr`) is **not** per-object separable, past
frames' stored memory is **never re-muxed** — the new object simply had a padding
(=absent) slot in every past frame, which is correct (it has no history). Only the
**current** frame is re-encoded to fold in the new object.

Our `MultiplexState.get_state(N)` sizes `num_buckets = ceil(N / cap)` and pads to
`num_buckets · cap`, so `available_slots = cap − (N mod cap)` free slots already
exist. For realistic counts (≤ `cap`, i.e. ≤ 16 objects) a mid-stream add fills a
free slot with **no bucket growth and no memory re-layout**. The only failure mode
is running out of slots (seed count a multiple of `cap`, or > `cap` total) — an
unrecoverable case we detect and raise on, rather than silently grow buckets.

## Scope

**In:**
- `tracker.add_new_masks_to_existing_state(...)` — port of upstream UP:3068–3216:
  demux existing pointers → `add_objects(allow_new_buckets=False)` → encode each
  new mask via `_use_mask_as_output` (subset) → `_append` pred_masks /
  pred_masks_high_res / object_score_logits (data space) + re-mux `obj_ptr` →
  `conditioning_objects.update` → `_encode_new_memory` (full grid) → returns the
  grown `out`.
- Predictor `_grow_mux_state(state, frame_idx, new_masks, are_from_pts, bf_int,
  bf_prop, num_frames)` — prep, call the tracker method, re-key the current frame
  to `cond_frame_outputs` (survives prune), extend `mux_obj_ids` + register on
  `bank`, return per-object masklets.
- `forward` routing for the three consumers (mask source differs; all funnel
  through `_grow_mux_state`):
  - **Detector spawn** — `det.masks_logits[det_idx]` for unmatched detections at
    frame > 0 (binarised, resized to `input_mask_size`). Replaces the raising
    `_seed_multiplex` call when `mux_state` exists.
  - **Interactive click** — points → interactive decode (`_forward_sam_heads`
    interactive) → mask, `are_from_pts=True`. Lifts the `_check_mux_geometry`
    mid-stream raise.
  - **Co-seed** — on the seed frame, create `mux_state` via the primary modality,
    then `_grow_mux_state` with the other modality's masks in the same call. Lifts
    the co-seed raise.

**Out:**
- Bucket growth / > `cap` total objects — raise a clear "capacity exceeded"
  error (unrecoverable given non-separable joint memory).
- Backward re-propagation (retroactive appearance) — forward-only, per the 1a
  decision.
- Refinement of an existing object mid-stream (`recondition_masks_in_existing_state`,
  UP:3222) — a follow-on; 1b only ADDS objects.

## Architecture

New standalone tracker method (does **not** refactor `track_step`): the predictor
already holds `bf_int` / `bf_prop` in `forward`, so
`add_new_masks_to_existing_state` is fed those directly (no `_track_step_aux`
split). Reused unchanged: `_use_mask_as_output`, `_encode_new_memory`,
`_forward_sam_heads` (interactive), `MultiplexState.{add_objects,
find_next_batch_of_available_indices, mux, demux}`, `_demux_outputs`,
`_masklet_from_lowres`, bank/tracklet lifecycle. New: an `_append` helper
(`torch.cat` dim 0 on the StageOutput tensors).

`object_ids` note: our seeds create `mux_state` without `object_ids`
(`state.mux_state.object_ids is None`), so `add_objects` is called with
`object_ids=None` and `state.mux_obj_ids` is extended by the predictor (which
already owns the idx→id map).

Sequencing in `forward`: `_grow_mux_state` runs AFTER `_propagate_multiplex` has
produced+stored the current frame's `out`, so `prev_output` is that `out`; after
growth the frame is re-keyed `non_cond → cond` at `frame_idx` (mirrors upstream
`forward_tracking`), which `_prune_mux_memory` keeps.

## Parity gate

Two goldens captured in `../sam3_reference/.venv` (reuse the 1a capture harness +
patches), committed as fixtures:

- **Model-find** (`dance.mp4` / `notebooks/videos/dance/*`): text `"person"`, N
  frames where the camera pans and a new person enters mid-video → the detector
  spawns a tracklet after frame 0. Golden = per-frame `{obj_id: mask}` incl. the
  mid-stream id.
- **Interactive mid-stream add** (`bedroom`): click a first object at frame 0,
  then click a SECOND object at frame k > 0 → both tracked. Golden = per-frame
  masks for both ids.

Tolerances (phase1): per-frame masklet IoU ≥ 0.99, matching object ids.

## Testing (TDD)

1. **`add_objects` capacity unit** (CPU) — a fresh `get_state(2)` has ≥ 1 free
   slot; `add_objects([2], allow_new_buckets=False)` grows `total_valid_entries`
   to 3 without changing `num_buckets`; `add_objects` past `cap` raises.
2. **`add_new_masks_to_existing_state` unit** (GPU, tiny) — seed 1 object, grow by
   1 synthetic mask; assert `out["pred_masks"].shape[0] == 2`, `obj_ptr` re-muxed
   to bucket space, `maskmem_features` present, `conditioning_objects` includes the
   new idx.
3. **Detector mid-stream parity** — `test_sam3p1_modelfind_parity` on `dance`,
   IoU ≥ 0.99, ids stable incl. mid-stream id.
4. **Interactive mid-stream parity** — `test_sam3p1_midstream_click_parity` on
   `bedroom`, IoU ≥ 0.99 for both ids.
5. **Co-seed smoke** (GPU) — text + click on the seed frame yields both; capacity
   headroom respected.
6. **Regression** — the 1a seed-click parity + mux text-tracking still pass; the
   `> cap` capacity error raises clearly.
7. **Notebook** — a model-find demo cell on `dance` (text, new person enters).

## Files

- `sam/modeling/tracking/sam3_multiplex_tracker.py` — `add_new_masks_to_existing_state`
  + `_append`.
- `sam/models/sam3_predictor.py` — `_grow_mux_state`; `forward` routing (detector /
  interactive / co-seed); narrow `_check_mux_geometry` (mid-stream now allowed).
- `tests/parity/reference_sam3/capture_sam3p1_modelfind_golden.py`,
  `capture_sam3p1_midstream_click_golden.py` + fixtures under
  `tests/parity/fixtures/sam3p1/`.
- `tests/parity/test_sam3p1_modelfind_parity.py`,
  `test_sam3p1_midstream_click_parity.py`; unit tests for capacity + grow.
- `notebooks/sam3_video_predictor_example.ipynb` — model-find demo cell.
- Ledger note (mark 1b done).

## Risks (blueprint §5, re-ranked after capacity finding)

1. **`add_new_masks_to_existing_state` correctness** — the append + re-mux +
   re-encode sequence must exactly mirror UP:3068–3216; verified by unit + the two
   goldens. Medium.
2. **Frame re-keying to `cond` + prune safety** — a newly-conditioned frame must
   survive `_prune_mux_memory`; small but silent if wrong (new object loses its
   only memory). Covered by the parity gates (IoU would collapse on later frames).
3. **Capacity edge** — only bites at > `cap` total or seed count a multiple of
   `cap`; detected + raised, not silently grown. Low for our scenarios.
4. **`no_obj_ptr` subset fidelity** — our `_use_mask_as_output` blend differs
   slightly from upstream's per-slot demux select; a parity nuance verified against
   the golden, not a crash. Low.

## CLAUDE.md fit

Reuses the existing mask-encode / memory-encode / mux primitives; one new tracker
method mirroring a single upstream function + one predictor orchestration method +
routing. Pure tensor ops in the tracker; state stays in the predictor/state that
owns it. Capacity-exceeded and other out-of-scope cases raise specific errors
rather than silently degrading.
