# Findings — `feat/sam3-integration`

Seven defects found by a review of this branch, each of which needed a decision about
intended behaviour rather than a mechanical correction, plus an eighth (#8) found while
fixing them. **All eight are now fixed** (2026-08-21), each with a CPU regression test
that reproduces the documented failure against the pre-fix code. The evidence and the
decision taken are kept here. Nothing is open.

Reviewed against `a48ee10`; fixed on top of it. Every decision was settled by reading
the upstream path in `../sam3_reference` (facebook `sam3` @ `5dd401d`) first, per the
`sam3-parity-architecture-preference` directive.

Suite: **195 passed, 24 skipped, 1 xfailed** (was 174 / 24 / 1).

---

## 1. Multiplex state survived its own teardown — FIXED

**`sam/modeling/multiplex.py:189`** · established by *read* + *reproduced*

`remove_objects` deleted emptied buckets, and when none survived it set
`self.assignments = None` and returned early — before `_initialize_assignments`. So
`num_buckets`, `total_valid_entries` and the mux/demux matrices kept their pre-removal
values while `assignments` was gone. Reachable through
`Sam3MultiplexVideoPredictor._purge_removed` → `_shrink_mux_state`: a false-positive
burst where every seeded detection dies in hotstart. `add_objects` and
`get_all_valid_object_idx` then raised `TypeError: 'NoneType' object is not iterable`;
`available_slots`, `mux` and `demux` answered from stale matrices.

**Decision: an objectless multiplex state is NOT a resting state.** Upstream agrees —
`video_tracking_multiplex_demo.py:458-464` nulls the state whenever a removal empties
it, and `_merge_singleton_interaction_result` rebuilds from the controller
(`need_state_reinit = ... or total_valid_entries == 0`) rather than reviving it.

- `MultiplexState._mark_empty` zeroes the derived bookkeeping with the sentinel, so
  every query is honest (0 slots, no valid indices) or fails loudly (`mux`/`demux`
  shape assertions). New `is_empty` property; `add_objects` on a spent state raises
  `ValueError`; `get_all_valid_object_idx` returns `set()`.
- `_shrink_mux_state` drops `state.mux_state` to `None` when the last object leaves,
  **and clears `state.mux_output_dict`** — the threaded bucket-space memory is keyed to
  the old grid, which the next seed will not reproduce. The next detection re-seeds
  through the existing `_seed_multiplex` path.

Tests: `tests/test_sam3p1_mux_growth.py` (state-level),
`tests/test_mux_state_lifecycle.py` (predictor-level).

## 2. Bucket slots were never reclaimed — FIXED

**`sam/modeling/multiplex.py:174`** · established by *read* + *reproduced*

Removed slots were stamped `_REMOVED_NUM`, counted as occupied by
`total_non_padding_entries`, and never reused: `add_objects` only refilled
`_PADDING_NUM`. With `add_new_masks_to_existing_state` hardcoding
`allow_new_buckets=False`, the grid was frozen at `ceil(N_seed / 16)` and the free pool
only shrank — the 14th mid-stream grow raised `AssertionError: not enough available
slots 0 < 1` from inside `@torch.inference_mode()`.

**Decision: recycle the slot** (`_REMOVED_NUM` → `_PADDING_NUM` during the re-index
pass of `remove_objects`). The doc's condition for this option — "bucket-space memory
for a removed object is inert once its pointers are dropped" — holds, and for a
stronger reason than expected: spatial memory is encoded **per bucket**, one tensor for
all K slots (`_encode_new_memory` muxes the masks into channels), and past object
pointers are attended by the whole bucket. A survivor of that bucket already carries
whatever the removed object left behind, so refilling the slot adds nothing to it.

Upstream instead passes `allow_new_buckets=True` when `available_slots < num_objects`.
That is not portable to our streaming loop: stored bucket-space memories carry the
bucket count they were encoded at, and a later `demux` asserts on it — so growing the
grid mid-clip would break every retained frame. Recycling keeps the grid fixed, which
is what our architecture requires, and makes `_shrink_mux_state`'s docstring true.

Tests: `test_removed_slots_are_reclaimed_by_the_next_grow`,
`test_a_long_churn_never_exhausts_the_grid`.

## 3. `_grow_mux_state` read a frame it may never have written — FIXED

**`sam/models/sam3_predictor.py:2011`** · established by *read*

`was_cond` inferred "this frame was a conditioning frame" from the *absence* of a
propagation entry, then indexed the conditioning dict unconditionally (`KeyError`).

**Decision: look the frame up in both stores and fail with a message.** Upstream does
exactly this two-key lookup and raises when neither holds it
(`_run_single_frame_inference`). We keep propagation-first ordering — that is the store
this call's own decode went to, and a re-run frame (see #7) can leave a stale cond
entry at the same index. The objectless-state route into this branch is gone with #1.

Test: `test_growing_at_an_unstored_frame_says_so`.

## 4. A skipped object still counted in the batch — FIXED

**`sam/models/sam2_predictor.py:870`** · established by *read* · pre-dated this branch

An object with no memories hit `continue` and never appended to `results`, but `n_objs`
and `all_obj_ids` still included it — desynchronising `m.expand((n_objs, ...))`,
`try_add_memories`'s `assert results.batch_size == n_objs`, and the
`zip(all_obj_ids, batched_results)` that pairs masks with ids.

**Decision: the skip is real, so the object leaves the batch.** A known object with
neither a prompt nor a memory is reachable (e.g. after a prune), and the alternative —
asserting — would turn a recoverable frame into a crash. `forward_embeddings` now
tracks `decoded_obj_ids` in result order and keys the batch size, the memory write, the
prune and the returned dict off it. A caller that prompted an object which produced
nothing this frame simply does not see it in the returned dict, which is the same
contract the empty-`results` early return already had.

Test: `tests/test_video_batch_alignment.py` (3 cases).

## 5. `t_diff_max == 0` gave an all-NaN mask — FIXED

**`sam/modeling/tracking/sam3_tracker.py:132`** · established by *read*

`t_diff_max = max_abs_pos - 1` with callers passing `min(num_frames, 16)`, and our base
predictor computing `num_frames = frame_idx + 1`. A non-init-cond `track_step` at
`frame_idx == 0` — a second, refining click on an object seeded on frame 0 — divided by
zero; `get_1d_sine_pe` turned it into an all-NaN encoding that propagated through
`obj_ptr_tpos_proj` with no exception raised.

**Decision: clamp to `max(max_abs_pos - 1, 1)`, and it is parity-safe.** Upstream has
the identical expression (`sam3_tracker_base.py:165`) — the difference is that its
`num_frames` is the whole clip length, so the divisor is never 0. The divergence is our
streaming semantics, not the formula. With a single frame every relative position is 0,
so the quotient is 0 under either divisor; the clamp changes nothing anywhere except
where the pre-fix code produced `0/0`.

Test: `tests/test_tpos_enc.py` (includes an unchanged-elsewhere check).

## 6. Interactive tracklets decayed even though they could not be killed — FIXED

**`sam/modeling/association/tracklet.py:210`** · established by *read* (mechanism);
end-to-end effect *reproduced*

`step()` gated the hotstart kill on `not info.interactive` but applied the keep-alive
decay to every tracklet, and `force_confirm` never touched `keep_alive`. In a click-only
session `_advance_lifecycle` still calls `step(set(), set(), ...)` every frame, so a
clicked object decayed out of `Emit.VISIBLE` and re-clicking did not restore it.

**Decision: exempt from the decay as well as the kill.** Upstream initialises
`trk_keep_alive` **only** for `new_det_obj_ids` — both on the CPU path
(`sam3_multiplex_base.py:2344`) and in the GPU-metadata concat (:1320) — so a clicked
object is outside the suppression hysteresis entirely, not merely outside the kill.
`step()` now skips interactive tracklets, and `force_confirm` pins `keep_alive` at
`max_keep_alive` (with `init_keep_alive=0` a click would otherwise be born exactly at
the visibility boundary). `_advance_lifecycle`'s docstring, which claimed click-seeded
objects were left *unmanaged*, was wrong about the mechanism and is corrected: they are
managed but inert.

Tests: `test_clicked_object_stays_visible_through_a_click_only_session`,
`test_clicking_an_existing_object_pins_it_visible`.

## 7. `_associate_and_update` recomputed the frame index — FIXED

**`sam/models/sam3_predictor.py:1370`** · established by *read*

`frame_idx = state.num_frames_processed - 1`, while everything else in the same call
used the `frame_idx` `forward` was given. Re-running a frame desynchronised `first_frame`
from the `within_hotstart` gate, so hotstart stopped expiring when it should.

**Decision: use the parameter.** It is now threaded in from both `forward`s. The
counter keeps its one remaining use — `spawn_thresh = 0.0 if num_frames_processed == 1`
— which means "the first forward of this session" (the prompt frame) and is correctly
independent of the frame index. `num_frames_processed` is deliberately **not** rewound
on a re-run: it counts forwards, and its other consumer is `started`.

Test: `tests/test_associate_frame_index.py`.

---

## 8. `_shrink_mux_state` discarded `buckets_to_keep` — FIXED

**`sam/models/sam3_predictor.py:1849`** · found while fixing #1/#2 · established by
*read*, then *reproduced* by the regression tests

`MultiplexState.remove_objects` returns the surviving bucket indices and compacts the
data-space object indices; `_shrink_mux_state` used neither, so every frame retained in
`state.mux_output_dict` kept the shape the removal had just invalidated — in **both**
spaces:

- **Data space** (`pred_masks`, `pred_masks_high_res`, `object_score_logits`, and the
  `conditioning_objects` index set) is sliced positionally: `_demux_mux_outputs` pairs
  row *i* with `mux_obj_ids[i]`. A leftover row shifts every mask onto the wrong object.
  This is reachable at **one bucket** — the doc's original note ("unreachable below 17
  concurrent objects") was wrong. A hotstart kill in step 3 and a new detection in step
  4 of the same frame make `_grow_mux_state` append to a row set the removal already
  invalidated, so the grown frame carries N+1 rows for N objects.
- **Bucket space** (`maskmem_features`, `maskmem_pos_enc`, `obj_ptr`) is batched by
  bucket, so a deleted bucket leaves every retained memory one row too long for the next
  `demux` — this half does need >16 concurrent objects.

**Decision: re-slice on removal, and let each frame say what its rows are.** Upstream
slices the same three bucket-space tensors by `buckets_to_keep`
(`video_tracking_multiplex_demo.py:3092-3100`), and keys the data-space re-slice off a
per-frame `local_obj_id_to_idx`, precisely because "when we add new objects,
`obj_id_to_idx` mapping could be different locally (at this past frame) versus globally
(at the current frame)". We have the same problem and now the same answer: each stored
output carries the `obj_ids` its rows were built from (stamped at all three sinks —
`_seed_mux_state`, `_propagate_multiplex`, `_grow_mux_state`), and
`_reslice_mux_memory` slices by that rather than by the live order. A frame predating
the removed object is left alone.

Ruled out as alternatives: keying the re-slice off "frames whose row count matches the
pre-removal total" is unsound (two different object sets can share a count), and
dropping the data-space fields of past frames outright would break `_grow_mux_state`,
which appends to the current frame's rows.

Tests: five cases in `tests/test_mux_state_lifecycle.py`, including the single-bucket
row drop, the `conditioning_objects` re-index, the untouched earlier frame, and the
bucket-space slice at 17 objects.

## Not a finding

The same review flagged `sam/utils/misc.py:102` for clobbering a user's
`TORCH_CUDA_ARCH_LIST` instead of using `setdefault`. Applying that broke
`test_jit_pins_arch_list_to_the_local_gpu`, whose docstring says why the clobber exists:
*"The JIT fallback must not inherit a list torch may reject"* — the test sets
`"8.6;10.1"` precisely because torch rejects part of it. The behaviour is deliberate; the
comment above it, which mentions only the conda-forge default, is what is imprecise.
