# Open findings — `feat/sam3-integration`

Seven defects found by a review of this branch that were **not** fixed, because each
needs a decision about intended behaviour rather than a mechanical correction. Recorded
here so the decision is deliberate and the evidence is not lost.

Reviewed 2026-08-21 against `a48ee10`. The contained fixes from the same review shipped
in that commit (duplication, dead code, two conditional-memory bugs); this file holds
only what remains.

**How to read the evidence.** Each finding says how it was established. *Read* means the
control flow was traced in the code cited. *Reproduced* means the review agent executed
it and reported the failure. Nothing here is fixed, so nothing here has a regression
test yet — writing one is part of the fix, and in at least one case (#5) writing it may
be the cheapest way to settle whether the path is reachable at all.

**Why the test suite is quiet about all seven.** Every one lives on a path the current
tests do not drive: bucket teardown after total tracklet loss, mid-stream growth past 16
cumulative objects, a finite `max_cond_frames_in_attn`, a re-run frame index, a second
click on a frame-0 object, a click-only session run long enough to decay, and an
ONNX-compat attention with no spatial memory. Suite state at time of writing: 174
passed, 24 skipped, 1 xfailed.

---

## 1. Multiplex state survives its own teardown

**`sam/modeling/multiplex.py:189`** · established by *read* + *reproduced*

`remove_objects` deletes emptied buckets, and when none survive it sets
`self.assignments = None` and returns early — before `_initialize_assignments`. So
`num_buckets`, `total_valid_entries` and the mux/demux matrices keep their pre-removal
values while `assignments` is gone.

Reachable through `Sam3MultiplexVideoPredictor._purge_removed` → `_shrink_mux_state`:
a false-positive burst where every seeded detection dies in hotstart empties the grid
while `state.mux_state` stays non-None. `add_objects` and `get_all_valid_object_idx`
then raise `TypeError: 'NoneType' object is not iterable`; `available_slots`, `mux` and
`demux` answer from stale matrices instead of failing.

**The decision.** What *is* an empty multiplex state? Either it is a legal resting state
(then `assignments = []`, the counters and matrices must be rebuilt empty, and every
consumer has to tolerate zero buckets), or it is not (then the predictor must drop
`state.mux_state = None` when the last object leaves, and re-seed on the next
detection). The second is closer to how the seed path already works, but it means
auditing every `mux_state is not None` test in `sam3_predictor.py`.

## 2. Bucket slots are never reclaimed

**`sam/modeling/multiplex.py:174`** · established by *read* + *reproduced*

Removed slots are stamped `_REMOVED_NUM`, counted as occupied by
`total_non_padding_entries`, and never reused: `add_objects` only refills `_PADDING_NUM`
slots. `add_new_masks_to_existing_state` hardcodes `allow_new_buckets=False`, so the grid
is frozen at `ceil(N_seed / 16)` and the free pool only shrinks.

Reproduced: the 14th mid-stream grow raises `AssertionError: not enough available slots
0 < 1`, from inside an `@torch.inference_mode()` block. A long clip that holds only two
or three objects at a time still dies after 16 cumulative detections.
`_shrink_mux_state`'s docstring already claims it "frees its bucket slot" — it does not.

**The decision.** Reclaiming a slot means deciding what the tracker may still be
carrying for the object that left. If bucket-space memory for a removed object is inert
once its pointers are dropped, `_REMOVED_NUM` can simply become `_PADDING_NUM` on the
next grow. If it is not inert, the grid needs a compaction step, which changes the
demux order and therefore every downstream index. Upstream's own behaviour here is worth
capturing as a golden before choosing.

## 3. `_grow_mux_state` reads a frame it may never have written

**`sam/models/sam3_predictor.py:2011`** · established by *read*

```python
prev = state.mux_output_dict["non_cond_frame_outputs"].get(frame_idx)
was_cond = prev is None
if was_cond:
    prev = state.mux_output_dict["cond_frame_outputs"][frame_idx]   # KeyError
```

`was_cond` infers "this frame was a conditioning frame" from the *absence* of a
propagation entry, then indexes the conditioning dict unconditionally. If `mux_state` is
alive but `active_ids` is empty — the state left by #1 and #2 — the propagation step
never ran, neither dict holds this frame, and the next detection takes the grow branch
into a `KeyError`.

**The decision.** Largely downstream of #1: if an objectless mux state cannot exist,
this is unreachable and wants an assertion documenting that. If it can, `was_cond` needs
to be a fact the caller passes rather than a guess from a missing key.

## 4. A skipped object still counts in the batch

**`sam/models/sam2_predictor.py:870`** · established by *read* · pre-dates this branch

An object with no memories hits `continue` and never appends to `results`, but
`n_objs` and `all_obj_ids` still include it. Downstream: `m.expand((n_objs, -1, -1, -1))`
against a `best_mask_logits` of length `len(results)`, `try_add_memories`'s
`assert results.batch_size == n_objs`, and `zip(all_obj_ids, batched_results)` pairing
masks with the wrong ids.

Carried over from `main`'s `sam2_generic_video_predictor.py`, but it lives in a file this
branch rewrote, so it is ours now.

**The decision.** When can a tracked object have no memories at all? If the answer is
"never, and the branch is defensive", it should be an assertion. If it can happen, the
object must be dropped from `all_obj_ids` and `n_objs` for that frame — which changes
what the caller gets back and needs a rule for what a caller sees when an object it
prompted is silently absent for a frame.

## 5. `t_diff_max == 0` gives an all-NaN mask

**`sam/modeling/tracking/sam3_tracker.py:132`** · established by *read*

```python
t_diff_max = max_abs_pos - 1 if max_abs_pos is not None else 1
pos_enc = torch.tensor(rel_pos_list).to(...) / t_diff_max
```

Callers pass `min(num_frames, 16)`, and the base predictor computes
`num_frames = frame_idx + 1`. A non-init-cond `track_step` at `frame_idx == 0` — a
second, refining click on an object seeded on frame 0, i.e. `_apply_geometry_prompt`
with `is_new=False` — therefore divides by zero. `get_1d_sine_pe` turns the resulting
inf/nan into an all-NaN encoding, which propagates through `obj_ptr_tpos_proj`. No
exception is raised: the call returns a NaN mask.

**The decision.** What should a temporal encoding mean when there is exactly one frame?
Clamping to `max(max_abs_pos - 1, 1)` makes every relative position 0, which is
plausible but is a numerics change on a parity-pinned path, so it should be justified
against upstream rather than chosen for convenience. Worth confirming reachability with
a two-click-on-frame-0 test first; the notebooks refine on frame 0 through
`session.process(..., frame_idx=0)`, which is the same shape of call.

## 6. Interactive tracklets decay even though they cannot be killed

**`sam/modeling/association/tracklet.py:210`** · established by *read* (mechanism);
end-to-end effect *reproduced* by the review agent

`step()` gates the hotstart kill on `not info.interactive`, but applies the keep-alive
decay to every tracklet:

```python
info.keep_alive = max(self.min_keep_alive,
                      min(self.max_keep_alive, info.keep_alive + (1 if matched else -1)))
```

`visible_ids()` is `not removed and keep_alive > 0`, and `force_confirm` — the call that
exists to make a clicked object permanent — never touches `keep_alive`. In a click-only
session `_advance_lifecycle` still calls `step(set(), set(), ...)` every frame, so a
clicked object decays to `min_keep_alive` and stops being emitted under `Emit.VISIBLE`.
Re-clicking does not restore it. This contradicts `_advance_lifecycle`'s own docstring
("click-seeded objects … neither decay nor die") and the premise of
`tests/test_emit_modes.py:83`.

**The decision.** Either interactive tracklets are exempt from decay as well as from the
kill (skip the update, or have `force_confirm` reset `keep_alive` to `max_keep_alive`),
or they are not and the docstring is wrong. The defaults matter to the choice:
`init_keep_alive=0` means a clicked object is *already* at the visibility boundary the
moment it is spawned, so under `Emit.VISIBLE` the first unmatched frame hides it.

## 7. `_associate_and_update` recomputes the frame index

**`sam/models/sam3_predictor.py:1370`** · established by *read*

```python
frame_idx = state.num_frames_processed - 1
```

Everything else in the same call uses the `frame_idx` that `forward` was given: bank
keys, `num_frames`, and `_apply_geometry_prompt`'s `spawn(obj_id, frame_idx, ...)`.
`VideoSession.process` documents a `frame_idx` override ("e.g. to re-run a frame"), and
the SAM 2 notebook uses it. Re-running a frame desynchronises the two: `first_frame` and
the `within_hotstart` gate end up on different scales, so hotstart stops expiring when
it should.

**The decision.** Almost certainly just "use the parameter" — but confirm there is no
caller relying on the counter, then decide whether re-running a frame should also rewind
`num_frames_processed`, which is the deeper question the override raises.

---

## Not a finding

The same review flagged `sam/utils/misc.py:102` for clobbering a user's
`TORCH_CUDA_ARCH_LIST` instead of using `setdefault`. Applying that broke
`test_jit_pins_arch_list_to_the_local_gpu`, whose docstring says why the clobber exists:
*"The JIT fallback must not inherit a list torch may reject"* — the test sets
`"8.6;10.1"` precisely because torch rejects part of it. The behaviour is deliberate; the
comment above it, which mentions only the conda-forge default, is what is imprecise.
