# SAM 3.1 — tracklet re-ID (dormant-object lifecycle)

**Date:** 2026-07-23
**Status:** design (fast-track; user approved "steps 1+2+3, full parity")
**Reference:** upstream `facebookresearch/sam3` @ `5dd401d` at `../sam3_reference/`

## Context

The model-find golden (Feature 1b) exposed a lifecycle divergence: when a tracked
object leaves frame and re-enters ~20 frames later, upstream keeps its **original
id**; our port mints a **fresh** id. Root cause, from the upstream map:

Upstream has **no re-association step**. It simply **never kills an established
object**. Removal (`remove_by_unmatch`, `sam3_multiplex_base.py:2230-2234`) is gated
to an object's **hotstart window** — the first `hotstart_delay=15` frames of its
life. After that, an absent object is only **suppressed** (hidden from output) via a
`trk_keep_alive` hysteresis counter (`:2155-2160, 2237-2244`) while its mux slot and
SAM 2 memory keep updating every frame (empty mask, `object_score = -10`). On
re-entry, memory-conditioned propagation reconstructs the mask, the detector
re-matches it by IoU (`assoc_iou_thresh=0.1`) to the surviving slot, and it
un-suppresses under the same id.

Our `TrackletManager` (`sam/modeling/association/tracklet.py`) collapses upstream's
two counters into one lethal, **ungated** kill (`kill_thresh=3` → DEAD) and then
`remove_object` **purges** the bank memory — destroying exactly what re-ID relies on.

This is a **shared** lifecycle (base `Sam3VideoPredictor` + mux both use
`_associate_and_update` + `TrackletManager`), so the fix benefits both.

## Goal

Match upstream's dormant-object lifecycle so a re-entering object keeps its id: port
(1) hotstart-gated kill, (2) suppress-not-purge, (3) the `trk_keep_alive` show/hide
hysteresis.

## Scope

**In:**
- `TrackletManager` rewrite to mirror upstream's two independent counters + states:
  - Per tracklet: `first_frame`, `consecutive_det_count` (confirm), `unmatched_count`
    (kill, hotstart-gated), `keep_alive` (suppress hysteresis).
  - `step(matched_ids, new_ids, frame_idx)` — `frame_idx` needed for the hotstart gate
    `is_within_hotstart = first_frame > frame_idx - hotstart_delay`.
  - States: PENDING → CONFIRMED (unchanged, confirm gate); a **REMOVED** flag (purge)
    only when `is_within_hotstart & unmatched_count >= hotstart_unmatch_thresh`; a
    **SUPPRESSED** (hidden, alive) status when `keep_alive <= 0` and not removed.
  - `keep_alive`: init `init_keep_alive=0`, `+1` when matched / `-1` when not, clamp
    `[min_keep_alive=-4, max_keep_alive=8]`; `visible = keep_alive > 0`.
  - Queries: `removed_ids()` (purge), `alive_ids()` (not removed — propagated),
    `visible_ids()` (alive and not suppressed — output).
- Params (upstream defaults, `model_builder.py:1174-1177` + `sam3_multiplex_base.py:223-230`):
  `hotstart_delay=15`, `hotstart_unmatch_thresh=8`, `confirmation_thresh=3`,
  `init_keep_alive=0`, `max_keep_alive=8`, `min_keep_alive=-4`.
- `_associate_and_update`: pass `frame_idx` to `step`; purge only `removed_ids()`;
  keep suppressed objects in `bank` + `mux_state` (they propagate empty → memory
  retained). Return newly-spawned as before.
- `forward` output (base + mux): `results` include only `visible_ids()`; suppressed
  objects propagate but are hidden.
- Mux slot lifecycle: a truly REMOVED object frees its mux slot
  (`MultiplexState.remove_objects`); suppressed objects keep their slot.

**Out:**
- Backward re-propagation, boxes/exemplars, negatives, multi-concept (separate).
- Any appearance/obj_ptr re-association — upstream has none; not added.

## Architecture & data flow

`TrackletManager` becomes the single source of lifecycle truth; the predictor reads
three id sets from it each frame:

```
step(matched, new, frame_idx)
  -> removed_ids  : purge bank + free mux slot (within-hotstart failures only)
  -> alive_ids    : propagate (includes suppressed; memory rewritten empty)
  -> visible_ids  : emit in results
```

`_associate_and_update` flow per frame: associate (unchanged) → spawn new (record
`first_frame = frame_idx`) → `step(matched, new, frame_idx)` → purge `removed_ids`
(bank + mux slot) → leave the rest. `forward` builds `results` from `visible_ids`
only (both the tracker-propagated and detector-seeded outputs are filtered).

The mux `_propagate_multiplex` already propagates every object in `mux_obj_ids`; a
suppressed object stays in that list, so its memory keeps updating (empty mask,
score −10) — no code change beyond not-removing it. Freeing a removed object's mux
slot needs `MultiplexState.remove_objects` + compaction of `mux_obj_ids` /
`mux_output_dict` (the counterpart to `_grow_mux_state`).

## Testing (TDD)

1. **TrackletManager units** (CPU):
   - established object (first_frame 0), absent > kill window, past hotstart → NOT
     removed, becomes suppressed; matched again → visible again, same id.
   - new object (within hotstart) unmatched ≥ `hotstart_unmatch_thresh` → removed.
   - `keep_alive` clamps at `[-4, 8]`; `visible` flips at 0.
2. **Model-find id parity** (dance, GPU) — strengthen the 1b test: assert the total
   distinct ids over the clip equals the golden's (no spurious fresh id on
   re-entrance) and per-frame visible id-set matches the golden within a small timing
   slack, with matched-mask mean IoU ≥ 0.95.
3. **Regression** — 1a seed-click parity, 1b growth + smoke, mux text-tracking, base
   video tracking all still pass.
4. **Constant-VRAM** — dormant objects retain memory but the forgetful bank still
   bounds non-conditional frames; peak VRAM stays flat over a long clip.

## Files

- `sam/modeling/association/tracklet.py` — lifecycle rewrite (counters, states,
  hotstart gate, keep-alive, `removed/alive/visible` queries).
- `sam/models/sam3_predictor.py` — `_associate_and_update` (frame_idx, purge-only-
  removed, free mux slot on removal); `forward` output filter (base + mux);
  `_alloc_obj_id` records `first_frame`; a `_shrink_mux_state` helper (mux slot free).
- `tests/test_tracklet_reid.py` (units); strengthen
  `tests/parity/test_sam3p1_modelfind_parity.py`.
- Ledger: mark tracklet re-ID done.

## Risks

1. **Mux slot leak on removal** — freeing a removed object's slot mid-stream mirrors
   `_grow_mux_state` in reverse (`MultiplexState.remove_objects` + compact the
   threaded `mux_output_dict`/`mux_obj_ids`). Get the bucket-space compaction wrong
   and memory misaligns. Medium; removal is rare (within-hotstart only) so a simple
   correct-but-not-optimal compaction suffices. Covered by the parity gate.
2. **Show/hide timing** — the `keep_alive` hysteresis must match upstream defaults
   exactly for per-frame visible-set parity; off-by-one in the clamp/threshold shifts
   visibility by frames. Covered by the strengthened parity test.
3. **Constant-VRAM** — retaining dormant objects must not defeat the forgetful bank;
   assert peak memory stays flat.

## CLAUDE.md fit

The change concentrates lifecycle logic in `TrackletManager` (one clear owner, pure
state machine, unit-testable without the model) and reads three explicit id-sets in
the predictor. No appearance-matching speculation (upstream has none). Removal frees
its mux slot rather than leaking. Suppressed-but-alive is an explicit state, not an
implicit side effect.
