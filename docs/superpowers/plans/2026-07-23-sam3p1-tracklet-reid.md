# SAM 3.1 tracklet re-ID — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline). Steps use `- [ ]`.

**Goal:** A re-entering tracked object keeps its original id (match upstream): never kill established objects, suppress-not-purge absent ones, port the keep-alive hysteresis.

**Architecture:** Concentrate the lifecycle in `TrackletManager` (two independent counters: hotstart-gated kill + keep-alive suppress). The predictor reads three id-sets (`removed / alive / visible`), purges only removed, propagates alive (suppressed included → memory retained), emits visible.

**Spec:** `docs/superpowers/specs/2026-07-23-sam3p1-tracklet-reid-design.md`

## Global Constraints

- Upstream defaults: `hotstart_delay=15`, `hotstart_unmatch_thresh=8`, `confirmation_thresh=3`, keep_alive `init=0 / max=8 / min=-4`.
- Never `sed`/`python -c` for edits. 80-col, Google style. Parity IoU ≥ 0.95.

---

### Task 1: TrackletManager lifecycle rewrite

**Files:** rewrite `sam/modeling/association/tracklet.py`; test `tests/test_tracklet_reid.py`.

**Interfaces:** `spawn(obj_id, frame_idx)`; `step(matched_ids, new_ids, frame_idx)`; `removed_ids()`, `alive_ids()`, `visible_ids()`; `remove(obj_id)`.

- [ ] **Step 1: Failing units**

```python
# tests/test_tracklet_reid.py
from sam.modeling.association.tracklet import TrackletManager


def _mgr():
    return TrackletManager(confirmation_thresh=3, hotstart_delay=15,
                           hotstart_unmatch_thresh=8)


def test_established_object_suppressed_not_removed_then_reid():
    m = _mgr()
    m.spawn(1, frame_idx=0)
    for f in range(1, 20):                 # matched every frame -> keep_alive saturates
        m.step({1}, set(), frame_idx=f)
    assert 1 in m.visible_ids()
    for f in range(20, 40):                # absent 20 frames, but past hotstart
        m.step(set(), set(), frame_idx=f)
    assert 1 not in m.removed_ids()        # NOT killed (established)
    assert 1 in m.alive_ids()              # stays alive (memory retained)
    assert 1 not in m.visible_ids()        # hidden while gone
    m.step({1}, set(), frame_idx=40)       # re-enters
    assert 1 in m.visible_ids()            # same id, visible again


def test_new_object_within_hotstart_is_removed():
    m = _mgr()
    m.spawn(5, frame_idx=100)              # spawned mid-clip
    for f in range(101, 101 + 8):         # unmatched for hotstart_unmatch_thresh
        m.step(set(), set(), frame_idx=f)
    assert 5 in m.removed_ids()


def test_keep_alive_clamps_and_flips_visibility():
    m = _mgr()
    m.spawn(1, frame_idx=0)
    for f in range(1, 30):
        m.step({1}, set(), frame_idx=f)
    assert m._tracks[1].keep_alive == 8    # max clamp
    for f in range(30, 50):
        m.step(set(), set(), frame_idx=f)
    assert m._tracks[1].keep_alive == -4   # min clamp
    assert 1 not in m.visible_ids()
```

- [ ] **Step 2: Run — FAIL** (`spawn` signature / new API).

- [ ] **Step 3: Rewrite `tracklet.py`** — replace `_TrackletInfo` + `TrackletManager` (keep the module docstring updated). New `_TrackletInfo` fields `first_frame`, `state` (PENDING/CONFIRMED — display only, no DEAD), `consecutive_det_count`, `unmatched_count`, `keep_alive`, `removed`. `__init__` gains `hotstart_delay`, `hotstart_unmatch_thresh`, `init_keep_alive=0`, `max_keep_alive=8`, `min_keep_alive=-4`. `spawn(obj_id, frame_idx)` sets `first_frame` + `keep_alive=init`. `step(matched, new, frame_idx)`:

```python
        all_matched = matched_ids | new_ids
        for obj_id, info in self._tracks.items():
            if info.removed:
                continue
            matched = obj_id in all_matched
            if matched:
                info.consecutive_det_count += 1
                info.unmatched_count = 0
                if (info.state is TrackletState.PENDING
                        and info.consecutive_det_count >= self.confirmation_thresh):
                    info.state = TrackletState.CONFIRMED
            else:
                info.consecutive_det_count = 0
                info.unmatched_count += 1
            info.keep_alive = max(
                self.min_keep_alive,
                min(self.max_keep_alive, info.keep_alive + (1 if matched else -1)),
            )
            within_hotstart = info.first_frame > frame_idx - self.hotstart_delay
            if within_hotstart and info.unmatched_count >= self.hotstart_unmatch_thresh:
                info.removed = True
```

Queries: `removed_ids()` = `{o for removed}`; `alive_ids()` = `{o for not removed}`; `visible_ids()` = `{o for not removed and keep_alive > 0}`.

- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** `feat(sam3p1): TrackletManager hotstart-gated kill + keep-alive suppress`.

---

### Task 2: Wire lifecycle into the predictor (purge-only-removed, emit-only-visible)

**Files:** `sam/models/sam3_predictor.py` — `_associate_and_update`, `forward` (base + mux), `_alloc_obj_id`/spawn call sites, a `_shrink_mux_state` helper.

- [ ] **Step 1: `_associate_and_update`** — record `first_frame` on spawn and pass `frame_idx` to `step`; purge only `removed_ids()`:

```python
        for det_idx in new_dets:
            oid = self._alloc_obj_id(state)
            new_objects.append((oid, int(det_idx)))
            state.tracklet_mgr.spawn(oid, state.num_frames_processed - 1)
        new_ids = {oid for oid, _ in new_objects}
        for o in active_ids:
            if o not in state.tracklet_mgr._tracks:
                state.tracklet_mgr.spawn(o, state.num_frames_processed - 1)
        state.tracklet_mgr.step(matched_track_ids, new_ids, state.num_frames_processed - 1)
        for oid in state.tracklet_mgr.removed_ids():
            if oid in state.bank.known_obj_ids:
                self.remove_object(state, oid)
                if state.mux_state is not None and oid in (state.mux_obj_ids or []):
                    self._shrink_mux_state(state, oid)
```

(The `frame_idx` passed is `state.num_frames_processed - 1` since `forward` increments it at entry. `_associate_and_update` runs inside `forward`; confirm the value equals the current `frame_idx`.)

- [ ] **Step 2: `forward` output filter** (both the base `Sam3VideoPredictor.forward` and mux) — emit only visible tracklet objects (objects not managed by the tracklet_mgr, e.g. click-seeded, are always visible):

```python
            managed = set(state.tracklet_mgr._tracks)
            visible = state.tracklet_mgr.visible_ids()
            results = {
                oid: r for oid, r in results.items()
                if oid not in managed or oid in visible
            }
            return results
```

- [ ] **Step 3: `_shrink_mux_state(state, obj_id)`** — free a removed object's mux slot:

```python
    def _shrink_mux_state(self, state, obj_id: int) -> None:
        """Drop a removed object from the live mux state (frees its bucket slot)."""
        idx = state.mux_obj_ids.index(obj_id)
        state.mux_state.remove_objects([idx])
        state.mux_obj_ids = [o for o in state.mux_obj_ids if o != obj_id]
        for store in ("cond_frame_outputs", "non_cond_frame_outputs"):
            for out in state.mux_output_dict[store].values():
                for key in ("pred_masks", "pred_masks_high_res", "object_score_logits"):
                    if key in out and out[key].shape[0] > len(state.mux_obj_ids):
                        out[key] = torch.cat(
                            [out[key][:idx], out[key][idx + 1:]], dim=0
                        )
```

(If `MultiplexState.remove_objects` re-keys slots such that the muxed `obj_ptr` /
maskmem no longer align, prefer the simpler correctness path: leave the slot marked
REMOVED in `mux_state` — `demux` already drops it — and only drop it from
`mux_obj_ids`. Verify against the parity gate; pick whichever keeps IoU ≥ 0.95.)

- [ ] **Step 4: Regression** — 1a/1b/mux text-tracking still pass.

Run: `pixi run -e notebooks pytest tests/test_sam3p1_interactive_smoke.py tests/test_sam3p1_mux_growth.py tests/parity/test_sam3p1_interactive_parity.py -q`

- [ ] **Step 5: Commit** `feat(sam3p1): dormant-object lifecycle in predictor (re-ID)`.

---

### Task 3: Strengthen model-find parity to id-reuse

**Files:** `tests/parity/test_sam3p1_modelfind_parity.py`.

- [ ] **Step 1: Add id-reuse assertions** — the golden reuses ids (distinct ids over the clip == 4, no fresh id on re-entrance). Assert our port's total distinct ids over the clip equals the golden's, and the per-frame visible id-set matches within a small timing slack, keeping the existing mean-IoU ≥ 0.95 / count-within-1 checks:

```python
    our_ids, gold_ids = set(), set()
    ...
    for i, fr in enumerate(frames):
        out = pred.forward(st, i, fr)
        our_ids.update(out)
        gold_ids.update(int(o) for o in g[f"frame{i}_obj_ids"])
        ...  # existing per-frame IoU + count checks
    assert len(our_ids) == len(gold_ids), (
        f"distinct ids {sorted(our_ids)} vs golden {sorted(gold_ids)} "
        "(a re-entering object should reuse its id, not spawn a fresh one)"
    )
```

- [ ] **Step 2: Run — PASS** (re-ID now keeps the id count at 4; before the fix it was 8).
- [ ] **Step 3: Commit** `test(sam3p1): assert id-reuse in model-find parity`.

---

### Task 4: Ledger

- [ ] **Step 1:** Mark tracklet re-ID `[x]` in `docs/superpowers/plans/2026-06-26-phase1-sam3-torch-inference.md`.
- [ ] **Step 2: Commit** `docs(sam3p1): tracklet re-ID ledger`.

---

## Self-Review

- **Spec coverage:** hotstart-gated kill + keep-alive + suppress (T1), purge-only-removed / emit-only-visible / mux slot free (T2), id-reuse parity (T3), ledger (T4), regression (T2 step 4), constant-VRAM (the forgetful bank is unchanged; dormant objects add bounded cond memory — exercised by the long dance clip in T3). ✓
- **Placeholders:** the `_shrink_mux_state` step names its fallback (mark-removed vs compact) and defers the choice to the parity gate rather than hand-waving. ✓
- **Type consistency:** `spawn(obj_id, frame_idx)`, `step(matched, new, frame_idx)`, `removed_ids/alive_ids/visible_ids`, `_shrink_mux_state(state, obj_id)` used consistently. ✓
