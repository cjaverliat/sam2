# SAM 3.1 video box prompt — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline). Steps use `- [ ]`.

**Goal:** A box `GeometryPrompt` in the mux video `forward` biases that frame's detection; detections seed/track via the existing pipeline (our streaming/forgetful-bank arch).

**Architecture:** `_detect` gains an optional `geo`; a cached `"geometric"` placeholder concept covers box-only prompts; `forward` routes boxes to the detector-geometry path (points still go to the click path). Downstream (association, `_seed_multiplex`/`_grow_mux_state`, bank) unchanged.

**Spec:** `docs/superpowers/specs/2026-07-23-sam3p1-video-box-prompt-design.md`

## Global Constraints

- Box → GEOMETRIC slot (caption `"geometric"`); text (if any) → TEXT slot. Reuse 2a `_pack_geometry` (xyxy→cxcywh-norm). Never `sed`/`python -c`. Parity id-agnostic mean IoU ≥ 0.95.

---

### Task 1: `_detect` geo + placeholder concept + forward box routing

**Files:** `sam/models/sam3_predictor.py`; test `tests/test_sam3p1_interactive_smoke.py`.

- [ ] **Step 1: Failing smoke** (append to the interactive smoke file):

```python
@needs_gpu
def test_video_box_prompt_spawns_and_tracks():
    pred = _build()
    frames = _bedroom(4)
    h, w, _ = frames[0].shape
    st = _state((h, w))
    import torch
    box = GeometryPrompt(obj_id=1, boxes=torch.tensor([[300.0, 150.0, 470.0, 420.0]]))
    out0 = pred.forward(st, 0, frames[0], geometry_prompts=[box])   # box, no text concept
    assert len(out0) >= 1                              # detector found the boxed person(s)
    out1 = pred.forward(st, 1, frames[1])              # tracks forward
    assert len(out1) >= 1


@needs_gpu
def test_video_mask_prompt_still_raises():
    pred = _build()
    st = _state((540, 960))
    import numpy as np, torch
    m = GeometryPrompt(obj_id=1, masks_logits=torch.zeros(540, 960))
    with pytest.raises(NotImplementedError):
        pred.forward(st, 0, np.zeros((540, 960, 3), dtype=np.uint8), geometry_prompts=[m])
```

- [ ] **Step 2: Run — FAIL** (box currently routed to the click path / no detection).

- [ ] **Step 3: `_detect` geo** — add `geo=None` to the mux `_detect` and pass it:

```python
        out = self.detector.forward_grounding(
            det_feats, det_pos, concept.text_emb, concept.text_mask, geo=geo,
        )
```

- [ ] **Step 4: Placeholder concept** — add to `Sam3MultiplexVideoPredictor`:

```python
    def _placeholder_concept(self):
        """A cached '<geometric>' ConceptState for box-only prompts (no text set)."""
        if getattr(self, "_geo_concept", None) is None:
            emb, mask = self.encode_text(ConceptPrompt("geometric"))
            self._geo_concept = ConceptState(0, ConceptPrompt("geometric"), emb, None, mask)
        return self._geo_concept
```

- [ ] **Step 5: `forward` box branch** — split geometry_prompts into boxes vs points; pack boxes and run detection with them. Where the mux `forward` computes `concept` and `det`, replace the detection gate:

```python
            concept = state.concepts[0] if state.concepts else None
            box_prompts = [p for p in geometry_prompts if p.boxes is not None]
            point_prompts = [p for p in geometry_prompts if p.points_coords is not None]
            box_geo = None
            if box_prompts:
                packed = [self._pack_geometry(p, (H, W), device) for p in box_prompts]
                box_geo = {
                    "box_coords": torch.cat([g["box_coords"] for g in packed], 0),
                    "box_labels": torch.cat([g["box_labels"] for g in packed], 0),
                }
            det_concept = concept or (self._placeholder_concept() if box_geo else None)
            det = (
                self._detect(det_f, det_p, det_concept, geo=box_geo)
                if det_concept is not None else None
            )
```

Then the point-prompt click branch (the post-results block from 1b) must use `point_prompts` instead of `geometry_prompts`:

```python
            if point_prompts:
                if state.mux_state is None:
                    results.update(self._seed_points_multiplex(
                        state, frame_idx, point_prompts, bf_int, bf_prop, num_frames))
                else:
                    click_masks, click_ids = self._click_masks_multiplex(
                        point_prompts, (H, W), bf_int)
                    results.update(self._grow_mux_state(
                        state, frame_idx, click_masks, True, click_ids, bf_int, bf_prop))
```

- [ ] **Step 6: `_check_mux_geometry`** — drop the box raise; keep the mask raise:

```python
        for prompt in geometry_prompts:
            if prompt.masks_logits is not None:
                raise NotImplementedError(
                    "mask geometry prompts are unsupported (no mask_encoder weights); "
                    "use box or point prompts"
                )
            if prompt.boxes is None and prompt.points_coords is None:
                raise ValueError("geometry prompt has neither points nor boxes")
```

- [ ] **Step 7: Run — PASS** (box spawns+tracks; mask raises). Verify `_detect` accepts `geo`.
- [ ] **Step 8: Commit** `feat(sam3p1): video box prompt (detector-geometry -> seed via association)`.

---

### Task 2: Video box parity golden

**Files:** `tests/parity/reference_sam3/capture_sam3p1_video_box_golden.py` + fixtures; `tests/parity/test_sam3p1_video_box_parity.py`.

- [ ] **Step 1: Capture** (reference env, `--patches`) — bedroom first N frames, ONE box add_prompt on frame 0 (normalized xywh), no text; `propagate_in_video` forward. Save per-frame `{obj_ids, masks}`. Delegate the reference-env run.
- [ ] **Step 2: Parity test** — stream our mux predictor with the same box on frame 0; id-agnostic per-frame mean matched-mask IoU ≥ 0.95, count within 1 (reuse the model-find matcher).
- [ ] **Step 3: Run — PASS.**
- [ ] **Step 4: Commit** `test(sam3p1): video box-prompt parity`.

---

### Task 3: Notebook + ledger

- [ ] **Step 1:** Add a mux video box-prompt demo cell (box on frame 0, track). Execute headless (exit 0).
- [ ] **Step 2:** Ledger: mark video box (2b) done; note base-path video box + exemplar/negatives/multi-concept open.
- [ ] **Step 3: Commit** `docs(sam3p1): video box demo + ledger`.

---

## Self-Review

- **Spec coverage:** `_detect` geo + placeholder + forward routing (T1), box-vs-point split (T1 step 5), mask raise (T1 step 6), parity (T2), notebook+ledger (T3), regression (rerun 1a/1b/re-ID/model-find with T1). ✓
- **Placeholders:** T2 flags the reference-env delegation. ✓
- **Type consistency:** `_detect(det_feats, det_pos, concept, geo=None)`; `_placeholder_concept() -> ConceptState`; `_pack_geometry` reused; box_geo dict `{box_coords, box_labels}`. ✓
```
