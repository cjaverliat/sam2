# SAM 3.1 dynamic mid-stream add-object — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline, review-later per user). Steps use `- [ ]`.

**Goal:** Add objects to a running SAM 3.1 multiplex video session mid-stream (detector spawn, interactive click, co-seed), with numerical parity vs upstream.

**Architecture:** Port upstream `add_new_masks_to_existing_state` as a standalone tracker method fed by the predictor's existing `bf_int`/`bf_prop`; the bucket grid stays invariant (slot-fill, no re-mux of past frames). A predictor `_grow_mux_state` orchestrates it; `forward` routes the three consumers through it.

**Spec:** `docs/superpowers/specs/2026-07-23-sam3p1-dynamic-mux-growth-design.md`
**Blueprint:** upstream `video_tracking_multiplex.py::add_new_masks_to_existing_state` (UP:3068–3216).

## Global Constraints

- Preserve checkpoint attribute names; never `sed`/`python -c` for source edits.
- Bucket grid INVARIANT: `add_objects(allow_new_buckets=False)`; raise on no free slot (never grow buckets — joint memory is non-separable).
- Parity IoU ≥ 0.99, matching ids. CUDA-only; tests skip without GPU/ckpt/golden.
- 80-col, Google style.

---

### Task 1: Capacity unit + `add_new_masks_to_existing_state` (tracker core)

**Files:**
- Modify: `sam/modeling/tracking/sam3_multiplex_tracker.py` (add `add_new_masks_to_existing_state`)
- Test: `tests/test_sam3p1_mux_growth.py` (create)

**Interfaces:**
- Produces `Sam3MultiplexTracker.add_new_masks_to_existing_state(prev_output, new_masks, bf_int, bf_prop, multiplex_state, is_mask_from_pts) -> (out, new_idx)`. `new_masks` = `(num_new, 1, ims, ims)` binarised; grows `prev_output` in place (data-space append + re-muxed `obj_ptr` + re-encoded maskmem) and returns the grown `out` + the new slot indices.

- [ ] **Step 1: Failing capacity + grow test**

```python
# tests/test_sam3p1_mux_growth.py
import os
import numpy as np
import pytest
import torch

CKPT = "checkpoints/sam3.1_multiplex.pt"
needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.isfile(CKPT),
    reason="needs CUDA + sam3.1_multiplex.pt",
)


def test_multiplexstate_slot_fill_no_bucket_growth():
    from sam.modeling.multiplex import MultiplexController
    ctrl = MultiplexController(multiplex_count=16, allowed_bucket_capacity=16)
    st = ctrl.get_state(2, torch.device("cpu"), torch.float32, random=False)
    assert st.num_buckets == 1 and st.total_valid_entries == 2
    assert st.available_slots >= 1
    idx = st.find_next_batch_of_available_indices(1, allow_new_buckets=False)
    st.add_objects(idx, object_ids=None, allow_new_buckets=False)
    assert st.total_valid_entries == 3 and st.num_buckets == 1


@needs_gpu
def test_add_new_masks_grows_output():
    from sam.build_sam import build_sam3_multiplex_video_predictor
    from sam.models.sam3_predictor import Sam3VideoPredictorState
    from sam.prompts import ConceptPrompt
    import numpy as np
    from PIL import Image

    pred = build_sam3_multiplex_video_predictor(
        config_file="configs/sam3/sam3.1.yaml", ckpt_path=CKPT, device="cuda")
    frame = np.asarray(Image.open("notebooks/videos/bedroom/00000.jpg").convert("RGB"))
    h, w, _ = frame.shape
    st = Sam3VideoPredictorState(video_hw=(h, w))
    pred.set_concept(st, ConceptPrompt("person"))
    out0 = pred.forward(st, 0, frame)
    n0 = len(st.mux_obj_ids)
    ims = pred.tracker.input_mask_size
    new_mask = torch.zeros(1, 1, ims, ims, device="cuda")
    new_mask[..., ims // 4: ims // 2, ims // 4: ims // 2] = 1.0  # a blob
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        prev = st.mux_output_dict["cond_frame_outputs"][0]
        bf_prop = pred._mux_backbone_features(*pred._reencode(frame))  # see impl note
    # smoke: the tracker method exists and grows the object count
    assert hasattr(pred.tracker, "add_new_masks_to_existing_state")
```

(The GPU grow test is a smoke; the real numerical check is the parity gate in T3/T4. If `_reencode` helper is awkward, assert only `hasattr` + the CPU capacity test here and rely on parity for correctness.)

- [ ] **Step 2: Run — FAIL** (`AttributeError: ... add_new_masks_to_existing_state`).

Run: `pixi run pytest tests/test_sam3p1_mux_growth.py -q`

- [ ] **Step 3: Implement `add_new_masks_to_existing_state`** in `sam3_multiplex_tracker.py` (mirror UP:3068–3216, reusing the track_step feature-prep at OURS:737–759):

```python
    def add_new_masks_to_existing_state(
        self, prev_output, new_masks, backbone_features_interactive,
        backbone_features_propagation, multiplex_state, is_mask_from_pts,
    ):
        """Add new objects to a live multiplex frame output (mirror upstream).

        Fills padding slots in the EXISTING bucket grid (no growth): demux the
        current pointers, allocate slots, encode each new mask via the interactive
        head, append the per-object tensors, re-mux the pointers, and re-encode the
        frame's spatial memory so it conditions on the new objects too.

        Args:
            prev_output: the current frame's ``track_step`` output (mutated in place).
            new_masks: ``(num_new, 1, ims, ims)`` binarised masks for the new objects.
            backbone_features_interactive / _propagation: the per-frame feature dicts.
            multiplex_state: the live state (grown in place via ``add_objects``).
            is_mask_from_pts: True for click-derived masks (memory binarisation).

        Returns:
            ``(prev_output, new_idx)`` — the grown output and the new slot indices.
        """
        num_new = new_masks.shape[0]
        int_feats = backbone_features_interactive["vision_feats"]
        int_sizes = backbone_features_interactive["feat_sizes"]
        int_hi = None
        if len(int_feats) > 1:
            int_hi = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(int_feats[:-1], int_sizes[:-1])
            ]
        prop_feats = backbone_features_propagation["vision_feats"]
        prop_sizes = backbone_features_propagation["feat_sizes"]

        existing_ptr = multiplex_state.demux(prev_output["obj_ptr"])  # (old, C)
        new_idx = multiplex_state.find_next_batch_of_available_indices(
            num_new, allow_new_buckets=False
        )
        multiplex_state.add_objects(new_idx, object_ids=None, allow_new_buckets=False)

        interactive_pix_feat = self._get_interactive_pix_mem(int_feats, int_sizes)
        sam_out = self._use_mask_as_output(
            interactive_pix_feat, int_hi, new_masks, multiplex_state,
            objects_in_mask=new_idx,
        )

        prev_output["pred_masks"] = torch.cat(
            [prev_output["pred_masks"], sam_out["low_res_masks"]], dim=0
        )
        prev_output["pred_masks_high_res"] = torch.cat(
            [prev_output["pred_masks_high_res"], sam_out["high_res_masks"]], dim=0
        )
        prev_output["object_score_logits"] = torch.cat(
            [prev_output["object_score_logits"], sam_out["object_score_logits"]], dim=0
        )
        combined_ptr = torch.cat([existing_ptr, sam_out["obj_ptr"]], dim=0)
        prev_output["obj_ptr"] = multiplex_state.mux(combined_ptr)
        prev_output["conditioning_objects"].update(new_idx)

        if self.num_maskmem > 0:
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                image=None, current_vision_feats=prop_feats, feat_sizes=prop_sizes,
                pred_masks_high_res=prev_output["pred_masks_high_res"],
                object_score_logits=prev_output["object_score_logits"],
                is_mask_from_pts=is_mask_from_pts,
                conditioning_objects=prev_output["conditioning_objects"],
                multiplex_state=multiplex_state,
            )
            prev_output["maskmem_features"] = maskmem_features
            prev_output["maskmem_pos_enc"] = maskmem_pos_enc
        return prev_output, new_idx
```

- [ ] **Step 4: Run — PASS** (CPU capacity test passes; GPU smoke passes/asserts `hasattr`).

- [ ] **Step 5: Commit** `feat(sam3p1): tracker add_new_masks_to_existing_state (dynamic add)`.

---

### Task 2: `_grow_mux_state` + detector mid-stream spawn routing

**Files:**
- Modify: `sam/models/sam3_predictor.py` — add `_grow_mux_state`; in `Sam3MultiplexVideoPredictor.forward` branch the `new_objects` seeding on `state.mux_state is None`.

**Interfaces:**
- Consumes Task 1 `tracker.add_new_masks_to_existing_state`.
- Produces `_grow_mux_state(state, frame_idx, new_masks, is_mask_from_pts, new_ids, bf_int, bf_prop) -> dict[int, MaskletResult]` — grows the live state, re-keys the current frame to `cond_frame_outputs`, extends `mux_obj_ids` + `bank`, returns per-object masklets for the new ids.

- [ ] **Step 1: Implement `_grow_mux_state`** (next to `_seed_multiplex`):

```python
    def _grow_mux_state(
        self, state, frame_idx, new_masks, is_mask_from_pts, new_ids, bf_int, bf_prop
    ) -> dict:
        """Add new objects to the live mux state at ``frame_idx`` (forward-only).

        The current frame's output must already be stored in
        ``mux_output_dict["non_cond_frame_outputs"][frame_idx]``; after growth the
        frame becomes a conditioning frame (re-keyed so ``_prune_mux_memory`` keeps
        it). Returns per-object masklets for ``new_ids``.
        """
        height, width = state.video_hw
        prev = state.mux_output_dict["non_cond_frame_outputs"].get(frame_idx)
        if prev is None:  # seed-frame co-seed: grow the cond-frame output
            prev = state.mux_output_dict["cond_frame_outputs"][frame_idx]
            was_cond = True
        else:
            was_cond = False
        out, new_idx = self.tracker.add_new_masks_to_existing_state(
            prev, new_masks, bf_int, bf_prop, state.mux_state, is_mask_from_pts
        )
        if not was_cond:
            state.mux_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
            state.mux_output_dict["cond_frame_outputs"][frame_idx] = out
        state.mux_obj_ids = state.mux_obj_ids + list(new_ids)
        for obj_id in new_ids:
            state.bank.known_obj_ids.add(obj_id)
        per_obj = self._demux_outputs(out, state.mux_state, state.mux_obj_ids)
        return {
            oid: self._masklet_from_lowres(
                per_obj[oid]["pred_masks"][0, 0].float(), per_obj[oid], height, width
            )
            for oid in new_ids
        }
```

- [ ] **Step 2: Route detector mid-stream spawn** — in `forward`, replace the
  `if new_objects: self._seed_multiplex(...)` block with a seed-vs-grow branch:

```python
            if new_objects:
                new_masks, new_ids = self._det_masks_for_seed(det, new_objects)
                if state.mux_state is None:
                    self._seed_multiplex(state, frame_idx, new_objects, det, bf_int, bf_prop, num_frames)
                else:
                    grown = self._grow_mux_state(
                        state, frame_idx, new_masks, False, new_ids, bf_int, bf_prop
                    )
```

Add the small helper `_det_masks_for_seed(det, new_objects)` (extract the mask-prep already inlined in `_seed_multiplex`: resize each `det.masks_logits[det_idx]` to `input_mask_size`, binarise, stack `(n,1,ims,ims)`; return `(masks, ids)`), and reuse it in `_seed_multiplex`. Ensure the final results dict includes the grown ids (they are in `state.mux_obj_ids`, so the existing per-frame results build over `active_ids`/`mux_obj_ids` picks them up next frame; for the spawn frame, merge `grown` into `results`).

- [ ] **Step 3: Run the smoke + 1a regression**

Run: `pixi run pytest tests/test_sam3p1_interactive_smoke.py tests/test_sam3p1_mux_growth.py -q`
Expected: PASS/SKIP; the mid-stream detector path no longer raises.

- [ ] **Step 4: Commit** `feat(sam3p1): _grow_mux_state + detector mid-stream spawn`.

---

### Task 3: Model-find golden + parity (dance.mp4)

**Files:**
- Create: `tests/parity/reference_sam3/capture_sam3p1_modelfind_golden.py` (+ fixtures)
- Create: `tests/parity/test_sam3p1_modelfind_parity.py`

**Interfaces:** golden `{frame{i}_obj_ids, frame{i}_obj{oid}}` from the facebook reference on `dance` frames with text `"person"`, where a new id appears mid-stream.

- [ ] **Step 1: Extract dance frames** (the capture + test read numbered frames):

```bash
mkdir -p notebooks/videos/dance
ffmpeg -y -i notebooks/videos/dance.mp4 -vf fps=24 notebooks/videos/dance/%05d.jpg
```
(Commit the first ~16 frames only if a mid-stream entrance occurs within them; else pick the frame window that contains an entrance — inspect with the capture's per-frame id counts.)

- [ ] **Step 2: Capture script** — clone `capture_sam3p1_interactive_golden.py`, swap the interactive click for a text concept `add_prompt(text_str="person")` and use the `dance` frames; save per-frame `out_obj_ids` + `out_binary_masks`. Run in the reference env with `--patches` (delegate the reference-env run as in 1a).

- [ ] **Step 3: Parity test** — stream our mux predictor (`set_concept("person")`, forward per frame) over the same window; assert per-frame IoU ≥ 0.99 for every golden id (matched by id), including the mid-stream-spawned id.

- [ ] **Step 4: Run — PASS.** Debug the first frame whose IoU drops (usually the spawn frame's memory re-key or the append order).

- [ ] **Step 5: Commit** `test(sam3p1): model-find (detector mid-stream spawn) parity on dance`.

---

### Task 4: Interactive mid-stream click + co-seed

**Files:**
- Modify: `sam/models/sam3_predictor.py` — narrow `_check_mux_geometry` (allow mid-stream + co-seed); route clicks through interactive decode → `_grow_mux_state`.
- Create: `tests/parity/reference_sam3/capture_sam3p1_midstream_click_golden.py` + fixtures; `tests/parity/test_sam3p1_midstream_click_parity.py`.

- [ ] **Step 1: Interactive decode helper + routing** — add `_click_masks_multiplex(prompts, bf_int)` that runs `_build_mux_point_inputs` → `tracker._forward_sam_heads(interactive)` to produce a binarised `(n,1,ims,ims)` mask per clicked object; in `forward`, when `geometry_prompts` and `state.mux_state is not None`, call `_grow_mux_state(..., is_mask_from_pts=True)`. Remove the mid-stream + co-seed raises in `_check_mux_geometry`; keep box/mask + capacity raises.

- [ ] **Step 2: Co-seed** — on the seed frame with both a concept and clicks, seed via detector (`_seed_multiplex`) then `_grow_mux_state` with the click masks in the same `forward`.

- [ ] **Step 3: Goldens + parity** — capture (reference env): (a) click obj at frame 0, click a 2nd obj at frame k on bedroom → both tracked. Parity test streams our predictor, IoU ≥ 0.99 both ids.

- [ ] **Step 4: Run — PASS.**

- [ ] **Step 5: Commit** `feat(sam3p1): interactive mid-stream click + co-seed`.

---

### Task 5: Notebook demo + ledger

- [ ] **Step 1:** Add a `dance` model-find demo cell (text `"person"`, show a new id appearing mid-video). Execute headless (exit 0).
- [ ] **Step 2:** Ledger: mark **Feature 1b** `[x]` in `docs/superpowers/plans/2026-06-26-phase1-sam3-torch-inference.md`; note remaining (boxes/exemplars, negatives, multi-concept, mid-stream refine).
- [ ] **Step 3: Commit** `docs(sam3p1): dance model-find demo + ledger`.

---

## Self-Review

- **Spec coverage:** tracker port (T1), `_grow_mux_state` + detector spawn (T2), model-find parity (T3), interactive mid-stream + co-seed (T4), demo/ledger (T5), capacity raise (T1 unit), regression (T2/T4 rerun 1a). ✓
- **Placeholders:** the two capture scripts + frame-window selection note the single fragile spot (reference-env run + which frames contain an entrance) rather than vague text; the T1 GPU test degrades to `hasattr` + CPU-capacity if the feature-prep smoke is awkward. ✓
- **Type consistency:** `add_new_masks_to_existing_state(prev_output, new_masks, bf_int, bf_prop, multiplex_state, is_mask_from_pts) -> (out, new_idx)`; `_grow_mux_state(state, frame_idx, new_masks, is_mask_from_pts, new_ids, bf_int, bf_prop) -> dict`; `_det_masks_for_seed(det, new_objects) -> (masks, ids)`; `_click_masks_multiplex(prompts, bf_int) -> (masks, ids)`. Used consistently. ✓
