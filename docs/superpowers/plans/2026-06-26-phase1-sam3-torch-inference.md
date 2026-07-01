# Phase 1 — SAM 3 PyTorch Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SAM 3 promptable-concept segmentation (single-image + streaming video, full multiplex K>1) to the unified `sam/` package by vendoring SAM 3's model code and wrapping it in this repo's predictor / pluggable-memory-bank / streaming architecture — validated bit-for-tolerance against the official SAM 3.

**Architecture:** Vendor-and-wrap (spec D1). SAM 3 = a vision encoder shared by a DETR **detector** and a SAM 2-lineage **tracker**. We vendor the upstream nn modules under `sam/modeling/…` with version-in-class-name (`Sam3*`), keep submodule attribute names so `sam3.pt` loads strict, and compose them in `Sam3Predictor` (image) / `Sam3VideoPredictor` (streaming). Multiplex (object-packing for joint decode) stays **internal to `Sam3Tracker`** behind a per-object data-space seam, so the existing `ObjectMemoryBank` + streaming loop are reused unchanged.

**Tech Stack:** Python ≥3.12 (SAM 3 floor), PyTorch ≥2.7, Hydra/OmegaConf, pixi, pytest, the Perception-Encoder backbone + a CLIP-style text tower (vendored), `huggingface_hub` (already a dep).

**This is Phase 1 of the SAM 3 integration.** Phase 0 (the `sam2`→`sam` refactor) is complete on `develop` (through `783aaec`). Design: `docs/superpowers/specs/2026-06-26-sam3-integration-design.md` (§6–§11 = the SAM 3 model/data-flow/multiplex/concept/association; §16 = EfficientSAM3 readiness). Phase 2 (SAM 3 ONNX) is a separate later plan.

## Global Constraints

- **pixi only** — `pixi run python`, `pixi add`. No bare python/pip/conda. (Verify the env satisfies the SAM 3 floor: Python ≥3.12, torch ≥2.7; if the current default env is below, add a dedicated SAM 3 pixi feature/env rather than downgrading the existing ones.)
- **NEVER edit files with `sed`/`awk` on this host.** Git Bash `sed -i` injects a UTF-8 BOM (it silently broke config loading and SPDX headers in Phase 0). Use the Edit/Write tools, or a Python script run via a `.py` file. **Never pass non-trivial code through `python -c "…"`** — the Windows shell mangles `\n`/`\x` escapes (corrupted a file in Phase 0). Write a temp `.py` and run it.
- **Preserve submodule attribute names** when vendoring so `checkpoints/sam3.pt` (and `sam3.1_multiplex.pt`) load under strict key-matching. Rename *classes* (`Sam3*`) and *files*, never the `nn.Module` attribute paths the checkpoint keys reference. If a remap is unavoidable, write an explicit key-map in the builder.
- **Licensing (spec D11):** every vendored SAM 3 file and every new file that imports SAM 3 code carries `# SPDX-License-Identifier: LicenseRef-SAM`. Add `LICENSE_sam` (the SAM License text from the upstream repo) and a SAM 3 row in the README license table + `NOTICE`. SAM 2 / EfficientTAM files stay Apache-2.0. Weights are **never vendored** (gated download only — `tools/download_sam3.py` already exists).
- **Multiplex is internal to `Sam3Tracker`** — all shared tracker seams (`ObjectMemoryBank`, `SamTrackerBase` block methods, the streaming loop) operate in **data space (per-object)**. `mux`→compute→`demux` wraps stay inside the tracker.
- **Concept guard (spec D9):** `MAX_CONCEPTS = 1`; reject a 2nd concept and reject `set_concept` after the first frame — enforced only in `set_concept`, never baked into the types (the state holds a `list[ConceptState]`).
- **Reference-parity is the acceptance gate** (spec §14): each SAM 3 component is validated against the **official** SAM 3 output, captured once as fixtures (Task 1). Tolerances: feature/embedding `atol=1e-2` (large VL models drift more than SAM 2); masks IoU ≥ 0.99; boxes `atol=2px`; scores `atol=1e-2`.
- **Reuse the Phase-0 test harness** (`tests/characterization`, `tests/parity`) patterns + helpers; don't reinvent.

---

## File Structure

New (all `LicenseRef-SAM` unless noted):
```
sam/
  modeling/
    encoders/perception_encoder.py     # Sam3VisionEncoder (PE) + vendored vitdet/necks/vl_combiner helpers
    text/text_encoder.py  text/tokenizer.py   # Sam3TextEncoder, Sam3Tokenizer
    text/__init__.py
    decoders/detr_decoder.py           # Sam3DetrDetector + presence head
    decoders/multiplex_mask_decoder.py # MultiplexMaskDecoder
    multiplex.py                       # MultiplexState/MultiplexController (vendored multiplex_utils)
    tracking/sam3_tracker.py           # Sam3Tracker (mux/demux internal; RoPE attn)
    association/__init__.py
    association/associate.py           # associate_det_trk (stateless, vendored)
    association/tracklet.py            # TrackletManager (pending->confirmed->dead)
  prompts.py    # += ConceptPrompt [sam3]   (GeometryPrompt stays Apache)
  results.py    # += Sam3DetectionResult [sam3]
  models/sam3_predictor.py             # Sam3Predictor, Sam3VideoPredictor, Sam3VideoPredictorState, ConceptState
  configs/sam3/sam3.yaml  configs/sam3/sam3.1.yaml   # hydra translation of upstream config.json
build_sam.py  # += build_sam3, build_sam3_hf, build_sam3_video_predictor (+ _hf)
LICENSE_sam   # NEW (SAM License text)
tests/
  parity/reference_sam3/capture_sam3_golden.py   # runs official sam3 -> fixtures (run once, isolated env)
  parity/fixtures/sam3/*.npz                      # committed golden (small)
  parity/test_sam3_parity.py                      # new sam3 vs golden
  characterization/test_sam3_build.py             # build_sam3* instantiate + concept-guard unit tests
```
Reused unchanged: `sam/modeling/memory/{bank,banks,forgetful,attention,encoder}.py`, `sam/modeling/tracking/tracker_base.py` (`SamTrackerBase`), the streaming-loop shape from `Sam2VideoPredictor`, `tools/download_sam3.py`.

---

## Task 1: Reference golden + SAM 3 scaffolding (the oracle)

**Files:**
- Create: `tests/parity/reference_sam3/capture_sam3_golden.py`, `tests/parity/reference_sam3/README.md`
- Create: `LICENSE_sam`; Modify: `NOTICE`, `README.md` (license table + SAM 3 row)
- Modify: `pyproject.toml` (SAM 3 deps / env if needed)
- Generated: `tests/parity/fixtures/sam3/{image,video}.npz`

**Interfaces:**
- Produces: golden fixtures `image.npz` (keys: `boxes`, `scores`, `presence`, `masks` for a fixed image+phrase) and `video.npz` (keys: `frame{0..k}_obj{ids}` masklets) captured from **official** SAM 3 on `facebook/sam3` weights; consumed by Tasks 8–9.

- [ ] **Step 1: Stand up the official SAM 3 in an isolated location** (do NOT pollute this env). Clone upstream into a sibling dir and install in its own venv/uv env per upstream README (Python 3.12+, torch 2.7+, CUDA 12.6+):
```bash
git clone https://github.com/facebookresearch/sam3 ../sam3_reference
# follow ../sam3_reference/README.md to create its own env (NOT pixi here) and `pip install -e .`
```
- [ ] **Step 2: Confirm gated weights are present** — `checkpoints/sam3.pt` already downloaded (`pixi run download-sam3` exists). The reference loader reads `facebook/sam3` via HF; point it at the local `sam3.pt` or let it use the cached HF download.
- [ ] **Step 3: Write `capture_sam3_golden.py`** — replicate the upstream `examples/sam3_image_predictor_example` (a fixed image, e.g. `notebooks/images/truck.jpg` downscaled, phrase `"truck"`) and `examples/sam3_video_predictor_example.ipynb` flow (the golden scenario; forward-only: add text@frame0 → propagate; per spec §14). Run it in the **reference env**, save `np.savez_compressed` of boxes/scores/presence/masks (image) and per-frame masklets (video) to `tests/parity/fixtures/sam3/`. Keep inputs small (downscale) so fixtures are < a few MB.
- [ ] **Step 4: Run capture** in the reference env; commit the fixtures (they let CI run without the upstream repo/weights — spec §14).
- [ ] **Step 5: License scaffolding** — copy the upstream `LICENSE` to `LICENSE_sam`; add a SAM 3 row to the README license table (`SAM 3 → SAM License`) and to `NOTICE`. (SPDX headers go on the vendored files as they land, Tasks 2–9.)
- [ ] **Step 6: Verify** the Phase-0 suite still passes and fixtures exist:
```bash
pixi run pytest tests/ -q
ls tests/parity/fixtures/sam3/*.npz
```
Expected: Phase-0 tests green; both fixtures present.
- [ ] **Step 7: Commit** `test(sam3): capture reference SAM 3 golden + add LICENSE_sam`.

> Note: Task 1 produces the oracle. Tasks 2–9 each load `sam3.pt` weights into the vendored `Sam3*` modules and validate against these fixtures (or against per-block reference activations captured the same way if a component needs finer-grained checks).

---

## Task 2: Vision encoder — `Sam3VisionEncoder` (Perception Encoder)

**Files:** Create `sam/modeling/encoders/perception_encoder.py` (+ vendor `vitdet.py`/`necks.py`/`vl_combiner.py` helpers into `sam/modeling/encoders/`). Modify `sam/build_sam.py` (a minimal `build_sam3` stub that builds just the encoder for this task's test).
**Interfaces:** Produces `Sam3VisionEncoder.forward(image) -> (feats: list[Tensor], pos: list[Tensor])` (multi-level pyramid), shared by detector + tracker. Implements the `VisionEncoder` seam (spec §5).

- [ ] **Step 1: Vendor** `sam3/model/{encoder,vitdet,necks,vl_combiner}.py` into `sam/modeling/encoders/`, rename the top class to `Sam3VisionEncoder`, fix imports to `sam.…`, add `# SPDX-License-Identifier: LicenseRef-SAM`. **Preserve attribute names.**
- [ ] **Step 2: Write the failing test** `tests/parity/test_sam3_parity.py::test_encoder_parity` — load the encoder sub-state-dict from `checkpoints/sam3.pt`, run on the fixed image, compare the feature pyramid to a reference activation (capture it in Task 1 Step 3 as `image.npz['enc_feat_lastlevel']`).
- [ ] **Step 3: Run it — expect FAIL** (encoder not wired / key mismatch).
- [ ] **Step 4: Wire** weight loading (strict on the encoder subtree) in a minimal `build_sam3` path; fix any attribute-name mismatch by aligning vendored names to the checkpoint keys (inspect with `torch.load(...).keys()`).
- [ ] **Step 5: Run — expect PASS** (`atol=1e-2`).
- [ ] **Step 6: Commit** `feat(sam3): vendor Perception-Encoder vision encoder`.

---

## Task 3: Text encoder + tokenizer — `Sam3TextEncoder`

**Files:** Create `sam/modeling/text/{__init__,text_encoder,tokenizer}.py`.
**Interfaces:** Produces `Sam3TextEncoder.encode(phrases: list[str]) -> Tensor` and `Sam3Tokenizer`. Kept behind a clean `encode_text` seam (EfficientSAM3/MobileCLIP swap later, spec §16).

- [ ] **Step 1: Vendor** `sam3/model/{text_encoder_ve,tokenizer_ve}.py` → `sam/modeling/text/{text_encoder,tokenizer}.py`; classes `Sam3TextEncoder`/`Sam3Tokenizer`; SPDX SAM. Pull any tokenizer vocab/merges the upstream ships (or via HF) — do not vendor weights.
- [ ] **Step 2: Failing test** `test_text_parity` — encode `"truck"`, compare to `image.npz['text_emb']` (capture in Task 1).
- [ ] **Step 3: FAIL → Step 4: wire** text-tower weights from `sam3.pt` → **Step 5: PASS** (`atol=1e-2`).
- [ ] **Step 6: Commit** `feat(sam3): vendor text encoder + tokenizer`.

---

## Task 4: Detector — `Sam3DetrDetector` + `MultiplexMaskDecoder` + presence

**Files:** Create `sam/modeling/decoders/detr_decoder.py`, `sam/modeling/decoders/multiplex_mask_decoder.py`, `sam/modeling/multiplex.py`. Modify `sam/results.py` (add `Sam3DetectionResult`).
**Interfaces:** Produces `Sam3DetrDetector.detect(feats, text_emb, exemplar_emb=None) -> Sam3DetectionResult` (`masks_logits (N,H,W)`, `boxes (N,4)`, `scores (N,)`, `presence: float`, `instance_ids`). `MultiplexState`/`MultiplexController` from `sam/modeling/multiplex.py`.

- [ ] **Step 1: Vendor** `sam3/model/{decoder,geometry_encoders,multiplex_mask_decoder,maskformer_segmentation,multiplex_utils}.py` → the files above; classes `Sam3DetrDetector`, `MultiplexMaskDecoder`, `MultiplexState`/`MultiplexController`; SPDX SAM; preserve attribute names. (Reference: spec §8 for the mux/demux semantics.)
- [ ] **Step 2: Add `Sam3DetectionResult`** to `sam/results.py` (SPDX SAM):
```python
class Sam3DetectionResult:
    def __init__(self, masks_logits, boxes, scores, presence, instance_ids):
        # masks_logits (N,H,W) · boxes (N,4) xyxy · scores (N,) · presence float · instance_ids (N,)
        ...
    def to(self, device): ...
```
- [ ] **Step 3: Failing test** `test_detector_parity` — feats (Task 2) + text_emb (Task 3) → `detect()`; compare boxes (`atol=2px`), scores (`atol=1e-2`), top-mask IoU ≥ 0.99, presence (`atol=1e-2`) to `image.npz`.
- [ ] **Step 4: FAIL → wire** detector weights from `sam3.pt`; run multiplex at the trained K. → **Step 5: PASS**.
- [ ] **Step 6: Commit** `feat(sam3): vendor DETR detector + multiplex mask head`.

---

## Task 5: Tracker — `Sam3Tracker` (multiplex + RoPE, memory)

**Files:** Create `sam/modeling/tracking/sam3_tracker.py`; vendor `sam3/sam/{rope,transformer,mask_decoder,prompt_encoder,common}.py` into `sam/modeling/decoders/` (RoPE variant) + `sam/modeling/tracking/`. Possibly extend `sam/modeling/memory/bank.py` `ObjectMemory` if SAM 3 carries extra per-object state.
**Interfaces:** Produces `Sam3Tracker` exposing the `SamTrackerBase` block methods in **data space** (`condition_on_memories`/`decode`/`encode_memory`), with `mux→compute→demux` internal. Consumes shared `ObjectMemoryBank`.

- [ ] **Step 1: Vendor** `sam3/model/{sam3_tracker_base,sam3_tracker_utils,memory}.py` + `sam3/sam/*` → the files above; class `Sam3Tracker(SamTrackerBase)` or composition; SPDX SAM. Inject RoPE as the attention module (spec §5); if RoPE needs coords threaded through the transformer (not a drop-in), add a `Sam3` transformer variant rather than mutating the shared one.
- [ ] **Step 2: Verify `ObjectMemory` payload** — confirm SAM 3 per-object memory == `(mem, pos, ptr)`; if richer, extend the dataclass (keep it pluggable). 
- [ ] **Step 3: Failing test** `test_tracker_step_parity` — given a frame's feats + a seeded memory, one `track_step` (eval_multiplex at trained K, demuxed to per-object) matches a reference tracker activation (`video.npz['trk_f1']`) IoU ≥ 0.99.
- [ ] **Step 4: FAIL → wire** tracker weights; ensure mux/demux round-trips per-object. → **Step 5: PASS**.
- [ ] **Step 6: Commit** `feat(sam3): vendor multiplex tracker (RoPE, memory)`.

---

## Task 6: Concept data types + state guard

**Files:** Modify `sam/prompts.py` (`ConceptPrompt`), create `sam/models/sam3_predictor.py` (skeleton: `ConceptState`, `Sam3VideoPredictorState`, `set_concept`). Create `tests/characterization/test_sam3_build.py`.
**Interfaces:** Produces `ConceptPrompt`, `ConceptState`, `Sam3VideoPredictorState`, `set_concept` (spec §9). Pure-Python, no model — fast CPU unit tests.

- [ ] **Step 1: Write the failing guard tests** (real code in the test): `set_concept` returns id 0 first time; raises on a 2nd concept (`MAX_CONCEPTS=1`); raises after `state.started`.
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** `ConceptPrompt`(text, exemplars, negative_phrases) [SPDX SAM] in `sam/prompts.py`; `ConceptState`/`Sam3VideoPredictorState`/`set_concept` exactly per spec §9 (the dataclasses + `MAX_CONCEPTS` guard). `encode_text` embeds positives **and** negatives.
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** `feat(sam3): concept prompt/state + guarded set_concept`.

---

## Task 7: Association + tracklet lifecycle

**Files:** Create `sam/modeling/association/{__init__,associate,tracklet}.py`. Extend `tests/characterization/test_sam3_build.py`.
**Interfaces:** Produces `associate_det_trk(det_masks, track_masks, iou_threshold, iou_threshold_trk, det_scores, new_det_thresh) -> (new_dets, unmatched_tracks, det2track, matched_scores)` (stateless) and `TrackletManager` (pending→confirmed→dead via `masklet_confirmation` + presence/object-score gating).

- [ ] **Step 1: Vendor** `sam3/perflib/associate_det_trk.py` → `sam/modeling/association/associate.py` (stateless Hungarian on `1−IoU`); SPDX SAM. Drop the Triton fast-paths (use the torch/scipy reference) for Phase 1.
- [ ] **Step 2: Failing unit tests** on synthetic masks: identical det/track → matched; a novel high-score det → `new_dets`; a track with no det for K frames → killed by `TrackletManager`.
- [ ] **Step 3: FAIL → implement** `TrackletManager` (explicit state machine, holds tracklet→state in `Sam3VideoPredictorState`). → **Step 4: PASS.**
- [ ] **Step 5: Commit** `feat(sam3): det<->track association + tracklet lifecycle`.

---

## Task 8: `Sam3Predictor` (image concept predict) + builders/configs

**Files:** Modify `sam/models/sam3_predictor.py` (`Sam3Predictor`), `sam/build_sam.py` (`build_sam3`, `build_sam3_hf`), create `sam/configs/sam3/sam3.yaml`.
**Interfaces:** Produces `Sam3Predictor.predict(image, ConceptPrompt) -> Sam3DetectionResult` and `build_sam3(config, ckpt, device, ...)`. The shared encoder is **owned by `Sam3Predictor`** and injected into detector + tracker (spec §5 layering).

- [ ] **Step 1: Translate** the upstream `config.json` (from `sam3.pt` repo) into `sam/configs/sam3/sam3.yaml` — hydra `_target_`s pointing at the vendored `Sam3*` classes with exact dims. Cross-check against the reference build (spec §13). (Mind the dotted-dir `@package` BOM lesson — author the yaml with Write, never sed.)
- [ ] **Step 2: Failing test** `test_sam3_image_parity` — `build_sam3(sam3.yaml, checkpoints/sam3.pt)`, `predict(truck.jpg, ConceptPrompt("truck"))`; compare to `image.npz` (boxes atol 2px, scores atol 1e-2, mask IoU ≥ 0.99, instance count).
- [ ] **Step 3: FAIL → implement** `Sam3Predictor` composing encode_image→encode_text→detect (spec §7/§10); `build_sam3`/`build_sam3_hf` mirroring the SAM 2 builders (compose-based; the raw model builder pattern). → **Step 4: PASS.**
- [ ] **Step 5: Commit** `feat(sam3): image concept predictor + build_sam3`.

---

## Task 9: `Sam3VideoPredictor` (streaming) + reference-parity gate

**Files:** Modify `sam/models/sam3_predictor.py` (`Sam3VideoPredictor`), `sam/build_sam.py` (`build_sam3_video_predictor` + `_hf`). Add `tests/parity/test_sam3_parity.py::test_sam3_video_parity`.
**Interfaces:** Produces `Sam3VideoPredictor.forward(state, frame_idx, frame, geometry_prompts=[]) -> {obj_id: MaskletResult}` — the streaming loop of spec §10 (encode→gated detect→tracker_step→associate→TrackletManager→bank add/prune). `remove_object(state, obj_id)`; multi-pass re-propagation by re-feeding frames.

- [ ] **Step 1: Failing test** `test_sam3_video_parity` — replicate the official `sam3_video_predictor_example.ipynb` scenario (set concept → stream frames → collect per-object masks) on `sam3.pt`; compare per-frame masklets to `video.npz` (IoU ≥ 0.99, matching object ids).
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** the streaming `forward` (spec §10 pseudocode), reusing the forgetful `ObjectMemoryBank` + the `Sam2VideoPredictor` loop shape; detection gated on `concept is not None`; association + `TrackletManager` own spawn/confirm/kill; `remove_object`; unified obj-id allocator.
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Add** `test_sam3_video_constant_vram` — stream an N-frame clip with the forgetful bank, assert peak CUDA memory is ~flat vs N (the fork's headline property; spec §1/§14).
- [ ] **Step 6: Commit** `feat(sam3): streaming video predictor + reference parity`.

---

## Task 10: Packaging, configs, docs, full-suite green

**Files:** `sam/configs/sam3/sam3.1.yaml`; `pyproject.toml` (any SAM 3 export task stubs — NOT ONNX, that's Phase 2); `README.md` (SAM 3 usage section); `tests/characterization/test_sam3_build.py` (build smoke for sam3 + sam3.1, no ckpt).
**Interfaces:** Final public surface: `build_sam3`, `build_sam3_hf`, `build_sam3_video_predictor(+_hf)`, `Sam3Predictor`, `Sam3VideoPredictor`, `ConceptPrompt`.

- [ ] **Step 1: Add** `sam3.1` config + `build_sam3*_hf` for `facebook/sam3.1` (weights present: `sam3.1_multiplex.pt`).
- [ ] **Step 2: Build smoke test** (CPU, no ckpt): `build_sam3` / `build_sam3_video_predictor` on `configs/sam3/sam3.yaml` instantiate (mirrors `test_build_instantiate`). Guard with skip-if-checkpoint-absent where a ckpt is needed.
- [ ] **Step 3: README** — SAM 3 concept-segmentation quickstart (image + streaming video) using the new API; confirm the snippet runs.
- [ ] **Step 4: Full suite green:** `pixi run pytest tests/ -q` (Phase-0 + SAM 3). Confirm `import sam` walk + Phase-0 parity still pass.
- [ ] **Step 5: Commit** `feat(sam3): sam3.1 config, build smoke, docs`.

---

## Self-Review (author)

- **Spec coverage:** encoder (T2), text (T3), detector+multiplex+presence (T4/§8), tracker+RoPE+memory (T5), ConceptPrompt/Sam3DetectionResult/guard (T6/§6/§9), association+lifecycle (T7/§11), image predictor+builders+configs (T8/§7/§13), streaming video+constant-VRAM (T9/§10), packaging/docs/sam3.1 (T10). Reference-parity gate (§14) = T1 + per-task. License D11 = T1 + per-file SPDX. ✓
- **Deferred correctly:** ONNX (Phase 2), training/eval, multi-concept (guard stays at 1), EfficientSAM3 (§16). ✓
- **Risk flags for the executor:** (a) **never `sed`** / never `python -c` for edits — Phase-0 BOM + escape footguns; (b) the SAM 3 floor (py3.12/torch2.7) may exceed the current default env → add a dedicated pixi env, don't downgrade; (c) `state_dict` key alignment is the main vendoring hazard — inspect `sam3.pt` keys and keep attribute names; (d) tolerances are looser than Phase 0 (large VL model) — `atol=1e-2`, IoU ≥ 0.99; (e) capturing reference golden needs the official sam3 env once — isolate it; (f) RoPE may force a `Sam3` transformer variant (not a drop-in).
- **Type consistency:** `build_sam3` (raw/predictor), `Sam3DetectionResult`, `MaskletResult` (reused per-object tracker output), `ConceptPrompt`, `Sam3VideoPredictorState` used consistently across T6/T8/T9.
