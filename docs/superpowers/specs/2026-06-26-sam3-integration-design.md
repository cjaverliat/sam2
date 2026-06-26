# SAM 3 Integration — Design Spec

- **Date:** 2026-06-26
- **Status:** Draft (awaiting user review)
- **Scope:** Unify SAM 2 / EfficientTAM / SAM 3 into a single `sam/` package; add SAM 3
  promptable-concept segmentation for images and streaming video, reusing this fork's
  modular-predictor / pluggable-memory-bank / streaming / ONNX architecture.

---

## 1. Goal

Integrate SAM 3 (detector + tracker sharing a vision encoder, 848M params) into this fork
so that:

1. SAM 2 and SAM 3 live in one `sam/` package, sharing flexible components; the *version*
   of any version-specific class is identifiable from its **class name** (`Sam2*` / `Sam3*`).
2. SAM 3 video runs through this fork's **streaming, constant-VRAM** loop with a **pluggable
   memory bank** — segment a concept across a video without loading the whole video into VRAM.
3. Readability is the top priority: clear names, well-separated responsibilities, no
   speculative abstraction.

## 2. Scope

**In scope (this plan):**
- Phase 0 — refactor `sam2/` → `sam/` (clean break) + responsibility-oriented reorg.
- Phase 1 — SAM 3 PyTorch inference: single-image concept predictor + streaming video predictor,
  with **full multiplex (K>1)** for reference parity.
- Phase 2 — SAM 3 ONNX/TRT export, **reuse-first** (PE encoder + text tower + tracker blocks).

**Deferred (follow-up specs):**
- DETR detector ONNX export (→ fully checkpoint-free SAM 3 ONNX).
- Multiple / mid-stream concepts (structure ready, behavior guarded off).
- Training + eval (SA-Co losses, matcher, datasets, HOTA/TETA/CGF1).
- Demo / backend (concept prompting UI + server).
- EfficientSAM3 port (distilled SAM 3, lightweight encoders) — see §16.

## 3. Decision log

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | **Vendor-and-wrap** SAM 3 model code into the new package; wrap in this repo's predictor/bank/streaming/ONNX architecture | Reuse the *deployment architecture*, not SAM 2 internals; SAM 3's NN is mostly net-new |
| D2 | **Clean break** `sam2` → `sam`; major version bump; no `sam2` import shim | Cleanest, most readable; downstream `import sam2` breaks (accepted) |
| D3 | **Approach B (phased)** responsibility-oriented reorg | Structure *and* class name both convey SAM2/SAM3/shared; phasing keeps each step test-green |
| D4 | Rename `Sam2Generic*` → **canonical** names; legacy monolithic demoted | "Generic" is vague; clean break frees the plain names |
| D5 | SAM 3 ONNX **reuse-first**: export PE encoder + text + tracker; detector stays torch | Detector (DETR set-prediction, RoPE, Triton-adjacent) is the riskiest graph |
| D6 | **Skip demo** in this plan | Focus on library + ONNX; demo is large and orthogonal |
| D7 | Distribution/wheel name = **`sam`** | Matches the import; publish privately / via release artifacts |
| D8 | SAM 3 checkpoints via **gated HF download task, fail loud** | Weights are access-gated; no silent fallback |
| D9 | Concepts = **guarded 0/1 collection** (structure multi-ready) | YAGNI: ship single-concept, keep multi-concept seam |
| D10 | Multiplex = **internal `Sam3Tracker` batching**, **full K>1 in Phase 1** (faithful) | Phase-1 parity gate needs real joint-decode; bank/streaming still see per-object via demux |
| D11 | **Mixed licensing**, per component: SAM2/ETAM = Apache-2.0 (+ BSD-3 cctorch), SAM3 (+ derivatives) = SAM License, EfficientSAM3 = SAM License + backbone licenses. Per-file SPDX; **explicit per-model license table in README**; weights never vendored | Keep each owner's grant intact; permissive parts stay independently usable; SAM3 parts carry SAM License terms |
| D12 | **Build a characterization harness first** (golden SAM2/ETAM torch+ONNX outputs) | Repo has **no tests**; the refactor needs a regression oracle |

### Naming map (D4)
```
SAM2Generic                -> Sam2Predictor            (model + block API + image predict)
SAM2GenericVideoPredictor  -> Sam2VideoPredictor       (adds streaming state)
SAM2VideoPredictor (legacy)-> Sam2LegacyVideoPredictor
build_sam2_generic_*       -> build_sam2_*
SAM2Base                   -> SamTrackerBase           (shared per-frame tracker network)
SAM2Prompt                 -> GeometryPrompt           (shared)
SAM2Result                 -> MaskletResult            (shared)
SAM3:  Sam3Predictor / Sam3VideoPredictor / Sam3Tracker
```
> Naming veto open: `GeometryPrompt` / `MaskletResult` (shared, neutral) vs keeping `Sam2Prompt` / `Sam2Result`.

## 4. Target package layout

`[shared]` = neutral name, ≥2 implementers · `[sam2]` / `[sam3]` = version in class name.

```
sam/
  __init__.py                     # hydra initialize_config_module("sam")
  build.py                        # build_sam2_* + build_sam3_*  (was build_sam.py)
  configs/
    sam2/  sam2.1/  efficienttam/ # unchanged yaml; _target_ paths updated
    sam3/                         # NEW
  prompts.py                      # GeometryPrompt [shared] · ConceptPrompt [sam3]
  results.py                      # MaskletResult [shared] · Sam3DetectionResult [sam3]
  modeling/
    position_encoding.py          # [shared]
    utils.py                      # [shared, was sam2_utils]
    encoders/                     # (was backbones/)
      image_encoder.py            # VisionEncoder base + Sam2ImageEncoder [shared/sam2]
      hiera.py  vitdet.py  utils.py
      perception_encoder.py       # Sam3VisionEncoder (PE) [sam3, vendored]
    text/                         # NEW
      text_encoder.py  tokenizer.py        # Sam3TextEncoder, Sam3Tokenizer [sam3]
    prompt/
      prompt_encoder.py           # geometry prompt encoder [shared]
    decoders/                     # (was modeling/sam/ — resolves name collision)
      transformer.py  mask_decoder.py  onnx_compat.py     # [shared]
      multiplex_mask_decoder.py   # MultiplexMaskDecoder [sam3, net-new]
      detr_decoder.py             # Sam3DetrDetector + presence head [sam3]
    memory/
      bank.py                     # ObjectMemoryBank ABC + ObjectMemory [shared]
      banks.py                    # Sam2ObjectMemoryBank [sam2]
      forgetful.py                # ForgetfulObjectMemoryBank [shared strategy]
      attention.py  encoder.py    # [shared]
    tracking/
      tracker_base.py             # SamTrackerBase [shared, was SAM2Base]
      sam3_tracker.py             # Sam3Tracker (mux/demux internal) [sam3]
    association/                  # NEW
      associate.py                # associate_det_trk (stateless) [sam3]
      tracklet.py                 # TrackletManager lifecycle [sam3]
  models/
    sam2_predictor.py             # Sam2Predictor, Sam2VideoPredictor [sam2]
    sam3_predictor.py             # Sam3Predictor, Sam3VideoPredictor [sam3]
    legacy_video_predictor.py     # Sam2LegacyVideoPredictor [sam2]
  onnx/
    ort_block.py  trt_options.py  # [shared]
    blocks/                       # per-block ORT wrappers (sam2 + sam3)
    sam2.py  sam3.py              # attach_onnx_blocks per version
  utils/{amg,misc,transforms}.py  # [shared]
```

## 5. Shared seams (each has ≥2 real implementers → abstraction justified)

| Seam | Contract (data space, per-object) | SAM2 | SAM3 |
|------|-----------------------------------|------|------|
| `VisionEncoder` | `forward(img) -> (feats[], pos[])` multi-level pyramid | `Sam2ImageEncoder` | `Sam3VisionEncoder` (PE), shared by detector+tracker |
| `MaskDecoder` / `TwoWayTransformer` | RoPE **injected as attention module** (DIP), not duplicated | std attn | RoPE attn |
| `ObjectMemoryBank` (ABC, exists) | select / add / prune, per-object | `Sam2ObjectMemoryBank`, `Forgetful…` | **unchanged** |
| `SamTrackerBase` (was `SAM2Base`) | block methods: `encode_image` / `encode_prompts` / `condition_on_memories` / `decode` / `encode_memory` | `Sam2Predictor` | `Sam3Tracker` |
| `OrtBlock` | `run(inputs) -> outputs` | 5 SAM2 wrappers | SAM3 wrappers |
| Predictor loop | `forward(state, frame_idx, frame, …)` | `Sam2VideoPredictor` | `Sam3VideoPredictor` |

**Stays concrete SAM3-only (no speculative Protocol/ABC):** `Sam3TextEncoder`, `Sam3Tokenizer`,
`Sam3DetrDetector`, `MultiplexMaskDecoder`, presence head, `associate.py`, `tracklet.py`.

**Key principle:** all shared tracker seams are defined in **data space (per-object in/out)**.
SAM3's multiplex bucketing is an *internal* detail of `Sam3Tracker`, invisible to the bank,
streaming loop, and `SamTrackerBase`.

**Layering (important).** SAM 2's tracker *is* the model — `Sam2Predictor` **inherits** `SamTrackerBase`.
SAM 3's tracker is *one component* — `Sam3Predictor` **composes** `Sam3Tracker` (+ encoder, text,
detector) and reaches tracker block methods via `self.tracker.…` (delegation, not inheritance). The
**shared vision encoder is owned by `Sam3Predictor`**, run once per frame, and its features are
*injected* into both detector and tracker (the tracker does not own `encode_image`).

## 6. New data types

- `ConceptPrompt` [sam3] — `text: str`, optional `exemplars` (boxes/masks on a ref frame),
  optional `negative_phrases`. Per-*concept*. `encode_text` embeds positives **and** negatives; both
  flow into `detect()` (negatives sharpen the presence head / suppress near-misses).
- `Sam3DetectionResult` [sam3] — `masks_logits (N,H,W)`, `boxes (N,4)`, `scores (N,)`,
  `presence: float`, `instance_ids`. The detector's per-frame set output.
- `GeometryPrompt` [shared, was `SAM2Prompt`] — points/boxes/mask per `obj_id`. **Dual role:**
  spawns a new object when `obj_id` is new, refines when it already exists (matches SAM 2).
- `MaskletResult` [shared, was `SAM2Result`] — per-object tracker output.

## 7. SAM 3 model decomposition

`Sam3Predictor` block methods (mirror SAM 2's seam + detector/text):
```
encode_image(frame)       -> PE feature pyramid    # shared by detector AND tracker, run once
encode_text(phrase)       -> text_emb              # cached once per concept
encode_exemplars(ex)      -> exemplar_emb           # geometry encoder, optional
detect(feats, text_emb, exemplar_emb) -> Sam3DetectionResult   # DETR set + presence
# delegated to the composed Sam3Tracker (predictor owns the shared encoder, not the tracker):
tracker.condition_on_memories(...) / tracker.decode(...) / tracker.encode_memory(...)
```
The vision encoder returns the full pyramid; detector and tracker each select the levels
they need (detector = multi-scale; tracker = stride-16 + hi-res stride-4).

## 8. Multiplex — what it is and why it doesn't touch the bank

Verified against `sam3/model/multiplex_utils.py`, `multiplex_mask_decoder.py`,
`sam3_multiplex_tracking.py`.

**What:** packing of multiple *objects* into shared **buckets** (not multi-hypothesis).
- data space `(num_objects, …)` ↔ multiplex space `(num_buckets, multiplex_count, …)`,
  `num_buckets = ceil(N / K)`, `K = multiplex_count`.
- `mux` / `demux` are precomputed permutation-matrix conversions.
- The mask decoder runs at `batch = num_buckets`: one shared feature map per bucket decodes
  all `K` objects in one transformer pass; their tokens self-attend (joint decode).
- In the tracker, memory attention runs at `batch = num_buckets`; `maskmem_features` / `obj_ptr`
  are **demuxed to per-object before storage** and **muxed back** on read.

**Why:** efficiency (image-feature + memory-attention cost ~ `N/K`, not `N`; matters when a
concept yields many instances) + joint cross-object decode (non-overlap / presence disambiguation).

**Impact on our design:**
- Memory bank: **none** — sees per-object tensors (demux precedes storage). Reuse holds.
- Streaming loop: **none** — orchestrates per-object; mux/demux is internal to the tracker step.
- Mask decoder: **must port `MultiplexMaskDecoder`** — token embeddings are sized `K`; SAM 3
  weights require it.
- `Sam3Tracker`: memory-attn / decode / memory-encode run in bucket space; `mux → compute → demux`
  wrap stays internal.

**Phase 1 implements full K>1 (D10).** Each frame, `Sam3Predictor` builds a `MultiplexState` from the
current active-object set, `mux`es per-object tensors into buckets, runs the tracker blocks at
`batch = num_buckets`, then `demux`es back to per-object before the bank stores them. So the bank and
streaming loop still see **per-object** data; bucketing is fully internal to `Sam3Tracker`. We do
**not** use the buckets-of-1 shortcut — the Phase-1 parity gate (match reference SAM 3) needs the real
joint-decode. ONNX (Phase 2): the tracker blocks gain a **dynamic `num_buckets` axis** (fixed
`multiplex_count = K`).

## 9. Concept state + guard (D9)

```python
@dataclass
class ConceptState:
    concept_id: int
    prompt: ConceptPrompt            # original (text, exemplars, negatives)
    text_emb: torch.Tensor           # encoded once
    exemplar_emb: torch.Tensor | None

@dataclass
class Sam3VideoPredictorState:
    video_hw: tuple[int, int]
    bank: ObjectMemoryBank = field(default_factory=ForgetfulObjectMemoryBank)
    concepts: list[ConceptState] = field(default_factory=list)   # 0..1 now; list keeps multi open
    num_frames_processed: int = 0
    _next_obj_id: int = 0

    @property
    def started(self) -> bool:
        return self.num_frames_processed > 0

MAX_CONCEPTS = 1   # relax (or remove) to enable multi-concept

def set_concept(self, state, concept: ConceptPrompt) -> int:
    if state.started:
        raise RuntimeError("concept must be set before the first frame is processed")
    if len(state.concepts) >= MAX_CONCEPTS:
        raise RuntimeError(f"at most {MAX_CONCEPTS} concept(s) supported")
    cid = len(state.concepts)
    text_emb = self.encode_text(concept.text)
    ex_emb = self.encode_exemplars(concept.exemplars) if concept.exemplars else None
    state.concepts.append(ConceptState(cid, concept, text_emb, ex_emb))
    return cid
```
All current limitations live in `set_concept` (one place). Extension path (no refactor):
raise `MAX_CONCEPTS` → relax `started` guard → loop detector over `state.concepts`, tag
tracklets with `concept_id`.

## 10. Data flow

**Image — `Sam3Predictor.predict(image, ConceptPrompt)`:**
`encode_image → encode_text → detect → (optional geometry refine) → masks`. Single frame,
multi-instance, no memory.

**Streaming video — `Sam3VideoPredictor.forward(state, frame_idx, frame, geometry_prompts=[])`:**
```
active_ids = state.bank.known_obj_ids
state.num_frames_processed += 1
feat = encode_image(frame)                              # PE, once per frame (shared)
concept = state.concepts[0] if state.concepts else None

# Concept-driven detection (gated): only runs when a concept is set -> finds new instances.
det = detect(feat, concept.text_emb, concept.exemplar_emb) if concept is not None else None

sel  = state.bank.select_memories(active_ids, frame_idx, ...)    # bounded / forgetful
trk  = tracker_step(feat, sel)                          # propagate tracklets -> MaskletResult/id

if det is not None:
    matches = associate(det, trk)                       # Hungarian/IoU + presence gate
    tracklet_mgr.apply(matches, det)                    # spawn / confirm / kill concept instances

# Geometry prompts reuse the SAM 2 tracker path: new obj_id -> spawn, existing -> refine.
route geometry_prompts to tracker per obj_id

mem = encode_memory(feat, masks)
state.bank.try_add_memories(...); state.bank.prune_memories(...) # bounds VRAM
return {obj_id: MaskletResult}
```
- Frame-at-a-time; **no cross-frame feature cache** (deliberate — constant VRAM, diverges from
  SAM 3's growing `feature_cache`).
- Detection is **gated**: the DETR detector runs only when a concept is set (free perf win when
  propagating). Geometry prompts spawn/refine via the reused SAM 2 tracker path.
- Deferred enhancement: SAM 3 also lets geometry/exemplars *condition the detector*
  (`allow_new_detections = has_text or has_geometric_prompt`); we add that later if needed.

**Session API (matches the golden notebook's flow).** Beyond `set_concept` + per-frame `forward`,
the predictor exposes **`remove_object(state, obj_id)`** (kill tracklet + purge its bank memories) and
supports **multi-pass re-propagation**: refinement (add text → propagate → remove obj → add clicks to
an obj → re-propagate) is done by **re-streaming** frames from the source for another pass — state and
bank persist; we keep no stored frames. A single **obj-id allocator** issues ids for both
detector-spawned and user-prompted objects, prevents collisions, and survives remove-then-re-add of an id.

## 11. Association + tracklet lifecycle

- `association/associate.py` — **stateless** port of `associate_det_trk`: Hungarian on `1 − IoU`
  + `iou_threshold_trk`, `new_det_thresh` for spawns; returns
  `(new_dets, unmatched_tracks, det→track map, matched_scores)`.
- `association/tracklet.py` — `TrackletManager` (in `state`): explicit
  **pending → confirmed → dead** state machine driven by `masklet_confirmation`
  + `object_score_logits` / presence gating (not a bare frame counter).
- Predictor orchestrates; matcher stays a pure function.

## 12. ONNX / TRT (Phase 2, reuse-first, hybrid)

- Export as `OrtBlock`s: `Sam3VisionEncoder` (PE), `Sam3TextEncoder`, tracker memory-attention /
  `MultiplexMaskDecoder` / memory-encoder.
- **Detector stays torch** (deferred). Consequence: SAM 3 ONNX is **hybrid and not
  checkpoint-free** (detector weights still loaded) until the detector is exported.
- `onnx/sam3.py::attach_onnx_blocks` mirrors the SAM 2 meta-build-then-swap flow; precision via
  exported graph + `TensorRTOptions` (same fp16-safe-block discipline; the recurrent tracker is
  fp16-sensitive).
- Association / NMS / Hungarian / lifecycle remain torch/Python (same boundary as the existing
  5-block design).

## 13. Build / config / packaging

- **Hydra:** `sam/__init__.py` → `initialize_config_module("sam")`. Every config `_target_`
  rewritten to `sam.…`. `configs/sam3/` added.
- **pyproject:** `name = "sam"`; `packages.find include = ["sam*"]`; `package-data` key `sam`;
  native ext `sam._C`; bump major version. Pixi tasks rewritten (`configs/sam3/…`, etc.).
- **Builders:** `build_sam3_*` (+ `build_sam3_hf`, `build_sam3_video_predictor`,
  `build_sam3_video_predictor_onnx`) mirror the SAM 2 builders (hydra compose/instantiate;
  meta-build for ONNX).
- **Checkpoints (gated, self-authenticating, fail loud):** `tools/download_sam3.py` first
  ensures a valid HF login — reuses a cached token / `HF_TOKEN`, else prompts once
  (interactive, then cached); fails loud if there is no TTY to prompt — then
  `hf_hub_download`s the weights into `checkpoints/`. A remaining gated 401 → guidance to
  accept access on the model page. Pixi tasks `download-sam3` / `download-sam3-1` with
  `outputs=[…]` cache-skip. Adds `huggingface_hub` (conda) as a dependency. Sets
  `HF_HUB_DISABLE_XET=1` (the Xet transfer backend can hang on some setups).
- **Licensing (D11) — mixed, per component:** no single umbrella. Ship every license text
  (`LICENSE_apache2` for SAM 2 / EfficientTAM, `LICENSE_cctorch` BSD-3 for the CUDA ext, `LICENSE_sam`
  for SAM 3) + a `NOTICE`. Every file carries an **SPDX header** naming its license: SAM3-derived +
  glue that imports SAM3 = `LicenseRef-SAM`; verbatim/derived SAM2/ETAM = `Apache-2.0` (or
  `BSD-3-Clause`). `pyproject` → `license = {file = "LICENSE"}`, where the top-level `LICENSE` states
  the mix and points to the per-component files + README table. **README carries an explicit per-model
  license table** (model → license → key restrictions). Permissive parts stay independently usable;
  using the SAM 3 parts binds you to the SAM License (no weapons/ITAR, trade controls, revocable).
  Weights never vendored (gated download only).
- **Checkpoint loading:** vendoring must **preserve module attribute names / `state_dict` keys** so
  `sam3.pt` (and SAM2/ETAM `.pt`) load under strict key-matching; otherwise add an explicit key remap.
- **SAM 3 hydra configs:** hand-translate SAM 3's `config.json` architecture into `configs/sam3/*.yaml`
  that instantiate the vendored `Sam3*` modules at exact dims — cross-check vs the reference build, as
  a silent dim mismatch fails the load or degrades output.

## 14. Phasing + verification gates

**Phase 0 — refactor `sam2` → `sam` (zero behavior change)**
- 0-pre *characterization harness* — the repo has **no tests**, so first freeze golden outputs
  (masks/logits, torch + ONNX) for SAM 2 + EfficientTAM on fixed inputs. This is the regression oracle.
- 0a *mechanical rename* (+ per-file SPDX headers, D11) → verify: harness reproduces frozen outputs;
  **preserve `state_dict` keys**.
- 0b *responsibility reorg + class renames + seam extraction* → verify: harness still green.

**Phase 1 — SAM 3 torch (image + streaming video, full multiplex K>1)**
- Vendor `Sam3VisionEncoder`, `Sam3TextEncoder` + tokenizer, `Sam3DetrDetector` +
  `MultiplexMaskDecoder`, `Sam3Tracker`; new data types; concept guard; `TrackletManager`;
  `associate_det_trk`; `build_sam3_*`; `configs/sam3/`; gated download. Add `LICENSE_sam` + the SAM3
  row in the README license table; SPDX headers on every vendored file.
- Verify: image masks match reference SAM 3 within tolerance; **reproduce the official
  `examples/sam3_video_predictor_example.ipynb` scenario end-to-end and match its masklets
  within tolerance** (golden reference, see Reference parity below); streaming masklets stable;
  **peak VRAM flat vs video length** with the forgetful bank.

**Phase 2 — SAM 3 ONNX/TRT (reuse-first, hybrid)**
- Export encoder + text + tracker blocks; `attach_onnx_blocks` for SAM 3; detector torch.
- Verify: ORT/TRT vs torch parity within tolerance; hybrid streaming runs on CUDA + TRT EP.

**Reference parity (acceptance gate).** After each SAM 3 phase, validate against the *official*
implementation, not only internal tests. The golden scenario is the upstream notebook
[`examples/sam3_video_predictor_example.ipynb`](https://github.com/facebookresearch/sam3/blob/main/examples/sam3_video_predictor_example.ipynb):
run the same video + concept prompt through both reference SAM 3 and our `Sam3VideoPredictor`,
then compare per-frame masklets (mask IoU within tolerance), instance counts, and object ids.
Our public API should make reproducing that notebook's flow natural (set concept → stream frames →
collect per-object masks). Phase 2 re-runs the identical scenario through the hybrid ORT/TRT path.
Capture the reference outputs once as fixtures so the check runs in CI without the upstream repo
or gated weights present.

## 15. Risks & mitigations

1. **No project tests** → build the Phase-0 characterization harness *first*; it is the only
   regression oracle for the rename.
2. **Mixed licensing** → per-component licenses (Apache/BSD for SAM2-ETAM, SAM License for SAM3),
   per-file SPDX, explicit per-model README table; legal review before any public release; weights never vendored.
3. **Phase-0 rename surface large** (hydra discovery, `_target_`, native ext, pixi, `state_dict` keys)
   → 0a pure-mechanical + harness check; preserve attribute names.
4. **Multiplex K>1 + RoPE faithful port** (mux/demux + `MultiplexMaskDecoder`) → fixed-input parity vs
   reference. RoPE may **not** be a drop-in attention (needs coords threaded) → could force a `Sam3`
   transformer variant (mild duplication).
5. **SAM 3 hydra config vs `config.json`** dim mismatch → cross-check architecture vs reference build.
6. **SAM 3 weights gated** → no-token CI → skip-if-absent tests; harness fixtures captured once.
7. **Bank conditionality** assumed prompt-driven → generalize "conditioning frame" to detection-spawned
   objects; verify `ObjectMemory` payload `(mem, pos, ptr)` matches SAM 3 (extend if richer).
8. **Detector runs every frame** when a concept is set (heaviest new block) → detection-cadence knob
   later if needed.
9. **PE + text tower** large new weights/deps; **hybrid ONNX not checkpoint-free** until detector export.

## 16. EfficientSAM3 readiness (forward-looking)

Planned follow-up: port **EfficientSAM3** (https://github.com/SimonZeng7108/efficientsam3) — a
*distillation* of SAM 3 (~848M → ~90M) that keeps SAM 3's detector, tracker, and memory and only
swaps the two encoders: vision (PE ViT-H → EfficientViT / RepViT / TinyViT) and text (heavy tower →
MobileCLIP / "litetext"). It is the EfficientTAM-of-SAM 3.

**Does it change this design? No — it validates it.** It slots into the same structure, exactly as
EfficientTAM coexists with SAM 2 today:
- **Vision encoder** — already a seam (`VisionEncoder`, multi-level pyramid contract). The efficient
  backbones (`EfficientSam3VisionEncoder` family: EfficientViT / RepViT / TinyViT) are additional
  implementers under `modeling/encoders/`. Keep the contract backbone-agnostic (CNN/hybrid, not
  PE-specific) so a non-ViT trunk drops in.
- **Text encoder** — currently concrete `Sam3TextEncoder` (one implementer → no abstraction, per
  KISS/YAGNI). EfficientSAM3 adds a 2nd real implementation (MobileCLIP); at that point `encode_text`
  generalizes into a small `TextEncoder` seam. Until then keep PE-text-tower specifics out of the
  predictor so the swap stays non-invasive.
- **Detector / tracker / memory / streaming / bank / multiplex** — retained from SAM 3, reused unchanged.
- **Naming / configs** — `EfficientSam3*` classes, `configs/efficientsam3/`, consistent with the
  EfficientTAM precedent and the version-in-class-name rule.

**Caveats for the future port (not this plan):**
- Built via multi-stage **knowledge distillation** → training-heavy; lands with the deferred training
  phase. Near-term piece is *inference* (load distilled checkpoints, run through the same
  `Sam3Predictor` / `Sam3VideoPredictor`).
- **Licensing** — vendors third-party backbones (EfficientViT, RepViT, TinyViT, MobileCLIP); some
  (notably Apple MobileCLIP) carry restrictive / research-only terms. Verify each before vendoring.
- Extra deps (e.g. EfficientViT ships a Triton RMSNorm kernel) — env + ONNX-export implications.

**Net:** no change to the Phase 0–2 design; add EfficientSAM3 as a later variant once SAM 3 is in.
Confirms the value of keeping encoders behind a clean seam with the version in class names.

## 17. Open questions

- Naming **resolved**: neutral shared data types (`GeometryPrompt`, `MaskletResult`).
- License **resolved** (D11): mixed, per component (SAM 2 + EfficientTAM = Apache-2.0, BSD-3 cctorch,
  SAM 3 = SAM License), explicit per-model README table. Remaining: legal review before any public release.

## 18. References

- SAM 3 repo: https://github.com/facebookresearch/sam3
- **Golden parity reference** — video predictor example notebook:
  https://github.com/facebookresearch/sam3/blob/main/examples/sam3_video_predictor_example.ipynb
- Checkpoint loader (repo / filenames): `sam3/model_builder.py::download_ckpt_from_hf`
- Multiplex internals: `sam3/model/multiplex_utils.py`, `multiplex_mask_decoder.py`,
  `sam3_multiplex_tracking.py`
