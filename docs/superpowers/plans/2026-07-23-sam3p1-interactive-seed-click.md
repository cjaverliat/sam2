# SAM 3.1 interactive seed-frame click — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the SAM 3.1 multiplex video predictor seed and track objects from point-clicks on the first frame (text-free interactive VOS), validated for parity against upstream.

**Architecture:** Pure predictor-level wiring. A new pure helper batches point-only `GeometryPrompt`s into the tracker's `point_inputs` dict; a new `_seed_points_multiplex` mirrors the proven `_seed_multiplex` (detector seeding) but feeds `point_inputs` instead of `mask_inputs`; `Sam3MultiplexVideoPredictor.forward` routes seed-frame point prompts there and narrows the blanket `NotImplementedError` to the genuinely-unsupported cases. No tracker changes (the interactive stack is already instantiated + strict-loaded).

**Tech Stack:** PyTorch, pixi (`notebooks` env has matplotlib; base env for pytest), the upstream `../sam3_reference/.venv` for golden capture.

**Design:** `docs/superpowers/specs/2026-07-23-sam3p1-interactive-seed-click-design.md`

## Global Constraints

- **Preserve checkpoint attribute names** — do not rename any `nn.Module` attribute the mux tracker exposes; the interactive submodules already strict-load (157 keys).
- **Never `sed` / never `python -c` for edits** (BOM / escape footguns) — use Write/Edit.
- **Parity tolerances (phase1):** per-frame masklet IoU ≥ 0.99, matching object ids.
- **CUDA required** for the mux predictor (bf16-only regime); parity + smoke tests skip if no GPU / checkpoint.
- **80-col, Google style, `X | None`** in new code.

---

### Task 1: Pure point-batching helper

**Files:**
- Modify: `sam/models/sam3_predictor.py` (add module-level `_build_mux_point_inputs` near `_masks_to_boxes`)
- Test: `tests/test_sam3p1_point_inputs.py` (create)

**Interfaces:**
- Produces: `_build_mux_point_inputs(prompts: list[GeometryPrompt], video_hw: tuple[int,int], image_size: int, device) -> tuple[dict, list[int]]` returning `({"point_coords": (n,P,2) float, "point_labels": (n,P) int32}, obj_ids)`. Ragged point counts are right-padded with coord `(0,0)` / label `-1` (SAM padding convention).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sam3p1_point_inputs.py
import torch
from sam.models.sam3_predictor import _build_mux_point_inputs
from sam.prompts import GeometryPrompt


def test_batches_and_scales_points():
    # two objects, ragged point counts (2 and 1) -> padded to P=2 with label -1.
    p0 = GeometryPrompt(
        obj_id=1,
        points_coords=torch.tensor([[480.0, 270.0], [0.0, 0.0]]),   # centre + origin
        points_labels=torch.tensor([1, 0]),
    )
    p1 = GeometryPrompt(
        obj_id=2,
        points_coords=torch.tensor([[960.0, 540.0]]),               # bottom-right corner
        points_labels=torch.tensor([1]),
    )
    inp, ids = _build_mux_point_inputs(
        [p0, p1], video_hw=(540, 960), image_size=1008, device=torch.device("cpu")
    )
    assert ids == [1, 2]
    assert inp["point_coords"].shape == (2, 2, 2)
    assert inp["point_labels"].shape == (2, 2)
    # obj0 centre -> half the 1008 grid; obj1 corner -> full grid.
    assert torch.allclose(inp["point_coords"][0, 0], torch.tensor([504.0, 504.0]))
    assert torch.allclose(inp["point_coords"][1, 0], torch.tensor([1008.0, 1008.0]))
    # obj1 second slot is padding.
    assert inp["point_labels"][1, 1].item() == -1
    assert inp["point_labels"][0].tolist() == [1, 0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_sam3p1_point_inputs.py -q`
Expected: FAIL — `ImportError: cannot import name '_build_mux_point_inputs'`

- [ ] **Step 3: Write minimal implementation**

Add near the top-level helpers in `sam/models/sam3_predictor.py` (after `_masks_to_boxes`):

```python
def _build_mux_point_inputs(prompts, video_hw, image_size, device):
    """Batch point-only ``GeometryPrompt``s into the tracker's ``point_inputs`` dict.

    Coords are scaled from video pixels (or ``[0,1]`` when ``is_normalized``) to the
    ``image_size`` prompt grid. Ragged point counts are right-padded with coord
    ``(0,0)`` and label ``-1`` (the SAM "no point" padding), so all objects batch
    into one interactive ``track_step``.

    Args:
        prompts: point-only prompts, one per object (each ``points_coords`` (P,2),
            ``points_labels`` (P,), no boxes/masks).
        video_hw: ``(H, W)`` of the source video.
        image_size: the tracker's square input resolution.
        device: target device for the built tensors.

    Returns:
        ``(point_inputs, obj_ids)`` where ``point_inputs`` has ``point_coords``
        ``(n, P, 2)`` float and ``point_labels`` ``(n, P)`` int32.
    """
    height, width = video_hw
    scale = torch.tensor([width, height], device=device, dtype=torch.float32)
    obj_ids, per_coords, per_labels = [], [], []
    for prompt in prompts:
        obj_ids.append(prompt.obj_id)
        coords = prompt.points_coords.to(device).float()
        coords = coords if prompt.is_normalized else coords / scale
        per_coords.append(coords * image_size)
        per_labels.append(prompt.points_labels.to(device).to(torch.int32))
    max_points = max(labels.shape[0] for labels in per_labels)
    n = len(prompts)
    coords = torch.zeros(n, max_points, 2, device=device)
    labels = torch.full((n, max_points), -1, device=device, dtype=torch.int32)
    for i, (obj_coords, obj_labels) in enumerate(zip(per_coords, per_labels)):
        coords[i, : obj_coords.shape[0]] = obj_coords
        labels[i, : obj_labels.shape[0]] = obj_labels
    return {"point_coords": coords, "point_labels": labels}, obj_ids
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run pytest tests/test_sam3p1_point_inputs.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sam/models/sam3_predictor.py tests/test_sam3p1_point_inputs.py
git commit -m "feat(sam3p1): pure point-prompt batcher for mux seeding"
```

---

### Task 2: Route seed-frame clicks + narrow the NotImplementedError

**Files:**
- Modify: `sam/models/sam3_predictor.py` — `Sam3MultiplexVideoPredictor.forward` (replace the top `if geometry_prompts: raise` at ~`:896-901`); add `_check_mux_geometry` + `_seed_points_multiplex` methods on that class.
- Test: `tests/test_sam3p1_interactive_smoke.py` (create) — guard units (CPU) + a GPU click-spawn smoke.

**Interfaces:**
- Consumes: `_build_mux_point_inputs` (Task 1); existing `self.tracker.track_step`, `self.tracker.multiplex_controller.get_state`, `self._demux_outputs`, `self._masklet_from_lowres`, `state.mux_state / mux_obj_ids / mux_output_dict / bank`.
- Produces: seed-frame point prompts spawn tracklets; `forward` returns `{obj_id: MaskletResult}` including click-seeded objects. Out-of-scope prompts raise with specific messages.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sam3p1_interactive_smoke.py
import os
import numpy as np
import pytest
import torch
from PIL import Image

from sam.prompts import ConceptPrompt, GeometryPrompt

CKPT = "checkpoints/sam3.1_multiplex.pt"
CFG = "configs/sam3/sam3.1.yaml"
needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.isfile(CKPT),
    reason="mux interactive needs CUDA + sam3.1_multiplex.pt",
)


def _build():
    from sam.build_sam import build_sam3_multiplex_video_predictor
    return build_sam3_multiplex_video_predictor(config_file=CFG, ckpt_path=CKPT, device="cuda")


def _state(hw):
    from sam.models.sam3_predictor import Sam3VideoPredictorState
    return Sam3VideoPredictorState(video_hw=hw)


@needs_gpu
def test_box_prompt_raises_pointing_to_exemplars():
    pred = _build()
    st = _state((540, 960))
    box = GeometryPrompt(obj_id=1, boxes=torch.tensor([[10.0, 10.0, 50.0, 50.0]]))
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    with pytest.raises(NotImplementedError, match="exemplar"):
        pred.forward(st, 0, frame, geometry_prompts=[box])


@needs_gpu
def test_click_prompt_with_concept_raises_coseed():
    pred = _build()
    st = _state((540, 960))
    pred.set_concept(st, ConceptPrompt("person"))
    click = GeometryPrompt(
        obj_id=1, points_coords=torch.tensor([[385.0, 230.0]]),
        points_labels=torch.tensor([1]),
    )
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    with pytest.raises(NotImplementedError, match="co-seed|concept"):
        pred.forward(st, 0, frame, geometry_prompts=[click])


@needs_gpu
def test_seed_frame_click_spawns_and_tracks():
    pred = _build()
    frames = [
        np.asarray(Image.open(f"notebooks/videos/bedroom/{i:05d}.jpg").convert("RGB"))
        for i in range(3)
    ]
    h, w, _ = frames[0].shape
    st = _state((h, w))
    click = GeometryPrompt(
        obj_id=1, points_coords=torch.tensor([[385.0, 230.0]]),
        points_labels=torch.tensor([1]),
    )
    out0 = pred.forward(st, 0, frames[0], geometry_prompts=[click])
    assert set(out0) == {1}
    assert out0[1].masks_logits.shape[-2:] == (h, w)
    assert (out0[1].masks_logits > 0).sum() > 0  # non-empty seed mask
    # mid-stream add now blocked (1b territory)
    with pytest.raises(NotImplementedError, match="mid-stream"):
        pred.forward(st, 1, frames[1], geometry_prompts=[
            GeometryPrompt(obj_id=2, points_coords=torch.tensor([[500.0, 300.0]]),
                           points_labels=torch.tensor([1]))])
    out1 = pred.forward(st, 1, frames[1])           # plain propagation still works
    assert set(out1) == {1}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/test_sam3p1_interactive_smoke.py -q`
Expected: FAIL — box/click currently hit the blanket `NotImplementedError` (wrong message), and the click-spawn returns nothing.

- [ ] **Step 3: Implement routing + seeding**

In `sam/models/sam3_predictor.py`, replace the top of `Sam3MultiplexVideoPredictor.forward`:

```python
        geometry_prompts = geometry_prompts or []
        if geometry_prompts:
            raise NotImplementedError(
                "multiplex (sam3.1) video predictor does not support geometry prompts yet; "
                "use the base build_sam3_video_predictor for click/box prompting"
            )
```

with:

```python
        geometry_prompts = geometry_prompts or []
        if geometry_prompts:
            self._check_mux_geometry(state, geometry_prompts)
```

and route the seed after the features are computed. Locate, inside the `with torch.autocast(...)` block, right after `concept = state.concepts[0] if state.concepts else None`, insert:

```python
            if geometry_prompts:
                return self._seed_points_multiplex(
                    state, frame_idx, geometry_prompts, bf_int, bf_prop, num_frames
                )
```

Then add these two methods to `Sam3MultiplexVideoPredictor` (next to `_seed_multiplex`):

```python
    def _check_mux_geometry(self, state, geometry_prompts) -> None:
        """Reject the geometry-prompt cases this predictor does not yet support.

        Feature 1a supports only point clicks that SEED the first frame (no text
        concept, no existing mux state). Boxes/masks are the detector geometry-
        encoder (exemplar) path; mid-stream add and text co-seed are Feature 1b.
        """
        for prompt in geometry_prompts:
            if prompt.boxes is not None or prompt.masks_logits is not None:
                raise NotImplementedError(
                    "multiplex (sam3.1): box/mask geometry prompts route through the "
                    "detector geometry-encoder (exemplar path), not implemented yet; "
                    "clicks (points) are supported"
                )
            if prompt.points_coords is None:
                raise ValueError("geometry prompt has no points")
        if state.mux_state is not None:
            raise NotImplementedError(
                "multiplex (sam3.1): mid-stream click add is unsupported (the mux "
                "state is fixed from the seed frame); all objects must appear on the "
                "seed frame"
            )
        if state.concepts:
            raise NotImplementedError(
                "multiplex (sam3.1): text concept + click co-seed on one frame is "
                "unsupported yet; use clicks alone (no set_concept) for interactive VOS"
            )

    def _seed_points_multiplex(
        self, state, frame_idx, prompts, bf_int, bf_prop, num_frames
    ) -> dict:
        """Seed click-prompted objects on the seed frame (interactive VOS, no text).

        Structural twin of :meth:`_seed_multiplex`: builds the persistent
        ``MultiplexState`` from a single JOINT interactive ``track_step`` (points ->
        interactive prompt encoder + mask decoder + joint memory encoder), records
        the obj-id map, registers ids on the bank, and returns per-object masklets.
        """
        height, width = state.video_hw
        point_inputs, new_ids = _build_mux_point_inputs(
            prompts, (height, width), self.tracker.image_size, self.device
        )
        mux_state = self.tracker.multiplex_controller.get_state(
            len(new_ids), self.device, torch.float32, random=False
        )
        out = self.tracker.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=True,
            backbone_features_interactive=bf_int,
            backbone_features_propagation=bf_prop,
            point_inputs=point_inputs,
            mask_inputs=None,
            output_dict={"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
            num_frames=num_frames,
            multiplex_state=mux_state,
        )
        state.mux_state = mux_state
        state.mux_obj_ids = new_ids
        state.mux_output_dict["cond_frame_outputs"][frame_idx] = out
        for obj_id in new_ids:
            state.bank.known_obj_ids.add(obj_id)
        per_obj = self._demux_outputs(out, mux_state, new_ids)
        return {
            obj_id: self._masklet_from_lowres(
                per_obj[obj_id]["pred_masks"][0, 0].float(), per_obj[obj_id], height, width
            )
            for obj_id in new_ids
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run pytest tests/test_sam3p1_interactive_smoke.py -q`
Expected: PASS (or SKIP where no GPU). If a shape error surfaces in `_demux_outputs`/`_masklet_from_lowres`, inspect the `track_step` `out` shapes for the interactive path and adjust the per-object indexing (do NOT change the tracker).

- [ ] **Step 5: Commit**

```bash
git add sam/models/sam3_predictor.py tests/test_sam3p1_interactive_smoke.py
git commit -m "feat(sam3p1): interactive seed-frame click add-object (points)"
```

---

### Task 3: Capture the upstream golden

**Files:**
- Create: `tests/parity/reference_sam3/capture_sam3p1_interactive_golden.py`
- Create (output, committed): `tests/parity/fixtures/sam3p1/interactive_noconcept.npz`, `tests/parity/fixtures/sam3p1/interactive_scenario.json`

**Interfaces:**
- Produces the golden `masklets` array `(num_frames, H, W)` bool + `scenario.json` (`frame0_click_xy`, `label`, `obj_id`, `num_frames`, `hw`) consumed by Task 4.

- [ ] **Step 1: Write the capture script**

```python
# tests/parity/reference_sam3/capture_sam3p1_interactive_golden.py
"""Capture the interactive seed-frame click golden from the OFFICIAL SAM 3.1.

Run ONCE in the isolated reference env (NOT this repo's pixi env):

    cd C:/Users/javerlia/PycharmProjects/sam3_reference
    ./.venv/Scripts/python.exe \
        ../sam2/tests/parity/reference_sam3/capture_sam3p1_interactive_golden.py \
        --frames ../sam2/notebooks/videos/bedroom --n 8 \
        --out ../sam2/tests/parity/fixtures/sam3p1

Scenario: frame-0 positive click at the native-pixel (x, y) below, obj_id=2, NO
text concept -> propagate forward N frames. Coords are converted to the relative
[0,1] the upstream add_prompt expects.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

CLICK_XY = (385.0, 230.0)   # native 960x540 pixel
OBJ_ID = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(42)
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    predictor = build_sam3_multiplex_video_predictor()

    frame_dir = Path(args.frames)
    paths = sorted(frame_dir.glob("*.jpg"))[: args.n]
    w0, h0 = Image.open(paths[0]).size

    r = predictor.handle_request(dict(type="start_session", resource_path=str(frame_dir)))
    sid = r["session_id"]
    predictor.handle_request(dict(
        type="add_prompt", session_id=sid, frame_index=0, obj_id=OBJ_ID,
        points=torch.tensor([[CLICK_XY[0] / w0, CLICK_XY[1] / h0]], dtype=torch.float32),
        point_labels=torch.tensor([1], dtype=torch.int32),
    ))

    masklets = np.zeros((args.n, h0, w0), dtype=bool)
    for resp in predictor.handle_stream_request(dict(type="propagate_in_video", session_id=sid)):
        idx = resp["frame_index"]
        if idx >= args.n:
            break
        out = resp["outputs"]                      # {obj_id: mask (H,W)}
        mask = out.get(OBJ_ID)
        if mask is not None:
            masklets[idx] = np.asarray(mask, dtype=bool)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "interactive_noconcept.npz", masklets=masklets)
    (out_dir / "interactive_scenario.json").write_text(json.dumps({
        "frame0_click_xy": list(CLICK_XY), "label": 1, "obj_id": OBJ_ID,
        "num_frames": args.n, "hw": [h0, w0],
    }, indent=2))
    print(f"saved {masklets.shape} masklets -> {out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run capture in the reference env**

Run:
```bash
cd C:/Users/javerlia/PycharmProjects/sam3_reference && ./.venv/Scripts/python.exe \
  ../sam2/tests/parity/reference_sam3/capture_sam3p1_interactive_golden.py \
  --frames ../sam2/notebooks/videos/bedroom --n 8 \
  --out ../sam2/tests/parity/fixtures/sam3p1
```
Expected: `saved (8, 540, 960) masklets -> .../fixtures/sam3p1`.
If the upstream `outputs`/`handle_stream_request` keys differ, inspect one `resp` dict and adjust the key access (the mask extraction is the only fragile part).

- [ ] **Step 3: Commit the script + fixtures**

```bash
cd C:/Users/javerlia/PycharmProjects/sam2
git add tests/parity/reference_sam3/capture_sam3p1_interactive_golden.py tests/parity/fixtures/sam3p1
git commit -m "test(sam3p1): capture interactive seed-click golden"
```

---

### Task 4: Parity test against the golden

**Files:**
- Create: `tests/parity/test_sam3p1_interactive_parity.py`

**Interfaces:**
- Consumes: fixtures from Task 3; `build_sam3_multiplex_video_predictor`, the Task 2 click path.

- [ ] **Step 1: Write the failing test**

```python
# tests/parity/test_sam3p1_interactive_parity.py
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from sam.prompts import GeometryPrompt

FIX = Path("tests/parity/fixtures/sam3p1")
CKPT = "checkpoints/sam3.1_multiplex.pt"
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.isfile(CKPT)
    or not (FIX / "interactive_noconcept.npz").is_file(),
    reason="needs CUDA + sam3.1_multiplex.pt + captured golden",
)


def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else inter / union


def test_interactive_seed_click_parity():
    from sam.build_sam import build_sam3_multiplex_video_predictor
    from sam.models.sam3_predictor import Sam3VideoPredictorState

    scenario = json.loads((FIX / "interactive_scenario.json").read_text())
    golden = np.load(FIX / "interactive_noconcept.npz")["masklets"]  # (n,H,W) bool
    n, h, w = golden.shape
    frames = [
        np.asarray(Image.open(f"notebooks/videos/bedroom/{i:05d}.jpg").convert("RGB"))
        for i in range(n)
    ]

    pred = build_sam3_multiplex_video_predictor(
        config_file="configs/sam3/sam3.1.yaml", ckpt_path=CKPT, device="cuda")
    st = Sam3VideoPredictorState(video_hw=(h, w))
    x, y = scenario["frame0_click_xy"]
    click = GeometryPrompt(
        obj_id=scenario["obj_id"],
        points_coords=torch.tensor([[x, y]]),
        points_labels=torch.tensor([scenario["label"]]),
    )
    for i, fr in enumerate(frames):
        out = pred.forward(st, i, fr, geometry_prompts=[click] if i == 0 else [])
        oid = scenario["obj_id"]
        assert oid in out, f"frame {i}: object {oid} lost"
        pred_mask = (out[oid].masks_logits[0, 0] > 0).cpu().numpy()
        iou = _iou(pred_mask, golden[i])
        assert iou >= 0.99, f"frame {i}: IoU {iou:.3f} < 0.99"
```

- [ ] **Step 2: Run to verify it fails without the fixture, passes with it**

Run: `pixi run pytest tests/parity/test_sam3p1_interactive_parity.py -q`
Expected: PASS if Task 3 captured the golden and Task 2 is correct; else it surfaces the first frame whose IoU drops below 0.99 (debug the coord scaling / seed path against that frame).

- [ ] **Step 3: Commit**

```bash
git add tests/parity/test_sam3p1_interactive_parity.py
git commit -m "test(sam3p1): interactive seed-click parity gate"
```

---

### Task 5: Notebook demo + ledger note

**Files:**
- Modify: `notebooks/sam3_video_predictor_example.ipynb` (§3 markdown + a new mux-click code cell)
- Modify: `docs/superpowers/plans/2026-06-26-phase1-sam3-torch-inference.md` (append a status note)

**Interfaces:** none (docs/demo only).

- [ ] **Step 1: Add a mux seed-click demo cell**

After the mux text-tracking cell in `notebooks/sam3_video_predictor_example.ipynb`, add a markdown cell and a code cell:

```python
# Interactive seed-frame click on SAM 3.1 (no text): click frame 0, track forward.
st_click = Sam3VideoPredictorState(video_hw=(H, W))
click = GeometryPrompt(obj_id=1,
                       points_coords=torch.tensor([[385.0, 230.0]], device=device),
                       points_labels=torch.tensor([1], device=device))
mux_click = collect(mux_predictor, st_click, frames, geometry_prompts_frame0=[click])
show_tracking(frames, mux_click, "mux SAM 3.1: seed-frame click", "click")
```

Update the §3 markdown: change "**Geometry prompts are not supported here**" to note that **seed-frame point clicks are now supported** (text+click co-seed and mid-stream add remain 1b).

- [ ] **Step 2: Execute the notebook end-to-end (headless) to verify**

Run (extract + run as before):
```bash
python - <<'PY'
import json
nb=json.load(open('notebooks/sam3_video_predictor_example.ipynb'))
src=[ ''.join(c['source']).replace('plt.show()','plt.close("all")')
      for c in nb['cells'] if c['cell_type']=='code']
open('scratch_run.py','w').write('\n\n'.join(src)); compile(open('scratch_run.py').read(),'x','exec')
PY
MPLBACKEND=Agg pixi run -e notebooks python scratch_run.py
rm -f scratch_run.py
git checkout pixi.lock 2>/dev/null || true
```
Expected: exit 0, the click cell prints a tracked object across frames.

- [ ] **Step 3: Append the ledger note**

Append to `docs/superpowers/plans/2026-06-26-phase1-sam3-torch-inference.md`:

```markdown
## Post-phase1: SAM 3.1 interactive parity (2026-07-23)

- [x] **Feature 1a** — mux seed-frame point-click add-object (interactive VOS, no
  text). Spec + plan: `docs/superpowers/{specs,plans}/2026-07-23-sam3p1-interactive-seed-click*`.
- [ ] **Feature 1b** — dynamic mux-state growth (port `add_new_masks_to_existing_state`):
  mid-stream click add + text+click co-seed + detector mid-stream spawn.
- [ ] **Boxes/exemplars**, **negative_phrases**, **multi-concept** — separate parity features.
```

- [ ] **Step 4: Commit**

```bash
git add notebooks/sam3_video_predictor_example.ipynb docs/superpowers/plans/2026-06-26-phase1-sam3-torch-inference.md
git commit -m "docs(sam3p1): notebook click demo + ledger note"
```

---

## Self-Review

- **Spec coverage:** seed-frame click (T1/T2), out-of-scope raises box→exemplar / mid-stream→1b / co-seed→1b (T2 `_check_mux_geometry`), golden capture (T3), parity gate IoU≥0.99 (T4), constant-VRAM (existing propagation path, exercised by T2 smoke + T4), notebook demo + ledger (T5). ✓
- **Placeholder scan:** every code/test step carries full code; capture + notebook steps note the single fragile spot (upstream `outputs` keys / result extraction) to inspect rather than a vague "handle errors". ✓
- **Type consistency:** `_build_mux_point_inputs(prompts, video_hw, image_size, device) -> (dict, list)` used identically in T1/T2; `_seed_points_multiplex(state, frame_idx, prompts, bf_int, bf_prop, num_frames) -> dict[obj_id, MaskletResult]`; `_check_mux_geometry(state, geometry_prompts)`; `forward` returns `{obj_id: MaskletResult}` throughout. ✓
```
