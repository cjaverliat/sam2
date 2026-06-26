# Phase 0 — `sam2` → `sam` Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the `sam2` package to `sam` (clean break) and reorganize `modeling/` by responsibility with version-in-class-name classes, with **zero behavior change**, proven by a characterization harness.

**Architecture:** First freeze golden block outputs for SAM 2 / EfficientTAM (the repo has no tests — this harness is the only regression oracle). Then do the rename in surgical, reviewed steps, keeping the harness green after each. Class renames and file moves preserve `state_dict` keys (keys are submodule *attribute* paths, unaffected by class names or file locations), so existing checkpoints keep loading.

**Tech Stack:** Python ≥3.10, PyTorch, Hydra/OmegaConf, pixi, pytest (added here), git.

**This is Phase 0 of 3.** Phase 1 (SAM 3 torch inference) and Phase 2 (SAM 3 ONNX) are separate plans, authored after this one lands (they reference the final module layout produced here). See `docs/superpowers/specs/2026-06-26-sam3-integration-design.md`.

## Global Constraints

- **pixi only** — run everything through pixi (`pixi run python ...`); deps via `pixi add`. No bare `python`/`pip`.
- **Clean break** — no `sam2` import shim; `import sam2` stops working by design.
- **Distribution name = `sam`**; hydra config module = `"sam"`.
- **Preserve `state_dict` keys** — move files / rename classes only; never rename a submodule *attribute* (would break checkpoint load).
- **Mixed licensing** — SAM 2 / EfficientTAM stay Apache-2.0 (+ BSD-3 `cctorch`). Every touched file keeps/gains an SPDX header. (SAM License arrives in Phase 1 with SAM 3 code.)
- **Do NOT rename these `sam2` literals** (they are not the package): HF ids `facebook/sam2-*` / `facebook/sam2.1-*`, checkpoint filenames (`sam2.1_hiera_*.pt`), download URLs (`dl.fbaipublicfiles.com/segment_anything_2/...`), hydra config-name paths (`configs/sam2.1/...`, `configs/sam2/...`), pixi task names (`download-sam2-*`, `export-onnx-sam2-*`), and README prose about upstream SAM 2.
- **Readability first** — clear names, one responsibility per file.

---

## File Structure

Created in this phase:
- `tests/characterization/test_refactor_parity.py` — the golden harness (Task 1).
- `tests/characterization/fixtures/*.npy` — committed golden arrays (Task 1).
- `LICENSE_apache2` — copy of the current Apache text for the SAM2/ETAM portion (Task 5).
- `NOTICE` — component → license map (Task 5).

Renamed / moved:
- `sam2/` → `sam/` (Task 2); native ext `sam2._C` → `sam._C` (Task 3).
- `sam/modeling/` reorganized into `encoders/`, `prompt/`, `decoders/`, `memory/`, `tracking/`, plus top-level `prompts.py`, `results.py`, `models/` (Task 6).

Modified: `pyproject.toml`, `setup.py`, every file importing `sam2`, all `configs/**/*.yaml` `_target_` lines, `README.md` (Tasks 2–6).

---

## Task 1: Characterization harness (golden oracle)

**Files:**
- Modify: `pyproject.toml` (add `pytest` dep + `test` task)
- Create: `tests/characterization/test_refactor_parity.py`
- Create: `tests/characterization/fixtures/etam_ti_image_emb.npy` (generated)

**Interfaces:**
- Consumes: existing `build_sam2_generic_video_predictor(config_file, ckpt_path, device, mode, use_half)` and `model.encode_image(frame) -> (list[Tensor], list[Tensor])`, `model.image_size`.
- Produces: a committed golden array + a pytest that re-derives it and asserts equality. Later tasks treat "harness green" as their gate.

- [ ] **Step 1: Add pytest to the env**

Run:
```bash
pixi add pytest
```
Expected: `Added pytest ...`; `pixi.lock` + `pyproject.toml` updated.

- [ ] **Step 2: Add a `test` task to `pyproject.toml`**

Add under the tasks section:
```toml
[tool.pixi.tasks.test]
cmd = "pytest tests/ -v"
```

- [ ] **Step 3: Write the harness (uses CURRENT pre-rename names)**

Create `tests/characterization/test_refactor_parity.py`:
```python
# SPDX-License-Identifier: Apache-2.0
"""Characterization harness for the sam2->sam refactor.

Freezes a deterministic block output (CPU, fp32, fixed seed) for EfficientTAM-tiny
so the rename can be proven behavior-preserving. Capture once with CAPTURE_GOLDEN=1,
then this test compares against the committed fixture. Skips if the checkpoint is absent.

During the refactor this file is renamed like any other (`sam2`->`sam`,
`build_sam2_generic_video_predictor`->`build_sam2_video_predictor`); the .npy fixture
is the invariant.
"""
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from sam2.build_sam import build_sam2_generic_video_predictor

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures"
CKPT = ROOT / "checkpoints" / "efficienttam_ti.pt"
CONFIG = "configs/efficienttam/efficienttam_ti.yaml"
ATOL = RTOL = 1e-4


@pytest.mark.skipif(not CKPT.is_file(), reason=f"checkpoint absent: {CKPT}")
def test_image_encode_parity():
    torch.manual_seed(0)
    model = build_sam2_generic_video_predictor(
        CONFIG, str(CKPT), device="cpu", mode="eval", use_half=False
    )
    frame = torch.rand(3, model.image_size, model.image_size)
    with torch.inference_mode():
        emb, _pos = model.encode_image(frame)
    got = emb[-1].float().cpu().numpy()  # lowest-res feature level

    golden = FIXTURES / "etam_ti_image_emb.npy"
    if os.environ.get("CAPTURE_GOLDEN"):
        FIXTURES.mkdir(parents=True, exist_ok=True)
        np.save(golden, got)
        pytest.skip("captured golden fixture")
    np.testing.assert_allclose(got, np.load(golden), atol=ATOL, rtol=RTOL)
```

- [ ] **Step 4: Capture the golden fixture**

Run:
```bash
CAPTURE_GOLDEN=1 pixi run pytest tests/characterization/test_refactor_parity.py -v
```
Expected: one `SKIPPED (captured golden fixture)`; `tests/characterization/fixtures/etam_ti_image_emb.npy` now exists.
(If it instead skips with "checkpoint absent", run `pixi run download-efficienttam-ti` first.)

- [ ] **Step 5: Verify the harness passes against its own fixture**

Run:
```bash
pixi run pytest tests/characterization/test_refactor_parity.py -v
```
Expected: `1 passed`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml pixi.lock tests/characterization/
git commit -m "test(sam): add characterization harness for the sam2->sam refactor"
```

---

## Task 2: Rename the package `sam2/` → `sam/`

**Files:**
- Move: `sam2/` → `sam/` (git)
- Modify: `sam/__init__.py` (hydra module), every `*.py` importing `sam2`, every `configs/**/*.yaml` `_target_`, `tests/characterization/test_refactor_parity.py`
- Note: native ext is handled in Task 3; packaging in Task 4.

**Interfaces:**
- Consumes: Task 1 harness.
- Produces: an importable `sam` package; `from sam.build_sam import build_sam2_generic_video_predictor` works. (Class/builder *renames* happen in Task 6.)

- [ ] **Step 1: Move the directory with git (preserves history)**

```bash
git mv sam2 sam
```

- [ ] **Step 2: Point hydra at the new module**

Edit `sam/__init__.py`: change `initialize_config_module("sam2", ...)` → `initialize_config_module("sam", ...)`.

- [ ] **Step 3: Enumerate the module references to change**

Run (review the list — these are the ONLY things that change):
```bash
rg -n --glob '!**/configs/**' '\b(import sam2\b|from sam2\b|sam2\.modeling|sam2\.utils|sam2\.build_sam|sam2\.sam2_|sam2\.onnx|sam2\.automatic_mask_generator|sam2\.benchmark)' sam training tools demo sav_dataset tests
rg -n '_target_:\s*sam2\.' sam/configs
```
Expected: a finite list across `sam/`, `training/` (train.py, model/sam2.py), `tools/`, `demo/`, `tests/characterization/`, and `_target_:` lines in `sam/configs/**`.

- [ ] **Step 4: Replace module references (NOT the protected literals)**

Apply, then eyeball with `git diff`:
```bash
# Python module paths: sam2.<module>  and import/from statements
rg -l --glob '*.py' '\b(import sam2\b|from sam2\b|sam2\.)' sam training tools demo sav_dataset tests \
  | xargs sed -i -E 's/\b(import )sam2\b/\1sam/; s/\bfrom sam2\b/from sam/g; s/\bsam2\.(modeling|utils|build_sam|onnx|automatic_mask_generator|benchmark|sam2_|version)/sam.\1/g'
# Hydra targets in configs
rg -l '_target_:\s*sam2\.' sam/configs | xargs sed -i -E 's/(_target_:\s*)sam2\./\1sam./g'
```
Then **manually review** `git diff` and confirm NONE of these changed: `facebook/sam2*` ids, `sam2.1_hiera_*.pt`, `segment_anything_2` URLs, `configs/sam2.1/...` strings, `download-sam2-*` task names. Revert any such accidental hit.

- [ ] **Step 5: Fix the harness import**

In `tests/characterization/test_refactor_parity.py` change `from sam2.build_sam import ...` → `from sam.build_sam import ...` (Step 4's sed already does this; confirm).

- [ ] **Step 6: Verify import + harness (native ext may warn — Task 3 fixes it)**

```bash
pixi run python -c "import sam; from sam.build_sam import build_sam2_generic_video_predictor; print('import ok')"
pixi run pytest tests/characterization/ -v
```
Expected: `import ok`; `1 passed`. (A `sam2._C`/`sam._C` warning is fine until Task 3.)

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor(sam): rename sam2 package to sam (clean break)"
```

---

## Task 3: Rename the native extension `sam2._C` → `sam._C`

**Files:**
- Modify: `setup.py` (ext `name=`, comments, error strings)
- Modify: `sam/utils/misc.py:62` (`from sam2 import _C`) and the JIT-load fallback

**Interfaces:**
- Consumes: Task 2 (package is `sam`).
- Produces: `from sam import _C` resolves (pre-built or JIT), `get_connected_components` works on CUDA; on CPU it is simply not exercised.

- [ ] **Step 1: Rename the extension target in `setup.py`**

Change `CUDAExtension(name="sam2._C", ...)` → `name="sam._C"`. Update the surrounding log/error strings (`sam2._C` → `sam._C`) for accuracy.

- [ ] **Step 2: Fix the loader in `sam/utils/misc.py`**

Change `from sam2 import _C` → `from sam import _C`. Leave the JIT `name="_C"` and the `csrc` path (`Path(__file__).parent.parent / "csrc"`) as-is (still correct under `sam/`).

- [ ] **Step 3: Verify the harness still passes (CPU path, no ext needed)**

```bash
pixi run pytest tests/characterization/ -v
```
Expected: `1 passed` (the CPU `encode_image` path does not touch `_C`).

- [ ] **Step 4: (CUDA hosts only) verify the ext import**

```bash
pixi run python -c "from sam import _C; print('ext ok')" || echo "no prebuilt ext (JIT/optional) - acceptable"
```
Expected: `ext ok`, or the acceptable-fallback message on a host without the built ext.

- [ ] **Step 5: Commit**

```bash
git add setup.py sam/utils/misc.py
git commit -m "refactor(sam): rename native extension sam2._C to sam._C"
```

---

## Task 4: Packaging — `pyproject.toml` + setuptools

**Files:**
- Modify: `pyproject.toml` (`[project].name`, `packages.find`, `package-data`, `[tool.pixi.pypi-options].no-build-isolation`, `[tool.pixi.pypi-dependencies]`)

**Interfaces:**
- Consumes: Tasks 2–3.
- Produces: wheel/dist name `sam`; package discovery + config data resolve under `sam/`.

- [ ] **Step 1: Rename the distribution + package globs**

In `pyproject.toml`:
- `[project]` → `name = "sam"`.
- `[tool.setuptools.packages.find]` → `include = ["sam*"]` (already `sam*`-shaped; confirm it no longer says `sam2*`). Keep the `exclude` list.
- `[tool.setuptools.package-data]` → rename the key `sam2 = [...]` to `sam = [...]` (same globs).
- `[tool.pixi.pypi-options]` → `no-build-isolation = ["sam"]`.
- `[tool.pixi.pypi-dependencies]` → `sam = { path = ".", editable = true }` (was `sam2`).

- [ ] **Step 2: Re-resolve the editable install**

```bash
pixi install
```
Expected: resolves; the editable `sam` package builds/installs without error.

- [ ] **Step 3: Verify config data + build resolve**

```bash
pixi run python -c "import sam, hydra; from hydra import compose; print(compose('configs/efficienttam/efficienttam_ti.yaml').model._target_)"
```
Expected: prints a `sam....` target (config discovery via the `sam` hydra module works).

- [ ] **Step 4: Harness regression**

```bash
pixi run pytest tests/characterization/ -v
```
Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml pixi.lock
git commit -m "build(sam): rename distribution + package data to sam"
```

---

## Task 5: Mixed-license groundwork (SPDX + NOTICE)

**Files:**
- Create: `LICENSE_apache2` (copy of the existing Apache text), `NOTICE`
- Modify: `README.md` (license section + per-component table)
- Modify: `sam/**/*.py` (add `# SPDX-License-Identifier: Apache-2.0` where missing)

**Interfaces:**
- Consumes: none.
- Produces: every `sam/` file carries an Apache SPDX header; a NOTICE + README table document the component→license map. (Phase 1 adds `LICENSE_sam` + SAM-License SPDX on vendored SAM 3 files.)

- [ ] **Step 1: Add the Apache copy + NOTICE**

```bash
cp LICENSE LICENSE_apache2
```
Create `NOTICE`:
```
This project bundles components under multiple licenses:
- SAM 2, EfficientTAM, and this fork's code: Apache-2.0 (LICENSE_apache2)
- CUDA connected-components extension (csrc): BSD-3-Clause (LICENSE_cctorch)
SAM 3 components (added later) are under the SAM License (LICENSE_sam); see README.
```

- [ ] **Step 2: Add SPDX headers to `sam/` Python files lacking one**

```bash
for f in $(rg -L --files sam -g '*.py'); do
  head -1 "$f" | rg -q 'SPDX-License-Identifier' || \
    sed -i '1i # SPDX-License-Identifier: Apache-2.0' "$f"
done
```
Eyeball `git diff --stat` (every `sam/*.py` gained one line).

- [ ] **Step 3: Add the README license table**

Add to `README.md` (license section):
```markdown
## Licenses

| Component | License |
|---|---|
| SAM 2 | Apache-2.0 |
| EfficientTAM | Apache-2.0 |
| CUDA connected-components ext (`csrc`) | BSD-3-Clause |
| This fork's code | Apache-2.0 |
```

- [ ] **Step 4: Verify nothing imports broke**

```bash
pixi run python -c "import sam; print('ok')" && pixi run pytest tests/characterization/ -v
```
Expected: `ok`; `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add LICENSE_apache2 NOTICE README.md sam
git commit -m "docs(sam): document mixed licensing + add SPDX headers"
```

---

## Task 6: Responsibility reorg + canonical class names (0b)

Do this as **independently-committed sub-steps**, harness green after each. `git mv` every move (preserves history); update imports immediately; never rename a submodule attribute.

**Files (moves):**
- `sam/modeling/backbones/` → `sam/modeling/encoders/` (`hieradet.py`→`hiera.py`, `image_encoder.py`, `vitdet.py`, `utils.py`)
- `sam/modeling/sam/` → `sam/modeling/decoders/` (`mask_decoder.py`, `transformer.py`, `onnx_compat.py`); `prompt_encoder.py` → `sam/modeling/prompt/prompt_encoder.py`
- `sam/modeling/{memory,memory_attention,memory_encoder,sam2_memory,sam2_forgetful_memory}.py` → `sam/modeling/memory/{bank,attention,encoder,banks,forgetful}.py`
- `sam/modeling/sam2_base.py` → `sam/modeling/tracking/tracker_base.py`
- `sam/modeling/sam2_utils.py` → `sam/modeling/utils.py`
- `sam/modeling/sam2_prompt.py` → `sam/prompts.py`; `sam/modeling/sam2_result.py` → `sam/results.py`
- `sam/modeling/sam2_generic.py` → `sam/models/sam2_predictor.py`; `sam/sam2_generic_video_predictor.py` → fold into `sam/models/sam2_predictor.py`; `sam/sam2_video_predictor.py` → `sam/models/legacy_video_predictor.py`; `sam/sam2_image_predictor.py` → `sam/models/image_predictor.py`

**Class / builder renames (attributes unchanged → `state_dict` keys unchanged):**
- `SAM2Base` → `SamTrackerBase`; `SAM2Generic` → `Sam2Predictor`; `SAM2GenericVideoPredictor` → `Sam2VideoPredictor`; `SAM2VideoPredictor` (legacy) → `Sam2LegacyVideoPredictor`
- `SAM2Prompt` → `GeometryPrompt`; `SAM2Result` → `MaskletResult` (update every reference, incl. `memory/bank.py` and the harness)
- Builders: `build_sam2_generic_video_predictor` → `build_sam2_video_predictor`; `build_sam2_generic` → `build_sam2`; legacy `build_sam2_video_predictor` → `build_sam2_legacy_video_predictor`; `build_sam2` (raw) → `build_sam2_legacy`. Update `configs/**` `_target_` to the moved class paths.

**Interfaces:**
- Consumes: Tasks 2–5.
- Produces: the final Phase-0 layout from the spec §4; `from sam.build_sam import build_sam2_video_predictor`, `from sam.prompts import GeometryPrompt`, `from sam.results import MaskletResult`, `from sam.modeling.tracking.tracker_base import SamTrackerBase`.

- [ ] **Step 1: Move encoders, update imports, harness green, commit**

```bash
mkdir -p sam/modeling/encoders
git mv sam/modeling/backbones/hieradet.py sam/modeling/encoders/hiera.py
git mv sam/modeling/backbones/image_encoder.py sam/modeling/encoders/image_encoder.py
git mv sam/modeling/backbones/vitdet.py sam/modeling/encoders/vitdet.py
git mv sam/modeling/backbones/utils.py sam/modeling/encoders/utils.py
git rm sam/modeling/backbones/__init__.py 2>/dev/null; touch sam/modeling/encoders/__init__.py
rg -l 'modeling\.backbones|modeling\.sam2_base' sam training tools | xargs sed -i -E 's/modeling\.backbones\.hieradet/modeling.encoders.hiera/g; s/modeling\.backbones/modeling.encoders/g'
rg -l '_target_:.*modeling\.backbones' sam/configs | xargs sed -i -E 's/modeling\.backbones\.hieradet/modeling.encoders.hiera/g; s/modeling\.backbones/modeling.encoders/g'
pixi run pytest tests/characterization/ -v
git add -A && git commit -m "refactor(sam): move backbones -> modeling/encoders"
```
Expected: `1 passed` before commit.

- [ ] **Step 2: Move decoders + prompt, update imports, harness green, commit**

```bash
mkdir -p sam/modeling/decoders sam/modeling/prompt
git mv sam/modeling/sam/mask_decoder.py sam/modeling/decoders/mask_decoder.py
git mv sam/modeling/sam/transformer.py sam/modeling/decoders/transformer.py
git mv sam/modeling/sam/onnx_compat.py sam/modeling/decoders/onnx_compat.py
git mv sam/modeling/sam/prompt_encoder.py sam/modeling/prompt/prompt_encoder.py
git rm sam/modeling/sam/__init__.py 2>/dev/null; touch sam/modeling/decoders/__init__.py sam/modeling/prompt/__init__.py
rg -l 'modeling\.sam\b|modeling\.sam\.' sam training tools | xargs sed -i -E 's/modeling\.sam\.mask_decoder/modeling.decoders.mask_decoder/g; s/modeling\.sam\.transformer/modeling.decoders.transformer/g; s/modeling\.sam\.onnx_compat/modeling.decoders.onnx_compat/g; s/modeling\.sam\.prompt_encoder/modeling.prompt.prompt_encoder/g'
rg -l '_target_:.*modeling\.sam\.' sam/configs | xargs sed -i -E 's/modeling\.sam\.mask_decoder/modeling.decoders.mask_decoder/g; s/modeling\.sam\.transformer/modeling.decoders.transformer/g; s/modeling\.sam\.prompt_encoder/modeling.prompt.prompt_encoder/g'
pixi run pytest tests/characterization/ -v
git add -A && git commit -m "refactor(sam): move modeling/sam -> decoders + prompt"
```
Expected: `1 passed` before commit.

- [ ] **Step 3: Move memory group, update imports, harness green, commit**

```bash
mkdir -p sam/modeling/memory
git mv sam/modeling/memory.py sam/modeling/memory/bank.py
git mv sam/modeling/memory_attention.py sam/modeling/memory/attention.py
git mv sam/modeling/memory_encoder.py sam/modeling/memory/encoder.py
git mv sam/modeling/sam2_memory.py sam/modeling/memory/banks.py
git mv sam/modeling/sam2_forgetful_memory.py sam/modeling/memory/forgetful.py
touch sam/modeling/memory/__init__.py
rg -l 'modeling\.memory\b|modeling\.memory_attention|modeling\.memory_encoder|modeling\.sam2_memory|modeling\.sam2_forgetful_memory' sam training tools | xargs sed -i -E 's/modeling\.memory_attention/modeling.memory.attention/g; s/modeling\.memory_encoder/modeling.memory.encoder/g; s/modeling\.sam2_memory/modeling.memory.banks/g; s/modeling\.sam2_forgetful_memory/modeling.memory.forgetful/g; s/modeling\.memory\b/modeling.memory.bank/g'
rg -l '_target_:.*modeling\.(memory|memory_attention|memory_encoder)' sam/configs | xargs sed -i -E 's/modeling\.memory_attention/modeling.memory.attention/g; s/modeling\.memory_encoder/modeling.memory.encoder/g; s/modeling\.memory\b/modeling.memory.bank/g'
pixi run pytest tests/characterization/ -v
git add -A && git commit -m "refactor(sam): consolidate memory modules under modeling/memory"
```
Expected: `1 passed` before commit. (Watch the order: rewrite the specific `memory_attention`/`memory_encoder` paths BEFORE the bare `modeling.memory` → `modeling.memory.bank`, as the sed above does.)

- [ ] **Step 4: Move tracker base + shared utils/data types, harness green, commit**

```bash
mkdir -p sam/modeling/tracking
git mv sam/modeling/sam2_base.py sam/modeling/tracking/tracker_base.py
git mv sam/modeling/sam2_utils.py sam/modeling/utils.py
git mv sam/modeling/sam2_prompt.py sam/prompts.py
git mv sam/modeling/sam2_result.py sam/results.py
touch sam/modeling/tracking/__init__.py
rg -l 'modeling\.sam2_base|modeling\.sam2_utils|modeling\.sam2_prompt|modeling\.sam2_result' sam training tools | xargs sed -i -E 's/modeling\.sam2_base/modeling.tracking.tracker_base/g; s/modeling\.sam2_utils/modeling.utils/g; s/modeling\.sam2_prompt/prompts/g; s/modeling\.sam2_result/results/g'
rg -l '_target_:.*modeling\.sam2_base' sam/configs | xargs sed -i -E 's/modeling\.sam2_base/modeling.tracking.tracker_base/g'
pixi run pytest tests/characterization/ -v
git add -A && git commit -m "refactor(sam): tracker_base + shared utils/prompts/results"
```
Expected: `1 passed` before commit.

- [ ] **Step 5: Create `sam/models/`, move predictors, harness green, commit**

```bash
mkdir -p sam/models && touch sam/models/__init__.py
git mv sam/modeling/sam2_generic.py sam/models/sam2_predictor.py
git mv sam/sam2_image_predictor.py sam/models/image_predictor.py
git mv sam/sam2_video_predictor.py sam/models/legacy_video_predictor.py
git mv sam/sam2_generic_video_predictor.py sam/models/_video_predictor_tmp.py  # fold in next step
rg -l 'sam2_generic\b|sam\.sam2_image_predictor|sam\.sam2_video_predictor|sam\.sam2_generic_video_predictor' sam training tools | xargs sed -i -E 's/sam\.modeling\.sam2_generic/sam.models.sam2_predictor/g; s/sam\.sam2_image_predictor/sam.models.image_predictor/g; s/sam\.sam2_generic_video_predictor/sam.models.sam2_predictor/g; s/sam\.sam2_video_predictor/sam.models.legacy_video_predictor/g'
```
Then manually merge `sam/models/_video_predictor_tmp.py` (the `Sam2GenericVideoPredictor` + state classes) into `sam/models/sam2_predictor.py`, delete the tmp file, and fix `build_sam.py` imports. Run:
```bash
pixi run pytest tests/characterization/ -v
git add -A && git commit -m "refactor(sam): group predictors under sam/models"
```
Expected: `1 passed` before commit.

- [ ] **Step 6: Rename classes + builders to canonical names, harness green, commit**

Apply the rename map (attributes untouched). Replace-all across `sam`, `training`, `tools`, `demo`, `tests`, `configs`:
```bash
rg -l '\bSAM2Base\b|\bSAM2Generic\b|\bSAM2GenericVideoPredictor\b|\bSAM2VideoPredictor\b|\bSAM2Prompt\b|\bSAM2Result\b' sam training tools demo tests \
 | xargs sed -i -E 's/\bSAM2GenericVideoPredictor\b/Sam2VideoPredictor/g; s/\bSAM2VideoPredictor\b/Sam2LegacyVideoPredictor/g; s/\bSAM2Generic\b/Sam2Predictor/g; s/\bSAM2Base\b/SamTrackerBase/g; s/\bSAM2Prompt\b/GeometryPrompt/g; s/\bSAM2Result\b/MaskletResult/g'
```
> Order matters: `SAM2GenericVideoPredictor` before `SAM2Generic`; `SAM2VideoPredictor` (legacy) is renamed *after* the generic one to avoid double-hitting.

Then rename the builders in `sam/build_sam.py` and their `_target_`s / callers:
```bash
rg -l 'build_sam2_generic_video_predictor|build_sam2_generic\b|build_sam2_video_predictor\b|build_sam2\b' sam training tools demo tests \
 | xargs sed -i -E 's/\bbuild_sam2_video_predictor\b/build_sam2_legacy_video_predictor/g; s/\bbuild_sam2_generic_video_predictor\b/build_sam2_video_predictor/g; s/\bbuild_sam2_generic\b/build_sam2/g; s/(_target_:\s*sam\.models\.sam2_predictor\.)SAM2GenericVideoPredictor/\1Sam2VideoPredictor/g'
```
Update the harness call `build_sam2_generic_video_predictor` → `build_sam2_video_predictor` (the sed above covers it; confirm). Manually verify `build_sam.py` hydra `++model._target_=` strings now point at the renamed classes/paths.

```bash
pixi run pytest tests/characterization/ -v
git add -A && git commit -m "refactor(sam): canonical predictor/class names (Sam2Predictor, SamTrackerBase, ...)"
```
Expected: `1 passed`. The fixture is unchanged → identical `encode_image` output proves state_dict keys + behavior survived the reorg.

- [ ] **Step 7: Full-tree import sanity sweep**

```bash
pixi run python - <<'PY'
import importlib, pkgutil, sam
ok = 0
for m in pkgutil.walk_packages(sam.__path__, "sam."):
    importlib.import_module(m.name); ok += 1
print(f"imported {ok} submodules ok")
PY
```
Expected: imports all submodules with no error.

---

## Self-Review (author)

- **Spec coverage:** Phase-0 items in spec §14 (0-pre harness, 0a mechanical rename + SPDX + state_dict, 0b reorg + class renames) → Tasks 1 / 2–5 / 6. Mixed-licensing groundwork (D11) → Task 5. Naming map (§3) → Task 6. ✓
- **Out of scope (correctly deferred to Phase 1+):** SAM 3 modules, `LICENSE_sam`, ONNX SAM3, multiplex. ✓
- **Type/name consistency:** builder `build_sam2_video_predictor` (canonical, post-Task-6) is the name later phases consume; legacy is `build_sam2_legacy_video_predictor`; data types `GeometryPrompt`/`MaskletResult`; base `SamTrackerBase`. Harness uses the pre-rename name at capture (Task 1) and the canonical name after Task 6 — consistent. ✓
- **Risk note for the executor:** the `sed` lines are starting points — **review every `git diff`** before committing; the protected-literals rule (Global Constraints) is the main hazard. EfficientTAM-tiny must be downloaded for the harness (`pixi run download-efficienttam-ti`).
