# Task 3 report — Text encoder + tokenizer (`Sam3TextEncoder` / `Sam3Tokenizer`)

**Status:** DONE (parity is BITWISE-EXACT, far inside the atol=1e-2 gate)
**Scope:** Phase 1, Task 3 — vendor SAM 3's text tower + BPE tokenizer, load the
language-backbone subtree from `checkpoints/sam3.pt`, and prove the output matches
the golden `text_emb`. In-place on `develop`. TDD (RED→GREEN parity-gated).

---

## 1. What was vendored (upstream file → new file)

All new files carry `# SPDX-License-Identifier: LicenseRef-SAM`. Vendored from
`../sam3_reference` @ `5dd401d`, trimmed to the text inference path.

| New file | Source | Notes |
|---|---|---|
| `sam/modeling/text/tokenizer.py` | `sam3/model/tokenizer_ve.py` | `Sam3Tokenizer` (renamed from `SimpleTokenizer`). Strips `iopath` (replaced by stdlib `gzip.open`). `ftfy` + `regex` kept — required for Unicode normalisation and the `\p{L}`/`\p{N}` regex pattern. Loads BPE vocab from bundled asset via `Path(__file__).parent / "assets"`. |
| `sam/modeling/text/text_encoder.py` | `sam3/model/text_encoder_ve.py` | `Sam3TextEncoder` (renamed from `VETextEncoder`). `LayerScale` inlined from `model_misc.py` (not used in checkpoint — `ls_init_value=None` → `nn.Identity()`; kept for forward-compat). `compile_mode` / `use_act_checkpoint` stripped (inference-only). `encode(phrases) -> Tensor` convenience method added. |
| `sam/modeling/text/__init__.py` | — | Package init: exports `Sam3TextEncoder`, `Sam3Tokenizer`. |
| `sam/modeling/text/assets/bpe_simple_vocab_16e6.txt.gz` | `sam3/assets/bpe_simple_vocab_16e6.txt.gz` | CLIP BPE vocabulary (1.36 MB). Origin: OpenAI CLIP, ships unchanged with SAM 3. Committed as **vocabulary data** (not model weights). Declared in `pyproject.toml` `package-data` so it's included in wheels. |

**Deps added:**
- `setup.py::BASE_DEPS`: `ftfy>=6.1.1`, `regex>=2024.1.1` (declared runtime deps for the tokenizer; installed into the pixi `default` env via `pip install`).
- `pyproject.toml`: package-data glob `modeling/text/assets/*.gz` for wheel inclusion.

**Stripped:** `iopath` (for BPE loading), `torch.compile` + activation-checkpoint wrapper (inference-only path).

## 2. Golden key selection — `text_emb` vs `text_embeds_pre`

**Target: `text_emb` (32, 1, 256) = `language_features`.**

Reasoning confirmed from reading `vl_combiner.py::_forward_text_no_ack_ckpt()`:

| Key | Shape | Origin | Pure text? | Target? |
|---|---|---|---|---|
| `text_emb` | (32, 1, 256) | `VETextEncoder.forward()` → `text_memory_resized` = `resizer(transformer_output.T)` | **Yes** — transformer + ln_final + resizer; no vision | **chosen** |
| `text_embeds_pre` | (32, 1, 1024) | `VETextEncoder.forward()` → `inputs_embeds.T` = token embedding lookup BEFORE transformer | Yes, but weaker | Rejected |

`text_embeds_pre` was rejected because it is only an embedding-table lookup (no transformer forward), making it a weaker parity gate. `text_emb` exercises the full 24-layer transformer + ln_final + the d_model resizer projection — the same path the DETR detector consumes.

**Neither requires vision input.** The VL early-fusion encoder (`TransformerEncoderFusion` in `encoder.py`) is a *separate downstream* module that takes `language_features` + `vision_features` together — that is Task 4. `VETextEncoder.forward()` runs independently with tokenised text only, as confirmed by tracing the `forward_text` hook path in `capture_sam3_golden.py`.

## 3. state_dict subtree + key handling

`sam3.pt` text subtree is **295 keys** under `detector.backbone.language_backbone.*`:

| Layer type | Key count |
|---|---|
| `encoder.token_embedding.weight` | 1 (49408 × 1024) |
| `encoder.positional_embedding` | 1 (32 × 1024) |
| `encoder.text_projection` | 1 (1024 × 512) — Parameter for discarded pooled output; loaded for strict= completeness |
| `encoder.ln_final.{weight,bias}` | 2 |
| `encoder.transformer.resblocks.{0..23}.{attn,ln_1,ln_2,mlp}.*` | 12 × 24 = 288 |
| `resizer.{weight,bias}` | 2 |
| **Total** | **295** |

`build_sam3_text_encoder` strips the `detector.backbone.language_backbone.` prefix and calls `load_state_dict(sub, strict=True)`:
```
Missing: []    Unexpected: []    Keys loaded: 295
```

No remap required. Attribute names (`encoder`, `resizer`) were preserved verbatim from the upstream `VETextEncoder`, so the prefix strip produces an exact match.

**No `ls_1`/`ls_2` keys** in the checkpoint (`ls_init_value=None` → `nn.Identity()` has no parameters). The `attn_mask` causal buffer is `persistent=False` — absent from checkpoint as expected.

## 4. Tokenization verification

For phrase `"truck"`:
```
token sequence: [49406, 4629, 49407, 0, 0, ..., 0]   (context_length=32)
                 SOT    "truck</w>"  EOT   pad×29
```

## 5. TDD evidence (RED → GREEN)

**RED** (`pixi run pytest tests/parity/test_sam3_parity.py::test_text_parity -q`):
```
E   ImportError: cannot import name 'build_sam3_text_encoder' from 'sam.build_sam'
1 failed in 1.98s
```
Feature missing — fixture loaded fine. Expected failure. ✓

**Iteration (1 round):** First GREEN attempt failed at import because `ftfy` and `regex` were in
`miniconda3` but not in the pixi `default` env's site-packages. Fixed by adding both to
`setup.py::BASE_DEPS` and installing with `pixi run pip install ftfy regex`. No changes to
text encoder or tokenizer logic.

**GREEN** (`pixi run pytest tests/parity/test_sam3_parity.py::test_text_parity -q`):
```
.                                                    [100%]
1 passed in 5.58s
```

**Margin at GREEN: max|Δ| = 0.0 (bitwise-exact)** vs gate atol=1e-2.

| metric | value |
|---|---|
| max\|Δ\| | **0.0** |
| mean\|Δ\| | 0.0 |
| % elements > 1e-2 | 0.000% |
| shape | (32, 1, 256) ✓ |

Bitwise reproducibility is consistent with the text tower being a standard MLP + MultiheadAttention + LayerNorm stack with no special fused ops (unlike Task 2's PE vision trunk `_addmm_activation`). Under bf16 autocast + determinism, the result is deterministically exact.

## 6. Files changed

- **New:** `sam/modeling/text/{__init__,text_encoder,tokenizer}.py` (SPDX `LicenseRef-SAM`), `sam/modeling/text/assets/bpe_simple_vocab_16e6.txt.gz` (BPE vocab data, no SPDX inside gz).
- **Modified:**
  - `sam/build_sam.py` — added `build_sam3_text_encoder` (SPDX `LicenseRef-SAM` section; Apache-2.0 reasserted before `_load_checkpoint`).
  - `setup.py` — added `ftfy>=6.1.1` + `regex>=2024.1.1` to `BASE_DEPS`.
  - `pyproject.toml` — added `modeling/text/assets/*.gz` to `package-data`.
  - `tests/parity/test_sam3_parity.py` — added `test_text_parity`.
  - `pixi.lock` — updated for `ftfy` + `regex` in the default env.
- No weights / `sam3_reference` content committed. BPE vocab `.gz` IS committed (required tokenizer data per the brief).

## 7. `pixi run pytest tests/ -q` result

```
....................
20 passed, 39 warnings in 15.96s
```
Phase-0's 18 + vision encoder parity (Task 2) + new text parity (Task 3) = 20 passed. No regressions.

## 8. Self-review

- **Strict load (295 keys, 0 missing/unexpected):** Confirms the vendored module mirrors the checkpoint structure exactly.
- **Bitwise-exact output (max|Δ|=0):** The text transformer has no special fused ops, so deterministic bf16 autocast is perfectly reproducible.
- **`encode()` seam is clean:** Takes `list[str]`, returns `(seq, batch, d_model)` tensor matching `language_features`. The full `forward()` (matching upstream `VETextEncoder.forward()`) is preserved for the DETR detector integration in Task 4.
- **BPE asset provenance documented:** Comment in `tokenizer.py` and commit message note the CLIP / SAM 3 origin.
- **No speculative ABCs:** `Sam3TextEncoder` is a concrete class; the spec §16 swap interface is deferred per the brief.

## 9. Concerns / decisions

1. **`ftfy` + `regex` not in pixi lock prior to this task.** They were in the system miniconda3 Python, not the pixi env. Added to `setup.py::BASE_DEPS` and `pixi.lock` updated. Anyone recreating the env from scratch will get them automatically.
2. **`encoder.text_projection` is loaded but never used in the `text_memory` path.** It feeds the discarded `pooled` output in `TextTransformer.forward()`. Necessary for strict=True and preserved with correct shape (1024 × 512).
3. **`LayerScale` is inlined but inactive** (`ls_init_value=None` in the standard build → `nn.Identity()`). Kept for forward-compat.
