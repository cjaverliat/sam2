# SAM 3 geometry-prompt encoder + image box/point — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline). Steps use `- [ ]`.

**Goal:** Activate box+point geometry encoding on the SAM 3 image detector: a box or point prompt biases `predict`/`detect`, matching upstream.

**Architecture:** Rewrite `Sam3GeometryEncoder.forward` to port upstream `_encode_points` (grid_sample) + `_encode_boxes` (roi_align) + `concat_padded_sequences` + cls + cross-attn; thread `geo_prompt` through `forward_grounding`/`detect`/`predict`. Null prompt reduces to today's CLS-only path (regression guard). All 76 geometry weights already load; no new deps.

**Spec:** `docs/superpowers/specs/2026-07-23-sam3-geometry-encoder-image-design.md`

## Global Constraints

- Box format at the encoder: **normalized cxcywh** `(N,B,4)`; point: **normalized xy** `(N,B,2)`; labels long `(N,B)` in {0,1}; masks bool `(B,N)` True=pad. Tokens seq-first, masks batch-first.
- `pos_enc = PositionEmbeddingSine(num_pos_feats=256)` built internally (no params → no state_dict impact).
- Text-only (`geo_prompt=None`) MUST stay bit-identical. Never `sed`/`python -c`. 80-col.
- Detector tolerances (phase1): boxes atol 2px, scores atol 1e-2, mask IoU ≥ 0.99, presence atol 1e-2.

---

### Task 1: `concat_padded_sequences` + geometry encoder box/point forward

**Files:** `sam/modeling/decoders/detr_decoder.py`; test `tests/test_geometry_encoder.py`.

**Interfaces:** `Sam3GeometryEncoder.forward(img_feats, img_pos_embeds, bs, img_sizes=None, box_coords=None, box_labels=None, point_coords=None, point_labels=None) -> (geo_feats, geo_mask)`.

- [ ] **Step 1: Failing tests** (GPU for the encode paths; a CPU null-path guard):

```python
# tests/test_geometry_encoder.py
import os
import pytest
import torch

CKPT = "checkpoints/sam3.1_multiplex.pt"
needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.isfile(CKPT),
    reason="needs CUDA + sam3.1_multiplex.pt",
)


def _detector():
    from sam.build_sam import build_sam3_multiplex_video_predictor
    return build_sam3_multiplex_video_predictor(
        config_file="configs/sam3/sam3.1.yaml", ckpt_path=CKPT, device="cuda").detector


@needs_gpu
def test_geometry_encoder_box_and_point_token_counts():
    ge = _detector().geometry_encoder
    B, C = 1, ge.d_model
    hw = 72
    img_feats = [torch.randn(hw * hw, B, C, device="cuda")]
    img_pos = [torch.randn(hw * hw, B, C, device="cuda")]
    # text-only (null) -> CLS-only: 1 token
    f0, m0 = ge(img_feats, img_pos, B)
    assert f0.shape[0] == 1 and m0.shape == (B, 1)
    # 2 boxes -> 2 + 1(cls) tokens
    box = torch.tensor([[[0.5, 0.5, 0.2, 0.2]], [[0.3, 0.3, 0.1, 0.1]]], device="cuda")
    lbl = torch.ones(2, B, device="cuda")
    fb, mb = ge(img_feats, img_pos, B, img_sizes=[(hw, hw)],
                box_coords=box, box_labels=lbl)
    assert fb.shape[0] == 3 and torch.isfinite(fb).all()
    # 1 point -> 1 + 1(cls) tokens
    pt = torch.tensor([[[0.4, 0.6]]], device="cuda")
    pl = torch.ones(1, B, device="cuda")
    fp, mp = ge(img_feats, img_pos, B, img_sizes=[(hw, hw)],
                point_coords=pt, point_labels=pl)
    assert fp.shape[0] == 2 and torch.isfinite(fp).all()
```

- [ ] **Step 2: Run — FAIL** (`forward` rejects the new kwargs).

- [ ] **Step 3: Add `concat_padded_sequences`** (module-level in `detr_decoder.py`, faithful port):

```python
def _concat_padded_sequences(seq1, mask1, seq2, mask2):
    """Concatenate two right-padded seq-first sequences -> one right-padded sequence.

    seq* are (L, B, C); mask* are (B, L) with True at padded positions.
    """
    l1, bs, c = seq1.shape
    l2 = seq2.shape[0]
    len1 = (~mask1).sum(dim=-1)
    len2 = (~mask2).sum(dim=-1)
    max_len = l1 + l2
    cat_mask = (
        torch.arange(max_len, device=seq2.device)[None].repeat(bs, 1)
        >= (len1 + len2)[:, None]
    )
    cat_seq = torch.zeros((max_len, bs, c), device=seq2.device, dtype=seq2.dtype)
    cat_seq[:l1] = seq1
    index = torch.arange(l2, device=seq2.device)[:, None].repeat(1, bs) + len1[None]
    cat_seq = cat_seq.scatter(0, index[:, :, None].expand(-1, -1, c), seq2)
    return cat_seq, cat_mask
```

- [ ] **Step 4: Wire `pos_enc` + `_encode_points`/`_encode_boxes` + rewrite `forward`.** In `Sam3GeometryEncoder.__init__` add `from sam.modeling.position_encoding import PositionEmbeddingSine` and `self.pos_enc = PositionEmbeddingSine(num_pos_feats=d_model)`. Add the two encode methods (faithful port of upstream `geometry_encoders.py:589-680`):

```python
    def _encode_points(self, points, points_mask, points_labels, img_feats):
        n_points, bs = points.shape[:2]
        emb = self.points_direct_project(points)
        grid = (points.transpose(0, 1).unsqueeze(2) * 2) - 1          # (B,N,1,2) in [-1,1]
        sampled = F.grid_sample(img_feats, grid, align_corners=False)  # (B,C,N,1)
        emb = emb + self.points_pool_project(sampled.squeeze(-1).permute(2, 0, 1))
        x, y = points.unbind(-1)
        enc_x, enc_y = self.pos_enc._encode_xy(x.flatten(), y.flatten())
        enc = torch.cat([enc_x.view(n_points, bs, -1), enc_y.view(n_points, bs, -1)], -1)
        emb = emb + self.points_pos_enc_project(enc)
        return self.label_embed(points_labels.long()) + emb, points_mask

    def _encode_boxes(self, boxes, boxes_mask, boxes_labels, img_feats):
        n_boxes, bs = boxes.shape[:2]
        emb = self.boxes_direct_project(boxes)
        h, w = img_feats.shape[-2:]
        boxes_xyxy = box_cxcywh_to_xyxy(boxes)
        scale = torch.tensor([w, h, w, h], device=boxes.device, dtype=boxes_xyxy.dtype)
        boxes_xyxy = boxes_xyxy * scale.view(1, 1, 4)
        sampled = torchvision.ops.roi_align(
            img_feats, boxes_xyxy.float().transpose(0, 1).unbind(0), self.roi_size
        )                                                              # (B*N,C,roi,roi)
        proj = self.boxes_pool_project(sampled).view(bs, n_boxes, self.d_model).transpose(0, 1)
        emb = emb + proj
        cx, cy, bw, bh = boxes.unbind(-1)
        enc = self.pos_enc.encode_boxes(cx.flatten(), cy.flatten(), bw.flatten(), bh.flatten())
        emb = emb + self.boxes_pos_enc_project(enc.view(n_boxes, bs, -1))
        return self.label_embed(boxes_labels.long()) + emb, boxes_mask
```

Rewrite `forward` (null -> current CLS-only path unchanged; else port upstream `:717-838` for the box+point, no-mask, `encode_boxes_as_points=False` case):

```python
    def forward(self, img_feats, img_pos_embeds, bs, img_sizes=None,
                box_coords=None, box_labels=None, point_coords=None, point_labels=None):
        seq_first_img_feats = img_feats[-1]
        seq_first_img_pos_embeds = (
            img_pos_embeds[-1] if img_pos_embeds is not None
            else torch.zeros_like(seq_first_img_feats)
        )
        has_geo = box_coords is not None or point_coords is not None
        if has_geo:
            h, w = img_sizes[-1]
            nchw = self.img_pre_norm(img_feats[-1])
            nchw = nchw.permute(1, 2, 0).view(bs, self.d_model, h, w)
            device = seq_first_img_feats.device
            if point_coords is not None:
                pm = torch.zeros(bs, point_coords.shape[0], dtype=torch.bool, device=device)
                final_embeds, final_mask = self._encode_points(point_coords, pm, point_labels, nchw)
            else:
                final_embeds = seq_first_img_feats.new_zeros(0, bs, self.d_model)
                final_mask = torch.zeros(bs, 0, dtype=torch.bool, device=device)
            if box_coords is not None:
                bm = torch.zeros(bs, box_coords.shape[0], dtype=torch.bool, device=device)
                be, bmask = self._encode_boxes(box_coords, bm, box_labels, nchw)
                final_embeds, final_mask = _concat_padded_sequences(final_embeds, final_mask, be, bmask)
            cls = self.cls_embed.weight.view(1, 1, self.d_model).repeat(1, bs, 1)
            cls_mask = torch.zeros(bs, 1, dtype=torch.bool, device=device)
            final_embeds, final_mask = _concat_padded_sequences(final_embeds, final_mask, cls, cls_mask)
        else:
            final_embeds = self.cls_embed.weight.view(1, 1, self.d_model).repeat(1, bs, 1)
            final_mask = torch.zeros(bs, 1, dtype=torch.bool, device=seq_first_img_feats.device)
        final_embeds = self.norm(self.final_proj(final_embeds))
        for lay in self.encode:
            final_embeds = lay(
                tgt=final_embeds, memory=seq_first_img_feats,
                tgt_key_padding_mask=final_mask, pos=seq_first_img_pos_embeds,
            )
        return self.encode_norm(final_embeds), final_mask
```

Ensure `import torchvision` + `import torch.nn.functional as F` present at the top.

- [ ] **Step 5: Run — PASS.** (Add a CPU null-path guard comparing the `has_geo=False` branch output to the pre-change implementation if a fixture exists; otherwise the existing image parity gate in Task 4 protects it.)
- [ ] **Step 6: Commit** `feat(sam3): activate box+point geometry encoder`.

---

### Task 2: Thread geometry through `forward_grounding` / `detect` / `predict`

**Files:** `sam/modeling/decoders/detr_decoder.py` (`forward_grounding`, `detect`); `sam/models/sam3_predictor.py` (`predict`, `_pack_geometry`).

**Interfaces:** `forward_grounding(feats, pos, text_emb, text_mask, geo=None)` where `geo` is a dict `{box_coords, box_labels, point_coords, point_labels}`; `Sam3Predictor.predict(image, concept, confidence_threshold=.5, geometry=None)` / `Sam3MultiplexPredictor.predict(...)`; `_pack_geometry(prompt, image_hw, device) -> dict | None`.

- [ ] **Step 1: `forward_grounding`** — drop the `exemplar_emb is None` assert; pass geometry + `img_sizes=[(h, w)]` to the encoder; the existing `cat([text_emb, geo_feats])` / `cat([text_mask, geo_mask], 1)` already handles >1 geo token:

```python
        if self.geometry_encoder is not None:
            geo = geo or {}
            geo_feats, geo_mask = self.geometry_encoder(
                img_feats=[img_feat_seq], img_pos_embeds=[img_pos_seq], bs=bs,
                img_sizes=[(h, w)], **geo,
            )
```

- [ ] **Step 2: `detect`** — accept `geo=None`, forward it to `forward_grounding`. `Sam3MultiplexPredictor.predict` (`sam3_predictor.py`) uses `forward_grounding` directly — add the same `geo` passthrough there.

- [ ] **Step 3: `_pack_geometry` + `predict`** in `sam3_predictor.py`:

```python
    @staticmethod
    def _pack_geometry(prompt, image_hw, device):
        """Pack a point/box GeometryPrompt into the encoder's normalized inputs."""
        if prompt is None:
            return None
        if prompt.masks_logits is not None:
            raise NotImplementedError(
                "mask geometry prompts are unsupported (no mask_encoder weights); "
                "use box or point prompts"
            )
        h, w = image_hw
        geo = {}
        if prompt.points_coords is not None:
            c = prompt.points_coords.to(device).float()
            c = c if prompt.is_normalized else c / torch.tensor([w, h], device=device)
            geo["point_coords"] = c[:, None, :]                       # (N,1,2)
            geo["point_labels"] = prompt.points_labels.to(device)[:, None]
        if prompt.boxes is not None:
            b = prompt.boxes.to(device).float()
            b = b if prompt.is_normalized else b / torch.tensor([w, h, w, h], device=device)
            cx = (b[:, 0] + b[:, 2]) / 2
            cy = (b[:, 1] + b[:, 3]) / 2
            bw = (b[:, 2] - b[:, 0]).abs()
            bh = (b[:, 3] - b[:, 1]).abs()
            geo["box_coords"] = torch.stack([cx, cy, bw, bh], -1)[:, None, :]  # (N,1,4)
            geo["box_labels"] = torch.ones(b.shape[0], 1, device=device)
        return geo or None
```

Wire `geometry` through `predict` (both base and multiplex): `geo = self._pack_geometry(geometry, image_hw, self.device)` then pass to `detect`/`forward_grounding`.

- [ ] **Step 4: Run** the geometry-encoder unit + a quick smoke (predict with a box returns detections):

```python
# smoke inside tests/test_geometry_encoder.py
@needs_gpu
def test_predict_with_box_runs():
    from sam.build_sam import build_sam3_multiplex_video_predictor
    import numpy as np
    from PIL import Image
    from sam.prompts import ConceptPrompt, GeometryPrompt
    pred = build_sam3_multiplex_video_predictor(
        config_file="configs/sam3/sam3.1.yaml", ckpt_path=CKPT, device="cuda")
    img = np.asarray(Image.open("notebooks/videos/bedroom/00000.jpg").convert("RGB"))
    # Sam3MultiplexPredictor image predict lives on the video predictor's detector path;
    # exercise via the detector forward_grounding through the image predictor build.
```

(If the image predictor isn't directly built here, defer the end-to-end smoke to the Task 3 parity test and keep Task 2's check to the encoder unit + a `_pack_geometry` CPU unit.)

- [ ] **Step 5: Commit** `feat(sam3): geometry prompt through forward_grounding/detect/predict`.

---

### Task 3: Image box-prompt parity golden

**Files:** `tests/parity/reference_sam3/capture_sam3_box_golden.py` + fixtures; `tests/parity/test_sam3_box_prompt_parity.py`.

- [ ] **Step 1: Capture** (reference env, `--patches`) — build the upstream SAM 3.1 image predictor (or the multiplex detector on a single frame), a fixed frame + text `"person"` + one normalized box prompt; save `boxes`, `scores`, `presence`, top-mask (uint8) to `fixtures/sam3/box_prompt.npz` + `scenario.json`. Delegate the reference-env run (as in the prior goldens).
- [ ] **Step 2: Parity test** — our `predict(frame, ConceptPrompt("person"), geometry=GeometryPrompt(boxes=...))` vs golden: boxes atol 2px, scores atol 1e-2, top-mask IoU ≥ 0.99, presence atol 1e-2.
- [ ] **Step 3: Run — PASS** (debug ROI denorm / pos_enc params against the first mismatch).
- [ ] **Step 4: Commit** `test(sam3): image box-prompt parity`.

---

### Task 4: Regression + ledger

- [ ] **Step 1:** Run `tests/parity/test_sam3_parity.py` (image detect, text-only) + the sam3.1 mux suite (interactive/model-find/growth) — all still pass (null geo path unchanged).
- [ ] **Step 2:** Ledger: mark box/point geometry (image, 2a) done; note 2b (video routing) + exemplar/mask still open.
- [ ] **Step 3: Commit** `docs(sam3): geometry encoder image ledger`.

---

## Self-Review

- **Spec coverage:** encoder box/point forward + null-guard (T1), forward_grounding/detect/predict threading + `_pack_geometry` (T2), image box parity (T3), regression + ledger (T4). ✓
- **Placeholders:** T2 step 4 names the fallback (defer end-to-end smoke to T3) rather than hand-waving; T3 step 1 flags the reference-env delegation + the ROI/pos_enc debug spot. ✓
- **Type consistency:** encoder `forward(..., box_coords, box_labels, point_coords, point_labels)`; `forward_grounding(..., geo=dict)`; `_pack_geometry(prompt, image_hw, device) -> dict|None`; `predict(..., geometry=GeometryPrompt)`. ✓
