# SPDX-License-Identifier: LicenseRef-SAM
"""Capture golden reference activations from the OFFICIAL SAM 3 (``facebook/sam3``).

This is the **parity oracle** for Phase 1 of the SAM 3 integration: every vendored
``Sam3*`` re-implementation (Tasks 2-10) validates its outputs against the fixtures
this script produces. It must therefore run the *unmodified upstream* model end to
end and save enough intermediate activations for per-component checks.

Run it ONCE, in the **isolated reference env** (NOT this repo's pixi env):

    # sibling clone facebookresearch/sam3 @ 5dd401d, its own venv (see README.md):
    ../sam3_reference/.venv/Scripts/python.exe \
        tests/parity/reference_sam3/capture_sam3_golden.py

Outputs (committed): ``tests/parity/fixtures/sam3/{image,video}.npz`` + ``scenario.json``.

Determinism mirrors ``tests/parity/run_pipelines.py::_determinism`` (seed 0, deterministic
algorithms, cuDNN deterministic, TF32 OFF) so the capture is reproducible and the later
vendored re-implementation -- run under the same regime -- matches within tolerance.

Precision: fp32 is attempted first (highest reproducibility, per the harness convention);
if the 848M model OOMs on the local 12 GB GPU, the section is retried under bf16 autocast
(the precision the official demo itself uses). The mode actually used is recorded in each
fixture (``precision_mode``) and in ``scenario.json`` so downstream parity runs match it.

Scenario (see ``scenario.json`` / ``README.md`` for the authoritative record):
  * IMAGE -- this repo's ``notebooks/images/truck.jpg`` resized to 384x512, phrase "truck"
            (mirrors the upstream ``sam3_image_predictor_example`` text-prompt flow).
  * VIDEO -- upstream ``assets/videos/0001`` (the dance clip from the upstream
            ``sam3_video_predictor_example``), first 4 frames resized to 288x512,
            phrase "person": add text @ frame 0 -> forward-propagate (the golden scenario).
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

UPSTREAM_COMMIT = "5dd401d1c5c1d5c3eedff06d41b77af824517619"

# ----- scenario constants -------------------------------------------------------------
IMAGE_HW = (384, 512)  # (H, W) the truck image is resized to before the processor
IMAGE_PHRASE = "truck"
VIDEO_HW = (288, 512)  # (H, W) each dance frame is resized to (keeps 16:9 of 1280x720)
VIDEO_PHRASE = "person"
VIDEO_NUM_FRAMES = 4  # frames 0..3
CONFIDENCE_THRESHOLD = 0.5


def determinism():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # NOTE: sam3.model_builder calls _setup_tf32() at import time (enables TF32); we
    # disable it here, AFTER import, so the capture is TF32-off like run_pipelines.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _np(t, dtype=None):
    """Detach a tensor to a contiguous numpy array, optionally down-casting."""
    if isinstance(t, torch.Tensor):
        a = t.detach().float().cpu().numpy()
    else:
        a = np.asarray(t)
    if dtype is not None:
        a = a.astype(dtype)
    return np.ascontiguousarray(a)


def _feat_levels(value):
    """Yield plain tensors from a backbone_fpn / pos list (handles NestedTensor)."""
    for lvl in value:
        yield lvl.tensors if hasattr(lvl, "tensors") else lvl


def _load_rgb(path, hw):
    """Load an image as an (H, W, 3) uint8 array resized to hw=(H, W)."""
    img = Image.open(path).convert("RGB").resize((hw[1], hw[0]))  # PIL is (W, H)
    return np.asarray(img, dtype=np.uint8)


# ======================================================================================
# IMAGE capture
# ======================================================================================
def capture_image(args):
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(
        bpe_path=args.bpe_path,
        device="cuda",
        eval_mode=True,
        checkpoint_path=args.checkpoint,
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
    )

    # Hook the text tower + the grounding head so we capture text_emb and the raw
    # detector outputs (incl. the presence logit) alongside the post-processed results.
    captured = {}
    orig_forward_text = model.backbone.forward_text

    def forward_text_spy(*a, **k):
        out = orig_forward_text(*a, **k)
        captured["text"] = out
        return out

    model.backbone.forward_text = forward_text_spy

    orig_forward_grounding = model.forward_grounding

    def forward_grounding_spy(*a, **k):
        out = orig_forward_grounding(*a, **k)
        captured["grounding"] = out
        return out

    model.forward_grounding = forward_grounding_spy

    image_rgb = _load_rgb(args.truck, IMAGE_HW)
    pil = Image.fromarray(image_rgb)

    def run():
        captured.clear()
        proc = Sam3Processor(
            model, resolution=1008, device="cuda",
            confidence_threshold=CONFIDENCE_THRESHOLD,
        )
        state = proc.set_image(pil)
        proc.reset_all_prompts(state)
        state = proc.set_text_prompt(prompt=IMAGE_PHRASE, state=state)
        return state

    state, mode = _run_section(run)

    backbone_out = state["backbone_out"]
    grounding = captured["grounding"]
    text = captured["text"]

    out = {}
    # --- final detect outputs (the primary golden) --------------------------------
    out["boxes"] = _np(state["boxes"], np.float32)               # (N,4) xyxy, image px
    out["scores"] = _np(state["scores"], np.float32)             # (N,) presence-weighted
    out["masks"] = _np(state["masks"].squeeze(1), np.uint8)      # (N,H,W) bool->uint8
    out["masks_logits"] = _np(state["masks_logits"].squeeze(1), np.float16)  # (N,H,W) prob
    # presence: the image-level presence token (raw logit + sigmoid)
    presence_logit = grounding["presence_logit_dec"]
    out["presence_logit"] = _np(presence_logit, np.float32)
    out["presence"] = _np(torch.as_tensor(presence_logit).float().sigmoid(), np.float32)
    # raw (pre-threshold) DETR set predictions, useful for detector parity
    out["pred_boxes_cxcywh"] = _np(grounding["pred_boxes"], np.float32)
    out["pred_logits"] = _np(grounding["pred_logits"], np.float32)

    # --- text embedding -----------------------------------------------------------
    out["text_emb"] = _np(text["language_features"], np.float16)
    if "language_embeds" in text:
        out["text_embeds_pre"] = _np(text["language_embeds"], np.float16)

    # --- vision-encoder feature pyramid -------------------------------------------
    # The principal / last level (72x72, stride-14) is the single level the DETR
    # detector consumes (num_feature_levels=1) -> committed in image.npz. The full
    # high-res pyramid (l0=288x288, l1=144x144) + positional encodings are large at
    # the model's fixed 1008 internal resolution (~38 MB), so they go to a SEPARATE,
    # git-ignored file (regenerate via this script). Positional encodings are a
    # deterministic sine function (computed, not learned), kept only for completeness.
    fpn = list(_feat_levels(backbone_out["backbone_fpn"]))
    pos = list(_feat_levels(backbone_out["vision_pos_enc"]))
    out["enc_feat_lastlevel"] = _np(backbone_out["vision_features"], np.float16)
    pyramid = {}
    for i, lvl in enumerate(fpn):
        pyramid[f"enc_feat_l{i}"] = _np(lvl, np.float16)
    for i, p in enumerate(pos):
        pyramid[f"enc_pos_l{i}"] = _np(p, np.float16)
    pyramid["precision_mode"] = np.array(mode)
    pyramid["upstream_commit"] = np.array(UPSTREAM_COMMIT)

    # --- inputs + metadata (byte-identical reproduction by later tasks) -----------
    out["image_input_rgb"] = image_rgb                            # (H,W,3) uint8
    out["image_phrase"] = np.array(IMAGE_PHRASE)
    out["image_hw"] = np.array(IMAGE_HW, np.int64)
    out["precision_mode"] = np.array(mode)
    out["upstream_commit"] = np.array(UPSTREAM_COMMIT)
    out["confidence_threshold"] = np.array(CONFIDENCE_THRESHOLD, np.float32)

    meta = {
        "num_detections": int(out["boxes"].shape[0]),
        "precision_mode": mode,
        "enc_levels": len(fpn),
        "enc_shapes": [list(fpn[i].shape) for i in range(len(fpn))],
        "enc_feat_lastlevel_shape": list(out["enc_feat_lastlevel"].shape),
        "text_emb_shape": list(out["text_emb"].shape),
        "pyramid_file": "image_encoder_pyramid.npz (NOT committed; regenerate via capture)",
    }
    del model
    torch.cuda.empty_cache()
    return out, pyramid, meta


# ======================================================================================
# VIDEO capture
# ======================================================================================
def capture_video(args):
    from sam3.model_builder import build_sam3_video_model

    model = build_sam3_video_model(
        checkpoint_path=args.checkpoint,
        load_from_HF=False,
        bpe_path=args.bpe_path,
        device="cuda",
        apply_temporal_disambiguation=True,  # the default / notebook config
    )

    # downscale the first VIDEO_NUM_FRAMES dance frames -> a temp dir of lossless PNGs
    # (PNG keeps the saved RGB array byte-identical to what the model loads).
    src_dir = Path(args.sam3_root) / "assets" / "videos" / "0001"
    tmp_dir = Path(tempfile.mkdtemp(prefix="sam3_ref_frames_"))
    frames_rgb = []
    try:
        for i in range(VIDEO_NUM_FRAMES):
            rgb = _load_rgb(src_dir / f"{i}.jpg", VIDEO_HW)
            frames_rgb.append(rgb)
            Image.fromarray(rgb).save(tmp_dir / f"{i}.png")
        frames_rgb = np.stack(frames_rgb)  # (T,H,W,3) uint8

        # hook the tracker mask decoder to grab the raw per-object low-res mask logits
        # at frame 1 (one tracker step), demuxed per-object (base SAM 3 is per-object).
        trk_capture = {"frame1": []}
        model._cap_frame = -1
        orig_rsfi = model._run_single_frame_inference

        def rsfi_spy(inference_state, frame_idx, reverse):
            model._cap_frame = frame_idx
            return orig_rsfi(inference_state, frame_idx, reverse)

        model._run_single_frame_inference = rsfi_spy

        def mask_decoder_hook(_module, _inp, output):
            if model._cap_frame == 1:
                low_res = output[0] if isinstance(output, (tuple, list)) else output
                trk_capture["frame1"].append(_np(low_res, np.float16))

        model.tracker.sam_mask_decoder.register_forward_hook(mask_decoder_hook)

        def run():
            trk_capture["frame1"].clear()
            state = model.init_state(resource_path=str(tmp_dir))
            model.add_prompt(state, frame_idx=0, text_str=VIDEO_PHRASE)
            masklets = {}
            for f_idx, fout in model.propagate_in_video(
                state,
                start_frame_idx=0,
                max_frame_num_to_track=VIDEO_NUM_FRAMES - 1,
                reverse=False,
            ):
                masklets[f_idx] = fout
            return masklets

        masklets, mode = _run_section(run)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    out = {}
    per_frame = {}
    for f_idx, fout in masklets.items():
        obj_ids = np.asarray(fout["out_obj_ids"], np.int64)
        masks = np.asarray(fout["out_binary_masks"])  # (N,H,W) bool
        probs = np.asarray(fout["out_probs"], np.float32)
        out[f"frame{f_idx}_obj_ids"] = obj_ids
        out[f"frame{f_idx}_scores"] = probs
        for j, oid in enumerate(obj_ids.tolist()):
            out[f"frame{f_idx}_obj{oid}"] = masks[j].astype(np.uint8)
        per_frame[int(f_idx)] = {"obj_ids": obj_ids.tolist(), "scores": probs.tolist()}

    # tracker-step activation at frame 1 (Task 5 tracker-step parity): the raw
    # sam_mask_decoder low-res mask logits, shape (num_obj, num_multimask=3, H, W),
    # per-object (base SAM 3 tracks per-object; no multiplex bucketing to demux).
    f1 = trk_capture["frame1"]
    if f1:
        try:
            out["trk_f1"] = np.concatenate(f1, axis=0)  # stack object decodes
        except ValueError:
            out["trk_f1"] = f1[0]  # shapes differ across calls -> keep the first
        out["trk_f1_num_calls"] = np.array(len(f1), np.int64)

    out["video_frames_rgb"] = frames_rgb                         # (T,H,W,3) uint8
    out["video_phrase"] = np.array(VIDEO_PHRASE)
    out["video_hw"] = np.array(VIDEO_HW, np.int64)
    out["video_frame_indices"] = np.arange(VIDEO_NUM_FRAMES, dtype=np.int64)
    out["precision_mode"] = np.array(mode)
    out["upstream_commit"] = np.array(UPSTREAM_COMMIT)

    meta = {
        "precision_mode": mode,
        "per_frame": per_frame,
        "trk_f1_present": bool(f1),
        "trk_f1_shape": (list(out["trk_f1"].shape) if f1 else None),
    }
    del model
    torch.cuda.empty_cache()
    return out, meta


# ======================================================================================
def _run_section(run):
    """Run ``run()`` under bf16 autocast -- the precision the official SAM 3 demo uses.

    Pure fp32 is NOT a supported inference mode upstream: perflib's fused ``addmm_act``
    (ViT MLP path) hardcodes ``.to(torch.bfloat16)``, so without an active autocast the
    next Linear raises a BFloat16-vs-Float dtype mismatch. The notebook image example and
    the base predictor's ``add_prompt`` both run under ``autocast(bf16)``. We therefore
    capture under bf16 autocast and keep every OTHER determinism lever from
    ``run_pipelines._determinism`` (seed, deterministic algos, cuDNN deterministic, TF32
    off) so the oracle is reproducible and later vendored parity matches in the same regime.
    """
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return run(), "bf16_autocast"


def _save(out, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    mb = path.stat().st_size / 1e6
    print(f"wrote {path}  ({mb:.2f} MB)  keys={sorted(out)}")
    return mb


def main():
    here = Path(__file__).resolve()
    repo_root = here.parents[3]  # sam2/
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(repo_root / "checkpoints" / "sam3.pt"))
    ap.add_argument("--sam3-root", default=str(repo_root.parent / "sam3_reference"))
    ap.add_argument("--truck", default=str(repo_root / "notebooks" / "images" / "truck.jpg"))
    ap.add_argument("--out-dir", default=str(repo_root / "tests" / "parity" / "fixtures" / "sam3"))
    ap.add_argument("--bpe-path", default=None)
    ap.add_argument("--only", choices=["image", "video"], default=None)
    args = ap.parse_args()
    if args.bpe_path is None:
        args.bpe_path = str(
            Path(args.sam3_root) / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        )

    assert torch.cuda.is_available(), "capture requires CUDA"
    determinism()
    out_dir = Path(args.out_dir)
    scenario = {"upstream_commit": UPSTREAM_COMMIT, "determinism": "seed=0, deterministic, TF32 off"}

    if args.only in (None, "image"):
        img_out, img_pyramid, img_meta = capture_image(args)
        _save(img_out, out_dir / "image.npz")
        _save(img_pyramid, out_dir / "image_encoder_pyramid.npz")  # git-ignored
        scenario["image"] = {
            "source": "notebooks/images/truck.jpg",
            "resize_hw": list(IMAGE_HW),
            "phrase": IMAGE_PHRASE,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            **img_meta,
        }

    if args.only in (None, "video"):
        vid_out, vid_meta = capture_video(args)
        _save(vid_out, out_dir / "video.npz")
        scenario["video"] = {
            "source": "sam3_reference/assets/videos/0001 (upstream dance clip)",
            "resize_hw": list(VIDEO_HW),
            "num_frames": VIDEO_NUM_FRAMES,
            "phrase": VIDEO_PHRASE,
            **vid_meta,
        }

    (out_dir / "scenario.json").write_text(json.dumps(scenario, indent=2))
    print("wrote", out_dir / "scenario.json")
    print(json.dumps(scenario, indent=2))


if __name__ == "__main__":
    main()
