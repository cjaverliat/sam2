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
# SAM 3.1 multiplex capture (addendum oracle)
# ======================================================================================
# Multiplex bucket capacity K. SAM 3.1's checkpoint sizes the memory encoder's mask
# channels + the MultiplexMaskDecoder token embeddings to this value, so it is fixed by
# the weights (do NOT change it -- a different value breaks state_dict loading). For N
# tracked objects the controller packs them into ``num_buckets = ceil(N / K)`` buckets and
# the SAM mask decoder + memory attention run at ``batch = num_buckets`` (joint decode).
MULTIPLEX_COUNT = 16


def determinism_sam31():
    """Determinism for the SAM 3.1 multiplex path.

    Same as ``determinism()`` EXCEPT we must NOT call
    ``torch.use_deterministic_algorithms(True)``: the multiplex memory attention
    (``sam3.model.decoder.functional_attention``) hardcodes
    ``sdpa_kernel(SDPBackend.FLASH_ATTENTION)``, and deterministic mode forbids the
    (nondeterministic) flash SDPA kernel -> ``RuntimeError: No available kernel``. We keep
    every other lever (seed 0, cuDNN deterministic, TF32 off); reproducibility is verified
    empirically (a second capture reproduces boxes/scores; masks IoU = 1.0).
    """
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _patch_multiplex_sdpa():
    """Allow the math/efficient SDPA fallback for the multiplex memory attention.

    ``decoder.functional_attention`` runs ``with sdpa_kernel(SDPBackend.FLASH_ATTENTION):``
    when ``use_fa3=False`` (we have no flash-attn-3 in the reference env). On this GPU the
    flash SDPA backend is unavailable for these inputs -> ``No available kernel``. We
    override the module-level ``sdpa_kernel`` so the forced-flash context instead permits
    every backend; SDPA then auto-selects (math is the exact reference attention). This is
    a capture-env kernel-dispatch concession (no weights/logic change), analogous to the
    bf16-autocast precision concession; parity gates are atol~1e-2 / IoU>=0.99, not bitwise.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel as _sk
    from sam3.model import decoder as _dec

    _all = [
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.MATH,
    ]
    _dec.sdpa_kernel = lambda *a, **k: _sk(_all)


def _build_sam31_predictor(args):
    """Build the SAM 3.1 multiplex video predictor (the recommended entry point).

    ``build_sam3_predictor(version="sam3.1")`` -> ``build_sam3_multiplex_video_predictor``
    -> ``Sam3MultiplexVideoPredictor`` wrapping ``Sam3MultiplexTrackingWithInteractivity``
    (detector ``Sam3MultiplexDetector`` + tracker ``Sam3VideoTrackingMultiplexDemo``, the
    ``VideoTrackingMultiplex`` whose ``sam_mask_decoder`` is the ``MultiplexMaskDecoder``).
    We load the LOCAL ``sam3.1_multiplex.pt`` (no HF) and force ``use_fa3=False`` (flash-attn-3
    is absent). The predictor ``__init__`` enters a global bf16 autocast and re-enables TF32,
    so we re-assert ``determinism_sam31()`` afterwards.
    """
    from sam3.model_builder import build_sam3_predictor

    _patch_multiplex_sdpa()
    predictor = build_sam3_predictor(
        version="sam3.1",
        checkpoint_path=args.sam31_checkpoint,
        bpe_path=args.bpe_path,
        multiplex_count=MULTIPLEX_COUNT,
        use_fa3=False,
        use_rope_real=False,
        compile=False,
        warm_up=False,
        async_loading_frames=False,
    )
    determinism_sam31()  # predictor.__init__ re-enables TF32 -> force it back off
    return predictor


def _save_frames_dir(frames_rgb, tmp_dir):
    """Save (T,H,W,3) uint8 frames as ``0.png .. {T-1}.png`` (the integer-named, lossless
    layout the upstream image-folder loader expects)."""
    for i, rgb in enumerate(frames_rgb):
        Image.fromarray(rgb).save(tmp_dir / f"{i}.png")


def capture_image_sam31(args, predictor):
    """SAM 3.1 image golden: the truck@384x512 / phrase "truck", run through the multiplex
    predictor's detector as a single-frame input (the DETR detector is set-prediction, NOT
    multiplex -- so this mainly validates the 3.1 detector weights load + run). Mirrors the
    base ``image.npz`` final-output keys."""
    model = predictor.model
    detector = model.detector

    captured = {}
    orig_forward_text = detector.backbone.forward_text

    def forward_text_spy(*a, **k):
        out = orig_forward_text(*a, **k)
        captured["text"] = out
        return out

    detector.backbone.forward_text = forward_text_spy

    # forward_grounding is inherited from Sam3Image; it yields the raw DETR set predictions
    # + the presence token, exactly like the base image capture.
    orig_forward_grounding = detector.forward_grounding

    def forward_grounding_spy(*a, **k):
        out = orig_forward_grounding(*a, **k)
        captured.setdefault("grounding", out)
        return out

    detector.forward_grounding = forward_grounding_spy

    # capture the principal encoder level (vision_features) from the shared PE backbone
    orig_forward_image = detector.backbone.forward_image

    def forward_image_spy(*a, **k):
        out = orig_forward_image(*a, **k)
        captured.setdefault("backbone_out", out)
        return out

    detector.backbone.forward_image = forward_image_spy

    image_rgb = _load_rgb(args.truck, IMAGE_HW)
    tmp_dir = Path(tempfile.mkdtemp(prefix="sam31_ref_image_"))
    try:
        Image.fromarray(image_rgb).save(tmp_dir / "0.png")

        def run():
            captured.clear()
            state = model.init_state(resource_path=str(tmp_dir),
                                     async_loading_frames=False)
            _fi, ap_out = model.add_prompt(state, frame_idx=0, text_str=IMAGE_PHRASE)
            return state, ap_out

        (state, ap_out), mode = _run_section(run)
    finally:
        detector.backbone.forward_text = orig_forward_text
        detector.forward_grounding = orig_forward_grounding
        detector.backbone.forward_image = orig_forward_image
        shutil.rmtree(tmp_dir, ignore_errors=True)

    H, W = IMAGE_HW
    out = {}
    # --- final detect outputs (denormalise the predictor's xywh boxes -> xyxy px) -------
    boxes_xywh = np.asarray(ap_out["out_boxes_xywh"], np.float32)  # normalised xywh
    if boxes_xywh.size:
        bx = boxes_xywh.copy()
        bx[:, 0] *= W; bx[:, 1] *= H; bx[:, 2] *= W; bx[:, 3] *= H
        boxes_xyxy = np.stack([bx[:, 0], bx[:, 1], bx[:, 0] + bx[:, 2], bx[:, 1] + bx[:, 3]], 1)
    else:
        boxes_xyxy = np.zeros((0, 4), np.float32)
    out["boxes"] = boxes_xyxy.astype(np.float32)
    out["scores"] = np.asarray(ap_out["out_probs"], np.float32)
    masks = np.asarray(ap_out["out_binary_masks"])
    out["masks"] = masks.astype(np.uint8)

    grounding = captured.get("grounding")
    if grounding is not None and "presence_logit_dec" in grounding:
        presence_logit = grounding["presence_logit_dec"]
        out["presence_logit"] = _np(presence_logit, np.float32)
        out["presence"] = _np(torch.as_tensor(presence_logit).float().sigmoid(), np.float32)
    if grounding is not None and "pred_boxes" in grounding:
        out["pred_boxes_cxcywh"] = _np(grounding["pred_boxes"], np.float32)
    if grounding is not None and "pred_logits" in grounding:
        out["pred_logits"] = _np(grounding["pred_logits"], np.float32)

    text = captured.get("text")
    if text is not None and "language_features" in text:
        out["text_emb"] = _np(text["language_features"], np.float16)
    if text is not None and "language_embeds" in text:
        out["text_embeds_pre"] = _np(text["language_embeds"], np.float16)

    backbone_out = captured.get("backbone_out")
    if backbone_out is not None and "vision_features" in backbone_out:
        out["enc_feat_lastlevel"] = _np(backbone_out["vision_features"], np.float16)

    out["image_input_rgb"] = image_rgb
    out["image_phrase"] = np.array(IMAGE_PHRASE)
    out["image_hw"] = np.array(IMAGE_HW, np.int64)
    out["precision_mode"] = np.array(mode)
    out["upstream_commit"] = np.array(UPSTREAM_COMMIT)
    out["confidence_threshold"] = np.array(CONFIDENCE_THRESHOLD, np.float32)
    out["model_version"] = np.array("sam3.1")

    meta = {
        "model_version": "sam3.1",
        "num_detections": int(out["boxes"].shape[0]),
        "precision_mode": mode,
        "captured_keys": sorted(out),
        "scores": out["scores"].tolist(),
        "enc_feat_lastlevel_shape": (
            list(out["enc_feat_lastlevel"].shape) if "enc_feat_lastlevel" in out else None
        ),
        "text_emb_shape": list(out["text_emb"].shape) if "text_emb" in out else None,
    }
    return out, meta


# module-level capture slot for the frame-1 multiplex tracker step
_MUX_CAP = {"frame": -1, "bucket_masks": None, "state": None}


def capture_video_sam31(args, predictor):
    """SAM 3.1 multiplex video golden: the dance clip, first 4 frames, phrase "person".

    ``add_prompt(0)`` conditions the detected objects on frame 0; ``propagate_in_video(
    start_frame_idx=1)`` tracks frames 1..3 (re-processing frame 0 is unsupported -- the
    cond-frame features are already consumed). Mirrors the base ``video.npz`` masklet keys
    and ADDS the multiplex-specific activations: the ``MultiplexController`` mapping
    (K / num_buckets / mux+demux matrices / assignments) and the frame-1 tracker step in
    BOTH spaces (``trk_f1_mux_buckets`` bucket-space, ``trk_f1_demux_perobj`` per-object).
    """
    model = predictor.model            # Sam3MultiplexTrackingWithInteractivity
    tracker = model.tracker.model      # VideoTrackingMultiplex (owns sam_mask_decoder)

    # frame spy: tag the frame currently being processed (mirrors the base capture)
    orig_rsfi = model._run_single_frame_inference

    def rsfi_spy(inference_state, frame_idx, reverse, **kw):
        _MUX_CAP["frame"] = frame_idx
        return orig_rsfi(inference_state, frame_idx, reverse, **kw)

    model._run_single_frame_inference = rsfi_spy

    # mask-decoder hook: grab the BUCKET-SPACE mask logits at frame 1. The MultiplexMaskDecoder
    # runs at batch = num_buckets and returns masks (num_buckets, K, 3, H, W) -- BEFORE the demux
    # that _forward_sam_heads applies next.
    def md_hook(_m, _inp, output):
        if _MUX_CAP["frame"] == 1 and _MUX_CAP["bucket_masks"] is None:
            if isinstance(output, dict) and "masks" in output:
                _MUX_CAP["bucket_masks"] = output["masks"].detach().float()

    tracker.sam_mask_decoder.register_forward_hook(md_hook)

    # _forward_sam_heads spy: capture the MultiplexState for the frame-1 propagation pass
    # (point_inputs / mask_inputs both None == the multiplexed propagation path).
    orig_fsh = tracker._forward_sam_heads

    def fsh_spy(*a, **k):
        out = orig_fsh(*a, **k)
        if (
            _MUX_CAP["frame"] == 1
            and _MUX_CAP["state"] is None
            and k.get("point_inputs") is None
            and k.get("mask_inputs") is None
            and k.get("multiplex_state") is not None
        ):
            _MUX_CAP["state"] = k["multiplex_state"]
        return out

    tracker._forward_sam_heads = fsh_spy

    # downscale the first VIDEO_NUM_FRAMES dance frames (same scenario as base video.npz)
    src_dir = Path(args.sam3_root) / "assets" / "videos" / "0001"
    tmp_dir = Path(tempfile.mkdtemp(prefix="sam31_ref_frames_"))
    frames_rgb = []
    _MUX_CAP.update({"frame": -1, "bucket_masks": None, "state": None})
    try:
        for i in range(VIDEO_NUM_FRAMES):
            rgb = _load_rgb(src_dir / f"{i}.jpg", VIDEO_HW)
            frames_rgb.append(rgb)
        frames_rgb = np.stack(frames_rgb)  # (T,H,W,3) uint8
        _save_frames_dir(frames_rgb, tmp_dir)

        def run():
            state = model.init_state(resource_path=str(tmp_dir),
                                     async_loading_frames=False)
            f0, ap_out = model.add_prompt(state, frame_idx=0, text_str=VIDEO_PHRASE)
            masklets = {f0: ap_out}
            for f_idx, fout in model.propagate_in_video(
                state, start_frame_idx=1, max_frame_num_to_track=None, reverse=False,
            ):
                masklets[f_idx] = fout
            return state, masklets

        (state, masklets), mode = _run_section(run)
    finally:
        model._run_single_frame_inference = orig_rsfi
        tracker._forward_sam_heads = orig_fsh
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

    # --- multiplex-specific activations (the NEW content) -------------------------------
    # The committed fixture carries the ESSENTIALS (the demux mapping + the per-object
    # demuxed frame-1 decode). The full BUCKET-SPACE tensor (16 slots incl. padding) is bulky
    # (~8 MB), so -- exactly like the base capture split its 38 MB encoder pyramid -- it goes
    # to a SEPARATE, git-ignored ``video_sam31_multiplex_internals.npz`` (regenerable here).
    ms = _MUX_CAP["state"]
    bucket_masks = _MUX_CAP["bucket_masks"]
    internals = {}
    mux_meta = {}
    if ms is not None:
        K = int(ms.multiplex_count)
        num_buckets = int(ms.num_buckets)
        out["multiplex_count"] = np.array(K, np.int64)
        out["num_buckets"] = np.array(num_buckets, np.int64)
        out["total_valid_entries"] = np.array(int(ms.total_valid_entries), np.int64)
        # assignments: (num_buckets, multiplex_count) int (object idx per slot; <0 = padding)
        out["mux_assignments"] = np.asarray(ms.assignments, np.int64)
        out["mux_matrix"] = _np(ms.mux_matrix, np.float32)      # (num_buckets*K, n_valid)
        out["demux_matrix"] = _np(ms.demux_matrix, np.float32)  # (n_valid, num_buckets*K)
        out["mux_valid_mask"] = _np(ms.get_valid_object_mask(), np.uint8)  # (num_buckets,K)
        mux_meta = {
            "multiplex_count_K": K,
            "num_buckets": num_buckets,
            "total_valid_entries": int(ms.total_valid_entries),
            "assignments": [list(map(int, b)) for b in ms.assignments],
            "mux_matrix_shape": list(out["mux_matrix"].shape),
            "demux_matrix_shape": list(out["demux_matrix"].shape),
        }
    if bucket_masks is not None:
        if ms is not None:
            # demux the SAME tensor back to per-object space -> the committed round-trip target
            # (num_objects, num_multimask=3, H*4, W*4). Analogous to the base ``trk_f1``.
            out["trk_f1_demux_perobj"] = _np(ms.demux(bucket_masks), np.float16)
            mux_meta["trk_f1_demux_perobj_shape"] = list(out["trk_f1_demux_perobj"].shape)
        # bucket space: (num_buckets, K, num_multimask=3, H*4, W*4) -- the MultiplexMaskDecoder
        # output BEFORE demux (joint decode of all K slots per bucket). Git-ignored (bulky).
        internals["trk_f1_mux_buckets"] = _np(bucket_masks, np.float16)
        internals["precision_mode"] = np.array(mode)
        internals["upstream_commit"] = np.array(UPSTREAM_COMMIT)
        mux_meta["trk_f1_mux_buckets_shape"] = list(internals["trk_f1_mux_buckets"].shape)
        mux_meta["trk_f1_mux_buckets_file"] = (
            "video_sam31_multiplex_internals.npz (NOT committed; regenerate via capture)"
        )

    out["video_frames_rgb"] = frames_rgb
    out["video_phrase"] = np.array(VIDEO_PHRASE)
    out["video_hw"] = np.array(VIDEO_HW, np.int64)
    out["video_frame_indices"] = np.arange(VIDEO_NUM_FRAMES, dtype=np.int64)
    out["precision_mode"] = np.array(mode)
    out["upstream_commit"] = np.array(UPSTREAM_COMMIT)
    out["model_version"] = np.array("sam3.1")

    meta = {
        "model_version": "sam3.1",
        "precision_mode": mode,
        "per_frame": per_frame,
        "multiplex": mux_meta,
    }
    return out, internals, meta


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


def _load_scenario(path):
    """Load the existing scenario.json so a second-model run augments (not clobbers) it."""
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"upstream_commit": UPSTREAM_COMMIT,
            "determinism": "seed=0, deterministic, TF32 off"}


def _run_sam3(args, out_dir, scenario):
    """Base sam3.pt (per-object) oracle -> image.npz / video.npz."""
    if args.only in (None, "image"):
        img_out, img_pyramid, img_meta = capture_image(args)
        _save(img_out, out_dir / "image.npz")
        _save(img_pyramid, out_dir / "image_encoder_pyramid.npz")  # git-ignored
        scenario["image"] = {
            "model_version": "sam3",
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
            "model_version": "sam3",
            "source": "sam3_reference/assets/videos/0001 (upstream dance clip)",
            "resize_hw": list(VIDEO_HW),
            "num_frames": VIDEO_NUM_FRAMES,
            "phrase": VIDEO_PHRASE,
            **vid_meta,
        }


def _run_sam31(args, out_dir, scenario):
    """SAM 3.1 multiplex oracle -> image_sam31.npz / video_sam31.npz (+ git-ignored
    multiplex internals). One predictor build serves both the image and the video."""
    determinism_sam31()
    predictor = _build_sam31_predictor(args)
    if args.only in (None, "image"):
        img_out, img_meta = capture_image_sam31(args, predictor)
        _save(img_out, out_dir / "image_sam31.npz")
        scenario["image_sam31"] = {
            "source": "notebooks/images/truck.jpg",
            "resize_hw": list(IMAGE_HW),
            "phrase": IMAGE_PHRASE,
            "entry_point": 'build_sam3_predictor(version="sam3.1").model.add_prompt',
            **img_meta,
        }
    if args.only in (None, "video"):
        vid_out, vid_internals, vid_meta = capture_video_sam31(args, predictor)
        _save(vid_out, out_dir / "video_sam31.npz")
        if vid_internals:
            _save(vid_internals, out_dir / "video_sam31_multiplex_internals.npz")  # git-ignored
        scenario["video_sam31"] = {
            "source": "sam3_reference/assets/videos/0001 (upstream dance clip)",
            "resize_hw": list(VIDEO_HW),
            "num_frames": VIDEO_NUM_FRAMES,
            "phrase": VIDEO_PHRASE,
            "entry_point": 'build_sam3_predictor(version="sam3.1").model '
                           "(add_prompt @f0 -> propagate_in_video start_frame_idx=1)",
            **vid_meta,
        }


def main():
    here = Path(__file__).resolve()
    repo_root = here.parents[3]  # sam2/
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(repo_root / "checkpoints" / "sam3.pt"))
    ap.add_argument("--sam31-checkpoint",
                    default=str(repo_root / "checkpoints" / "sam3.1_multiplex.pt"))
    ap.add_argument("--sam3-root", default=str(repo_root.parent / "sam3_reference"))
    ap.add_argument("--truck", default=str(repo_root / "notebooks" / "images" / "truck.jpg"))
    ap.add_argument("--out-dir", default=str(repo_root / "tests" / "parity" / "fixtures" / "sam3"))
    ap.add_argument("--bpe-path", default=None)
    ap.add_argument("--model", choices=["sam3", "sam3.1"], default="sam3",
                    help="which oracle to capture (default: sam3 base, preserves prior behavior)")
    ap.add_argument("--only", choices=["image", "video"], default=None)
    args = ap.parse_args()
    if args.bpe_path is None:
        args.bpe_path = str(
            Path(args.sam3_root) / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        )

    assert torch.cuda.is_available(), "capture requires CUDA"
    out_dir = Path(args.out_dir)
    scenario = _load_scenario(out_dir / "scenario.json")

    if args.model == "sam3":
        determinism()
        _run_sam3(args, out_dir, scenario)
    else:
        _run_sam31(args, out_dir, scenario)

    (out_dir / "scenario.json").write_text(json.dumps(scenario, indent=2))
    print("wrote", out_dir / "scenario.json")
    print(json.dumps(scenario, indent=2))


if __name__ == "__main__":
    main()
