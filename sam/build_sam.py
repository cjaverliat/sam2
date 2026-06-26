# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

import sam

from sam.models.sam2_predictor import Sam2Predictor
from sam.models.sam2_predictor import Sam2VideoPredictor

# Check if the user is running Python from the parent directory of the sam2 repo
# (i.e. the directory where this repo is cloned into) -- this is not supported since
# it could shadow the sam2 package and cause issues.
if os.path.isdir(os.path.join(sam.__path__[0], "sam")):
    # If the user has "sam2/sam2" in their path, they are likey importing the repo itself
    # as "sam2" rather than importing the "sam2" python package (i.e. "sam2/sam2" directory).
    # This typically happens because the user is running Python from the parent directory
    # that contains the sam2 repo they cloned.
    raise RuntimeError(
        "You're likely running Python from the parent directory of the sam2 repository "
        "(i.e. the directory where https://github.com/facebookresearch/sam2 is cloned into). "
        "This is not supported since the `sam2` Python package could be shadowed by the "
        "repository name (the repository is also named `sam2` and contains the Python package "
        "in `sam2/sam2`). Please run Python from another directory (e.g. from the repo dir "
        "rather than its parent dir, or from your home directory) after installing SAM 2."
    )


HF_MODEL_ID_TO_FILENAMES = {
    "facebook/sam2-hiera-tiny": (
        "configs/sam2/sam2_hiera_t.yaml",
        "sam2_hiera_tiny.pt",
    ),
    "facebook/sam2-hiera-small": (
        "configs/sam2/sam2_hiera_s.yaml",
        "sam2_hiera_small.pt",
    ),
    "facebook/sam2-hiera-base-plus": (
        "configs/sam2/sam2_hiera_b+.yaml",
        "sam2_hiera_base_plus.pt",
    ),
    "facebook/sam2-hiera-large": (
        "configs/sam2/sam2_hiera_l.yaml",
        "sam2_hiera_large.pt",
    ),
    "facebook/sam2.1-hiera-tiny": (
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1_hiera_tiny.pt",
    ),
    "facebook/sam2.1-hiera-small": (
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_hiera_small.pt",
    ),
    "facebook/sam2.1-hiera-base-plus": (
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_base_plus.pt",
    ),
    "facebook/sam2.1-hiera-large": (
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "sam2.1_hiera_large.pt",
    ),
}


def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    use_half=False,
) -> Sam2Predictor:
    hydra_overrides = [
        "++model._target_=sam.models.sam2_predictor.Sam2Predictor",
        f"++model.use_half={use_half}",
    ]
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def _meta_build_generic(
    config_file,
    target,
    use_half,
    apply_postprocessing,
    hydra_overrides_extra,
):
    """Instantiate a Sam2Predictor / Sam2VideoPredictor on the ``meta`` device.

    Builds the full orchestration skeleton (so every dim/sub-module is derived from
    the real config) but allocates **no** weights — meta tensors carry shape only.
    The ONNX path then swaps the 5 heavy blocks and materializes the small glue from
    ``weights.npz``, so no real weights are ever built-then-discarded. ``ckpt_path``
    is intentionally absent: the path is checkpoint-free."""
    hydra_overrides = [
        f"++model._target_={target}",
        f"++model.use_half={use_half}",
    ]
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # sigmoid the mask logits on interacted frames so encoded masks match clicks
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area`
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    with torch.device("meta"):
        model = instantiate(cfg.model, _recursive_=True)
    return model


def build_sam2_image_predictor_onnx(
    config_file,
    onnx_dir,
    device="cuda",
    use_half=False,
    use_trt=True,
    trt_opts=None,
    apply_postprocessing=True,
    hydra_overrides_extra=[],
) -> Sam2Predictor:
    """Build a Sam2Predictor image model and swap its 5 heavy blocks for the ONNX /
    TensorRT wrappers exported by tools/export_onnx.py.

    ``onnx_dir`` accepts an extracted export directory, a ``.zip`` of one (e.g. a CI
    release artifact), or an ``http(s)`` URL to such a zip (downloaded + unpacked into
    a cache; override the location with ``SAM2_ONNX_CACHE``).

    Checkpoint-free: every weight the orchestration used to read from torch (the
    mask-decoder hi-res convs ``conv_s0``/``conv_s1``, the dense positional encoding,
    ``no_mask_embed``, and the memory-path glue) is baked into the export artifacts.
    The orchestration skeleton is instantiated on the ``meta`` device (no weights
    allocated), the 5 heavy blocks are swapped for their ONNX wrappers, and the small
    glue is materialized from ``weights.npz`` — so no real weights are ever
    built-then-discarded. ``use_half`` only affects the torch glue; the ONNX sessions own
    their own precision via the exported graph + ``trt_opts``. No ``vos_optimized`` knob
    — ONNX/TRT is already the optimized path."""
    from sam.onnx.sam2_generic_onnx import attach_onnx_blocks

    model = _meta_build_generic(
        config_file,
        "sam.models.sam2_predictor.Sam2Predictor",
        use_half=use_half,
        apply_postprocessing=apply_postprocessing,
        hydra_overrides_extra=hydra_overrides_extra,
    )
    return attach_onnx_blocks(
        model, onnx_dir, device=torch.device(device), use_trt=use_trt, trt_opts=trt_opts
    )


# Backwards-compatible alias for the original name.
build_sam2_onnx = build_sam2_image_predictor_onnx


def build_sam2_video_predictor_onnx(
    config_file,
    onnx_dir,
    device="cuda",
    use_half=False,
    use_trt=True,
    trt_opts=None,
    apply_postprocessing=True,
    hydra_overrides_extra=[],
) -> Sam2VideoPredictor:
    """Build a Sam2VideoPredictor and swap its 5 heavy blocks for the ONNX /
    TensorRT wrappers exported by tools/export_onnx.py.

    ``onnx_dir`` accepts an extracted export directory, a ``.zip`` of one (e.g. a CI
    release artifact), or an ``http(s)`` URL to such a zip (downloaded + unpacked into
    a cache; override the location with ``SAM2_ONNX_CACHE``).

    Same checkpoint-free contract as
    :func:`build_sam2_image_predictor_onnx`: the skeleton is meta-built (no
    weights allocated), the 5 blocks are swapped and the glue materialized from
    ``weights.npz``, ``use_half`` only touches the torch glue, and ``trt_opts`` tunes
    the TensorRT engine build (cache, workspace, ...). Precision is owned by the exported
    graph + ``trt_opts``. No ``vos_optimized`` knob — ONNX/TRT is already the optimized
    path."""
    from sam.onnx.sam2_generic_onnx import attach_onnx_blocks

    model = _meta_build_generic(
        config_file,
        "sam.models.sam2_predictor.Sam2VideoPredictor",
        use_half=use_half,
        apply_postprocessing=apply_postprocessing,
        hydra_overrides_extra=hydra_overrides_extra,
    )
    return attach_onnx_blocks(
        model, onnx_dir, device=torch.device(device), use_trt=use_trt, trt_opts=trt_opts
    )


def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    vos_optimized=False,
    use_half=False,
    compile_image_encoder=False,
    **kwargs,
) -> Sam2VideoPredictor:
    hydra_overrides = [
        "++model._target_=sam.models.sam2_predictor.Sam2VideoPredictor",
        f"++model.use_half={use_half}",
    ]

    if vos_optimized:
        # VOS always compiles the image encoder; don't append compile_image_encoder
        # again below to avoid a duplicate hydra override.
        hydra_overrides = [
            "++model._target_=sam.models.sam2_predictor.Sam2VideoPredictorVOS",
            "++model.compile_image_encoder=True",  # Let sam2_base handle this
            f"++model.use_half={use_half}",
        ]
    elif compile_image_encoder:
        hydra_overrides.append("++model.compile_image_encoder=True")

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(
        cfg.model,
        _recursive_=True
    )
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def _hf_download(model_id):
    from huggingface_hub import hf_hub_download

    config_name, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
    return config_name, ckpt_path


def build_sam2_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2(config_file=config_name, ckpt_path=ckpt_path, **kwargs)


def build_sam2_video_predictor_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2_video_predictor(
        config_file=config_name, ckpt_path=ckpt_path, **kwargs
    )

def build_sam2_predictor_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2_predictor(
        config_file=config_name, ckpt_path=ckpt_path, **kwargs
    )


# SPDX-License-Identifier: LicenseRef-SAM
# --- SAM 3 (Perception Encoder) -------------------------------------------------------
# Minimal direct-construction builder for the SAM 3 vision encoder (Phase 1, Task 2).
# Full hydra configs for SAM 3 arrive in a later task; this stub builds JUST the PE vision
# trunk + Simple-FPN neck and strict-loads the encoder subtree from a local sam3.pt, which
# is enough to parity-check the encoder against the captured golden. The config mirrors
# sam3/model_builder.py::_create_vit_backbone / _create_vit_neck (the SAM 3 image model).


def build_sam3_vision_encoder(ckpt_path=None, device="cuda", add_sam2_neck=False):
    """Build the SAM 3 Perception-Encoder vision encoder and (optionally) load weights.

    Args:
        ckpt_path: path to a local ``sam3.pt``. The encoder subtree
            (``detector.backbone.vision_backbone.{trunk,convs}.*``) is loaded with
            ``strict=True``. With ``add_sam2_neck=False`` (default, the detector path) the SAM 2
            dual-neck (``sam2_convs.*``) is filtered out; with ``add_sam2_neck=True`` (the
            tracker path) it is built and loaded too, so ``forward(..., return_sam2=True)``
            yields the ``sam2_backbone_out`` pyramid the tracker consumes. If ``None`` the model
            is returned with init weights.
        device: device to move the model to.
        add_sam2_neck: build + load the SAM 2 ("propagation") neck used by the tracker.

    Returns:
        A ``Sam3VisionEncoder`` in eval mode. ``forward(image)`` takes the preprocessed
        (B, 3, 1008, 1008) tensor and returns ``(features, pos)`` pyramids.
    """
    from sam.modeling.encoders.necks import Sam3DualViTDetNeck
    from sam.modeling.encoders.pe_vitdet import ViT
    from sam.modeling.encoders.perception_encoder import Sam3VisionEncoder
    from sam.modeling.position_encoding import PositionEmbeddingSine

    trunk = ViT(
        img_size=1008,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
    )
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        warmup_cache=False,
    )
    neck = Sam3DualViTDetNeck(
        trunk=trunk,
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        add_sam2_neck=add_sam2_neck,
    )
    encoder = Sam3VisionEncoder(vision_backbone=neck, scalp=1)

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]
        sub_prefix = "detector.backbone.vision_backbone."
        skip_prefix = "detector.backbone.vision_backbone.sam2_convs."
        sub = {
            k[len("detector.backbone."):]: v
            for k, v in ckpt.items()
            if k.startswith(sub_prefix)
            and (add_sam2_neck or not k.startswith(skip_prefix))
        }
        encoder.load_state_dict(sub, strict=True)

    encoder = encoder.to(device)
    encoder.eval()
    return encoder


# SPDX-License-Identifier: LicenseRef-SAM
# --- SAM 3 text encoder -------------------------------------------------------
# Minimal direct-construction builder for the SAM 3 text tower (Phase 1, Task 3).
# Mirrors build_sam3_vision_encoder: construct Sam3TextEncoder with the PE-text
# dims, strict-load the language_backbone subtree (295 keys) from a local sam3.pt.
# Config mirrors sam3/model_builder.py::_create_language_backbone (VETextEncoder
# init with d_model=256, width=1024, heads=16, layers=24, context_length=32).


def build_sam3_text_encoder(ckpt_path=None, device="cuda"):
    """Build the SAM 3 text encoder (PE text tower + resizer) and optionally load weights.

    Args:
        ckpt_path: path to a local ``sam3.pt``. The language-backbone subtree
            (``detector.backbone.language_backbone.*``) is loaded with
            ``strict=True`` (295 keys, 0 missing, 0 unexpected). If ``None``
            the model is returned with init weights.
        device: device to move the model to.

    Returns:
        A ``Sam3TextEncoder`` in eval mode. ``encode(phrases)`` takes a list of
        strings and returns ``(seq=32, N, d_model=256)`` language_features.
    """
    from sam.modeling.text.text_encoder import Sam3TextEncoder
    from sam.modeling.text.tokenizer import Sam3Tokenizer

    tokenizer = Sam3Tokenizer()
    text_encoder = Sam3TextEncoder(
        d_model=256,
        tokenizer=tokenizer,
        width=1024,
        heads=16,
        layers=24,
        context_length=32,
        vocab_size=49408,
        use_ln_post=True,
    )

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]
        prefix = "detector.backbone.language_backbone."
        sub = {
            k[len(prefix):]: v
            for k, v in ckpt.items()
            if k.startswith(prefix)
        }
        text_encoder.load_state_dict(sub, strict=True)

    text_encoder = text_encoder.to(device)
    text_encoder.eval()
    return text_encoder


# SPDX-License-Identifier: LicenseRef-SAM
# --- SAM 3 DETR detector ------------------------------------------------------
# Minimal direct-construction builder for the base (per-object) SAM 3 detector
# (Phase 1, Task 4). Mirrors build_sam3_vision_encoder / build_sam3_text_encoder:
# construct the VL fusion encoder + 200-query set decoder (presence token, log-boxRPB,
# box-refine) + dot-product scorer + per-object mask head + the (text-only) geometry
# cls-token encoder, then strict-load the detector subtree from a local sam3.pt. The
# config mirrors sam3/model_builder.py (_create_transformer_encoder / _decoder /
# _dot_product_scoring / _segmentation_head / _geometry_encoder, all d_model=256).


def build_sam3_detector(ckpt_path=None, device="cuda"):
    """Build the base SAM 3 DETR detector and optionally load weights.

    Args:
        ckpt_path: path to a local ``sam3.pt``. The detector subtree (``detector.*``
            minus the shared ``detector.backbone.*`` vision/language backbones, which are
            built by ``build_sam3_vision_encoder`` / ``build_sam3_text_encoder``) is loaded
            with ``strict=True``. If ``None`` the model is returned with init weights.
        device: device to move the model to.

    Returns:
        A ``Sam3DetrDetector`` in eval mode. ``forward_grounding(feats, pos, text_emb,
        text_mask)`` returns the raw DETR set; ``detect(...)`` returns a
        ``Sam3DetectionResult`` (presence-weighted boxes/scores/masks).
    """
    from sam.modeling.decoders.detr_decoder import (
        DotProductScoring,
        MLP,
        MultiheadAttention,
        Sam3DetrDetector,
        Sam3GeometryEncoder,
        TransformerDecoder,
        TransformerDecoderLayer,
        TransformerEncoderFusion,
        TransformerEncoderLayer,
        TransformerWrapper,
    )
    from sam.modeling.decoders.maskformer_segmentation import (
        PixelDecoder,
        UniversalSegmentationHead,
    )

    d_model = 256

    # VL early-fusion encoder (6 layers; pre-norm; image self-attn + text cross-attn).
    encoder_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=d_model,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=True,
        pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8, dropout=0.1, embed_dim=d_model, batch_first=True
        ),
        cross_attention=MultiheadAttention(
            num_heads=8, dropout=0.1, embed_dim=d_model, batch_first=True
        ),
    )
    encoder = TransformerEncoderFusion(
        layer=encoder_layer,
        num_layers=6,
        d_model=d_model,
        num_feature_levels=1,
        add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )

    # 200-query set decoder (box-refine, log-boxRPB, presence token).
    decoder_layer = TransformerDecoderLayer(
        activation="relu",
        d_model=d_model,
        dim_feedforward=2048,
        dropout=0.1,
        cross_attention=MultiheadAttention(
            num_heads=8, dropout=0.1, embed_dim=d_model
        ),
        n_heads=8,
        use_text_cross_attention=True,
    )
    decoder = TransformerDecoder(
        layer=decoder_layer,
        num_layers=6,
        num_queries=200,
        return_intermediate=True,
        box_refine=True,
        num_o2m_queries=0,
        dac=True,
        boxRPB="log",
        d_model=d_model,
        dac_use_selfatt_ln=True,
        resolution=1008,
        stride=14,
        presence_token=True,
    )
    transformer = TransformerWrapper(encoder=encoder, decoder=decoder, d_model=d_model)

    # Dot-product scorer (per-query class logits + the seg-head presence; here unused).
    prompt_mlp = MLP(
        input_dim=d_model,
        hidden_dim=2048,
        output_dim=d_model,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=torch.nn.LayerNorm(d_model),
    )
    dot_prod_scoring = DotProductScoring(
        d_model=d_model, d_proj=d_model, prompt_mlp=prompt_mlp
    )

    # Per-object mask head (PixelDecoder over the FPN pyramid + per-query MaskPredictor).
    pixel_decoder = PixelDecoder(
        num_upsampling_stages=3, interpolation_mode="nearest", hidden_dim=d_model
    )
    cross_attend_prompt = MultiheadAttention(num_heads=8, dropout=0, embed_dim=d_model)
    segmentation_head = UniversalSegmentationHead(
        hidden_dim=d_model,
        upsampling_stages=3,
        aux_masks=False,
        presence_head=False,
        dot_product_scorer=None,
        cross_attend_prompt=cross_attend_prompt,
        pixel_decoder=pixel_decoder,
    )

    # Geometry cls-token encoder (text-only path active; box/point/mask encoders dormant).
    geo_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=d_model,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8, dropout=0.1, embed_dim=d_model, batch_first=False
        ),
        cross_attention=MultiheadAttention(
            num_heads=8, dropout=0.1, embed_dim=d_model, batch_first=False
        ),
    )
    geometry_encoder = Sam3GeometryEncoder(d_model=d_model, layer=geo_layer, num_layers=3)

    detector = Sam3DetrDetector(
        transformer=transformer,
        geometry_encoder=geometry_encoder,
        segmentation_head=segmentation_head,
        dot_prod_scoring=dot_prod_scoring,
        num_feature_levels=1,
        o2m_mask_predict=True,
    )

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]
        prefix = "detector."
        skip_prefix = "detector.backbone."  # vision + language backbones built elsewhere
        sub = {
            k[len(prefix):]: v
            for k, v in ckpt.items()
            if k.startswith(prefix) and not k.startswith(skip_prefix)
        }
        detector.load_state_dict(sub, strict=True)

    detector = detector.to(device)
    detector.eval()
    return detector


# SPDX-License-Identifier: LicenseRef-SAM
# --- SAM 3 per-object tracker -------------------------------------------------
# Minimal direct-construction builder for the base (per-object) SAM 3 tracker (Phase 1, Task 5).
# Mirrors build_sam3_detector: construct the memory-attention transformer (4 RoPE self+cross
# layers) + memory encoder (SimpleMaskEncoder) + SAM prompt encoder / mask decoder + the
# temporal / no-object embeddings, then strict-load the ``tracker.*`` subtree (309 keys) from a
# local sam3.pt. Config mirrors sam3/model_builder.py (_create_tracker_transformer /
# _create_tracker_maskmem_backbone / build_tracker). NO multiplex (that is SAM 3.1).


def build_sam3_tracker(ckpt_path=None, device="cuda"):
    """Build the base SAM 3 per-object tracker and optionally load weights.

    Args:
        ckpt_path: path to a local ``sam3.pt``. The tracker subtree (``tracker.*``, 309 keys)
            is loaded with ``strict=True`` (the ``tracker.`` prefix is stripped). If ``None``
            the model is returned with init weights.
        device: device to move the model to.

    Returns:
        A ``Sam3Tracker`` in eval mode. It is weights-only (no vision backbone): callers pass
        the SAM 2-neck feature pyramid (from ``build_sam3_vision_encoder(add_sam2_neck=True)``,
        ``forward(..., return_sam2=True)``) into ``track_step`` / the data-space block methods.
    """
    from sam.modeling.decoders.sam3_transformer import (
        RoPEAttention,
        TransformerDecoderLayerv2,
        TransformerEncoderCrossAttention,
        TransformerWrapper,
    )
    from sam.modeling.memory.sam3_memory_encoder import (
        CXBlock,
        SimpleFuser,
        SimpleMaskDownSampler,
        SimpleMaskEncoder,
    )
    from sam.modeling.position_encoding import PositionEmbeddingSine
    from sam.modeling.tracking.sam3_tracker import Sam3Tracker

    d_model = 256

    # Memory encoder (mask + pixel-feature -> 64-ch spatial memory at 72x72).
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=64, normalize=True, scale=None, temperature=10000,
        warmup_cache=False,
    )
    mask_downsampler = SimpleMaskDownSampler(
        kernel_size=3, stride=2, padding=1, interpol_size=[1152, 1152]
    )
    cx_block = CXBlock(
        dim=256, kernel_size=7, padding=3, layer_scale_init_value=1.0e-06, use_dwconv=True
    )
    fuser = SimpleFuser(layer=cx_block, num_layers=2)
    maskmem_backbone = SimpleMaskEncoder(
        out_dim=64, position_encoding=position_encoding,
        mask_downsampler=mask_downsampler, fuser=fuser,
    )

    # Memory attention (4 layers; RoPE self-attn over frame feats + RoPE cross-attn to memory).
    self_attention = RoPEAttention(
        embedding_dim=d_model, num_heads=1, downsample_rate=1, dropout=0.1,
        rope_theta=10000.0, feat_sizes=[72, 72], use_rope_real=False,
    )
    cross_attention = RoPEAttention(
        embedding_dim=d_model, num_heads=1, downsample_rate=1, dropout=0.1, kv_in_dim=64,
        rope_theta=10000.0, feat_sizes=[72, 72], rope_k_repeat=True, use_rope_real=False,
    )
    encoder_layer = TransformerDecoderLayerv2(
        cross_attention_first=False, activation="relu", dim_feedforward=2048, dropout=0.1,
        pos_enc_at_attn=False, pre_norm=True, self_attention=self_attention, d_model=d_model,
        pos_enc_at_cross_attn_keys=True, pos_enc_at_cross_attn_queries=False,
        cross_attention=cross_attention,
    )
    encoder = TransformerEncoderCrossAttention(
        remove_cross_attention_layers=[], batch_first=True, d_model=d_model, frozen=False,
        pos_enc_at_input=True, layer=encoder_layer, num_layers=4, use_act_checkpoint=False,
    )
    transformer = TransformerWrapper(encoder=encoder, decoder=None, d_model=d_model)

    tracker = Sam3Tracker(
        transformer=transformer,
        maskmem_backbone=maskmem_backbone,
        image_size=1008,
        num_maskmem=7,
        backbone_stride=14,
        multimask_output_in_sam=True,
        multimask_output_for_tracking=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        non_overlap_masks_for_mem_enc=False,
        max_cond_frames_in_attn=4,
        sam_mask_decoder_extra_args={
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
        use_memory_selection=True,
    )

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]
        prefix = "tracker."
        sub = {
            k[len(prefix):]: v
            for k, v in ckpt.items()
            if k.startswith(prefix)
        }
        tracker.load_state_dict(sub, strict=True)

    tracker = tracker.to(device)
    tracker.eval()
    return tracker


# SPDX-License-Identifier: Apache-2.0
def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")
