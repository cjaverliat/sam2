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


def _build_sam3_tracker_module():
    """Construct the base SAM 3 per-object ``Sam3Tracker`` (weights-only, no load).

    Shared by :func:`build_sam3_tracker` (image/isolated tracker) and
    :func:`build_sam3_video_predictor` (streaming) so the proven 309-key module tree is built
    in exactly one place. Returns the tracker on CPU with init weights.
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
    return tracker


def _load_sam3_tracker_subtree(tracker, ckpt_path):
    """Strict-load the ``tracker.*`` subtree (309 keys, prefix stripped) into a ``Sam3Tracker``."""
    if ckpt_path is None:
        return
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    prefix = "tracker."
    sub = {k[len(prefix):]: v for k, v in ckpt.items() if k.startswith(prefix)}
    tracker.load_state_dict(sub, strict=True)


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
    tracker = _build_sam3_tracker_module()
    _load_sam3_tracker_subtree(tracker, ckpt_path)
    tracker = tracker.to(device)
    tracker.eval()
    return tracker


# SPDX-License-Identifier: LicenseRef-SAM
# --- SAM 3 image concept predictor --------------------------------------------
# Hydra-compose builder for the full Sam3Predictor (image, Phase 1, Task 8): the shared PE
# vision encoder + text tower + DETR detector, composed from configs/sam3/sam3.yaml and
# strict-loading the detector.* subtree of a local sam3.pt. Mirrors build_sam2_predictor /
# build_sam2_hf (compose -> instantiate -> load). The hydra config is a hand-translation of
# sam3/model_builder.py::build_sam3_image_model cross-checked vs the build_sam3_* stubs.


HF_SAM3_MODEL_ID_TO_CONFIG = {
    "facebook/sam3": ("configs/sam3/sam3.yaml", "sam3.pt"),
}


def _load_sam3_image_checkpoint(model, ckpt_path):
    """Strict-load the ``detector.*`` subtree (1156 keys) of ``sam3.pt`` into a Sam3Predictor.

    The predictor separates the upstream ``Sam3Image.backbone`` into the OWNED
    ``vision_encoder`` + ``text_encoder`` (spec §5), so the checkpoint keys are remapped in
    three groups (the upstream image loader only strips the flat ``detector.`` prefix because
    its ``Sam3Image`` keeps the combined ``backbone`` submodule):

      ``detector.backbone.vision_backbone.*``   -> ``vision_encoder.vision_backbone.*``  (464, incl. the 22 sam2_convs)
      ``detector.backbone.language_backbone.*`` -> ``text_encoder.*``                    (295)
      ``detector.*`` (minus ``detector.backbone.*``) -> ``detector.*``                    (397, the head)

    ``tracker.*`` (309) is ignored — the image predictor has no tracker. The vision encoder
    MUST be built with ``add_sam2_neck=True`` (set in configs/sam3/sam3.yaml) so the
    ``sam2_convs`` load and the 1156-key load is STRICT (0 missing / 0 unexpected).
    """
    if ckpt_path is None:
        return
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    vb_prefix = "detector.backbone.vision_backbone."
    lb_prefix = "detector.backbone.language_backbone."
    sub = {}
    for k, v in ckpt.items():
        if k.startswith(vb_prefix):
            sub["vision_encoder.vision_backbone." + k[len(vb_prefix):]] = v
        elif k.startswith(lb_prefix):
            sub["text_encoder." + k[len(lb_prefix):]] = v
        elif k.startswith("detector.") and not k.startswith("detector.backbone."):
            sub[k] = v  # detector head keys (transformer/geometry/seg/scoring) map 1:1
    model.load_state_dict(sub, strict=True)


def build_sam3(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    **kwargs,
):
    """Build a SAM 3 image concept predictor (``Sam3Predictor``) and load weights.

    Mirrors :func:`build_sam2_predictor`: hydra-compose ``config_file`` (e.g.
    ``"configs/sam3/sam3.yaml"``) -> instantiate the owned encoder / text tower / detector
    via their ``_target_``s -> strict-load the ``detector.*`` subtree of ``ckpt_path`` (a
    local ``sam3.pt``). Returns the predictor on ``device``, in eval mode when
    ``mode == "eval"``.
    """
    hydra_overrides = list(hydra_overrides_extra)
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_sam3_image_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam3_hf(model_id, **kwargs):
    """Build a SAM 3 image predictor from a HuggingFace model id (downloads the checkpoint).

    The hydra config is OURS (``configs/sam3/sam3.yaml`` — a hand-translation of the upstream
    architecture); only the gated weights (``sam3.pt``) are pulled from HF. Mirrors
    :func:`build_sam2_hf`.
    """
    from huggingface_hub import hf_hub_download

    config_file, ckpt_name = HF_SAM3_MODEL_ID_TO_CONFIG.get(
        model_id, ("configs/sam3/sam3.yaml", "sam3.pt")
    )
    ckpt_path = hf_hub_download(repo_id=model_id, filename=ckpt_name)
    return build_sam3(config_file=config_file, ckpt_path=ckpt_path, **kwargs)


# SPDX-License-Identifier: LicenseRef-SAM
# --- EfficientSAM3 image concept predictor (Phase A, Task A6) -------------------
# Hydra-compose builder for the EfficientSAM3 Sam3Predictor (image): the SAME base SAM 3
# lineage as build_sam3 (shared vision encoder + text tower + DETR detector composed from a
# hydra config -> strict load), but with the lightweight EfficientSam3Trunk (RepViT-M1.1) vision
# trunk and MobileClipTextEncoder (MobileCLIP-S0, ctx16) text tower, a SINGLE detection neck
# (no SAM 2 neck) and NO geometry encoder. The EfficientSAM3 checkpoint is detector-ROOT (keys
# begin ``backbone.`` / ``transformer.`` / ``dot_prod_scoring.`` / ``segmentation_head.`` --
# one level shallower than base ``sam3.pt``'s ``detector.`` root), so the remap is the SAME
# 3-group split as base but on the un-prefixed keys (see _load_efficientsam3_image_checkpoint).


HF_EFFICIENTSAM3_MODEL_ID_TO_FILES = {
    "repvit": (
        "configs/efficientsam3/efficientsam3_repvit.yaml",
        "efficientsam3_ft/efficientsam3_repvit.pt",
    ),
    "tinyvit": (
        "configs/efficientsam3/efficientsam3_tinyvit.yaml",
        "efficientsam3_ft/efficientsam3_tinyvit.pt",
    ),
    "efficientvit": (
        "configs/efficientsam3/efficientsam3_efficientvit.yaml",
        "efficientsam3_ft/efficientsam3_efficientvit.pt",
    ),
}


def _load_efficientsam3_image_checkpoint(model, ckpt_path):
    """Strict-load the EfficientSAM3 image checkpoint (1107 keys) into a ``Sam3Predictor``.

    The checkpoint is ``{'model': state_dict, ...}`` and is **detector-root**: its keys begin
    ``backbone.`` / ``transformer.`` / ``dot_prod_scoring.`` / ``segmentation_head.`` (NO
    ``detector.`` prefix -- one level shallower than base ``sam3.pt``). ``Sam3Predictor`` OWNS
    the vision encoder + text tower separately from the detector (spec §5), so -- exactly like
    base :func:`_load_sam3_image_checkpoint`, but on the un-prefixed keys -- the load is remapped
    in three groups:

      ``backbone.vision_backbone.*``   -> ``vision_encoder.vision_backbone.*``  (675: RepViT trunk + single neck convs)
      ``backbone.language_backbone.*`` -> ``text_encoder.*``                    (111: MobileCLIP encoder + projector)
      everything else                  -> ``detector.*``                        (321: transformer / dot_prod_scoring / segmentation_head)

    = **1107 keys, strict (0 missing / 0 unexpected)**. There is NO geometry encoder, NO SAM 2 /
    interactive neck and NO tracker in this checkpoint, so the config builds none of them. The
    load is loud: ``strict=False`` followed by assertions on the missing / unexpected sets, so a
    drift surfaces as a clear "N missing / N unexpected" message rather than a silent partial load.
    """
    if ckpt_path is None:
        return
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    vb_prefix = "backbone.vision_backbone."
    lb_prefix = "backbone.language_backbone."
    remapped = {}
    for k, v in ckpt.items():
        if k.startswith(vb_prefix):
            remapped["vision_encoder.vision_backbone." + k[len(vb_prefix):]] = v
        elif k.startswith(lb_prefix):
            remapped["text_encoder." + k[len(lb_prefix):]] = v
        else:
            remapped["detector." + k] = v  # transformer / dot_prod_scoring / segmentation_head
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    assert not missing, (
        f"EfficientSAM3 strict load: {len(missing)} missing key(s) "
        f"(model params not in the {len(remapped)}-key checkpoint), e.g. {missing[:5]}"
    )
    assert not unexpected, (
        f"EfficientSAM3 strict load: {len(unexpected)} unexpected key(s) "
        f"(checkpoint params not in the model), e.g. {unexpected[:5]}"
    )


def build_efficientsam3(
    config_file="configs/efficientsam3/efficientsam3_repvit.yaml",
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    **kwargs,
):
    """Build an EfficientSAM3 image concept predictor (``Sam3Predictor``) and load weights.

    Mirrors :func:`build_sam3`: hydra-compose ``config_file`` (default
    ``"configs/efficientsam3/efficientsam3_repvit.yaml"``) -> instantiate the owned
    EfficientSam3Trunk vision encoder / MobileCLIP text tower / DETR detector via their
    ``_target_``s -> strict-load the 1107-key EfficientSAM3 checkpoint (detector-root 3-group
    remap). Returns the predictor on ``device``, in eval mode when ``mode == "eval"``.
    """
    cfg = compose(config_name=config_file, overrides=list(hydra_overrides_extra))
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_efficientsam3_image_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_efficientsam3_hf(model_id="repvit", **kwargs):
    """Build an EfficientSAM3 image predictor from a HuggingFace model id (downloads the ckpt).

    The hydra config is OURS; only the weights are pulled from HF
    (``Simon7108528/EfficientSAM3``). ``model_id`` selects the variant (default ``"repvit"`` ->
    ``efficientsam3_ft/efficientsam3_repvit.pt``). Mirrors :func:`build_sam3_hf`.
    """
    from huggingface_hub import hf_hub_download

    config_file, ckpt_name = HF_EFFICIENTSAM3_MODEL_ID_TO_FILES[model_id]
    ckpt_path = hf_hub_download(repo_id="Simon7108528/EfficientSAM3", filename=ckpt_name)
    return build_efficientsam3(config_file=config_file, ckpt_path=ckpt_path, **kwargs)


# SPDX-License-Identifier: LicenseRef-SAM
# --- SAM3-LiteText base-lineage VIDEO predictor (D1) --------------------------
# SAM3-LiteText is the base SAM 3 lineage VIDEO model: PE-ViT vision encoder
# (unchanged, add_sam2_neck=True), TRAINED geometry encoder (kept), base Sam3Tracker,
# and a MobileCLIP text encoder instead of SAM 3's PE text tower.  The EXISTING
# build_sam3_video_predictor + _load_sam3_video_checkpoint already handle this path
# because _load_sam3_video_checkpoint is text-encoder-agnostic: it remaps the 111
# language_backbone keys onto whichever text_encoder the config instantiates.
# 1281 keys = 464 vision + 111 MobileCLIP language + 397 detector head (incl. geo 76)
# + 309 tracker.  Checkpoint: Simon7108528/EfficientSAM3 (public, no token).


HF_EFFICIENTSAM3_LITETEXT_MODEL_ID_TO_FILES = {
    "litetext-s0-ctx16": (
        "configs/efficientsam3/sam3_litetext_s0_ctx16.yaml",
        "sam3_litetext/sam3_litetext_mobileclip_s0_ctx16.pt",
    ),
}


def build_efficientsam3_litetext_video_predictor_hf(model_id="litetext-s0-ctx16", **kwargs):
    """Build a SAM3-LiteText base-video predictor from a HuggingFace model id.

    Downloads the checkpoint from the PUBLIC repo ``Simon7108528/EfficientSAM3`` (no
    token required) and delegates to the existing :func:`build_sam3_video_predictor`
    with the matching hydra config.  The ``_load_sam3_video_checkpoint`` loader
    performs the same 3-group remap as the base ``sam3.pt`` path, remapping the 111
    MobileCLIP ``language_backbone`` keys onto the ``MobileClipTextEncoder`` submodule
    (strict=True, 0 missing / 0 unexpected).

    Args:
        model_id: variant key (default ``"litetext-s0-ctx16"`` ->
            ``sam3_litetext/sam3_litetext_mobileclip_s0_ctx16.pt``).
        **kwargs: forwarded to :func:`build_sam3_video_predictor`
            (``device``, ``mode``, ``hydra_overrides_extra``, ...).

    Returns:
        A ``Sam3VideoPredictor`` in eval mode with MobileCLIP-S0 text encoder,
        base Sam3Tracker, and trained geometry encoder.
    """
    from huggingface_hub import hf_hub_download

    config_file, ckpt_name = HF_EFFICIENTSAM3_LITETEXT_MODEL_ID_TO_FILES[model_id]
    ckpt_path = hf_hub_download(repo_id="Simon7108528/EfficientSAM3", filename=ckpt_name)
    return build_sam3_video_predictor(config_file=config_file, ckpt_path=ckpt_path, **kwargs)


# --- SAM 3 streaming video concept predictor (Phase 1, Task 9) -----------------
# Mirrors build_sam2_video_predictor: hydra-compose the shared encoder/text/detector from
# configs/sam3/sam3.yaml, build the proven 309-key tracker module, wrap them in
# Sam3VideoPredictor, and strict-load the FULL sam3.pt = detector.* (1156, the Task 8 3-group
# remap) + tracker.* (309) = 1465 keys.


def _load_sam3_video_checkpoint(model, ckpt_path):
    """Strict-load the FULL ``sam3.pt`` (1465 keys) into a ``Sam3VideoPredictor``.

    Combines the Task-8 image 3-group remap (``detector.backbone.vision_backbone.*`` ->
    ``vision_encoder.vision_backbone.*`` 464; ``detector.backbone.language_backbone.*`` ->
    ``text_encoder.*`` 295; ``detector.*`` head 397) with the tracker subtree
    (``tracker.*`` -> ``tracker.*`` 309) = **1465 keys, strict (0 missing / 0 unexpected)**.
    """
    if ckpt_path is None:
        return
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    vb_prefix = "detector.backbone.vision_backbone."
    lb_prefix = "detector.backbone.language_backbone."
    sub = {}
    for k, v in ckpt.items():
        if k.startswith(vb_prefix):
            sub["vision_encoder.vision_backbone." + k[len(vb_prefix):]] = v
        elif k.startswith(lb_prefix):
            sub["text_encoder." + k[len(lb_prefix):]] = v
        elif k.startswith("detector.") and not k.startswith("detector.backbone."):
            sub[k] = v  # detector head keys map 1:1
        elif k.startswith("tracker."):
            sub[k] = v  # tracker subtree maps 1:1 (model.tracker.*)
    model.load_state_dict(sub, strict=True)


def build_sam3_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    **kwargs,
):
    """Build a SAM 3 streaming video concept predictor (``Sam3VideoPredictor``) and load weights.

    Composes the shared PE vision encoder + text tower + DETR detector from ``config_file``
    (e.g. ``"configs/sam3/sam3.yaml"`` — the encoder built ``add_sam2_neck=True`` so the
    tracker's SAM 2 pyramid view is available), builds the per-object ``Sam3Tracker``, wraps
    them in ``Sam3VideoPredictor``, and strict-loads the full ``sam3.pt`` (1465 keys). The
    forgetful bank owns temporal memory selection, so the tracker runs with
    ``use_memory_selection=False``.
    """
    from sam.models.sam3_predictor import Sam3VideoPredictor

    cfg = compose(config_name=config_file, overrides=list(hydra_overrides_extra))
    OmegaConf.resolve(cfg)
    vision_encoder = instantiate(cfg.model.vision_encoder, _recursive_=True)
    text_encoder = instantiate(cfg.model.text_encoder, _recursive_=True)
    detector = instantiate(cfg.model.detector, _recursive_=True)
    tracker = _build_sam3_tracker_module()
    # The bank performs temporal memory selection; the tracker conditions on exactly the frames
    # the bank returns (so it must NOT additionally filter via its SAM2Long heuristic).
    tracker.use_memory_selection = False

    model = Sam3VideoPredictor(
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        detector=detector,
        tracker=tracker,
    )
    _load_sam3_video_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam3_video_predictor_hf(model_id, **kwargs):
    """Build a SAM 3 video predictor from a HuggingFace model id (downloads the checkpoint).

    The hydra config is OURS; only the gated ``sam3.pt`` is pulled from HF. Mirrors
    :func:`build_sam2_video_predictor_hf`.
    """
    from huggingface_hub import hf_hub_download

    config_file, ckpt_name = HF_SAM3_MODEL_ID_TO_CONFIG.get(
        model_id, ("configs/sam3/sam3.yaml", "sam3.pt")
    )
    ckpt_path = hf_hub_download(repo_id=model_id, filename=ckpt_name)
    return build_sam3_video_predictor(config_file=config_file, ckpt_path=ckpt_path, **kwargs)


# SPDX-License-Identifier: LicenseRef-SAM
# --- SAM 3.1 multiplex tracker (M1) -------------------------------------------
# Direct-construction builders for the SAM 3.1 MULTIPLEX tracker + its vision encoder. The
# sam3.1 tracker is the upstream ``VideoTrackingMultiplex`` (decoupled memory attention +
# ``MultiplexMaskDecoder`` + separate interactive heads); its ``tracker.*`` subtree (457 keys,
# all under ``tracker.model.*``) is NOT bit-shared with the base per-object tracker (309 keys) --
# only ``obj_ptr_proj`` is identical. The sam3.1 vision encoder is ALSO not bit-shared with base
# (different PE-trunk weights + a 3-level tri-neck), so the tracker's propagation features are
# loaded from the sam3.1 checkpoint's ``propagation_convs`` (mapped onto the existing dual-neck's
# ``sam2_convs`` slot). Config mirrors sam3/model_builder.py (_create_multiplex_transformer /
# _create_multiplex_maskmem_backbone / build_sam3_multiplex_video_model).


def build_sam3_multiplex_vision_encoder(ckpt_path=None, device="cuda"):
    """Build the SAM 3.1 vision encoder (PE trunk + propagation neck) and optionally load weights.

    Unlike base ``sam3.pt``, the sam3.1 PE trunk is fine-tuned and the neck is a 3-level tri-neck
    (``convs`` / ``interactive_convs`` / ``propagation_convs``). This builds the shared ViT trunk
    + a dual neck with ``scale_factors=[4,2,1]`` (``scalp=0`` so the pyramid is the 3-level
    [288,144,72]) and strict-loads from ``sam3.1_multiplex.pt``: trunk -> ``trunk``,
    ``interactive_convs`` -> ``convs`` (the SAM-3 head slot), ``propagation_convs`` ->
    ``sam2_convs``. The detector ``convs`` slot is repurposed for ``interactive_convs`` so a
    SINGLE trunk pass yields BOTH necks: ``vision_backbone(x)`` returns the interactive pyramid
    (``sam3_*``, for the cond-frame object-pointer head) and the propagation pyramid (``sam2_*``,
    via ``forward(..., return_sam2=True)``, the ``sam2_backbone_out`` the tracker propagates).
    The detector neck itself is unused by the tracker.
    """
    from sam.modeling.encoders.necks import Sam3DualViTDetNeck
    from sam.modeling.encoders.pe_vitdet import ViT
    from sam.modeling.encoders.perception_encoder import Sam3VisionEncoder
    from sam.modeling.position_encoding import PositionEmbeddingSine

    trunk = ViT(
        img_size=1008, pretrain_img_size=336, patch_size=14, embed_dim=1024, depth=32,
        num_heads=16, mlp_ratio=4.625, norm_layer="LayerNorm", drop_path_rate=0.1,
        qkv_bias=True, use_abs_pos=True, tile_abs_pos=True, global_att_blocks=(7, 15, 23, 31),
        use_rope=True, use_interp_rope=True, window_size=24, pretrain_use_cls_token=True,
        retain_cls_token=False, ln_pre=True, ln_post=False, return_interm_layers=False,
        bias_patch_embed=False,
    )
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=256, normalize=True, scale=None, temperature=10000, warmup_cache=False,
    )
    neck = Sam3DualViTDetNeck(
        trunk=trunk, position_encoding=position_encoding, d_model=256,
        scale_factors=[4.0, 2.0, 1.0], add_sam2_neck=True,
    )
    encoder = Sam3VisionEncoder(vision_backbone=neck, scalp=0)

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]
        vb = "detector.backbone.vision_backbone."
        sub = {}
        for k, v in ckpt.items():
            if not k.startswith(vb):
                continue
            rel = k[len(vb):]
            if rel.startswith("convs."):
                continue  # detector neck unused by the tracker
            if rel.startswith("interactive_convs."):
                rel = "convs." + rel[len("interactive_convs."):]  # SAM-3 slot = interactive
            elif rel.startswith("propagation_convs."):
                rel = "sam2_convs." + rel[len("propagation_convs."):]
            sub["vision_backbone." + rel] = v
        encoder.load_state_dict(sub, strict=True)

    encoder = encoder.to(device)
    encoder.eval()
    return encoder


def _build_sam3_multiplex_tracker_module(multiplex_count=16):
    """Construct the SAM 3.1 ``Sam3MultiplexTracker`` (weights-only, no load, on CPU)."""
    from sam.modeling.decoders.multiplex_memory_attention import (
        DecoupledTransformerDecoderLayerv2,
        SimpleRoPEAttention,
        TransformerEncoderDecoupledCrossAttention,
    )
    from sam.modeling.decoders.sam3_transformer import TransformerWrapper
    from sam.modeling.memory.sam3_memory_encoder import (
        CXBlock,
        SimpleFuser,
        SimpleMaskDownSampler,
        SimpleMaskEncoder,
    )
    from sam.modeling.multiplex import MultiplexController
    from sam.modeling.position_encoding import PositionEmbeddingSine
    from sam.modeling.tracking.sam3_multiplex_tracker import Sam3MultiplexTracker

    d_model = 256

    # Multiplex memory encoder (per-bucket: K mask channels + K conditioning channels -> 256-ch).
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=256, normalize=True, scale=None, temperature=10000, warmup_cache=False,
    )
    mask_downsampler = SimpleMaskDownSampler(
        kernel_size=3, stride=2, padding=1, interpol_size=[1152, 1152],
        multiplex_count=multiplex_count, starting_out_chan=4, input_channel_multiplier=2,
    )
    cx_block = CXBlock(
        dim=256, kernel_size=7, padding=3, layer_scale_init_value=1.0e-06, use_dwconv=True
    )
    fuser = SimpleFuser(layer=cx_block, num_layers=2)
    maskmem_backbone = SimpleMaskEncoder(
        out_dim=256, position_encoding=position_encoding,
        mask_downsampler=mask_downsampler, fuser=fuser,
    )

    # Decoupled memory attention (4 layers; RoPE self-attn + image/spatial cross-attn).
    self_attention = SimpleRoPEAttention(
        d_model=d_model, num_heads=8, dropout_p=0.1, rope_theta=10000.0,
        feat_sizes=[72, 72], use_rope_real=False,
    )
    cross_attention = SimpleRoPEAttention(
        d_model=d_model, num_heads=8, dropout_p=0.1, rope_theta=10000.0,
        feat_sizes=[72, 72], rope_k_repeat=True, use_rope_real=False,
    )
    encoder_layer = DecoupledTransformerDecoderLayerv2(
        activation="gelu", d_model=d_model, num_heads=8, dropout=0.1, dim_feedforward=2048,
        pos_enc_at_attn=False, pre_norm=True, pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False, self_attention_rope=self_attention,
        cross_attention_rope=cross_attention,
    )
    encoder = TransformerEncoderDecoupledCrossAttention(
        d_model=d_model, frozen=False, pos_enc_at_input=True, use_image_in_output=False,
        layer=encoder_layer, num_layers=4, use_act_checkpoint=False, batch_first=True,
    )
    transformer = TransformerWrapper(encoder=encoder, decoder=None, d_model=d_model)

    multiplex_controller = MultiplexController(
        multiplex_count=multiplex_count, eval_multiplex_count=multiplex_count,
    )

    tracker = Sam3MultiplexTracker(
        transformer=transformer,
        maskmem_backbone=maskmem_backbone,
        multiplex_controller=multiplex_controller,
        image_size=1008, num_maskmem=7, backbone_stride=14,
        multimask_output_in_sam=True, multimask_output_for_tracking=True,
        multimask_min_pt_num=0, multimask_max_pt_num=1, num_multimask_outputs=3,
        use_multimask_token_for_obj_ptr=True, non_overlap_masks_for_mem_enc=False,
        max_cond_frames_in_attn=4, pred_obj_scores=True, pred_obj_scores_mlp=True,
        sam_mask_decoder_extra_args={
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
        use_memory_selection=False,
    )
    return tracker


def _load_sam3_multiplex_tracker_subtree(tracker, ckpt_path):
    """Strict-load the ``tracker.model.*`` subtree (457 keys, prefix stripped) into the tracker."""
    if ckpt_path is None:
        return
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    prefix = "tracker.model."
    sub = {k[len(prefix):]: v for k, v in ckpt.items() if k.startswith(prefix)}
    tracker.load_state_dict(sub, strict=True)


def build_sam3_multiplex_tracker(ckpt_path=None, device="cuda"):
    """Build the SAM 3.1 multiplex tracker and optionally load weights.

    Args:
        ckpt_path: path to a local ``sam3.1_multiplex.pt``. The tracker subtree
            (``tracker.model.*``, 457 keys) is loaded with ``strict=True`` (the
            ``tracker.model.`` prefix is stripped). If ``None`` the model is returned with init
            weights.
        device: device to move the model to.

    Returns:
        A ``Sam3MultiplexTracker`` in eval mode. It is weights-only (no vision backbone): callers
        pass the propagation feature pyramid (from
        ``build_sam3_multiplex_vision_encoder(...).forward(..., return_sam2=True)``) into
        ``track_step`` (mux/demux internal; outputs demuxed per-object).
    """
    tracker = _build_sam3_multiplex_tracker_module()
    _load_sam3_multiplex_tracker_subtree(tracker, ckpt_path)
    tracker = tracker.to(device)
    tracker.eval()
    return tracker


# SPDX-License-Identifier: LicenseRef-SAM
# --- SAM 3.1 multiplex image concept predictor (M2) ---------------------------
# Hydra-compose builder for the full Sam3MultiplexPredictor (image): the SAM 3.1 PE vision
# encoder (DETECTION tri-neck head `convs`) + text tower + DETR detector
# (supervise_joint_box_scores=true), composed from configs/sam3/sam3.1.yaml and strict-loading
# the relevant detector.* subtree of a local sam3.1_multiplex.pt. Mirrors build_sam3 (the base
# image builder): compose -> instantiate -> load. The SAM 3.1 detector HEAD is architecturally
# identical to base (same 397 keys); the +10 detector.* keys are the tri-neck's extra conv set.


def _load_sam3_multiplex_image_checkpoint(model, ckpt_path):
    """Strict-load the relevant ``detector.*`` subtree of ``sam3.1_multiplex.pt`` into a
    ``Sam3MultiplexPredictor`` (1130 of the 1166 ``detector.*`` keys).

    The predictor separates the upstream ``Sam3MultiplexDetector.backbone`` into the OWNED
    ``vision_encoder`` + ``text_encoder``, so the checkpoint keys are remapped in three groups:

      ``detector.backbone.vision_backbone.{trunk,convs}.*`` -> ``vision_encoder.vision_backbone.*``  (420 trunk + 18 detection `convs` = 438)
      ``detector.backbone.language_backbone.*``             -> ``text_encoder.*``                     (295)
      ``detector.*`` (minus ``detector.backbone.*``)        -> ``detector.*``                         (397, the DETR head)

    The tri-neck's ``interactive_convs`` / ``propagation_convs`` (18 + 18 = 36 keys) are
    TRACKER-only (M1) and are SKIPPED -- the image detector consumes only the detection neck.
    ``tracker.*`` (457) is ignored (the image predictor has no tracker). The load is STRICT
    over the predictor's 1130 params (0 missing / 0 unexpected).
    """
    if ckpt_path is None:
        return
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    vb_prefix = "detector.backbone.vision_backbone."
    lb_prefix = "detector.backbone.language_backbone."
    sub = {}
    for k, v in ckpt.items():
        if k.startswith(vb_prefix):
            rel = k[len(vb_prefix):]
            if rel.startswith("interactive_convs.") or rel.startswith("propagation_convs."):
                continue  # tracker-only necks (M1); the image detector uses detection `convs`
            sub["vision_encoder.vision_backbone." + rel] = v
        elif k.startswith(lb_prefix):
            sub["text_encoder." + k[len(lb_prefix):]] = v
        elif k.startswith("detector.") and not k.startswith("detector.backbone."):
            sub[k] = v  # detector head keys (transformer/geometry/seg/scoring) map 1:1
    model.load_state_dict(sub, strict=True)


def build_sam3_multiplex(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    **kwargs,
):
    """Build a SAM 3.1 multiplex image concept predictor (``Sam3MultiplexPredictor``).

    Mirrors :func:`build_sam3`: hydra-compose ``config_file`` (e.g.
    ``"configs/sam3/sam3.1.yaml"``) -> instantiate the owned SAM 3.1 vision encoder /
    text tower / detector via their ``_target_``s -> strict-load the relevant ``detector.*``
    subtree of ``ckpt_path`` (a local ``sam3.1_multiplex.pt``). Returns the predictor on
    ``device``, in eval mode when ``mode == "eval"``.
    """
    cfg = compose(config_name=config_file, overrides=list(hydra_overrides_extra))
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_sam3_multiplex_image_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


# SPDX-License-Identifier: LicenseRef-SAM
# --- SAM 3.1 multiplex streaming video predictor (M3) -------------------------
# Mirrors build_sam3_video_predictor (base) but for the SAM 3.1 multiplex path: compose the
# detector + text tower from configs/sam3/sam3.1.yaml, build a TRI-neck sam3.1 vision encoder
# (detection + interactive + propagation necks from one trunk) + the M1 Sam3MultiplexTracker, wrap
# them in Sam3MultiplexVideoPredictor, and strict-load the FULL sam3.1_multiplex.pt = detector.*
# (1166: vision_backbone 474 + language 295 + DETR head 397) + tracker.model.* (457) = 1623 keys.


def _build_sam3_multiplex_video_vision_encoder_module():
    """Construct the SAM 3.1 TRI-neck vision encoder (weights-only, no load, on CPU).

    Unlike the M1 ``build_sam3_multiplex_vision_encoder`` (a DUAL neck: interactive->convs,
    propagation->sam2_convs -- the tracker-only path), the video predictor also runs the DETECTOR,
    so it needs all THREE necks from one trunk pass: detection ``convs`` (-> detector),
    ``interactive_convs`` (-> the tracker's cond-frame object-pointer head), and
    ``propagation_convs`` (-> ``sam2_convs``, the tracker's per-frame propagation). ``scalp=0`` so
    each pyramid is the 3-level [288,144,72] the sam3.1 model uses.
    """
    from sam.modeling.encoders.necks import Sam3DualViTDetNeck
    from sam.modeling.encoders.pe_vitdet import ViT
    from sam.modeling.encoders.perception_encoder import Sam3VisionEncoder
    from sam.modeling.position_encoding import PositionEmbeddingSine

    trunk = ViT(
        img_size=1008, pretrain_img_size=336, patch_size=14, embed_dim=1024, depth=32,
        num_heads=16, mlp_ratio=4.625, norm_layer="LayerNorm", drop_path_rate=0.1,
        qkv_bias=True, use_abs_pos=True, tile_abs_pos=True, global_att_blocks=(7, 15, 23, 31),
        use_rope=True, use_interp_rope=True, window_size=24, pretrain_use_cls_token=True,
        retain_cls_token=False, ln_pre=True, ln_post=False, return_interm_layers=False,
        bias_patch_embed=False,
    )
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=256, normalize=True, scale=None, temperature=10000, warmup_cache=False,
    )
    neck = Sam3DualViTDetNeck(
        trunk=trunk, position_encoding=position_encoding, d_model=256,
        scale_factors=[4.0, 2.0, 1.0], add_sam2_neck=True, add_interactive_neck=True,
    )
    return Sam3VisionEncoder(vision_backbone=neck, scalp=0)


def _load_sam3_multiplex_video_checkpoint(model, ckpt_path):
    """Strict-load the FULL ``sam3.1_multiplex.pt`` (1623 keys) into a ``Sam3MultiplexVideoPredictor``.

      ``detector.backbone.vision_backbone.trunk.*``             -> ``vision_encoder.vision_backbone.trunk.*``           (420)
      ``detector.backbone.vision_backbone.convs.*`` (detection) -> ``vision_encoder.vision_backbone.convs.*``           (18)
      ``detector.backbone.vision_backbone.interactive_convs.*`` -> ``vision_encoder.vision_backbone.interactive_convs.*`` (18)
      ``detector.backbone.vision_backbone.propagation_convs.*`` -> ``vision_encoder.vision_backbone.sam2_convs.*``       (18)
      ``detector.backbone.language_backbone.*``                 -> ``text_encoder.*``                                    (295)
      ``detector.*`` (minus ``detector.backbone.*``)            -> ``detector.*`` (DETR head)                           (397)
      ``tracker.model.*``                                       -> ``tracker.*`` (multiplex tracker)                    (457)

    474 + 295 + 397 + 457 = **1623 keys, strict (0 missing / 0 unexpected)**.
    """
    if ckpt_path is None:
        return
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    vb_prefix = "detector.backbone.vision_backbone."
    lb_prefix = "detector.backbone.language_backbone."
    sub = {}
    for k, v in ckpt.items():
        if k.startswith(vb_prefix):
            rel = k[len(vb_prefix):]
            if rel.startswith("propagation_convs."):
                rel = "sam2_convs." + rel[len("propagation_convs."):]
            # detection ``convs.*``, ``interactive_convs.*`` and ``trunk.*`` map 1:1
            sub["vision_encoder.vision_backbone." + rel] = v
        elif k.startswith(lb_prefix):
            sub["text_encoder." + k[len(lb_prefix):]] = v
        elif k.startswith("detector.") and not k.startswith("detector.backbone."):
            sub[k] = v  # detector head keys (transformer/geometry/seg/scoring) map 1:1
        elif k.startswith("tracker.model."):
            sub["tracker." + k[len("tracker.model."):]] = v  # multiplex tracker (457)
    model.load_state_dict(sub, strict=True)


def build_sam3_multiplex_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    **kwargs,
):
    """Build a SAM 3.1 multiplex streaming video predictor (``Sam3MultiplexVideoPredictor``).

    Mirrors :func:`build_sam3_video_predictor` (base): compose ``config_file`` (e.g.
    ``"configs/sam3/sam3.1.yaml"``) for the SAM 3.1 text tower + DETR detector
    (``supervise_joint_box_scores=true``), build the TRI-neck sam3.1 vision encoder + the M1
    ``Sam3MultiplexTracker`` directly, wrap them, and strict-load the full ``sam3.1_multiplex.pt``
    (1623 keys). The forgetful bank owns temporal memory selection, so the tracker runs with
    ``use_memory_selection=False`` (already set in ``_build_sam3_multiplex_tracker_module``).
    """
    from sam.models.sam3_predictor import Sam3MultiplexVideoPredictor

    cfg = compose(config_name=config_file, overrides=list(hydra_overrides_extra))
    OmegaConf.resolve(cfg)
    text_encoder = instantiate(cfg.model.text_encoder, _recursive_=True)
    detector = instantiate(cfg.model.detector, _recursive_=True)
    vision_encoder = _build_sam3_multiplex_video_vision_encoder_module()
    tracker = _build_sam3_multiplex_tracker_module()

    model = Sam3MultiplexVideoPredictor(
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        detector=detector,
        tracker=tracker,
    )
    _load_sam3_multiplex_video_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


HF_SAM3P1_MODEL_ID_TO_CONFIG = {
    "facebook/sam3.1": ("configs/sam3/sam3.1.yaml", "sam3.1_multiplex.pt"),
}


def build_sam3_multiplex_hf(model_id, **kwargs):
    """Build a SAM 3.1 multiplex IMAGE predictor from a HuggingFace model id (downloads weights).

    The hydra config is OURS (``configs/sam3/sam3.1.yaml``); only the gated
    ``sam3.1_multiplex.pt`` is pulled from HF. Mirrors :func:`build_sam3_hf` for the SAM 3.1
    image path; the video counterpart is :func:`build_sam3_multiplex_video_predictor_hf`.
    """
    from huggingface_hub import hf_hub_download

    config_file, ckpt_name = HF_SAM3P1_MODEL_ID_TO_CONFIG.get(
        model_id, ("configs/sam3/sam3.1.yaml", "sam3.1_multiplex.pt")
    )
    ckpt_path = hf_hub_download(repo_id=model_id, filename=ckpt_name)
    return build_sam3_multiplex(config_file=config_file, ckpt_path=ckpt_path, **kwargs)


def build_sam3_multiplex_video_predictor_hf(model_id, **kwargs):
    """Build a SAM 3.1 multiplex video predictor from a HuggingFace model id (downloads weights).

    The hydra config is OURS (``configs/sam3/sam3.1.yaml``); only the gated
    ``sam3.1_multiplex.pt`` is pulled from HF. Mirrors :func:`build_sam3_video_predictor_hf`.
    """
    from huggingface_hub import hf_hub_download

    config_file, ckpt_name = HF_SAM3P1_MODEL_ID_TO_CONFIG.get(
        model_id, ("configs/sam3/sam3.1.yaml", "sam3.1_multiplex.pt")
    )
    ckpt_path = hf_hub_download(repo_id=model_id, filename=ckpt_name)
    return build_sam3_multiplex_video_predictor(
        config_file=config_file, ckpt_path=ckpt_path, **kwargs
    )


# SPDX-License-Identifier: LicenseRef-SAM
# --- SAM3.1-LiteText MULTIPLEX VIDEO predictor (E1) --------------------------
# SAM3.1-LiteText is the SAM 3.1 MULTIPLEX video stack with only the text encoder
# swapped from the PE text tower to MobileCLIP-S0 (ctx16).  The EXISTING
# build_sam3_multiplex_video_predictor + _load_sam3_multiplex_video_checkpoint already
# handle this path: the loader is text-encoder-agnostic (remaps language_backbone.*
# onto whatever text_encoder the config instantiates — 111 keys for MobileCLIP).
# 1439 keys = vision 474 + MobileCLIP 111 + detector head 397 + tracker 457.
# Checkpoint: Simon7108528/EfficientSAM3 (public, no token).


HF_EFFICIENTSAM3P1_LITETEXT_MODEL_ID_TO_FILES = {
    "sam3p1-litetext-s0-ctx16": (
        "configs/efficientsam3/sam3p1_litetext_s0_ctx16.yaml",
        "sam3p1_litetext/efficient_sam3p1_litetext_mobileclip_s0_ctx16.pt",
    ),
}


def build_efficientsam3p1_litetext_video_predictor_hf(
    model_id="sam3p1-litetext-s0-ctx16", **kwargs
):
    """Build a SAM3.1-LiteText multiplex-video predictor from a HuggingFace model id.

    Downloads the checkpoint from the PUBLIC repo ``Simon7108528/EfficientSAM3`` (no
    token required) and delegates to the existing
    :func:`build_sam3_multiplex_video_predictor` with the matching hydra config.  The
    ``_load_sam3_multiplex_video_checkpoint`` loader performs the 4-group strict remap:
    474 vision + 111 MobileCLIP language + 397 detector head + 457 multiplex tracker =
    **1439 keys strict (0 missing / 0 unexpected)**.

    Unlike the PE tower variant (295 text keys), the text encoder here is MobileCLIP-S0
    (ctx16) — only the text backbone is swapped; the tri-neck PE-ViT vision encoder,
    trained geometry encoder (76 keys inside the 397-key head), and the full 457-key
    multiplex tracker are kept unchanged.

    Args:
        model_id: variant key (default ``"sam3p1-litetext-s0-ctx16"`` ->
            ``sam3p1_litetext/efficient_sam3p1_litetext_mobileclip_s0_ctx16.pt``).
        **kwargs: forwarded to :func:`build_sam3_multiplex_video_predictor`
            (``device``, ``mode``, ``hydra_overrides_extra``, ...).

    Returns:
        A ``Sam3MultiplexVideoPredictor`` in eval mode with MobileCLIP-S0 text encoder,
        Sam3MultiplexTracker (457 keys), and trained geometry encoder.
    """
    from huggingface_hub import hf_hub_download

    config_file, ckpt_name = HF_EFFICIENTSAM3P1_LITETEXT_MODEL_ID_TO_FILES[model_id]
    ckpt_path = hf_hub_download(repo_id="Simon7108528/EfficientSAM3", filename=ckpt_name)
    return build_sam3_multiplex_video_predictor(
        config_file=config_file, ckpt_path=ckpt_path, **kwargs
    )


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
