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

from sam.modeling.sam2_generic import SAM2Generic
from sam.sam2_generic_video_predictor import SAM2GenericVideoPredictor

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


def build_sam2_generic(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    use_half=False,
) -> SAM2Generic:
    hydra_overrides = [
        "++model._target_=sam.sam2_generic_video_predictor.SAM2Generic",
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
    """Instantiate a SAM2Generic / SAM2GenericVideoPredictor on the ``meta`` device.

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


def build_sam2_generic_image_predictor_onnx(
    config_file,
    onnx_dir,
    device="cuda",
    use_half=False,
    use_trt=True,
    trt_opts=None,
    apply_postprocessing=True,
    hydra_overrides_extra=[],
) -> SAM2Generic:
    """Build a SAM2Generic image model and swap its 5 heavy blocks for the ONNX /
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
        "sam.sam2_generic_video_predictor.SAM2Generic",
        use_half=use_half,
        apply_postprocessing=apply_postprocessing,
        hydra_overrides_extra=hydra_overrides_extra,
    )
    return attach_onnx_blocks(
        model, onnx_dir, device=torch.device(device), use_trt=use_trt, trt_opts=trt_opts
    )


# Backwards-compatible alias for the original name.
build_sam2_generic_onnx = build_sam2_generic_image_predictor_onnx


def build_sam2_generic_video_predictor_onnx(
    config_file,
    onnx_dir,
    device="cuda",
    use_half=False,
    use_trt=True,
    trt_opts=None,
    apply_postprocessing=True,
    hydra_overrides_extra=[],
) -> SAM2GenericVideoPredictor:
    """Build a SAM2GenericVideoPredictor and swap its 5 heavy blocks for the ONNX /
    TensorRT wrappers exported by tools/export_onnx.py.

    ``onnx_dir`` accepts an extracted export directory, a ``.zip`` of one (e.g. a CI
    release artifact), or an ``http(s)`` URL to such a zip (downloaded + unpacked into
    a cache; override the location with ``SAM2_ONNX_CACHE``).

    Same checkpoint-free contract as
    :func:`build_sam2_generic_image_predictor_onnx`: the skeleton is meta-built (no
    weights allocated), the 5 blocks are swapped and the glue materialized from
    ``weights.npz``, ``use_half`` only touches the torch glue, and ``trt_opts`` tunes
    the TensorRT engine build (cache, workspace, ...). Precision is owned by the exported
    graph + ``trt_opts``. No ``vos_optimized`` knob — ONNX/TRT is already the optimized
    path."""
    from sam.onnx.sam2_generic_onnx import attach_onnx_blocks

    model = _meta_build_generic(
        config_file,
        "sam.sam2_generic_video_predictor.SAM2GenericVideoPredictor",
        use_half=use_half,
        apply_postprocessing=apply_postprocessing,
        hydra_overrides_extra=hydra_overrides_extra,
    )
    return attach_onnx_blocks(
        model, onnx_dir, device=torch.device(device), use_trt=use_trt, trt_opts=trt_opts
    )


def build_sam2_generic_video_predictor(
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
) -> SAM2GenericVideoPredictor:
    hydra_overrides = [
        "++model._target_=sam.sam2_generic_video_predictor.SAM2GenericVideoPredictor",
        f"++model.use_half={use_half}",
    ]

    if vos_optimized:
        # VOS always compiles the image encoder; don't append compile_image_encoder
        # again below to avoid a duplicate hydra override.
        hydra_overrides = [
            "++model._target_=sam.sam2_generic_video_predictor.SAM2GenericVideoPredictorVOS",
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


def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    vos_optimized=False,
    **kwargs,
):
    hydra_overrides = [
        "++model._target_=sam.sam2_video_predictor.SAM2VideoPredictor",
    ]
    if vos_optimized:
        hydra_overrides = [
            "++model._target_=sam.sam2_video_predictor.SAM2VideoPredictorVOS",
            "++model.compile_image_encoder=True",  # Let sam2_base handle this
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

def build_sam2_generic_video_predictor_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2_generic_video_predictor(
        config_file=config_name, ckpt_path=ckpt_path, **kwargs
    )

def build_sam2_generic_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2_generic(
        config_file=config_name, ckpt_path=ckpt_path, **kwargs
    )


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
