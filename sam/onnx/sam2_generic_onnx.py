# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch

from sam.onnx.image_encoder_onnx import ImageEncoderOnnx
from sam.onnx.mask_decoder_onnx import MaskDecoderOnnx
from sam.onnx.memory_attention_onnx import MemoryAttentionOnnx
from sam.onnx.memory_encoder_onnx import MemoryEncoderOnnx
from sam.onnx.prompt_encoder_onnx import PromptEncoderOnnx
from sam.onnx.trt_options import TensorRTOptions


def _cache_root() -> Path:
    """Where downloaded zips and zip extractions live.

    Override with ``SAM2_ONNX_CACHE``; otherwise ``$XDG_CACHE_HOME/sam2/onnx``
    (``~/.cache/sam2/onnx`` if unset)."""
    base = os.environ.get("SAM2_ONNX_CACHE")
    if base:
        return Path(base)
    xdg = os.environ.get("XDG_CACHE_HOME")
    root = Path(xdg) if xdg else Path.home() / ".cache"
    return root / "sam2" / "onnx"


def _find_manifest_dir(root: Path) -> Path:
    """Return the directory holding ``manifest.json`` at or below ``root``.

    The CI release zips wrap the export in an ``opset18/`` / ``opset23/`` subdir,
    so accept it one (or more) levels down — but reject an ambiguous tree that
    contains several exports."""
    if (root / "manifest.json").is_file():
        return root
    hits = list(root.rglob("manifest.json"))
    if not hits:
        raise FileNotFoundError(f"no manifest.json found under {root}")
    if len(hits) > 1:
        raise ValueError(
            f"multiple manifest.json found under {root} ({len(hits)} exports); "
            "point onnx_dir at a single export directory (e.g. .../opset23)"
        )
    return hits[0].parent


def _download(url: str) -> Path:
    """Fetch ``url`` into the cache, skipping the download if already present."""
    cache = _cache_root()
    cache.mkdir(parents=True, exist_ok=True)
    name = url.split("?")[0].rstrip("/").rsplit("/", 1)[-1] or "download.zip"
    digest = hashlib.sha256(url.encode()).hexdigest()[:16]
    dst = cache / f"{digest}_{name}"
    if not dst.exists():
        tmp = dst.with_name(dst.name + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dst)
    return dst


def _extract_zip(zip_path: Path) -> Path:
    """Extract ``zip_path`` into the cache (keyed by path + mtime + size) and
    reuse a prior extraction when the zip is unchanged."""
    st = zip_path.stat()
    key = f"{zip_path.resolve().as_posix()}|{st.st_mtime_ns}|{st.st_size}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    dst = _cache_root() / f"{zip_path.stem}_{digest}"
    done = dst / ".extracted"
    if not done.exists():
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dst)
        done.touch()
    return dst


def _resolve_onnx_dir(src: str | Path) -> Path:
    """Resolve an ONNX export source to the directory holding ``manifest.json``.

    ``src`` may be:
      * a directory (an extracted export, or a parent containing ``opset*/``),
      * a ``.zip`` file (extracted to the cache), or
      * an ``http(s)`` URL to such a zip (downloaded then extracted)."""
    if isinstance(src, str) and src.startswith(("http://", "https://")):
        src = _download(src)
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(src)
    if src.is_file():
        if src.suffix.lower() != ".zip":
            raise ValueError(
                f"expected a directory, a .zip file, or an http(s) URL; got {src}"
            )
        src = _extract_zip(src)
    return _find_manifest_dir(src)


def _load_glue(model: torch.nn.Module, onnx_dir: Path, device: torch.device) -> None:
    """Materialize the orchestration glue from ``weights.npz`` onto ``device``.

    The glue is every param outside the 5 swapped blocks (``obj_ptr_proj``,
    ``no_mem_embed``, ``maskmem_tpos_enc``, ``no_obj_ptr``, ``no_obj_embed_spatial``,
    ``obj_ptr_tpos_proj``, ``mask_downsample``, ...), shipped under the ``param.*``
    keys.

    ``assign=True``: the model is built on a meta device, so its glue params have no
    storage to copy into — they are replaced outright by the loaded tensors. After the
    swap the model's state_dict is *exactly* the glue (the ONNX wrappers hold no
    params), so any missing/unexpected key means a stale export."""
    w = np.load(onnx_dir / "weights.npz")
    sd = {
        k[len("param."):]: torch.from_numpy(w[k])
        for k in w.files
        if k.startswith("param.")
    }
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(
            "weights.npz glue (param.*) does not match the model "
            f"(missing={missing}, unexpected={unexpected}); "
            "stale export — re-run tools/export_onnx.py for this config"
        )
    # assign=True kept the loaded tensors on CPU (numpy origin); move glue to device.
    model.to(device)


def _assert_materialized(model: torch.nn.Module) -> None:
    """Fail loud if any param/buffer is still on the meta device after the swap — i.e.
    something outside the 5 swapped blocks was *not* covered by ``weights.npz``
    (e.g. a non-persistent buffer the export missed)."""
    leftover = [
        n
        for n, t in (*model.named_parameters(), *model.named_buffers())
        if t.is_meta
    ]
    if leftover:
        raise RuntimeError(
            f"meta tensors survived the ONNX swap (not provided by weights.npz): "
            f"{leftover} — re-run tools/export_onnx.py for this config"
        )


def attach_onnx_blocks(
    model: torch.nn.Module,
    onnx_dir: str | Path,
    device: torch.device,
    use_trt: bool = True,
    trt_opts: TensorRTOptions | None = None,
) -> torch.nn.Module:
    """Turn a *meta-built* Sam2Predictor / Sam2VideoPredictor into a runnable
    ONNX/TensorRT model: replace its 5 heavy blocks with ONNX-backed ``nn.Module``
    wrappers and materialize the orchestration glue from ``weights.npz``.

    The model is expected to be instantiated under ``torch.device("meta")`` (see
    :func:`sam.build_sam.build_sam2_video_predictor_onnx`): no real weights
    are ever allocated, so nothing is built-then-discarded. The 5 blocks are swapped
    with plain ``setattr`` (the wrappers are ``nn.Module``\\ s), then the glue params
    are given real storage on ``device`` via :func:`_load_glue`.

    ``onnx_dir`` may be an extracted export directory, a ``.zip`` of one (e.g. a CI
    release artifact), or an ``http(s)`` URL to such a zip — zips/URLs are fetched
    and unpacked into a cache (see :func:`_cache_root`).

    Checkpoint-free: every weight the orchestration reads from torch — the hi-res
    convs and dense PE / no-mask embed (baked into the graphs / ``prompt.*`` keys) plus
    the memory-path glue (``param.*`` keys, loaded here via :func:`_load_glue`) —
    comes from the export artifacts."""
    onnx_dir = _resolve_onnx_dir(onnx_dir)
    device = torch.device(device)
    manifest = json.loads((onnx_dir / "manifest.json").read_text())

    # A mixed-precision export is for the CUDA/CPU EP only: its baked fp16 weights + cast
    # islands can't build a TensorRT engine (memory_attention dies with TRT error 10,
    # "could not find any implementation for node"). Reject loud here instead of after
    # a multi-minute build.
    if use_trt and manifest["precision"] == "mixed-precision":
        raise RuntimeError(
            f"{onnx_dir} is a mixed-precision export; it is for the CUDA/CPU EP only and "
            f"cannot be used with TensorRT — its baked fp16 weights and cast islands "
            f"block the strongly-typed TRT build (memory_attention fails with 'could not "
            f"find any implementation for node'). For TensorRT use the fp32 export with "
            f"TensorRTOptions(bf16_enable=True), or fp16_enable=True (applied to the 2 "
            f"heavy blocks only); or pass use_trt=False to run this export on the CUDA EP."
        )

    # Swap the 5 heavy blocks for their ONNX-backed nn.Module wrappers. Plain setattr
    # works now that the wrappers are nn.Modules; the meta params they replace carry no
    # storage, so nothing real is freed.
    model.image_encoder = ImageEncoderOnnx(
        onnx_dir, manifest["num_levels"], device, use_trt, trt_opts)
    model.memory_encoder = MemoryEncoderOnnx(onnx_dir, device, use_trt, trt_opts)
    model.memory_attention = MemoryAttentionOnnx(onnx_dir, device, use_trt, trt_opts)
    model.sam_mask_decoder = MaskDecoderOnnx(
        onnx_dir, manifest["use_high_res"], device, use_trt=use_trt, trt_opts=trt_opts)
    model.sam_prompt_encoder = PromptEncoderOnnx(onnx_dir, device, use_trt, trt_opts)

    # Note: the export's --max-cond-frames only *sizes* the memory_attention TRT engine
    # profile (memory_*_max_tokens in the manifest); it does NOT cap the runtime. The
    # model keeps its configured max_cond_frames_in_attn (-1 = unbounded). If the actual
    # memory ever exceeds the profile, ORT just builds an extra engine for that shape —
    # safe now that RoPE is decomposed (no myelin failure), just a one-time build.

    # Give the remaining (orchestration glue) params real storage on `device`.
    _load_glue(model, onnx_dir, device)
    _assert_materialized(model)
    return model
