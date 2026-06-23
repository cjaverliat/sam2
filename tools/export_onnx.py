# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Export the 5 SAM2 neural-network blocks of a generic model to ONNX.

Produces, in --out-dir:
    image_encoder.onnx
    prompt_encoder_points.onnx, prompt_encoder_mask.onnx
    mask_decoder_mm0.onnx, mask_decoder_mm1.onnx
    memory_attention.onnx
    memory_encoder.onnx
    weights.npz                  (prompt.dense_pe, prompt.no_mask_embed, param.*)
    manifest.json                (num_levels, use_high_res, image_size, dims)

Works for both default SAM2 (Hiera) and EfficientTAM (ViT) configs — build_sam2_generic
overrides the model target and the block shapes are read off the built model.

Exports at opset 18 (default): attention and RoPE are fully decomposed. The native
opset-23 ONNX Attention / RotaryEmbedding ops are avoided because they break on the ONNX
Runtime TensorRT EP (see sam2.modeling.sam.onnx_compat).

Run from the repo root:
    pixi run python tools/export_onnx.py \
        --config configs/sam2.1/sam2.1_hiera_b+.yaml \
        --ckpt checkpoints/sam2.1_hiera_base_plus.pt \
        --out-dir onnx_sam2

Two artifacts, one per runtime (both opset 18):
  * fp32 (default) -> the TensorRT path. Let TensorRT convert + per-layer auto-tune from
    the fp32 graph: TensorRTOptions(fp16_enable=True) or (bf16_enable=True) on Ampere+.
    This weakly-typed path fuses freely (it matches the torch baseline even on Turing);
    bf16 keeps fp32 range and is Ampere's native precision. For a deployment engine use
    builder_optimization_level=5 (built once, cached).
  * --fp16 -> a mixed-precision copy for the CUDA/CPU EP, which can't convert precision
    at runtime. Do NOT feed this baked-fp16 graph to the TensorRT precision flags (the
    weights are already fp16; the baked cast-islands also block fusion).
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

# torch.onnx logs ✅/❌ status glyphs; the default Windows console codec (cp1252)
# raises UnicodeEncodeError on them. Force UTF-8 on the std streams so the export
# runs without a PYTHONUTF8 / PYTHONIOENCODING wrapper.
for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn

from sam2.build_sam import build_sam2_generic
from sam2.modeling.sam.onnx_compat import set_export_opset
from sam2.onnx import image_encoder_onnx as ienc
from sam2.onnx import mask_decoder_onnx as mdec
from sam2.onnx import memory_attention_onnx as mattn
from sam2.onnx import memory_encoder_onnx as menc
from sam2.onnx import prompt_encoder_onnx as penc

Dim = torch.export.Dim


def _force_fp32_io(model) -> None:
    """Make every graph input/output fp32, wrapping fp16 boundaries with Cast nodes.

    ``convert_float_to_float16(keep_io_types=True)`` is supposed to do this, but it
    leaves some boundaries fp16 when it can't infer types through the native opset-23
    Attention op (e.g. an Identity feeding a graph output stays fp16). The runtime
    pipes these tensors between blocks as fp32 torch tensors, so a stray fp16 boundary
    raises "Unexpected input data type" at the next block. Normalise deterministically:
    external IO is fp32, with a Cast to/from the fp16 internals."""
    from onnx import TensorProto, helper

    F16, F32 = TensorProto.FLOAT16, TensorProto.FLOAT
    g = model.graph
    for inp in g.input:
        tt = inp.type.tensor_type
        if tt.elem_type == F16:
            internal = inp.name + "_f16"
            for n in g.node:
                n.input[:] = [internal if i == inp.name else i for i in n.input]
            g.node.insert(
                0, helper.make_node("Cast", [inp.name], [internal], to=F16,
                                    name="cast_in_" + inp.name)
            )
            tt.elem_type = F32
    for out in g.output:
        tt = out.type.tensor_type
        if tt.elem_type == F16:
            internal = out.name + "_f16"
            for n in g.node:
                n.output[:] = [internal if o == out.name else o for o in n.output]
            g.node.append(
                helper.make_node("Cast", [internal], [out.name], to=F32,
                                 name="cast_out_" + out.name)
            )
            tt.elem_type = F32


def _to_mixed_fp16(path: Path) -> None:
    """Rewrite a saved fp32 ONNX graph in place as mixed precision for the CUDA/CPU EP
    (which can't convert precision at runtime; the TensorRT path keeps fp32 and uses
    trt_fp16/bf16_enable). fp16 weights/compute, graph IO kept fp32, with LayerNorm
    (variance) and Identity (converter correctness) held fp32. Shape inference stays on
    so the converter types its casts; ``_force_fp32_io`` then repairs any fp16 boundary
    the converter left behind."""
    import onnx
    from onnxconverter_common import float16

    block = list(float16.DEFAULT_OP_BLOCK_LIST) + ["LayerNormalization", "Identity"]
    model16 = float16.convert_float_to_float16(
        onnx.load(str(path)),
        keep_io_types=True,
        op_block_list=block,
        disable_shape_infer=False,
    )
    _force_fp32_io(model16)
    onnx.save(model16, str(path))
    print(f"  -> mixed fp16 {path.name}")


def _export(
    module: nn.Module, args, input_names, output_names, dynamic_shapes, path, opset,
    fp16=False,
):
    module = module.eval()
    onnx_program = torch.onnx.export(
        module,
        args,
        input_names=input_names,
        output_names=output_names,
        dynamic_shapes=dynamic_shapes,
        dynamo=True,
        opset_version=opset,
    )
    onnx_program.save(str(path))
    print(f"  saved {path.name}")
    if fp16:
        _to_mixed_fp16(path)
    return onnx_program


def _parity(onnx_path, feeds: dict, torch_outs, names):
    """torch (fp32 reference) vs ONNX Runtime (CPU) on identical inputs. For a mixed
    fp16 graph this is exactly the mixed-precision-vs-fp32 check: the torch outputs are
    the fp32 reference and the ORT outputs come from the converted graph. Reports the
    worst absolute and relative deviation across outputs."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    np_feeds = {k: v.detach().cpu().numpy() for k, v in feeds.items()}
    ort_outs = sess.run(names, np_feeds)
    worst_abs = 0.0
    worst_rel = 0.0
    for ref, got in zip(torch_outs, ort_outs):
        ref = ref.detach().cpu().numpy()
        abs_err = float(np.abs(ref - got).max())
        worst_abs = max(worst_abs, abs_err)
        worst_rel = max(worst_rel, abs_err / (float(np.abs(ref).max()) + 1e-9))
    print(f"  parity {onnx_path.name}: max|Δ| = {worst_abs:.3e}  max rel = {worst_rel:.2e}")
    return worst_abs


# --- thin export modules: bake non-tensor args, return flat tuples ---


class _ImageEncoder(nn.Module):
    """Image encoder + the two SAM hi-res projection convs folded in.

    The orchestration normally runs ``conv_s0`` / ``conv_s1`` (mask-decoder
    submodules) on backbone levels 0/1 right after the encoder. Baking them into
    this graph moves their weights into ONNX, so the runtime needs no checkpoint
    for them (the wrapper sets the torch convs to identity)."""

    def __init__(self, enc, conv_s0=None, conv_s1=None):
        super().__init__()
        self.enc = enc
        self.conv_s0 = conv_s0
        self.conv_s1 = conv_s1

    def forward(self, image):
        o = self.enc(image)
        fpn = list(o["backbone_fpn"])
        if self.conv_s0 is not None:
            fpn[0] = self.conv_s0(fpn[0])
            fpn[1] = self.conv_s1(fpn[1])
        return (o["vision_features"], *fpn, *o["vision_pos_enc"])


class _PromptPoints(nn.Module):
    def __init__(self, pe):
        super().__init__()
        self.pe = pe

    def forward(self, coords, labels):
        return self.pe._embed_points(coords, labels, pad=True)


class _PromptMask(nn.Module):
    def __init__(self, pe):
        super().__init__()
        self.pe = pe

    def forward(self, mask):
        return self.pe._embed_masks(mask)


class _MaskDecoder(nn.Module):
    def __init__(self, dec, multimask, use_hr):
        super().__init__()
        self.dec = dec
        self.multimask = multimask
        self.use_hr = use_hr

    def forward(self, image_embeddings, image_pe, sparse, dense, hr0=None, hr1=None):
        hr = [hr0, hr1] if self.use_hr else None
        return self.dec(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=self.multimask,
            repeat_image=False,
            high_res_features=hr,
        )


class _MemoryAttention(nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn

    def forward(self, curr, curr_pos, memory_rope, memory_norope, mpos_rope, mpos_norope):
        memory = torch.cat([memory_rope, memory_norope], dim=0)
        memory_pos = torch.cat([mpos_rope, mpos_norope], dim=0)
        num_obj_ptr_tokens = memory_norope.shape[0]
        return self.attn(
            curr=curr,
            curr_pos=curr_pos,
            memory=memory,
            memory_pos=memory_pos,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )


class _MemoryEncoder(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, pix_feat, masks):
        o = self.enc(pix_feat, masks, skip_mask_sigmoid=True)
        return o["vision_features"], o["vision_pos_enc"][0]


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out-dir", required=True)
    # opset 18: decomposed attention/RoPE. The native opset-23 ONNX Attention/
    # RotaryEmbedding ops break on the ORT TensorRT EP (Attention -> CUDA fallback /
    # ~25-way fragmentation; dynamic RoPE -> myelin build crash). See sam2.modeling.sam.
    # onnx_compat. Keep 18 unless TensorRT+ORT fix opset-23 fused-op support.
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--no-parity", action="store_true", help="skip torch vs ORT check")
    p.add_argument(
        "--fp16", action="store_true",
        help="also write a mixed-precision (fp16) copy of the heavy blocks (image encoder, "
             "memory attention) for the CUDA/CPU EP, which can't convert precision at "
             "runtime. Graph IO stays fp32; LayerNorm + Identity stay fp32. Needs "
             "onnxconverter-common. (For TensorRT, keep fp32 and use trt_fp16/bf16_enable.)",
    )
    p.add_argument(
        "--max-cond-frames", type=int, default=16,
        help="SIZES the memory-attention TensorRT engine profile only: build one engine "
             "covering up to this many conditional (prompted-frame) memories (+ the "
             "num_maskmem-1 recent frames + object pointers). It does NOT cap the "
             "runtime — the model keeps its configured max_cond_frames_in_attn "
             "(typically -1 = unbounded); memory beyond the profile just triggers a "
             "one-time ORT engine rebuild at runtime (safe with decomposed RoPE). Use -1 "
             "to leave the memory dim fully unbounded (no profile; ORT builds per shape).",
    )
    args = p.parse_args()

    set_export_opset(args.opset)
    if args.opset >= 23:
        warnings.warn(
            f"opset {args.opset} >= 23 emits the native ONNX RotaryEmbedding / Attention "
            "ops. These break on the ONNX Runtime TensorRT EP (TRT 10.16.x): the dynamic "
            "RoPE path fails the engine build with a myelin SSA error, and Attention is "
            "not placed on TensorRT (CUDA fallback -> heavy subgraph fragmentation). Use "
            "opset 18 (the default, fully decomposed) for the TensorRT path.",
            stacklevel=2,
        )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Always export on CPU: keeps the trace free of device-specific ops/guards and the
    # graphs portable. The runtime (video_predictor_example_onnx.py) picks the EP.
    device = torch.device("cpu")

    # Runtime conditional-memory cap is left at the config value (typically -1 =
    # unbounded); --max-cond-frames only sizes the TensorRT profile below.
    model = build_sam2_generic(
        args.config, args.ckpt, device="cpu", use_half=False
    ).eval()

    S = model.image_size
    C = model.hidden_dim
    G = model.sam_image_embedding_size
    nfl = model.num_feature_levels
    use_hr = model.use_high_res_features_in_sam
    mem_dim = model.mem_dim
    mask_in = model.sam_prompt_encoder.mask_input_size

    # Memory-attention length bounds for the TensorRT shape profile. Spatial (rope)
    # memory = up to (profile_cond + num_maskmem-1) frames, each G*G tokens. Object-
    # pointer (no-rope) memory = up to max_obj_ptrs pointers, each split into
    # (hidden_dim // mem_dim) tokens. profile_cond=-1 (or unbounded obj ptrs) leaves the
    # dim uncapped (no profile; ORT builds one engine per memory length at runtime).
    HW = G * G
    tokens_per_ptr = max(1, C // mem_dim)
    profile_cond = args.max_cond_frames
    rope_max = (
        (profile_cond + model.num_maskmem - 1) * HW if profile_cond != -1 else None
    )
    ptr_max = (
        model.max_obj_ptrs_in_encoder * tokens_per_ptr
        if model.max_obj_ptrs_in_encoder != -1 else None
    )

    # Batch is static: the Hiera image encoder bakes a batch-dependent guard that
    # blocks a dynamic batch, and the video pipeline runs one (frame, object) at a
    # time. Re-export for a different batch size. The genuinely variable axes — point
    # count and memory length — stay dynamic, but memory length is capped to the bounds
    # above so TensorRT can build a single profiled engine. (min=1: examples use size
    # 1/2, and torch.export Dim defaults to min=2.)
    npts = Dim("npts", min=1)
    # Spatial (rope) memory is always num_frames * HW tokens — each frame contributes a
    # G*G grid. Express the length as a derived dim (HW * num_frames) so the per-frame
    # landmark fold + avg_pool2d in EfficientRoPEAttention traces under torch.export
    # instead of specializing the frame axis (a plain Dim("mem_len") collapses to
    # min=max=1 on the single-frame example). Non-landmark models are unaffected: their
    # rope memory is a multiple of HW too, so the derived dim is just a tighter (correct)
    # bound on the same dynamic axis.
    nframes_max = rope_max // HW if rope_max is not None else None
    nframes = Dim("num_frames", min=1, max=nframes_max)
    mlen = HW * nframes
    plen = Dim("ptr_len", min=1, max=ptr_max)

    # ---------- image encoder (with hi-res convs folded in) ----------
    print("image encoder")
    img = torch.randn(1, 3, S, S, device=device)
    bb = model.image_encoder(img)
    num_levels = len(bb["backbone_fpn"])
    onames = ienc.output_names(num_levels)
    conv_s0 = model.sam_mask_decoder.conv_s0 if use_hr else None
    conv_s1 = model.sam_mask_decoder.conv_s1 if use_hr else None
    _export(
        _ImageEncoder(model.image_encoder, conv_s0, conv_s1),
        (img,),
        ienc.INPUT_NAMES,
        onames,
        {"image": None},
        out / ienc.FILENAME,
        args.opset,
        fp16=args.fp16,
    )

    # high-res projected features (as encode_image does) for the decoder example
    fpn = list(bb["backbone_fpn"])
    if use_hr:
        fpn[0] = conv_s0(fpn[0])
        fpn[1] = conv_s1(fpn[1])
    img_embeddings = fpn[-nfl:]
    low = img_embeddings[-1]
    high = img_embeddings[:-1] if use_hr else None

    # ---------- prompt encoder ----------
    print("prompt encoder")
    coords = torch.randn(1, 2, 2, device=device)
    labels = torch.ones(1, 2, dtype=torch.int64, device=device)
    pe_mod = _PromptPoints(model.sam_prompt_encoder)
    _export(
        pe_mod, (coords, labels), ["coords", "labels"], ["sparse"],
        {"coords": {1: npts}, "labels": {1: npts}},
        out / penc.POINTS_FILE, args.opset,
    )
    mask_in_t = torch.randn(1, 1, mask_in[0], mask_in[1], device=device)
    _export(
        _PromptMask(model.sam_prompt_encoder), (mask_in_t,), ["mask"], ["dense"],
        {"mask": None}, out / penc.MASK_FILE, args.opset,
    )
    sparse, dense = model.sam_prompt_encoder(points=(coords, labels), boxes=None, masks=None)
    image_pe = model.sam_prompt_encoder.get_dense_pe()

    # Bake everything the runtime needs that is NOT an ONNX graph into a single
    # weights.npz dict, so the ONNX/TRT predictor needs no torch checkpoint:
    #   prompt.dense_pe / prompt.no_mask_embed — the two data-independent
    #     prompt-encoder constants (get_dense_pe + the no-mask dense embedding),
    #     fixed tensors derived only from weights.
    #   param.<k> — every orchestration param/buffer NOT inside one of the 5
    #     swapped blocks (no_mem_embed, no_mem_pos_enc, maskmem_tpos_enc,
    #     obj_ptr_proj.*, obj_ptr_tpos_proj.*, no_obj_ptr, no_obj_embed_spatial,
    #     ...). These live in the torch glue on the video/memory path. The ONNX
    #     predictor builds its orchestration skeleton on the meta device and
    #     materializes exactly these via load_state_dict(assign=True), so this set
    #     must cover the full glue state_dict.
    no_mask = model.sam_prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
    swapped = ("image_encoder.", "memory_attention.", "memory_encoder.",
               "sam_prompt_encoder.", "sam_mask_decoder.")
    weights = {
        f"param.{k}": v.detach().cpu().numpy()
        for k, v in model.state_dict().items()
        if not k.startswith(swapped)
    }
    weights["prompt.dense_pe"] = image_pe.detach().cpu().numpy()
    weights["prompt.no_mask_embed"] = no_mask.detach().cpu().numpy()
    np.savez(out / "weights.npz", **weights)
    n_param = sum(k.startswith("param.") for k in weights)
    print(f"  saved weights.npz ({n_param} glue params + dense_pe + no_mask_embed)")

    # ---------- mask decoder (two graphs) ----------
    print("mask decoder")
    dec_inputs = mdec.BASE_INPUTS + (mdec.HR_INPUTS if use_hr else [])
    dec_args = [low, image_pe, sparse, dense]
    dyn = {
        "image_embeddings": None,
        "image_pe": None,
        "sparse": {1: npts},
        "dense": None,
    }
    if use_hr:
        dec_args += list(high)
        dyn["hr0"] = None
        dyn["hr1"] = None
    for mm in (False, True):
        _export(
            _MaskDecoder(model.sam_mask_decoder, mm, use_hr),
            tuple(dec_args), dec_inputs, mdec.OUTPUT_NAMES, dyn,
            out / mdec.filename(mm), args.opset,
        )

    # ---------- memory attention ----------
    print("memory attention")
    # Multi-frame memory example: the rope-memory length must be num_frames * HW for the
    # derived `mlen` dim, and >1 frame so (a) the landmark frame axis stays genuinely
    # dynamic and (b) the eager parity reference also takes the landmark branch (its
    # num_k_rope <= nq guard would otherwise pick full attention on a single frame and
    # disagree with the always-landmark exported graph).
    nf_example = 3
    curr = torch.randn(HW, 1, C, device=device)
    curr_pos = torch.randn(HW, 1, C, device=device)
    mem_rope = torch.randn(nf_example * HW, 1, mem_dim, device=device)
    mem_norope = torch.randn(4, 1, mem_dim, device=device)
    mpos_rope = torch.randn(nf_example * HW, 1, mem_dim, device=device)
    mpos_norope = torch.randn(4, 1, mem_dim, device=device)
    _export(
        _MemoryAttention(model.memory_attention),
        (curr, curr_pos, mem_rope, mem_norope, mpos_rope, mpos_norope),
        mattn.INPUT_NAMES, mattn.OUTPUT_NAMES,
        {
            "curr": None,
            "curr_pos": None,
            "memory_rope": {0: mlen},
            "memory_norope": {0: plen},
            "mpos_rope": {0: mlen},
            "mpos_norope": {0: plen},
        },
        out / mattn.FILENAME, args.opset,
        fp16=args.fp16,
    )

    # ---------- memory encoder ----------
    print("memory encoder")
    pix_feat = torch.randn(1, C, G, G, device=device)
    mask_for_mem = torch.randn(1, 1, S, S, device=device)
    _export(
        _MemoryEncoder(model.memory_encoder),
        (pix_feat, mask_for_mem), menc.INPUT_NAMES, menc.OUTPUT_NAMES,
        {"pix_feat": None, "masks": None},
        out / menc.FILENAME, args.opset,
    )

    manifest = {
        "num_levels": num_levels,
        "use_high_res": bool(use_hr),
        "image_size": S,
        "hidden_dim": C,
        "mem_dim": mem_dim,
        "sam_image_embedding_size": G,
        "mask_input_size": list(mask_in),
        "config": args.config,
        # Memory-attention TRT profile bounds. MemoryAttentionOnnx builds one engine
        # over [1 .. *_max_tokens]; the runtime is NOT capped to profile_max_cond_frames
        # (it keeps the model's max_cond_frames_in_attn) — over-profile lengths just
        # rebuild an engine on the fly (safe with decomposed RoPE). null = unbounded.
        "profile_max_cond_frames": profile_cond,
        "num_maskmem": model.num_maskmem,
        "memory_rope_max_tokens": rope_max,
        "memory_norope_max_tokens": ptr_max,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out/'manifest.json'}")

    if not args.no_parity:
        print("\nparity (torch vs ORT-CPU):")
        cpu = torch.device("cpu")
        m = model.to(cpu)
        _parity(
            out / ienc.FILENAME,
            {"image": img.to(cpu)},
            _ImageEncoder(
                m.image_encoder,
                m.sam_mask_decoder.conv_s0 if use_hr else None,
                m.sam_mask_decoder.conv_s1 if use_hr else None,
            )(img.to(cpu)),
            ienc.output_names(num_levels),
        )
        _parity(
            out / penc.POINTS_FILE,
            {"coords": coords.to(cpu), "labels": labels.to(cpu)},
            (_PromptPoints(m.sam_prompt_encoder)(coords.to(cpu), labels.to(cpu)),),
            ["sparse"],
        )
        _parity(
            out / menc.FILENAME,
            {"pix_feat": pix_feat.to(cpu), "masks": mask_for_mem.to(cpu)},
            _MemoryEncoder(m.memory_encoder)(pix_feat.to(cpu), mask_for_mem.to(cpu)),
            menc.OUTPUT_NAMES,
        )
        _parity(
            out / mattn.FILENAME,
            {
                "curr": curr.to(cpu), "curr_pos": curr_pos.to(cpu),
                "memory_rope": mem_rope.to(cpu), "memory_norope": mem_norope.to(cpu),
                "mpos_rope": mpos_rope.to(cpu), "mpos_norope": mpos_norope.to(cpu),
            },
            (_MemoryAttention(m.memory_attention)(
                curr.to(cpu), curr_pos.to(cpu), mem_rope.to(cpu),
                mem_norope.to(cpu), mpos_rope.to(cpu), mpos_norope.to(cpu)),),
            mattn.OUTPUT_NAMES,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
