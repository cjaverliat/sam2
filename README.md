# SAM 2 — Streaming, ONNX & EfficientTAM fork

A community fork of [**SAM 2: Segment Anything in Images and Videos**](https://github.com/facebookresearch/sam2) focused on **deployment** and **memory-efficient streaming video segmentation**.

![SAM 2 architecture](assets/model_diagram.png?raw=true)

This fork keeps the original SAM 2 models and weights bit-for-bit, and adds:

- **ONNX / TensorRT inference** for both images and video — a checkpoint-free runtime that runs the model's heavy blocks as ONNX Runtime sessions (CPU, CUDA, or TensorRT execution providers).
- **On-the-fly (streaming) video processing** — frames are fed one at a time instead of loading the entire video into memory. Peak memory stays roughly constant with video length.
- **A pluggable memory bank** with custom strategies for storing, selecting, and pruning per-object memories (e.g. a sliding-window "forgetful" bank), instead of the original's unbounded store.
- **Integrated [EfficientTAM](https://github.com/yformer/EfficientTAM)** — the lightweight ViT-based Track-Anything models, usable through the same APIs as SAM 2.
- **Pre-built wheels with compiled CUDA kernels** attached to each release, so `pip install` does not require a local CUDA toolchain in the common case.

> This is an unofficial fork. The underlying models, weights, and research are the work of Meta AI (FAIR). See [Citation](#citation).

---

## Table of contents

- [Why this fork](#why-this-fork)
- [Installation](#installation)
- [Download checkpoints](#download-checkpoints)
- [Usage](#usage)
  - [Image prediction (PyTorch)](#image-prediction-pytorch)
  - [Streaming video prediction (PyTorch)](#streaming-video-prediction-pytorch)
  - [Custom memory bank](#custom-memory-bank)
  - [ONNX / TensorRT inference](#onnx--tensorrt-inference)
- [EfficientTAM](#efficienttam)
- [SAM 3](#sam-3--concept-segmentation)
  - [Image — detect all instances of a concept](#image--detect-all-instances-of-a-concept)
  - [Streaming video — track a concept across frames](#streaming-video--track-a-concept-across-frames)
  - [SAM 3.1 (multiplex)](#sam-31-multiplex--up-to-16-objects-in-one-joint-forward-pass)
  - [Geometry prompts — clicks, boxes, masks](#geometry-prompts--clicks-boxes-masks)
  - [Adding objects mid-stream](#adding-objects-mid-stream)
  - [Output policy — when a tracked object becomes visible](#output-policy--when-a-tracked-object-becomes-visible)
  - [Several concepts at once](#several-concepts-at-once)
- [EfficientSAM3](#efficientsam3--lightweight-concept-segmentation)
- [Benchmarks](#benchmarks)
- [License](#license)
- [Citation](#citation)

---

## Why this fork

The upstream video predictor (`SAM2VideoPredictor`) builds an inference state that holds **every frame of the video in memory** before tracking, which scales linearly with video length and is impractical for long or live streams.

This fork adds a **generic predictor** (`SAM2GenericVideoPredictor`) that processes **one frame at a time**. The only per-object state retained between frames lives in a **memory bank**, which you can cap or prune. Combined with the ONNX/TensorRT runtime, this makes the model practical for long videos, live camera feeds, and resource-constrained deployment.

| | Upstream `SAM2VideoPredictor` | This fork `SAM2GenericVideoPredictor` |
|---|---|---|
| Frame ingestion | Whole video loaded into `init_state` | One frame per `forward()` call |
| Memory footprint | Grows with video length | Bounded by the memory bank |
| Memory policy | Unbounded store | Pluggable (`ObjectMemoryBank`), prunable |
| Backends | PyTorch | PyTorch **and** ONNX Runtime (CPU / CUDA / TensorRT) |

---

## Installation

The code requires `python>=3.10`, `torch>=2.5.1`, and `torchvision>=0.20.1`.

### Option A — pip with pre-built wheels (recommended)

At install time, `setup.py` first tries to download a **pre-built wheel** (with the compiled `sam2._C` CUDA kernels) from this repo's GitHub Releases, matched to your exact `(torch, CUDA, Python, platform, C++ ABI)`. If none matches, it compiles the extension from source; if the source build fails, it falls back to a pure-Python wheel that JIT-compiles `_C` on first use.

```bash
git clone https://github.com/cjaverliat/sam2.git && cd sam2
pip install -e .
```

Useful environment toggles (see [`setup.py`](setup.py) for the full list):

| Variable | Effect |
|---|---|
| `SAM2_FORCE_BUILD=1` | Skip the pre-built wheel download, always compile from source |
| `SAM2_BUILD_CUDA=0` | Build a pure-Python wheel (no `_C` extension) |
| `SAM2_ALLOW_BUILD_ERRORS=0` | Fail hard instead of falling back to pure-Python |
| `SAM2_WHEEL_BASE_URL=...` | Override the GitHub Releases base URL |

See [`INSTALL.md`](./INSTALL.md) for FAQs and troubleshooting.


### Option B — From source with pixi

This fork ships a [pixi](https://pixi.sh) workspace that pins the full toolchain (matching torch + CUDA, build tooling, ONNX runtimes) per environment. It handles the CUDA build for you.

```bash
git clone https://github.com/cjaverliat/sam2.git && cd sam2

# Default environment: CUDA 12.8 torch + compiled CUDA kernels
pixi shell           # drop into the environment
# or run a single command:
pixi run python -c "import sam; print(sam.__version__)"
```

Available environments include `default` (CUDA 12.8 torch), `notebooks`, `dev`, and the ONNX tiers described in [ONNX / TensorRT inference](#onnx--tensorrt-inference).

If you plan to commit, point git at the repo's hooks once: `git config core.hooksPath .githooks`. The `pre-commit` hook clears cell output from the notebooks you stage — your working copy keeps the figures it rendered — so `.ipynb` diffs stay readable.

> **Windows:** `nvcc` only supports the MSVC `cl.exe` host compiler, which conda/pixi cannot ship. Source the MSVC environment (`vcvars64.bat`) before any `pixi` command that builds the extension. See [`pyproject.toml`](pyproject.toml) for the exact path.

---

## Download checkpoints

Checkpoints download via pixi tasks (cross-platform, with resume and caching). Files land in `checkpoints/`.

```bash
# SAM 2.1 — all four sizes
pixi run download-sam2

# or an individual size
pixi run download-sam2-tiny      # also: -small, -base-plus, -large

# EfficientTAM — all variants
pixi run download-efficienttam
```

SAM 2.1 checkpoints can also be fetched individually:

- [sam2.1_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt)
- [sam2.1_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt)
- [sam2.1_hiera_base_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt)
- [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)

---

## Usage

### Image prediction (PyTorch)

Encode the image once, then prompt it (see [`notebooks/sam2_image_predictor_example.ipynb`](notebooks/sam2_image_predictor_example.ipynb)):

```python
import numpy as np
import torch
from PIL import Image
from sam.build_sam import build_sam2_predictor

model = build_sam2_predictor(
    "configs/sam2.1/sam2.1_hiera_l.yaml",
    "./checkpoints/sam2.1_hiera_large.pt",
    device="cuda",
    use_half=True,
)

image = np.array(Image.open("images/truck.jpg").convert("RGB"))   # HWC uint8
orig_hw = image.shape[:2]
img_embeddings, _ = model.encode_image(
    torch.as_tensor(image, device=model.device).permute(2, 0, 1)
)

prompt_embeddings = model.encode_prompts(
    orig_hw=orig_hw,
    batch_size=1,
    points_coords=torch.tensor([[[500, 375]]], dtype=torch.float32, device=model.device),
    points_labels=torch.tensor([[1]], dtype=torch.int, device=model.device),
)
result = model.generate_masks(
    orig_hw=orig_hw,
    img_embeddings=img_embeddings,
    prompt_embeddings=prompt_embeddings,
    multimask_output=True,
)
masks = result.masks_logits[0] > model.mask_threshold
```

### Streaming video prediction (PyTorch)

Build the predictor once, open a **session**, then feed it one frame at a time. The session owns the state and counts frames; memory is carried in `state.memory_bank` — no full-video buffer (see [`notebooks/sam2_video_predictor_example.ipynb`](notebooks/sam2_video_predictor_example.ipynb)):

```python
import torch
from sam.build_sam import build_sam2_video_predictor
from sam.prompts import GeometryPrompt

checkpoint = "./checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
device = torch.device("cuda")

predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
session = predictor.start_session()  # the video size comes from the first frame

# Frame 0: add a prompt for object id 1
masklets = session.process(frame_0, prompts=[GeometryPrompt.click(1, (210, 350))])

# Subsequent frames: just feed pixels — tracking continues from the memory bank.
for frame in stream:  # any iterable of (C, H, W) tensors
    masklets = session.process(frame)
    masks = {obj_id: (m.best_mask_logits > 0) for obj_id, m in masklets.items()}
```

`frame` is a `(C, H, W)` float tensor in `[0, 1]` with `H, W` matching `video_hw`, so frames can be read lazily from any decoder (OpenCV, PyAV, a live camera) without materializing the whole clip.

SAM 2 has no concept, so `start_session()` is its only session kind — SAM 3's
`start_concept_session` simply does not exist here, and asking for it raises `AttributeError`.
Neither takes a `video_hw`: the size always comes from the first frame, so `session.state` is
`None` until then. The explicit form stays public for callers that manage frame indices
themselves, and is how you pin the size up front or install a custom memory bank:

```python
from sam.models.sam2_predictor import Sam2VideoPredictorState

# State holds only (H, W) and the memory bank — not the frames.
state = Sam2VideoPredictorState.create(video_hw=(height, width))
results = predictor.forward(
    state=state, frame_idx=0, frame=frame_0,
    prompts=[GeometryPrompt.click(1, (210, 350))],
)
```

### Custom memory bank

The memory bank is a pluggable component. Two implementations ship in the box:

| Class | Strategy |
|---|---|
| `Sam2ObjectMemoryBank` | Default. Unbounded store, faithful to the original SAM 2 paper (no forgetting). |
| `ForgetfulObjectMemoryBank` | Sliding window. Conditional (prompted) memories are kept forever; non-conditional memories outside `[frame - window, frame + window]` are pruned. |

```python
from sam.modeling.memory.forgetful import ForgetfulObjectMemoryBank
from sam.models.sam2_predictor import Sam2VideoPredictorState

bank = ForgetfulObjectMemoryBank(memory_window_size=7)
state = Sam2VideoPredictorState.create(video_hw=(height, width), memory_bank=bank)
```

Write your own policy by subclassing the abstract base
[`ObjectMemoryBank`](sam/modeling/memory/bank.py) (or `Sam2ObjectMemoryBank`) and overriding
`try_add_memories`, `select_memories`, and `prune_memories` — for example, to cap memory
count, score-based eviction, or temporal striding.

### ONNX / TensorRT inference

Run the model with ONNX Runtime instead of PyTorch — for both images and video — via the same generic predictor. This path is **checkpoint-free**: all weights (including the small "glue" tensors) are baked into the export artifacts, so no `.pt` checkpoint is needed at runtime.

**1. Export the graphs** (one-time, device-agnostic, CPU is fine):

```bash
# Single model — fp32 + mixed-precision graphs into outputs/onnx/<model>/...
pixi run -e onnx-export export-onnx-sam2-base-plus

# Or all models / variants at once
pixi run -e onnx-export export-onnx
```

This writes the 5 neural-network blocks (image encoder, prompt encoder, mask decoder, memory attention, memory encoder) as ONNX graphs at opset 18, plus a `weights.npz`. You can also call the exporter directly:

```bash
pixi run -e onnx-export python tools/export_onnx.py \
    --config configs/sam2.1/sam2.1_hiera_b+.yaml \
    --ckpt checkpoints/sam2.1_hiera_base_plus.pt \
    --out-dir onnx_sam2 --opset 18        # add --mixed-precision for a CUDA/CPU-EP graph
```

**2. Run inference.** Build the ONNX-backed predictor and feed frames exactly like the PyTorch streaming API:

```python
import torch
from sam.build_sam import build_sam2_video_predictor_onnx
from sam.onnx.trt_options import TensorRTOptions

predictor = build_sam2_video_predictor_onnx(
    "configs/sam2.1/sam2.1_hiera_b+.yaml",
    onnx_dir="onnx_sam2",          # extracted dir, a .zip, or an http(s) URL to one
    device="cuda",
    use_trt=True,                  # TensorRT EP; set False for the CUDA EP
    trt_opts=TensorRTOptions(),    # engine cache / workspace tuning
)
```

For image-only inference, use `build_sam2_image_predictor_onnx` with the same arguments.

Notes:
- `onnx_dir` accepts an extracted export directory, a `.zip` of one (e.g. a release artifact), or an `http(s)` URL (downloaded and cached; override with `SAM2_ONNX_CACHE`).
- Execution provider tiers map to pixi environments: `onnx` (CPU), `onnx-gpu-cu12` / `onnx-gpu-cu13` (CUDA EP), and `onnx-tensorrt-cu12` / `onnx-tensorrt-cu13` (TensorRT EP + CUDA fallback). The CPU and GPU ONNX runtimes conflict, so each tier is its own environment.
- Precision:
  - **TensorRT** — use the **fp32** export. Run it as fp32, or set `TensorRTOptions(bf16_enable=True)` on Ampere+ (recommended: fp32 range, safe on all blocks). `fp16_enable=True` is allowed but TensorRT applies fp16 only to the 2 heavy blocks (image encoder, memory attention); the rest stay fp32, since fp16 error compounds through the recurrent memory loop and destroys the masklet. Each block logs whether fp16 was applied.
  - **CUDA / CPU EP** — these EPs can't convert precision at runtime, so use a **mixed-precision** export (`--mixed-precision`), which bakes fp16 into the 2 heavy blocks. It is **CUDA/CPU-only and rejected on TensorRT**.

---

## EfficientTAM

[EfficientTAM](https://github.com/yformer/EfficientTAM) (Efficient Track Anything) is integrated as a first-class model (`sam.modeling.efficienttam_base.EfficientTAMBase`) with a plain ViT image encoder. It uses the **same build and predictor APIs** as SAM 2 — point a builder at an EfficientTAM config and checkpoint:

```python
from sam.build_sam import build_sam2_video_predictor

predictor = build_sam2_video_predictor(
    "configs/efficienttam/efficienttam_ti.yaml",
    "./checkpoints/efficienttam_ti.pt",
)
```

Available configs live in [`sam/configs/efficienttam/`](sam/configs/efficienttam) (`s` / `ti` sizes, `512x512`, and the `_1` / `_2` variants). The SAM 2 notebooks have EfficientTAM counterparts that differ only in those two lines: [`efficienttam_image_predictor_example.ipynb`](notebooks/efficienttam_image_predictor_example.ipynb), [`efficienttam_video_predictor_example.ipynb`](notebooks/efficienttam_video_predictor_example.ipynb), [`efficienttam_automatic_mask_generator_example.ipynb`](notebooks/efficienttam_automatic_mask_generator_example.ipynb). EfficientTAM models export to ONNX through the same `tools/export_onnx.py` tasks (`pixi run -e onnx-export export-onnx-efficienttam-ti`, etc.).

---

## SAM 3 — Concept Segmentation

SAM 3 segments by **concept**: a text phrase, a box, or a click — every instance of it, tracked across a video. Build with `build_sam3` (image) or `build_sam3_video_predictor` (streaming video), then `predict(image, ...)` for one image or a session + `process(frame)` per frame for a stream. SAM 3 weights are access-gated — request via [Meta AI](https://ai.meta.com/sam) and download with:

```bash
pixi run download-sam3       # sam3.pt
pixi run download-sam3-1     # sam3.1_multiplex.pt (SAM 3.1)
```

### Image — detect all instances of a concept

```python
import numpy as np
from PIL import Image
from sam.build_sam import build_sam3
from sam.prompts import ConceptPrompt

predictor = build_sam3("configs/sam3/sam3.yaml", "./checkpoints/sam3.pt", device="cuda")

image = np.array(Image.open("images/truck.jpg").convert("RGB"))  # (H, W, 3) uint8

result = predictor.predict(image, ConceptPrompt(text="wheel"))
# result.masks_logits: (N, H, W)  — per-instance mask logits  (> 0 → foreground)
# result.boxes:        (N, 4)     — xyxy pixel bounding boxes
# result.scores:       (N,)       — per-instance confidence scores
# result.presence:     float      — 0..1 concept-in-image presence score

binary_masks = result.masks_logits > 0.0
```

Concept text, `confidence_threshold`, presence, and box / click exemplars are walked through in [`notebooks/sam3_image_predictor_example.ipynb`](notebooks/sam3_image_predictor_example.ipynb).

Prompting one image more than once — sweeping the threshold, trying several phrases, comparing exemplars — is cheaper through a session. The vision encoder is ~70% of a `predict` call and does not depend on the prompt, so the session pays it once (measured: a five-threshold sweep drops from 1862 ms to 490 ms, bit-identical results):

```python
session = predictor.start_image_session(image)   # encodes once

for threshold in (0.3, 0.5, 0.7):
    result = session.process(ConceptPrompt(text="wheel"), confidence_threshold=threshold)
```

`process` is the same verb the video sessions use, and takes the same `geometry=` exemplars as `predict`. There is no concept/interactive split here: an image predictor only detects, and the concept is a per-call argument rather than session-scoped.

The concept can be a `ConceptPrompt`, a bare phrase, or `predictor.PLACEHOLDER` for this lineage's box-only caption — `"visual"` on base SAM 3, `"<text placeholder>"` on the 3.1 multiplex — which is what upstream encodes when geometry arrives with no text. Ask for it by name rather than typing the string: the wrong lineage's caption returns nothing.

```python
result = predictor.predict(image, predictor.PLACEHOLDER,
                           geometry=GeometryPrompt.exemplar_box((x0, y0, x1, y1)))
```

The image predictor owns no tracker, so a `GeometryPrompt.box` / `.click` raises rather than being reinterpreted as an exemplar; it names `exemplar_box` / `exemplar_point` and the video predictor's `start_session()` as the two things you might have meant.


### Streaming video — track a concept across frames

A **session** owns one video: it holds the state and counts frames. There are two kinds, and
they are separate methods because they are separate contracts:

| | `start_session()` | `start_concept_session(concept)` |
|---|---|---|
| Available on | SAM 2 **and** SAM 3 | SAM 3 lineage only |
| Detection | off | runs every frame |
| Objects | only what you prompt | every instance the concept matches |
| `obj_id` | yours, on the prompt | minted by the detector |
| Geometry prompts | seed new objects | **refine** ids already returned |

A concept is declared at birth, because it is session-scoped and immutable.

```python
from sam.build_sam import build_sam3_video_predictor

predictor = build_sam3_video_predictor(
    "configs/sam3/sam3.yaml", "./checkpoints/sam3.pt", device="cuda"
)

session = predictor.start_concept_session("wheel")  # size comes from the first frame

for frame in frames:                                # frames: (H, W, 3) uint8 arrays
    masklets = session.process(frame)               # {obj_id: MaskletResult, ...}
    masks = {obj_id: (m.masks_logits > 0) for obj_id, m in masklets.items()}
```

`concept` takes a phrase or a `ConceptPrompt`. Sessions are independent: hold several over one
loaded model and interleave them freely, each counting its own frames.

The explicit form (`Sam3VideoPredictorState` + `forward(state, frame_idx, frame)`) stays public
for callers that manage frame indices themselves — a session is exactly one `forward` call per
frame, so the two are interchangeable frame for frame, and `session.state` is the same state
object:

```python
from sam.prompts import ConceptPrompt
from sam.models.sam3_predictor import Sam3VideoPredictorState

# State holds memory bank + tracklet bookkeeping — not the frames.
state = Sam3VideoPredictorState(video_hw=(height, width))
predictor.set_concept(state, ConceptPrompt(text="wheel"))

for frame_idx, frame in enumerate(frames):
    masklets = predictor.forward(state, frame_idx, frame)
```

Per-object memory is managed by the internal `ForgetfulObjectMemoryBank` (non-conditional frames outside a 7-frame window are pruned), so VRAM stays constant regardless of clip length.

### SAM 3.1 (multiplex) — up to 16 objects in one joint forward pass

SAM 3.1 packs all tracked objects into one joint forward via `build_sam3_multiplex` (image) and `build_sam3_multiplex_video_predictor` (video). The calling API is **identical** to base SAM 3; mux/demux is internal to the tracker.

**Scope:** the multiplex video predictor tracks up to K=16 objects jointly and supports the same prompts as the base predictor — concept text, seed-frame and mid-stream clicks, box prompts, and mid-stream instance spawn (see the sections below). Mask prompts raise `NotImplementedError`: neither checkpoint ships `mask_encoder` weights. VRAM stays bounded because the joint bucket-space spatial memory is threaded internally (not via the per-object `ObjectMemoryBank`) and non-conditional frames outside the 7-frame forgetful window are pruned.

```python
from sam.build_sam import build_sam3_multiplex, build_sam3_multiplex_video_predictor
from sam.prompts import ConceptPrompt

# Image
image_predictor = build_sam3_multiplex(
    "configs/sam3/sam3.1.yaml", "./checkpoints/sam3.1_multiplex.pt", device="cuda"
)
result = image_predictor.predict(image, ConceptPrompt(text="wheel"))

# Video: same session API as base SAM 3
video_predictor = build_sam3_multiplex_video_predictor(
    "configs/sam3/sam3.1.yaml", "./checkpoints/sam3.1_multiplex.pt", device="cuda"
)
wheel_session = video_predictor.start_concept_session("wheel")
for frame in frames:
    masklets = wheel_session.process(frame)
    masks = {obj_id: (m.masks_logits > 0) for obj_id, m in masklets.items()}
```

### Geometry prompts — selecting an object vs steering a search

A `GeometryPrompt` is one of two things, and the constructor you pick says which:

| | selects ONE object | steers the concept search |
|---|---|---|
| point | `GeometryPrompt.click(obj_id, (x, y))` | `GeometryPrompt.exemplar_point((x, y))` |
| box | `GeometryPrompt.box(obj_id, (x0, y0, x1, y1))` | `GeometryPrompt.exemplar_box((x0, y0, x1, y1))` |
| mask | `GeometryPrompt.mask(obj_id, mask)` | — no detector mask slot in either checkpoint |
| goes to | the tracker's prompt encoder (SAM 2 semantics) | the detector's geometric slot (SAM 3 only) |
| needs | nothing else | a concept: a phrase or `PLACEHOLDER` |
| ids | the `obj_id` you passed | minted by detection |

The `concept_*` constructors take **no** `obj_id`: they name nothing, so the ids they produce
are the detector's to hand out. `label=0` makes either one a counter-example — "everything
matching the concept EXCEPT this". Constructors take plain tuples, lists, or arrays in pixel
coordinates; the generic `GeometryPrompt(...)` form stays available (with
`route=PromptRoute.DETECTOR` for the detector slot, `is_normalized=True` for `[0, 1]`
coordinates), though the constructors set the route for you.

On an **image** only the `concept_*` family applies — there is no tracker to select with, so a
`click` or `box` raises and names the alternative. A negative box drops the enclosed detection's
score sharply (0.95 → 0.66 in the notebook's example, below a 0.7 threshold); a negative point
barely moves it, having no extent to say which instance you meant.

```python
from sam.prompts import ConceptPrompt, GeometryPrompt

box = GeometryPrompt.exemplar_box((300, 150, 470, 420))
result = image_predictor.predict(image, ConceptPrompt("person"), geometry=box)

# "everything matching the concept EXCEPT this one"
neg = GeometryPrompt.exemplar_box((300, 150, 470, 420), label=0)
```

On **video**, prompts are passed per frame, and they take one of two routes — the same split
upstream makes.

**Interactive VOS** is the default, and is the SAM 2 interface unchanged: **points**, a
**box**, or a **mask** go to the tracker's prompt encoder and seed exactly one object under the
`obj_id` you choose. A box is encoded as its two corner points, exactly as
`Sam2VideoPredictor` does. No detection runs, nothing else can appear, and the object tracks
until the clip ends.

**Detection-driven** is SAM 3 only and must be asked for: `GeometryPrompt.exemplar_box` and
`GeometryPrompt.exemplar_point` send their geometry to the detector's geometric slot, where it
biases that frame's detection. Detection then runs on every frame, so every instance the
concept matches is tracked, not only the one you marked — which is why this route needs the
session's concept. Upstream adopts a placeholder caption implicitly the moment a box arrives
without text; here it is an argument you can read.

The route belongs to the prompt, and the constructor sets it: `click` / `box` / `mask` mark one
object for the tracker, `exemplar_point` / `exemplar_box` describe what the search should look
for. You never name `PromptRoute` yourself.

> **`exemplar_point` on video is this fork's, not upstream's.** The detector's point slot is
> trained and ships in both checkpoints, and the image path has always used it, but upstream's
> video inference hardcodes `point_embeddings=None`, so there is no golden pinning it. Boxes
> are the parity-tested detector route.

```python
# interactive VOS — one object per prompt, no concept
interactive_session = video_predictor.start_session()
masklets = interactive_session.process(frames[0], prompts=[
    GeometryPrompt.box(1, (300, 150, 470, 420)),
    GeometryPrompt.click(2, (640, 300)),           # label=0 for a negative click
    GeometryPrompt.mask(3, first_frame_mask),      # (H, W) bool or logits, base only
])
for frame in frames[1:]:
    masklets = interactive_session.process(frame)

# detection hint — upstream's box-only mode, both choices explicit
hint_session = video_predictor.start_concept_session(video_predictor.PLACEHOLDER)
masklets = hint_session.process(frames[0], prompts=[
    GeometryPrompt.exemplar_box((300, 150, 470, 420)),      # label=0: "everything except this"
    GeometryPrompt.exemplar_point((640, 300)),              # the point form, same contract
])
```

Sending a `exemplar_box` into a `start_session()` session raises `ValueError` rather than picking
a caption for you. Making the interactive session adopt `PLACEHOLDER` on its own would silently
switch detection on for every frame under the box-only caption — so instead of the one object you
boxed you would get every instance that caption matches, for the rest of the clip. Ask for it by
name (`start_concept_session(predictor.PLACEHOLDER)`) or use `GeometryPrompt.box(obj_id, xyxy)`,
which tracks only what you boxed.

Objects seeded through the tracker are force-confirmed and exempt from the hotstart kill,
matching upstream's `add_tracker_new_points`: with no concept set nothing can ever re-match
them, so without the exemption the lifecycle would purge them after 8 unmatched frames.

Mask prompts are base-lineage only, on the tracker's own `sam_prompt_encoder.mask_downscaling`
weights, and there is no `concept_mask`: a mask on the detector route raises
`NotImplementedError` when you build it, because `mask_encoder` has no weights in either
checkpoint.

### Adding objects mid-stream

Prompts are not restricted to frame 0 — passing a `GeometryPrompt` on any later frame adds
that object to the running state, alongside whatever the detector spawns on its own. An object
that leaves the frame and comes back keeps its original id (tracklet re-identification;
upstream has no re-association and would issue a new id).

### Output policy — when a tracked object becomes visible

Every alive tracklet is propagated and keeps its memory regardless of policy; `emit` only
selects what a `process` / `forward` call returns, and empty masks are always dropped.

```python
from sam.results import Emit

video_predictor.emit = Emit.CONFIRMED  # default: matched by a detection 3 frames running
video_predictor.emit = Emit.VISIBLE    # alive and not suppressed — what upstream's goldens show
video_predictor.emit = Emit.ALIVE      # every tracklet, suppressed ones included

for obj_id, m in session.process(frame).items():
    m.tracklet_state   # each result carries its lifecycle state, for your own policy
```

`CONFIRMED` is causal: a one-frame detector false positive never reaches the caller, at the
cost of an object's first frames. Upstream decides visibility non-causally (it buffers 15
frames of output), which a streaming predictor cannot reproduce.

### Several concepts at once

Use one session per concept and merge with an id offset. Upstream's own multi-concept path does
exactly this — it loops the phrases, propagates the whole video per phrase, and offsets the
ids — so concepts never share association or memory:

```python
sessions = {c: video_predictor.start_concept_session(c) for c in ("person", "dog")}

for frame in frames:
    per_concept = {c: sess.process(frame) for c, sess in sessions.items()}
```

Each session numbers its objects from its own counter, so offset the ids yourself before
merging the dicts (upstream offsets by phrase index).

A worked end-to-end walkthrough of all of the above lives in
[`notebooks/sam3_video_predictor_example.ipynb`](notebooks/sam3_video_predictor_example.ipynb).

---

## EfficientSAM3 — Lightweight Concept Segmentation

EfficientSAM3 is a distilled, mobile-friendly version of SAM 3 (same DETR detector
and text-conditioned concept API, lighter vision backbone and text encoder).
It identifies and segments all instances of a text concept in images and across
streaming video (per-object and multiplex tracking) — see the [variant matrix](#variant-matrix).

### Download

EfficientSAM3 weights are **publicly available** (no HF login required):

```bash
pixi run download-efficientsam3        # RepViT-M1.1 variant (efficientsam3_repvit.pt)
```

Or download manually:

```python
from sam.build_sam import build_efficientsam3_hf
model = build_efficientsam3_hf("repvit", device="cuda")  # downloads on first call
```

### Image — detect all instances of a concept

```python
import numpy as np
from PIL import Image
from sam.build_sam import build_efficientsam3
from sam.prompts import ConceptPrompt

model = build_efficientsam3(
    "configs/efficientsam3/efficientsam3_repvit.yaml",
    "./checkpoints/efficientsam3_repvit.pt",
    device="cuda",
)

image = np.array(Image.open("dog_person.jpeg").convert("RGB"))  # (H, W, 3) uint8

result = model.predict(image, ConceptPrompt(text="dog"), confidence_threshold=0.1)
# result.masks_logits: (N, H, W)  — per-instance logits  (> 0 → foreground)
# result.boxes:        (N, 4)     — xyxy pixel boxes
# result.scores:       (N,)       — per-instance confidence

binary_masks = result.masks_logits > 0.0
```

### Variant matrix

All four published EfficientSAM3 families are integrated (image + streaming video). Parity is
measured against **efficientsam3's own reference** (its `build_efficientsam3_*` builders / the
`stage1_sam3.1` multiplex code), not facebook SAM 3. Read the ❌ note under the table before
using a distilled-vision variant for video.

| Family | Vision backbone | Text encoder | Tracker | Parity vs efficientsam3 reference |
|---|---|---|---|---|
| **EfficientSAM3** (image) | RepViT-M1.1 · TinyViT-11M · EfficientViT-B1 | MobileCLIP-S0 (ctx16) | — | ✅ mask IoU ≥ 0.99 |
| **EfficientSAM3** (video) | RepViT-M1.1 · TinyViT-11M · EfficientViT-B1 | PE text (ctx32) | base, 309 + geometry | detection ✅; propagation ❌ not supported |
| **SAM3-LiteText** (video) | PE-ViT (SAM 3) | MobileCLIP-S0 (ctx16) | base, 309 | ✅ streaming IoU ≥ 0.99 |
| **SAM3.1-LiteText** (video) | PE-ViT (SAM 3) | MobileCLIP-S0 (ctx16) | multiplex, 457 | ✅ min 0.994 / mean 0.998 |
| **EfficientSAM3.1** (video) | RepViT-M1.1 (stage1) | MobileCLIP-S0 (ctx16) | multiplex, 457 | detection ✅; propagation ❌ not supported |

❌ **The distilled-vision variants are image/detection models — do not use them for video
tracking.** Upstream released the Stage 1 encoder distillation but never Stage 2 (memory-bank
alignment, trained on SA-V — still unchecked on its roadmap), and the published video checkpoints
are Stage 1 lineage, so the tracker propagates features it was never trained on. Per-frame
**detection** reproduces the reference exactly (≥ 0.99); **propagation** degrades from frame 1 —
1–4% for the base lineage, down to min 0.7412 for EfficientSAM3.1 RepViT-M. Measured against
efficientsam3's own native reference with identical strict-loaded weights, so this is upstream's
gap, not an integration bug, and no change here can close it. Tracked as `xfail(strict=True)` —
see [`tests/parity/reference_efficientsam3/`](tests/parity/reference_efficientsam3/).

For streaming video use the PE-vision lineage: base `sam3.pt` / `sam3.1_multiplex.pt`, or the
**LiteText** variants above — they swap only the text tower and keep PE vision, so they track at
full fidelity while still shrinking the text encoder.

### Image benchmark — EfficientSAM3 vs SAM 3 (image only)

RTX 3080 Ti, `dog_person.jpeg` (2048×1365), prompt `"dog"`, 30 iters (`tools/benchmark_efficientsam3.py`). **autocast (bfloat16)** — the deployment path (works for every model; base SAM 3 is bf16-only):

| Model | Vision | Prompt+detect | **End-to-end** |
|---|---|---|---|
| EfficientSAM3 · EfficientViT-B1 | 25.3 ms · 39.5 FPS | 40.5 ms | **65.1 ms · 15.4 FPS** |
| EfficientSAM3 · RepViT-M1.1 | 27.4 ms · 36.6 FPS | 40.2 ms | **67.7 ms · 14.8 FPS** |
| EfficientSAM3 · TinyViT-11M | 29.3 ms · 34.2 FPS | 40.3 ms | **69.4 ms · 14.4 FPS** |
| **SAM 3 · PE ViT-H** | **147.2 ms · 6.8 FPS** | 52.0 ms | **200.1 ms · 5.0 FPS** |

fp32 (TF32) — EfficientSAM3 only (base SAM 3 fp32 is unsupported — perflib hardcodes bf16): EfficientViT 99.4 ms · 10.1 FPS · RepViT 101.5 ms · 9.8 FPS · TinyViT 104.9 ms · 9.5 FPS end-to-end. (Matches upstream `_bench.py`: fp32 vision ~31.5 ms.)

- **Distilled vision encoder is ~5–6× faster** than SAM 3's PE ViT-H (25–29 ms vs 147 ms) — the EfficientSAM3 win.
- **~3× faster end-to-end** (15 vs 5 FPS). The reused SAM 3 DETR detector (prompt phase, ~40–52 ms) now dominates (~⅔ of e2e) — the next speedup lever.
- These are **image** numbers; streaming-video parity/throughput is separate (see the variant matrix + `tests/parity/reference_efficientsam3/`).

Reproduce: `pixi run python tools/benchmark_efficientsam3.py --ckpt <ckpt> --config <cfg>` (add `--sam3` for the base SAM 3 model).

---

## Benchmarks

Per-frame **streaming** throughput of `Sam2VideoPredictor` across every model variant and backend. Measured with `tools/benchmark_torch.py` (PyTorch) and `tools/benchmark_onnx.py` (ONNX Runtime + TensorRT), which share one timing loop (`tools/bench_utils.py`): one point prompt on frame 0, propagate the masklet, time each `forward()` over 200 frames (5 warm-up frames discarded), forgetful memory bank (window 7) stored on GPU, `apply_postprocessing=True`.

**Hardware / stack:** NVIDIA RTX 3080 Ti (12 GB, sm86). PyTorch rows: torch 2.11.0+cu128. ONNX-TRT rows: TensorRT 10.16.1 + onnxruntime-gpu 1.27.0 (CUDA 13). Video: `notebooks/videos/bedroom.mp4` (960×540), single object.

Throughput (FPS, higher is better; **bold** = fastest per row):

| Model | Res | torch bf16 | torch fp16 | ONNX-TRT fp16 | ONNX-TRT bf16 |
|---|---|--:|--:|--:|--:|
| SAM 2.1 tiny | 1024 | 19.3 | 19.7 | **26.1** | 19.7 |
| SAM 2.1 small | 1024 | 18.3 | 19.1 | **30.9** | 18.4 |
| SAM 2.1 base+ | 1024 | 13.8 | 15.8 | **30.2** | 20.2 |
| SAM 2.1 large | 1024 | 9.0 | 9.9 | **14.3** | 12.6 |
| EfficientTAM-Ti | 1024 | 24.1 | 24.2 | **45.8** | 23.2 |
| EfficientTAM-Ti/1 | 1024 | 28.4 | 29.8 | **33.5** | 30.6 |
| EfficientTAM-Ti/2 | 1024 | 30.6 | 31.7 | **63.1** | 32.0 |
| EfficientTAM-Ti 512² | 512 | 45.2 | 46.0 | **117.2** | 95.1 |
| EfficientTAM-S | 1024 | 19.9 | 21.6 | **42.8** | 26.2 |
| EfficientTAM-S/1 | 1024 | 22.3 | 25.9 | **43.9** | 20.7 |
| EfficientTAM-S/2 | 1024 | 26.6 | 27.4 | **53.1** | 38.4 |
| EfficientTAM-S 512² | 512 | 46.2 | 46.6 | 76.9 | **77.5** |

Notes:
- **ONNX-TRT fp16 is fastest almost everywhere** (≈1.5–2.6× over torch bf16). TRT fp16 applies to the 2 heavy blocks (image encoder, memory attention); the rest stay fp32.
- **bf16 < fp16 on this GPU**: TensorRT's bf16 autotune is less optimized than fp16 here, and torch fp16 ≈ bf16 (Ampere). On newer GPUs the ordering can differ.
- Numbers are **streaming** (per-frame decode + transforms + mask postprocess included), so they sit below encoder-only / offline throughput.
- `_2` = efficient memory (2×2 pooled cross-attention); `512²` halves input resolution. Both trade accuracy for speed — see [EfficientTAM](#efficienttam).

Reproduce (any model / config / checkpoint):

```bash
# PyTorch (default env) — --precision {auto,bf16,fp16,fp32}
pixi run python tools/benchmark_torch.py \
    --model-cfg configs/efficienttam/efficienttam_s_2.yaml \
    --checkpoint checkpoints/efficienttam_s_2.pt --precision fp16

# ONNX + TensorRT — export once, then benchmark (--trt-fp16 / --trt-bf16)
pixi run -e onnx-export export-onnx-efficienttam-s-2
pixi run -e onnx-tensorrt-cu13 python tools/benchmark_onnx.py \
    --onnx-dir outputs/onnx/efficienttam_s_2/opset18 \
    --model-cfg configs/efficienttam/efficienttam_s_2.yaml --trt-fp16
```

---

## License

The SAM 2 model, code, and checkpoints are released by Meta AI under [Apache 2.0](./LICENSE). This fork is distributed under the same license. EfficientTAM is integrated from its [upstream repository](https://github.com/yformer/EfficientTAM); see that project for its license and terms. Third-party code includes a GPU connected-components algorithm adapted from [`cc_torch`](https://github.com/zsef123/Connected_components_PyTorch) ([license](./LICENSE_cctorch)). SAM 3 components (and code derived from or importing them) are released by Meta AI under the [SAM License](./LICENSE_sam) — using them binds you to that license's terms (e.g. acceptable-use and trade-control restrictions); the permissive Apache-2.0 / BSD-3 parts remain independently usable. SAM 3 weights are gated and never vendored.

| Component | License |
|---|---|
| SAM 2 | Apache-2.0 |
| EfficientTAM | Apache-2.0 |
| SAM 3 (+ derivatives) | SAM License |
| RepViT backbone (`sam/modeling/encoders/repvit.py`) | Apache-2.0 ([LICENSE_repvit](./LICENSE_repvit)) |
| TinyViT backbone (`sam/modeling/encoders/tiny_vit.py`) | MIT ([LICENSE_tinyvit](./LICENSE_tinyvit)) |
| EfficientViT backbone (`sam/modeling/encoders/efficientvit/`) | Apache-2.0 ([LICENSE_apache2](./LICENSE_apache2)) |
| MobileCLIP text transformer (`sam/modeling/text/mobile_clip.py`) | Apple ML Research License — **non-commercial** ([LICENSE_mobileclip](./LICENSE_mobileclip)) |
| CUDA connected-components ext (`csrc`) | BSD-3-Clause |
| This fork's code | Apache-2.0 |

---

## Citation

This fork builds directly on [SAM 2](https://github.com/facebookresearch/sam2). If you use it in your research, please cite the original SAM 2 paper:

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}
```

If you use the EfficientTAM models, please also cite [EfficientTAM](https://github.com/yformer/EfficientTAM):

```bibtex
@article{xiong2024efficienttam,
  title={Efficient Track Anything},
  author={Yunyang Xiong, Chong Zhou, Xiaoyu Xiang, Lemeng Wu, Chenchen Zhu, Zechun Liu, Saksham Suri, Balakrishnan Varadarajan, Ramya Akula, Forrest Iandola, Raghuraman Krishnamoorthi, Bilge Soran, Vikas Chandra},
  journal={preprint arXiv:2411.18933},
  year={2024}
}
```