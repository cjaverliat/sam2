## SAM 2 release notes

### 06/24/2026 -- v1.2.0: ONNX/TensorRT export hardening, benchmark throughput controls, and torch-agnostic builds

- **Features**
  * Callers can now force the autocast half precision (`_half_dtype`, fp16 or bf16) instead of relying on the capability-based default.
  * The torch benchmark gains throughput knobs (`--cudnn-benchmark`, `--channels-last`) and a bounded "forgetful" memory bank (`--forgetful` / `--memory-window`).
  * ONNX export now rejects mixed-precision exports on TensorRT, where they are unsupported.
- **Performance**
  * The ONNX image encoder caches its constant position encodings instead of re-binding ~87 MiB of frame-invariant tensors every frame, saving ~22 ms/frame on the TensorRT/CUDA execution providers. Masks are unchanged.
- **Bug Fixes**
  * Editable `sam2` metadata is kept torch-agnostic, preventing unsatisfiable solves and silent CPU fallbacks when a pixi environment runs a different torch line than the one used at build time.
  * ONNX export fails loudly on TensorRT bf16 with `onnxruntime < 1.23` rather than producing a broken model.
- **Breaking Changes**
  * The ONNX "fp16" export was renamed to "mixed-precision". Update any export invocations that referenced the old fp16 naming.
- **Developer Experience**
  * The torch and ONNX benchmarks now share a single CLI and timing loop.
  * CI pins pixi to `v0.59.0` and runs frozen against the committed lock for reproducible builds.
  * Dropped unused mask-decoder conv params and refreshed `pixi.lock`.
- **Documentation**
  * Added a benchmarks section to the README with a throughput table across the SAM 2.1 and EfficientTAM variants.

### 12/11/2024 -- full model compilation for a major VOS speedup and a new `SAM2VideoPredictor` to better handle multi-object tracking

- We now support `torch.compile` of the entire SAM 2 model on videos, which can be turned on by setting `vos_optimized=True` in `build_sam2_video_predictor` (it uses the new `SAM2VideoPredictorVOS` predictor class in `sam2/sam2_video_predictor.py`).
  * Compared to the previous setting (which only compiles the image encoder backbone), the new full model compilation gives a major speedup in inference FPS.
  * In the VOS prediction script `tools/vos_inference.py`, you can specify this option in `tools/vos_inference.py` via the `--use_vos_optimized_video_predictor` flag.
  * Note that turning on this flag might introduce a small variance in the predictions due to numerical differences caused by `torch.compile` of the full model.
  * **PyTorch 2.5.1 is the minimum version for full support of this feature**. (Earlier PyTorch versions might run into compilation errors in some cases.) Therefore, we have updated the minimum PyTorch version to 2.5.1 accordingly in the installation scripts.
- We also update the implementation of the `SAM2VideoPredictor` class for the SAM 2 video prediction in `sam2/sam2_video_predictor.py`, which allows for independent per-object inference. Specifically, in the new `SAM2VideoPredictor`:
  * Now **we handle the inference of each object independently** (as if we are opening a separate session for each object) while sharing their backbone features.
  * This change allows us to relax the assumption of prompting for multi-object tracking. Previously (due to the batching behavior in inference), if a video frame receives clicks for only a subset of objects, the rest of the (non-prompted) objects are assumed to be non-existent in this frame (i.e., in such frames, the user is telling SAM 2 that the rest of the objects don't appear). Now, if a frame receives clicks for only a subset of objects, we do not make any assumptions about the remaining (non-prompted) objects (i.e., now each object is handled independently and is not affected by how other objects are prompted). As a result, **we allow adding new objects after tracking starts** after this change (which was previously a restriction on usage).
  * We believe that the new version is a more natural inference behavior and therefore switched to it as the default behavior. The previous implementation of `SAM2VideoPredictor` is backed up to in `sam2/sam2_video_predictor_legacy.py`. All the VOS inference results using `tools/vos_inference.py` should remain the same after this change to the `SAM2VideoPredictor` class.

### 09/30/2024 -- SAM 2.1 Developer Suite (new checkpoints, training code, web demo) is released

- A new suite of improved model checkpoints (denoted as **SAM 2.1**) are released. See [Model Description](#model-description) for details.
  * To use the new SAM 2.1 checkpoints, you need the latest model code from this repo. If you have installed an earlier version of this repo, please first uninstall the previous version via `pip uninstall SAM-2`, pull the latest code from this repo (with `git pull`), and then reinstall the repo following [Installation](#installation) below.
- The training (and fine-tuning) code has been released. See [`training/README.md`](training/README.md) on how to get started.
- The frontend + backend code for the SAM 2 web demo has been released. See [`demo/README.md`](demo/README.md) for details.

### 07/29/2024 -- SAM 2 is released

- We release Segment Anything Model 2 (SAM 2), a foundation model towards solving promptable visual segmentation in images and videos.
  * SAM 2 code: https://github.com/facebookresearch/sam2
  * SAM 2 demo: https://sam2.metademolab.com/
  * SAM 2 paper: https://arxiv.org/abs/2408.00714
