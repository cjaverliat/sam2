# SPDX-License-Identifier: LicenseRef-SAM
"""SAM 3 image preprocessing utilities.

Shared between the parity tests (Phase 1, Task 2) and the SAM 3 predictor
(Phase 1, Task 8) so both use exactly the same on-device preprocessing path.
"""
from __future__ import annotations


def preprocess_to_1008(image_rgb, device: str = "cuda"):
    """Replicate ``Sam3Processor``'s exact preprocessing of a (H,W,3) uint8 NumPy
    array into the ``(1,3,1008,1008)`` float32 tensor that the PE-ViTDet backbone
    consumes.

    Pipeline (mirrors ``sam3/model/sam3_image_processor.py::Sam3Processor``):

    1. Convert to a ``(3,H,W)`` uint8 CHW tensor via PIL and move to *device*.
    2. Resize to ``(1008, 1008)`` using torchvision v2 default (bilinear + antialias).
    3. Scale pixel values from ``[0,255]`` to ``[0.0,1.0]``.
    4. Normalise with mean=0.5, std=0.5 → ``[-1.0,1.0]``.

    **Why on-device resize is required**: upstream ``set_image`` calls
    ``to_image(image).to(device)`` *before* ``transform(...)``, so the resize
    executes on the GPU.  Performing the equivalent resize on CPU produces small
    numerical differences (floating-point order-of-operations in bilinear
    interpolation differs by hardware) that the deep ViT amplifies to
    ``max|Δ| ≈ 0.97`` on the encoder output — well outside the ``atol=1e-2``
    parity gate.  Always call this function with ``device="cuda"`` when
    reproducing golden fixtures.

    Args:
        image_rgb: ``(H, W, 3)`` uint8 NumPy array (RGB).
        device: Target device string passed to ``tensor.to()``.  Defaults to
            ``"cuda"``.

    Returns:
        ``torch.Tensor`` of shape ``(1, 3, 1008, 1008)``, dtype ``float32``,
        on *device*.
    """
    import torch
    from PIL import Image
    from torchvision.transforms import v2

    transform = v2.Compose(
        [
            v2.Resize(size=(1008, 1008)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    img = v2.functional.to_image(Image.fromarray(image_rgb)).to(device)  # (3,H,W) uint8
    return transform(img).unsqueeze(0)


def preprocess_to_1008_video(image_rgb, device: str = "cuda", image_size: int = 1008):
    """Replicate the SAM 3.1 VIDEO demo's exact per-frame preprocessing.

    The sam3.1 multiplex video golden (``video_sam31.npz``) was captured by saving each
    ``(H,W,3)`` uint8 frame as a lossless PNG and loading it through the upstream image-folder
    video loader (``sam3/model/io_utils.py::_load_img_as_tensor`` +
    ``load_video_frames_from_image_folder``), which is **NOT** the same pipeline as the image
    :func:`preprocess_to_1008` (``Sam3Processor``-mirroring) path:

    * **resize backend** -- the video loader resizes the *PIL* image with
      ``torchvision.transforms.functional.resize`` (the v1 functional API: a CPU, uint8-domain
      PIL bilinear resample), whereas :func:`preprocess_to_1008` moves a uint8 CHW tensor to the
      GPU first and resizes with ``torchvision.transforms.v2.Resize`` (a tensor-domain
      bilinear+antialias on-device). These two bilinear implementations are NOT numerically
      identical (different antialiasing kernel + PIL uint8 rounding), giving a broad
      ``enc_feat`` median delta ~0.037 -- harmless for a single detection (M2: box exact) but a
      *systematic* bias that the video tracker propagates across frames via memory.
    * **dtype** -- the loader stores frames as ``float16`` (``to_tensor`` -> float32 ``[0,1]``
      cast into a float16 buffer) and normalises (``-= mean; /= std``) in float16; the
      ``mean = std = 0.5`` map to ``[-1, 1]`` matches :func:`preprocess_to_1008`.

    This function reproduces the loader EXACTLY (same ``TF.resize`` call on the same PIL image)
    so the encoder input matches the golden's regime. :func:`preprocess_to_1008` is intentionally
    left UNCHANGED (the M2 image parity depends on it). Designed to be called inside the
    ``autocast(bf16)`` region the predictor runs in (the float16 storage is washed out by the
    bf16 cast at the first conv; the resize backend is the load-bearing difference).

    Args:
        image_rgb: ``(H, W, 3)`` uint8 NumPy array (RGB) -- one already-resized video frame
            (e.g. ``video_sam31['video_frames_rgb'][i]``, the 288x512 frame).
        device: target device string. Defaults to ``"cuda"``.
        image_size: the square model resolution (1008 for the PE-ViTDet backbone).

    Returns:
        ``torch.Tensor`` of shape ``(1, 3, image_size, image_size)``, dtype ``float16``, on
        *device* -- normalised to ``[-1, 1]``, byte-faithful to the upstream video loader.
    """
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image

    img = Image.fromarray(image_rgb).convert("RGB")
    img = TF.resize(img, size=(image_size, image_size))  # PIL bilinear (v1 functional)
    img = TF.to_tensor(img)                               # (3,H,W) float32 in [0,1]
    img = img.to(dtype=torch.float16)                     # match the loader's float16 buffer
    img = img.unsqueeze(0).to(device)                     # (1,3,image_size,image_size) f16
    mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float16, device=img.device).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float16, device=img.device).view(1, 3, 1, 1)
    img = img - mean
    img = img / std
    return img
