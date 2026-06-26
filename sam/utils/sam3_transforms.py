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
