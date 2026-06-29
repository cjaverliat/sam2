# EfficientSAM3 RepViT Golden Fixtures

## Provenance

**Upstream Repository:** https://github.com/SimonZeng7108/efficientsam3

**Upstream Commit:** `d063e00b1837f8dd285eb517d2dd40faabc34555` (short: `d063e00`)
- Branch: `main`
- Date: 2026-06-24

**Checkpoint:** `efficientsam3_ft/efficientsam3_repvit.pt`
- RepViT-M1.1 vision backbone
- MobileCLIP-S0 language backbone
- Context length: 16 instances

**Image:** `sam3/assets/dog_person.jpeg` from upstream repo
- Resolution: 2048 × 1365
- Prompts: "dog", "person" (text-based semantic segmentation)

## How Captured

Built the upstream model via:
```python
build_efficientsam3_image_model(
    checkpoint_path="efficientsam3_ft/efficientsam3_repvit.pt",
    backbone_type="repvit",
    model_name="m1_1",
    text_encoder_type="MobileCLIP-S0",
    text_encoder_context_length=16,
    load_from_HF=False
)
```

Then ran inference with:
```python
Sam3Processor(model, confidence_threshold=0.1).set_image(dog_person.jpeg)
processor.set_text_prompt("dog")   # First prompt
processor.set_text_prompt("person") # Second prompt
```

**Notes:**
- Ran in float32 precision (no autocast)
- Context length forced to 16: upstream `build_efficientsam3_image_model` builds the student text encoder at context_length=77, but the post-load truncation occurs after the load. The strict load would fail without this override.

## Generation Details

**Inference Configuration:**
- Confidence threshold: 0.1
- Precision: float32 (no autocast)
- Device: CUDA (RTX 3080 Ti)
- PyTorch: 2.11.0+cu128

**Outputs:**
- `efficientsam3_repvit_summary.json`: Aggregated metrics per prompt
  - `masks.sha1`: Truncated SHA1 hash of mask data for integrity
  - `masks.sum`: Sum of sigmoid probabilities across all instances
  - `masks.mean`: Mean probability value
  - `num_instances`: Total detections per prompt
  - `boxes`: Bounding boxes (shape [N, 4])
  - `scores`: Confidence scores per instance
  
- `efficientsam3_repvit_masks_dog.npz`: Mask tensors for "dog" prompt
- `efficientsam3_repvit_masks_person.npz`: Mask tensors for "person" prompt
  - Shape: [num_instances, 1, height, width]
  - Format: Float32 (sigmoid probabilities)
  - Resolution: Original (1365 × 2048)

## Purpose

These fixtures serve as the oracle for A6's parity test, verifying that the SAM2 port of EfficientSAM3 produces identical outputs (masks, boxes, scores) to the upstream reference implementation.
