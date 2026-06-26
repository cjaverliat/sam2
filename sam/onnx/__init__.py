# SPDX-License-Identifier: Apache-2.0
# ONNX / TensorRT drop-in replacements for the 5 SAM2 neural-network blocks.
#
# Each wrapper in this package mirrors the forward() signature of the torch
# module it replaces, so SAM2GenericONNX can swap them in without changing any
# of the orchestration in sam.models.sam2_predictor.
from sam.onnx.trt_options import TensorRTOptions
from sam.onnx.ort_block import OrtBlock

__all__ = ["TensorRTOptions", "OrtBlock"]
