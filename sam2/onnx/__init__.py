# ONNX / TensorRT drop-in replacements for the 5 SAM2 neural-network blocks.
#
# Each wrapper in this package mirrors the forward() signature of the torch
# module it replaces, so SAM2GenericONNX can swap them in without changing any
# of the orchestration in sam2.modeling.sam2_generic.
from sam2.onnx.trt_options import TensorRTOptions
from sam2.onnx.ort_block import OrtBlock

__all__ = ["TensorRTOptions", "OrtBlock"]
