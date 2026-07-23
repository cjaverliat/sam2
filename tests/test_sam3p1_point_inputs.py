# SPDX-License-Identifier: LicenseRef-SAM
import torch

from sam.models.sam3_predictor import _build_mux_point_inputs
from sam.prompts import GeometryPrompt


def test_batches_and_scales_points():
    # two objects, ragged point counts (2 and 1) -> padded to P=2 with label -1.
    p0 = GeometryPrompt(
        obj_id=1,
        points_coords=torch.tensor([[480.0, 270.0], [0.0, 0.0]]),   # centre + origin
        points_labels=torch.tensor([1, 0]),
    )
    p1 = GeometryPrompt(
        obj_id=2,
        points_coords=torch.tensor([[960.0, 540.0]]),               # bottom-right corner
        points_labels=torch.tensor([1]),
    )
    inp, ids = _build_mux_point_inputs(
        [p0, p1], video_hw=(540, 960), image_size=1008, device=torch.device("cpu")
    )
    assert ids == [1, 2]
    assert inp["point_coords"].shape == (2, 2, 2)
    assert inp["point_labels"].shape == (2, 2)
    # obj0 centre -> half the 1008 grid; obj1 corner -> full grid.
    assert torch.allclose(inp["point_coords"][0, 0], torch.tensor([504.0, 504.0]))
    assert torch.allclose(inp["point_coords"][1, 0], torch.tensor([1008.0, 1008.0]))
    # obj1 second slot is padding.
    assert inp["point_labels"][1, 1].item() == -1
    assert inp["point_labels"][0].tolist() == [1, 0]
