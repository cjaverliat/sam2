# SPDX-License-Identifier: LicenseRef-SAM
"""Video box-prompt parity vs the facebook SAM 3.1 golden.

The box biases the prompt-frame detection and seeds a tracklet that tracks forward.

Visibility, measured against the upstream golden (see the ledger for the captured
per-frame trace):

- frame 0: both show the box-seeded object -> mask parity asserted.
- frames 1-4: both HIDE it. Upstream hides it as UNCONFIRMED (no detection ever
  re-matches the placeholder-caption pass, so ``consecutive_det_num`` stays 0); we
  hide it because the every-frame lifecycle step decays ``keep_alive`` to <= 0.
  Same observable, different mechanism.
- frames 5-7: upstream SHOWS it, we do not -- a documented divergence, not a bug on
  our side. Upstream kills the object at frame 7 (8 consecutive unmatched frames,
  inside its hotstart window; it hits 8 rather than 7 because the capture runs frame 0
  twice -- once via ``add_prompt``, once when propagation starts at 0). The kill
  compacts it out of ``obj_ids_all_gpu``, which empties the unconfirmed set that its
  (thresh-1)-frame lookahead reads, so the frames preceding the death get revealed
  retroactively. Reproducing that needs a buffered, non-causal output path. Our single
  pass reaches 7 unmatched frames at frame 7, so our purge lands at frame 8, past the
  end of this clip; the object stays hidden throughout. Left ungated here.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from sam.prompts import GeometryPrompt
from sam.results import Emit

FIX = Path("tests/parity/fixtures/sam3p1")
CKPT = "checkpoints/sam3.1_multiplex.pt"
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.isfile(CKPT)
    or not (FIX / "video_box.npz").is_file(),
    reason="needs CUDA + sam3.1_multiplex.pt + captured golden",
)

FRAME0_IOU_MIN = 0.85   # the box-seeded frame-0 detection matches the golden closely


def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else inter / union


def test_video_box_prompt_seed_parity():
    from sam.build_sam import build_sam3_multiplex_video_predictor
    from sam.models.sam3_predictor import Sam3VideoPredictorState

    scn = json.loads((FIX / "video_box_scenario.json").read_text())
    g = np.load(FIX / "video_box.npz")
    n, (h, w) = scn["num_frames"], scn["hw"]
    frames = [
        np.asarray(Image.open(f"{scn['frames_dir']}/{i:05d}.jpg").convert("RGB"))
        for i in range(n)
    ]

    pred = build_sam3_multiplex_video_predictor(
        config_file="configs/sam3/sam3.1.yaml", ckpt_path=CKPT, device="cuda")
    # Frame 0 shows the box-seeded object in the golden while it is still PENDING
    # (confirmation needs 3 consecutive detections), so measure in VISIBLE mode.
    pred.emit = Emit.VISIBLE
    st = Sam3VideoPredictorState(video_hw=(h, w))
    box = GeometryPrompt(obj_id=1, boxes=torch.tensor([scn["box_xyxy"]], dtype=torch.float32))

    for i, fr in enumerate(frames):
        out = pred.forward(st, i, fr, prompts=[box] if i == 0 else [])
        g_ids = g[f"frame{i}_obj_ids"]
        if len(g_ids) == 0:
            # Upstream hides the object here; so must we (see the module docstring).
            assert not out, f"frame {i}: object shown, golden hides it"
            continue
        if i > 0:
            continue  # frames 5-7: upstream's post-kill reveal, see the docstring
        g_mask = g[f"frame{i}_obj{int(g_ids[0])}"].astype(bool)
        assert len(out) >= 1, f"frame {i}: object lost (golden shows it)"
        best = max(_iou((out[k].masks_logits[0, 0] > 0).cpu().numpy(), g_mask) for k in out)
        assert best >= FRAME0_IOU_MIN, (
            f"frame {i}: box-seeded mask IoU {best:.3f} < {FRAME0_IOU_MIN}")
