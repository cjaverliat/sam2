# SPDX-License-Identifier: LicenseRef-SAM
# Vendored from facebookresearch/sam3 @ 5dd401d (sam3/model/sam3_tracker_utils.py): ONLY the
# helpers the tracker's ``track_step`` needs. ``select_closest_cond_frames`` carries the SAM 3
# ``keep_first_cond_frame`` knob the shared (Apache) ``sam.modeling.utils`` variant lacks.
# ``fill_holes_in_mask_scores`` is the inference mask post-processor (used by the Task-9
# predictor; a no-op at the tracker's ``fill_hole_area=0``). The triton ``edt`` /
# ``sample_box_points`` / training samplers are intentionally NOT vendored; connected components
# fall back to a lazy CPU (skimage) implementation so importing this module pulls in no triton.
"""SAM 3 tracker helper functions."""

import torch


def select_closest_cond_frames(
    frame_idx, cond_frame_outputs, max_cond_frame_num, keep_first_cond_frame=False
):
    """Select up to ``max_cond_frame_num`` conditioning frames temporally closest to
    ``frame_idx`` (optionally always keeping the first annotated frame)."""
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        selected_outputs = cond_frame_outputs
        unselected_outputs = {}
    else:
        assert max_cond_frame_num >= 2, "we should allow using 2+ conditioning frames"
        selected_outputs = {}
        if keep_first_cond_frame:
            idx_first = min(
                (t for t in cond_frame_outputs if t < frame_idx), default=None
            )
            if idx_first is None:
                # Maybe we are tracking in reverse
                idx_first = max(
                    (t for t in cond_frame_outputs if t > frame_idx), default=None
                )
            if idx_first is not None:
                selected_outputs[idx_first] = cond_frame_outputs[idx_first]
        # the closest conditioning frame before `frame_idx` (if any)
        idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
        if idx_before is not None:
            selected_outputs[idx_before] = cond_frame_outputs[idx_before]

        # the closest conditioning frame after `frame_idx` (if any)
        idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
        if idx_after is not None:
            selected_outputs[idx_after] = cond_frame_outputs[idx_after]

        # add other temporally closest conditioning frames until reaching a total
        # of `max_cond_frame_num` conditioning frames.
        num_remain = max_cond_frame_num - len(selected_outputs)
        inds_remain = sorted(
            (t for t in cond_frame_outputs if t not in selected_outputs),
            key=lambda x: abs(x - frame_idx),
        )[:num_remain]
        selected_outputs.update((t, cond_frame_outputs[t]) for t in inds_remain)
        unselected_outputs = {
            t: v for t, v in cond_frame_outputs.items() if t not in selected_outputs
        }

    return selected_outputs, unselected_outputs


def fill_holes_in_mask_scores(
    mask,
    max_area=None,
    fill_holes=True,
    remove_sprinkles=True,
    fill_hole_area=None,
    sprinkle_removal_area=None,
):
    """Fill small holes / remove small sprinkles in mask score maps (areas under ``max_area``).
    A no-op when ``max_area <= 0`` (the tracker's ``fill_hole_area=0`` default)."""
    # Support onevision-style keyword args
    if fill_hole_area is not None and max_area is None:
        max_area = fill_hole_area

    if max_area <= 0:
        return mask  # nothing to fill in this case

    if fill_holes:
        # Small background connected components -> foreground (small positive score 0.1).
        mask_bg = mask <= 0
        bg_area_thresh = max_area
        _, areas_bg = _get_connected_components_with_padding(mask_bg)
        small_components_bg = mask_bg & (areas_bg <= bg_area_thresh)
        mask = torch.where(small_components_bg, 0.1, mask)

    if remove_sprinkles:
        # Small foreground connected components -> background (small negative score -0.1).
        mask_fg = mask > 0
        fg_area_thresh = torch.sum(mask_fg, dim=(2, 3), keepdim=True, dtype=torch.int32)
        fg_area_thresh.floor_divide_(2).clamp_(max=max_area)
        _, areas_fg = _get_connected_components_with_padding(mask_fg)
        small_components_fg = mask_fg & (areas_fg <= fg_area_thresh)
        mask = torch.where(small_components_fg, -0.1, mask)
    return mask


def _get_connected_components_with_padding(mask):
    """Connected-component areas per pixel for a (B,1,H,W) binary mask, on CPU via skimage
    (lazy import -- avoids a hard triton/cc_torch dependency at import time)."""
    from skimage.measure import label as _label

    device = mask.device
    mask_np = mask.to(torch.uint8).cpu().numpy()
    B, _, H, W = mask_np.shape
    areas = torch.zeros((B, 1, H, W), dtype=torch.int32)
    labels = torch.zeros((B, 1, H, W), dtype=torch.int32)
    for b in range(B):
        lab = _label(mask_np[b, 0], connectivity=1)
        lab_t = torch.from_numpy(lab.astype("int32"))
        labels[b, 0] = lab_t
        if lab.max() > 0:
            counts = torch.bincount(lab_t.reshape(-1))
            areas[b, 0] = counts[lab_t]
    return labels.to(device), areas.to(device)
