# SPDX-License-Identifier: LicenseRef-SAM
"""Shared plotting and bookkeeping helpers for the example notebooks.

Nothing here is part of the library — it is the scaffolding the notebooks use so
that the cells you read are about SAM, not about matplotlib. Import it with::

    import nb_utils as nb

Both the SAM 2 and the SAM 3 notebook use the same helpers, so moving between
them means learning the model, not a new set of plotting conventions.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

__all__ = [
    "REPO_ROOT",
    "use_repo_root",
    "pick_device",
    "require_checkpoints",
    "use_dark",
    "to_mask",
    "box_from_mask",
    "show_mask",
    "show_box",
    "show_points",
    "show_frames",
    "collect",
    "free",
]

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
def use_repo_root() -> Path:
    """Run from the repo root so the ``configs/...`` and ``checkpoints/...`` paths work.

    Safe to call twice — it is a no-op once you are already there.
    """
    os.chdir(REPO_ROOT)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return REPO_ROOT


def pick_device(require_cuda: bool = False) -> torch.device:
    """Best available device, printed so the notebook output records what ran."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        # TF32 on Ampere+ is a free speedup for these models.
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif not require_cuda and torch.backends.mps.is_available():
        device = torch.device("mps")
        name = "Apple MPS (preliminary support; results may differ from CUDA)"
    elif not require_cuda:
        device = torch.device("cpu")
        name = "CPU (expect this to be slow)"
    else:
        raise RuntimeError(
            "This notebook needs a CUDA GPU: SAM 3 does its own GPU preprocessing."
        )
    print(f"device: {device} — {name}")
    return device


def require_checkpoints(*specs: tuple[str, str]) -> None:
    """Check checkpoints exist before a long load, and say how to get missing ones.

    Args:
        *specs: ``(path, how_to_get_it)`` pairs, e.g.
            ``("checkpoints/sam3.pt", "pixi run download-sam3")``.
    """
    missing = [(p, how) for p, how in specs if not Path(p).is_file()]
    for path, _ in specs:
        if Path(path).is_file():
            size = Path(path).stat().st_size / 1e9
            print(f"  found {path} ({size:.1f} GB)")
    if missing:
        lines = "\n".join(f"  {p}   ->  {how}" for p, how in missing)
        raise FileNotFoundError(f"Missing checkpoint(s):\n{lines}")


def use_dark(enabled: bool = True) -> None:
    """Switch the figures to a dark palette (for dark-themed editors)."""
    plt.style.use("dark_background" if enabled else "default")


# ---------------------------------------------------------------------------
# Results -> pixels
# ---------------------------------------------------------------------------
def to_mask(masklet) -> np.ndarray:
    """A ``MaskletResult`` (or raw logits) as a plain ``(H, W)`` boolean array.

    Both predictors return mask *logits* — positive means foreground — with
    leading batch dimensions. This flattens that away so the notebooks can just
    plot the result.
    """
    logits = getattr(masklet, "best_mask_logits", masklet)
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu()
    arr = np.asarray(logits)
    return arr.reshape(arr.shape[-2:]) > 0.0


def box_from_mask(mask_bool) -> np.ndarray | None:
    """Tight ``xyxy`` pixel box around a boolean mask, or None if it is empty.

    The video predictors return masks, not boxes, so every box drawn over a video
    frame in these notebooks is derived here. The image detector predicts real
    boxes — see ``result.boxes``.
    """
    mask = np.asarray(mask_bool, dtype=bool)
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=float)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def _color(obj_id):
    return plt.get_cmap("tab10")(0 if obj_id is None else int(obj_id) % 10)


def show_mask(mask, ax, obj_id=None, alpha: float = 0.6) -> None:
    """Overlay a boolean mask on ``ax``, coloured stably by object id."""
    mask = np.asarray(mask.detach().cpu() if isinstance(mask, torch.Tensor) else mask)
    mask = mask.reshape(mask.shape[-2:])
    color = np.array([*_color(obj_id)[:3], alpha])
    ax.imshow(mask[..., None] * color.reshape(1, 1, -1))


def show_box(box, ax, obj_id=None, label=None, style: str = "-") -> None:
    """Draw an ``xyxy`` box, coloured by object id, with an optional text tag."""
    if isinstance(box, torch.Tensor):
        box = box.detach().cpu().numpy()
    x0, y0, x1, y1 = np.asarray(box, dtype=float)
    color = _color(obj_id)[:3]
    ax.add_patch(
        plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor=color,
                      facecolor="none", lw=2, linestyle=style)
    )
    if label is not None:
        ax.text(x0, max(y0 - 4, 2), label, color="white", fontsize=9,
                bbox=dict(facecolor=color, edgecolor="none", pad=1))


def show_points(coords, labels, ax, marker_size: int = 220) -> None:
    """Plot clicks: green stars are positive (add), red stars negative (remove)."""
    coords = np.asarray(coords.detach().cpu() if isinstance(coords, torch.Tensor) else coords)
    labels = np.asarray(labels.detach().cpu() if isinstance(labels, torch.Tensor) else labels)
    coords = coords.reshape(-1, 2)
    for value, color in ((1, "lime"), (0, "red")):
        pts = coords[labels.reshape(-1) == value]
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], color=color, marker="*",
                       s=marker_size, edgecolor="black", linewidth=1.0, zorder=5)


def show_frames(frames, per_frame, title, idxs=None, label_prefix=None,
                boxes: bool = True, extra=None, width: float = 6.0):
    """Plot a few frames of a run side by side, with each object's mask overlaid.

    Args:
        frames: the frames themselves (anything ``imshow`` accepts).
        per_frame: one ``{obj_id: mask}`` (or ``{obj_id: {"mask": ...}}``) per frame.
        title: figure title.
        idxs: which frames to show; defaults to first / middle / last.
        label_prefix: if set, tag each box ``"<prefix> #<obj_id>"``.
        boxes: draw the mask-derived box around each object.
        extra: ``fn(ax, frame_index)`` hook, for drawing prompts on the seed frame.
    """
    idxs = list(idxs) if idxs is not None else [0, len(frames) // 2, len(frames) - 1]
    fig, axes = plt.subplots(1, len(idxs), figsize=(width * len(idxs), width * 0.82))
    fig.suptitle(title)
    for ax, i in zip(np.atleast_1d(axes), idxs):
        ax.set_title(f"frame {i}")
        ax.imshow(frames[i])
        ax.axis("off")
        for obj_id, entry in per_frame[i].items():
            mask = entry["mask"] if isinstance(entry, dict) else entry
            show_mask(mask, ax, obj_id=obj_id)
            if boxes:
                box = box_from_mask(mask)
                if box is not None:
                    tag = f"{label_prefix} #{obj_id}" if label_prefix else None
                    show_box(box, ax, obj_id=obj_id, label=tag)
        if extra is not None:
            extra(ax, i)
    plt.tight_layout()
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Running a session
# ---------------------------------------------------------------------------
def collect(session, frames, prompts_frame0=None, prompts_at=None):
    """Stream ``frames`` through a session; return one ``{obj_id: mask}`` per frame.

    Args:
        session: a predictor session (``start_session`` / ``start_concept_session``).
        frames: the clip, in whatever form the predictor takes.
        prompts_frame0: prompts to send with the first frame.
        prompts_at: ``{frame_index: prompts}`` for prompting mid-stream.
    """
    schedule = dict(prompts_at or {})
    if prompts_frame0:
        schedule[0] = prompts_frame0
    per_frame = []
    for i, frame in enumerate(frames):
        with torch.inference_mode():
            masklets = session.process(frame, prompts=schedule.get(i))
        per_frame.append({obj_id: to_mask(m) for obj_id, m in masklets.items()})
    return per_frame


def free(namespace, *names: str) -> None:
    """Drop big models from the notebook namespace and empty the CUDA cache.

    The SAM 3 checkpoints are several GB each and do not comfortably co-reside,
    so the notebook frees one before loading the next. Names that are not defined
    (because you ran the cells out of order) are skipped rather than raising.
    """
    dropped = [n for n in names if namespace.pop(n, None) is not None]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"freed: {', '.join(dropped) if dropped else '(nothing to free)'}")
