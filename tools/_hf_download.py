# SPDX-License-Identifier: LicenseRef-SAM
"""Shared Hugging Face Hub download helper for the tools/download_*.py scripts."""

import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

# The Xet transfer backend can hang on some setups; disable it for these downloads
# unless the caller explicitly overrides. Must be set before any huggingface_hub import
# (importing this module runs it, and the scripts import it before huggingface_hub).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def download_to(
    repo_id: str,
    hf_filename: str,
    dst: Path,
    *,
    on_error: Callable[[Exception], None],
    success_msg: Callable[[Path], str],
    announce: bool = False,
) -> None:
    """Fetch ``hf_filename`` from HF repo ``repo_id`` and copy it to ``dst``.

    Skips (and returns) if ``dst`` already exists. ``on_error(exc)`` handles an
    ``hf_hub_download`` failure and is expected to exit the process. ``success_msg(dst)``
    builds the line printed after a successful copy. ``announce`` prints a
    "Downloading ..." line once a download is actually going to happen.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"{dst} already present, skipping download.")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "ERROR: huggingface_hub is not installed (expected in the pixi env).",
            file=sys.stderr,
        )
        sys.exit(1)

    if announce:
        print(f"Downloading {repo_id}:{hf_filename} ...")
    try:
        cached = hf_hub_download(repo_id=repo_id, filename=hf_filename)
    except Exception as e:
        on_error(e)
        sys.exit(1)  # safety net: on_error is expected to exit

    # hf_hub_download returns a cache path; copy the weights into the project dir.
    shutil.copyfile(cached, dst)
    print(success_msg(dst))
