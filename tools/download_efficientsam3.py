# SPDX-License-Identifier: LicenseRef-SAM
"""Download EfficientSAM3 checkpoints from the Hugging Face Hub.

EfficientSAM3 weights live in the **public** repo ``Simon7108528/EfficientSAM3``
(no gating, no login required).  The script still resolves the HF cache correctly
and avoids the Xet transfer backend that hangs on some setups.

Usage:
    python tools/download_efficientsam3.py [--variant repvit] [--out-dir checkpoints]
"""

import argparse
import shutil
import sys
import os
from pathlib import Path

# The Xet transfer backend can hang on some setups; disable it for these downloads
# unless the caller explicitly overrides. Must be set before any huggingface_hub import.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

_REPO_ID = "Simon7108528/EfficientSAM3"

# variant -> (hf_filename, local_filename)
# hf_filename   : path inside the HF repo (may include subdirs)
# local_filename: flat name written under --out-dir
_VARIANTS = {
    "repvit": (
        "efficientsam3_ft/efficientsam3_repvit.pt",
        "efficientsam3_repvit.pt",
    ),
    "tinyvit": (
        "efficientsam3_ft/efficientsam3_tinyvit.pt",
        "efficientsam3_tinyvit.pt",
    ),
    "efficientvit": (
        "efficientsam3_ft/efficientsam3_efficientvit.pt",
        "efficientsam3_efficientvit.pt",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download EfficientSAM3 checkpoints from the Hugging Face Hub."
    )
    parser.add_argument(
        "--variant",
        choices=sorted(_VARIANTS),
        default="repvit",
        help="Model variant to download (default: repvit).",
    )
    parser.add_argument(
        "--out-dir",
        default="checkpoints",
        help="Directory to write the checkpoint into (default: checkpoints).",
    )
    args = parser.parse_args()

    hf_filename, local_filename = _VARIANTS[args.variant]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dst = out_dir / local_filename
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

    print(f"Downloading {_REPO_ID}:{hf_filename} ...")
    try:
        cached = hf_hub_download(repo_id=_REPO_ID, filename=hf_filename)
    except Exception as e:
        print(
            "\n".join(
                [
                    "",
                    "ERROR: could not download the EfficientSAM3 checkpoint.",
                    f"  reason: {type(e).__name__}: {e}",
                    "",
                    f"  repo:   {_REPO_ID}",
                    f"  file:   {hf_filename}",
                    "",
                    "Check your network connection.  If the repo has been moved or",
                    "renamed, update _REPO_ID / _VARIANTS in this script.",
                    "",
                ]
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    # hf_hub_download returns a cache path; copy the weights into the project dir.
    shutil.copyfile(cached, dst)
    print(f"Saved to {dst}")


if __name__ == "__main__":
    main()
