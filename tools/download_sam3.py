# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Download gated SAM 3 checkpoints from the Hugging Face Hub.

SAM 3 weights live in access-gated repos (``facebook/sam3``, ``facebook/sam3.1``).
This script ensures you are authenticated first (reusing a cached login if one is
valid, otherwise prompting once for a token), then downloads the checkpoint. On any
failure it prints actionable guidance and exits non-zero rather than continuing
silently.

Repo / filename mapping mirrors the SAM 3 reference loader
(``sam3/model_builder.py::download_ckpt_from_hf``).

Usage:
    python tools/download_sam3.py [--version sam3|sam3.1] [--out-dir checkpoints]
"""

import argparse
import sys
from pathlib import Path

from _hf_download import download_to

# version -> (repo_id, checkpoint filename)
_REPOS = {
    "sam3": ("facebook/sam3", "sam3.pt"),
    "sam3.1": ("facebook/sam3.1", "sam3.1_multiplex.pt"),
}


def _fail(reason: str, repo_id: str) -> None:
    """Print actionable guidance and exit non-zero."""
    print(
        "\n".join(
            [
                "",
                "ERROR: could not download the SAM 3 checkpoint.",
                f"  reason: {reason}",
                "",
                f"You are authenticated, but access to '{repo_id}' is gated. To fix:",
                f"  1. Open https://huggingface.co/{repo_id} (signed in) and accept the",
                "     conditions / request access. Approval may not be instant.",
                "  2. Once access shows as granted on that page, re-run this task.",
                "",
                "If instead the token is the problem, reset it with: pixi run hf auth login --force",
                "",
            ]
        ),
        file=sys.stderr,
    )
    sys.exit(1)


def _ensure_authenticated() -> None:
    """Guarantee a valid HF token, reusing a cached login when possible.

    Order: a valid stored token / ``HF_TOKEN`` is reused silently; otherwise we prompt
    once (interactive) and the token is cached to disk for next time. With no TTY and no
    valid token (e.g. CI) we fail loud instead of hanging."""
    try:
        from huggingface_hub import login, whoami
    except ImportError:
        print(
            "ERROR: huggingface_hub is not installed (expected in the pixi env).",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        user = whoami()  # uses HF_TOKEN or the cached login token
        print(f"Hugging Face: already logged in as {user['name']}.")
        return
    except Exception:
        pass  # no token, or the stored one is invalid -> log in below

    if not sys.stdin.isatty():
        print(
            "\n".join(
                [
                    "",
                    "ERROR: not authenticated to Hugging Face and no terminal to prompt.",
                    "  Set HF_TOKEN, or run `pixi run hf auth login` in a terminal first.",
                    "",
                ]
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    print("Not logged in to Hugging Face. Paste a token with access to the gated SAM 3 repos.")
    print("Create one at https://huggingface.co/settings/tokens")
    login()  # interactive prompt; validates and caches the token to disk


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download gated SAM 3 checkpoints from the Hugging Face Hub."
    )
    parser.add_argument("--version", choices=sorted(_REPOS), default="sam3")
    parser.add_argument("--out-dir", default="checkpoints")
    args = parser.parse_args()

    repo_id, ckpt_name = _REPOS[args.version]
    dst = Path(args.out_dir) / ckpt_name
    # Check up front so an already-present checkpoint skips authentication entirely
    # (the shared helper re-checks, but authenticating first would prompt/fail needlessly).
    if dst.exists():
        print(f"{dst} already present, skipping download.")
        return

    _ensure_authenticated()
    download_to(
        repo_id, ckpt_name, dst,
        on_error=lambda e: _fail(f"{type(e).__name__}: {e}", repo_id),
        success_msg=lambda d: f"Downloaded {repo_id}:{ckpt_name} -> {d}",
    )


if __name__ == "__main__":
    main()
