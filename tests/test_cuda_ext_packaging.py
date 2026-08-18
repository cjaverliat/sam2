# SPDX-License-Identifier: Apache-2.0
"""Guards for the sam._C CUDA extension: how it is built and how it is shipped.

Regressions these pin down, both observed in the wild:

1. conda-forge's cuda-nvcc activation exports a toolkit-wide
   TORCH_CUDA_ARCH_LIST (...;10.0;10.1;12.0+PTX). torch's
   _get_cuda_arch_flags() raises on any entry it does not know -- torch 2.11
   knows 10.0/10.3/12.0/12.1 but not 10.1 -- so an inherited list killed the
   compile on a fully supported GPU.
2. The tolerant build path then dropped the extension but kept the CUDA local
   version label, producing a wheel whose name promised a _C it did not
   contain. Such a wheel gets cached by pip/uv and reused indefinitely.
"""

import os
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_jit_pins_arch_list_to_the_local_gpu(monkeypatch):
    """The JIT fallback must not inherit a list torch may reject."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a visible CUDA device")

    import torch.utils.cpp_extension as cpp_extension

    from sam.utils import misc

    # Force step 1 (the prebuilt extension) to fail so the JIT path runs.
    import sam

    monkeypatch.delattr(sam, "_C", raising=False)
    monkeypatch.setitem(sys.modules, "sam._C", None)
    monkeypatch.setattr(misc, "_connected_components_ext", None)

    # A list torch accepts only in part: 10.1 is not in its supported arches.
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.6;10.1")

    seen = {}

    def fake_load(**kwargs):
        seen["arch_list"] = os.environ.get("TORCH_CUDA_ARCH_LIST")
        return object()

    monkeypatch.setattr(cpp_extension, "load", fake_load)

    misc._load_connected_components_ext()

    major, minor = torch.cuda.get_device_capability()
    assert seen["arch_list"] == f"{major}.{minor}"

    # And the pinned value must actually be usable by torch.
    cpp_extension._get_cuda_arch_flags()


@pytest.mark.parametrize("arch_list", ["8.6;10.1", "10.1", "Blackwell"])
def test_pinned_arch_list_survives_a_polluted_environment(monkeypatch, arch_list):
    """Whatever was inherited, torch must accept what we hand it."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a visible CUDA device")

    import torch.utils.cpp_extension as cpp_extension

    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", arch_list)
    major, minor = torch.cuda.get_device_capability()
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")

    cpp_extension._get_cuda_arch_flags()


def test_cuda_labelled_wheels_contain_the_extension():
    """A +cuXXXtorchYY wheel without _C is the artefact that poisons caches."""
    dist = REPO_ROOT / "dist"
    wheels = sorted(dist.glob("*.whl")) if dist.is_dir() else []
    if not wheels:
        pytest.skip("no built wheels in dist/")

    cuda_wheels = [w for w in wheels if "+cu" in w.name]
    if not cuda_wheels:
        pytest.skip("no CUDA-labelled wheels in dist/")

    for wheel in cuda_wheels:
        names = zipfile.ZipFile(wheel).namelist()
        binaries = [n for n in names if n.endswith((".pyd", ".so"))]
        assert binaries, (
            f"{wheel.name} is labelled as a CUDA build but ships no compiled "
            "extension"
        )
