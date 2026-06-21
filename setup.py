# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Build backend for sam2's CUDA extension (sam2._C).

flash-attn-style install: at `pip install` / wheel-build time we first try to
download a prebuilt wheel matching the user's exact (torch, CUDA, Python,
platform, C++ ABI) from GitHub Releases. If none matches, we fall back to
compiling the extension from source. Pure-Python install (no extension) is
available via SAM2_SKIP_CUDA_BUILD=1.

Environment toggles:
  SAM2_SKIP_CUDA_BUILD=1   build a pure-Python wheel (no _C extension)
  SAM2_FORCE_BUILD=1       skip the prebuilt-wheel download, always compile
  SAM2_FORCE_CUDA=1        configure the CUDA extension even if CUDA is not
                           detected at build time (cross-compile on CI)
  SAM2_WHEEL_BASE_URL=...  override the GitHub Releases base URL
"""

import os
import platform
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

from setuptools import setup

# ---------------------------------------------------------------------------
# Static metadata
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).parent.resolve()
PACKAGE_NAME = "sam2"
GH_OWNER_REPO = "cjaverliat/sam2"

# GitHub Releases asset layout: one release per upstream version (tag v1.0.3),
# many wheels attached, each filename carrying its +cuXXXtorchYY local label.
BASE_WHEEL_URL = os.environ.get(
    "SAM2_WHEEL_BASE_URL",
    "https://github.com/{owner_repo}/releases/download/v{version}/{wheel_name}",
)

FORCE_BUILD = os.getenv("SAM2_FORCE_BUILD", "0") == "1"
SKIP_CUDA_BUILD = os.getenv("SAM2_SKIP_CUDA_BUILD", "0") == "1"
FORCE_CUDA = os.getenv("SAM2_FORCE_CUDA", "0") == "1"


def _point_cuda_home_at_conda():
    """Make torch's build use the active pixi/conda nvcc, not a system one.

    torch.utils.cpp_extension resolves nvcc via CUDA_HOME (else PATH). A system
    CUDA on PATH that differs from the toolkit torch was built with triggers a
    fatal version-mismatch. The pixi env's nvcc matches its torch, so prefer it.
    Mirrors the runtime JIT logic in sam2/utils/misc.py. Must run before
    torch.utils.cpp_extension is imported (it caches CUDA_HOME at import).
    """
    conda = os.environ.get("CONDA_PREFIX") or os.environ.get("MAMBA_ROOT_PREFIX")
    if not conda:
        return
    base = os.path.join(conda, "Library") if sys.platform == "win32" else conda
    nvcc = os.path.join(base, "bin", "nvcc.exe" if sys.platform == "win32" else "nvcc")
    if os.path.isfile(nvcc):
        os.environ["CUDA_HOME"] = base


def get_package_version():
    text = (THIS_DIR / "sam2" / "version.py").read_text()
    return re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text).group(1)


# Non-torch runtime deps (torch/torchvision appended dynamically below).
BASE_DEPS = [
    "numpy>=1.24.4",
    "tqdm>=4.66.1",
    "hydra-core>=1.3.2",
    "iopath>=0.1.10",
    "pillow>=9.4.0",
    "opencv-python>=4.7.0",
]


def _minor_pin(name, ver):
    # "2.11.0+cu128" -> "name==2.11.*" : lock the ABI-critical minor, allow patches.
    major, minor = ver.split("+")[0].split(".")[:2]
    return f"{name}=={major}.{minor}.*"


def get_install_requires(torch, cpu=False):
    """Runtime deps. For a CUDA wheel, pin torch (and torchvision) to the exact
    minor it was built against (ABI lock) so a mismatched torch errors at install.
    The CPU wheel is pure-Python -> keep torch loose (works with any torch)."""
    if torch is None or cpu:
        return BASE_DEPS + ["torch>=2.5.1", "torchvision>=0.20.1"]
    reqs = BASE_DEPS + [_minor_pin("torch", torch.__version__)]
    try:
        import torchvision

        reqs.append(_minor_pin("torchvision", torchvision.__version__))
    except ImportError:
        reqs.append("torchvision>=0.20.1")
    return reqs


# ---------------------------------------------------------------------------
# Build-environment fingerprint (torch / CUDA / python / platform / C++ ABI)
# These pieces compose both the wheel's local version label and the prebuilt
# wheel filename we look for on GitHub Releases. Keep the two in lock-step.
# ---------------------------------------------------------------------------
def get_platform_tag():
    if sys.platform.startswith("linux"):
        return f"linux_{platform.uname().machine}"  # e.g. linux_x86_64
    if sys.platform == "win32":
        return "win_amd64"
    if sys.platform == "darwin":
        # CUDA wheels are not produced for macOS; only relevant for sdist builds.
        return "macosx_11_0_arm64"
    raise RuntimeError(f"Unsupported platform: {sys.platform}")


def get_cuda_tag(torch):
    # "12.8" -> "cu128". None when torch is a CPU build.
    cuda = getattr(torch.version, "cuda", None)
    return f"cu{cuda.replace('.', '')}" if cuda else "cpu"


def get_torch_tag(torch):
    # "2.8.0+cu128" -> "torch28"
    major, minor = torch.__version__.split("+")[0].split(".")[:2]
    return f"torch{major}{minor}"


def get_cxx_abi_tag(torch):
    # Linux torch ships two C++ ABIs; mismatched ABI = unimportable extension.
    return "cxx11abitrue" if torch._C._GLIBCXX_USE_CXX11_ABI else "cxx11abifalse"


def get_local_label(torch):
    """PEP440 local version label, e.g. cu128torch28cxx11abitrue (lowercased)."""
    return f"{get_cuda_tag(torch)}{get_torch_tag(torch)}{get_cxx_abi_tag(torch)}".lower()


def get_python_tag():
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def get_wheel_filename(version, local_label):
    # Matches the filename pip emits for this build, so CI uploads it as-is and
    # this function reconstructs it for download. Local label is normalized
    # lowercase by pip; we already lowercase it.
    py = get_python_tag()
    return (
        f"{PACKAGE_NAME}-{version}+{local_label}-{py}-{py}-{get_platform_tag()}.whl"
    )


# ---------------------------------------------------------------------------
# Extension definition (only when building from source)
# ---------------------------------------------------------------------------
def get_extensions(torch):
    from torch.utils.cpp_extension import CUDAExtension

    cxx_args = []
    nvcc_args = [
        "-DCUDA_HAS_FP16=1",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]
    # Pin C++17 on Windows. torch >= 2.12 requests C++20, but nvcc + MSVC ignores
    # -std=c++20 and falls back to a pre-C++17 default, breaking torch headers
    # ("one-argument static_assert not enabled"). Setting -std explicitly stops
    # torch from injecting C++20; C++17 satisfies the headers. (Linux/gcc builds
    # C++20 fine, so leave them untouched.)
    if sys.platform == "win32":
        # /Zc:preprocessor: CUDA 13.2+ CCCL headers hard-error (C1189) on MSVC's
        # traditional preprocessor; force the standard-conforming one. Harmless on
        # older CUDA.
        cxx_args += ["/std:c++17", "/Zc:preprocessor"]
        nvcc_args += [
            "-std=c++17",
            "-Xcompiler", "/std:c++17",
            "-Xcompiler", "/Zc:preprocessor",
        ]

    # setuptools requires /-separated paths relative to setup.py, never absolute.
    return [
        CUDAExtension(
            name="sam2._C",
            sources=[
                "sam2/csrc/connected_components.cu",
                "sam2/csrc/connected_components_binding.cpp",
            ],
            extra_compile_args={"cxx": cxx_args, "nvcc": nvcc_args},
        )
    ]


# ---------------------------------------------------------------------------
# Assemble setup() kwargs depending on torch availability / skip flag
# ---------------------------------------------------------------------------
version = get_package_version()
ext_modules = []
cmdclass = {}
local_label = None

# torch may be absent during metadata-only steps (e.g. uv's editable metadata
# prep). Degrade gracefully: no torch -> plain version, no extension. The runtime
# JIT fallback in sam2/utils/misc.py still builds _C on first use for dev/editable
# installs on a machine that has a CUDA toolchain.
try:
    import torch  # noqa: F401
except ImportError:
    torch = None

# CPU build: explicit skip, or torch is a CPU-only build (no CUDA to link).
# Produces a pure-Python wheel (no _C) labelled +cpu, valid on any platform.
is_cpu_build = SKIP_CUDA_BUILD or (
    torch is not None and getattr(torch.version, "cuda", None) is None
)

if is_cpu_build and torch is not None:
    local_label = "cpu"
    # No extension, no download command: a pure-Python build needs no toolchain
    # and is instant, so source-build is always the right fallback.
elif torch is not None:
    _point_cuda_home_at_conda()
    from torch.utils.cpp_extension import BuildExtension
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel

    local_label = get_local_label(torch)

    if torch.cuda.is_available() or FORCE_CUDA:
        ext_modules = get_extensions(torch)
    elif not FORCE_BUILD:
        # No CUDA and not forcing a local build: a download attempt (below) is
        # the only way to get _C; leave ext_modules empty so the fallback build
        # produces a pure-Python wheel rather than erroring on missing nvcc.
        pass
    else:
        ext_modules = get_extensions(torch)

    class CachedWheelsCommand(_bdist_wheel):
        """Try a prebuilt GitHub Releases wheel before compiling (flash-attn style)."""

        def run(self):
            if FORCE_BUILD:
                return super().run()

            wheel_name = get_wheel_filename(version, local_label)
            url = BASE_WHEEL_URL.format(
                owner_repo=GH_OWNER_REPO, version=version, wheel_name=wheel_name
            )
            try:
                os.makedirs(self.dist_dir, exist_ok=True)
                impl_tag, abi_tag, plat_tag = self.get_tag()
                dest = os.path.join(
                    self.dist_dir,
                    f"{self.wheel_dist_name}-{impl_tag}-{abi_tag}-{plat_tag}.whl",
                )
                print(f"sam2: fetching prebuilt wheel\n  {url}")
                urllib.request.urlretrieve(url, dest)
                print(f"sam2: using prebuilt wheel -> {dest}")
            except (urllib.error.HTTPError, urllib.error.URLError) as e:
                print(
                    f"sam2: no matching prebuilt wheel ({e}); building from source."
                )
                super().run()

    cmdclass = {
        "bdist_wheel": CachedWheelsCommand,
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
    }

# Local version label distinguishes one torch/CUDA build from another on the
# same upstream version (1.0.3+cu128torch28cxx11abitrue).
full_version = version if local_label is None else f"{version}+{local_label}"

setup(
    version=full_version,
    install_requires=get_install_requires(torch, cpu=is_cpu_build),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
