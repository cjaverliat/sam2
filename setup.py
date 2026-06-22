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
available via SAM2_BUILD_CUDA=0.

Environment toggles:
  SAM2_BUILD_CUDA=1        force-build the _C CUDA extension even if no GPU is
                           visible at build time (CI cross-compile)
  SAM2_BUILD_CUDA=0        build a pure-Python wheel (no _C extension)
                           (unset = auto: build _C when the installed torch is a
                           CUDA build with a usable runtime)
  SAM2_ALLOW_BUILD_ERRORS=1  (default) on a _C build failure fall back to a
                           pure-Python wheel; _C is JIT-compiled at runtime on
                           first use. Set 0 (CI) to fail hard on a broken build.
  SAM2_FORCE_BUILD=1       skip the prebuilt-wheel download, always compile
  SAM2_WHEEL_BASE_URL=...  override the GitHub Releases base URL
"""

import os
import platform
import re
import sys
import urllib.error
import urllib.request
import warnings
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
# Tri-state _C build switch: "1" force the build (even with no GPU visible, e.g.
# CI cross-compile), "0" skip it (pure-Python wheel), unset = auto (decided from
# the installed torch below). None means "unset".
BUILD_CUDA = os.getenv("SAM2_BUILD_CUDA")
# On by default: if the from-source _C compile fails, drop the extension and ship
# a pure-Python wheel (the runtime JIT-compiles _C on first use) instead of
# aborting the install. CI sets this to 0 so a broken kernel build fails loudly.
ALLOW_BUILD_ERRORS = os.getenv("SAM2_ALLOW_BUILD_ERRORS", "1") == "1"


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
class _BuildError(Exception):
    """A recoverable _C build problem. Raised before/at compile so the tolerant
    build path (SAM2_ALLOW_BUILD_ERRORS) can degrade to a pure-Python wheel."""


def _require_msvc():
    """Check MSVC's cl.exe is on PATH (Windows); raise ``_BuildError`` otherwise.

    nvcc compiles the host side of sam2._C with MSVC. Without an activated MSVC
    environment the build otherwise dies deep inside ninja with an opaque,
    swallowed error. Check up front and tell the user how to fix it.
    """
    if sys.platform != "win32":
        return
    import shutil

    if shutil.which("cl") is None:
        raise _BuildError(
            "MSVC compiler 'cl.exe' was not found on PATH. Building the sam2._C "
            "CUDA extension on Windows needs the MSVC x64 toolchain active: build "
            "from an \"x64 Native Tools Command Prompt for VS\" (or run vcvars64.bat "
            "first), then retry."
        )


def get_extensions(torch):
    from torch.utils.cpp_extension import CUDAExtension

    _require_msvc()

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

# torch must come from the target environment (not an isolated build env): the
# build keys the CUDA/CPU choice, C++ ABI tag and install_requires pin off the
# *installed* torch. An isolated build would silently produce a mismatched
# (typically CPU-only) wheel, so we refuse to build without torch.
#
# Exception: metadata-only steps (egg_info / dist_info / --version, e.g. uv's
# editable metadata prep during a pixi solve) run *before* torch is installed
# into the env. Those must degrade gracefully (no torch -> plain version, no
# extension) so the solve can bootstrap; the real build happens later with torch
# present. So only hard-fail on commands that actually build a distribution.
_BUILD_COMMANDS = {
    "build",
    "build_ext",
    "bdist_wheel",
    "bdist_egg",
    "editable_wheel",
    "install",
    "develop",
}
_is_building = not _BUILD_COMMANDS.isdisjoint(sys.argv)

try:
    import torch  # noqa: F401
except ImportError:
    if _is_building and BUILD_CUDA != "0":
        raise SystemExit(
            "\n"
            "sam2 build error: PyTorch is not importable in the build environment.\n"
            "\n"
            "sam2 inspects the installed torch (CUDA vs CPU, version, C++ ABI) to\n"
            "select/build the right native extension and to pin its torch dependency.\n"
            "It therefore must NOT be built with build isolation. Do this instead:\n"
            "\n"
            "  1. Install torch for your target platform first, e.g.\n"
            "       pip install torch torchvision\n"
            "     (or the CUDA build from https://pytorch.org for GPU kernels)\n"
            "  2. Reinstall sam2 with build isolation disabled:\n"
            "       pip install --no-build-isolation "
            "git+https://github.com/cjaverliat/sam2.git\n"
            "\n"
            "If torch reports a CPU-only build, sam2 produces a pure-Python (no _C)\n"
            "wheel; a CUDA torch enables the prebuilt-or-source compiled kernels.\n"
        )
    # Metadata-only step (or explicit SAM2_BUILD_CUDA=0): degrade gracefully.
    torch = None

# CPU build: no torch yet (metadata-only), explicit SAM2_BUILD_CUDA=0, or torch is
# a CPU-only build (no CUDA to link). Produces a pure-Python wheel (no _C) +cpu.
is_cpu_build = (
    BUILD_CUDA == "0"
    or torch is None
    or getattr(torch.version, "cuda", None) is None
)

if torch is None:
    # Metadata-only: plain version, no extension, loose deps.
    pass
elif is_cpu_build:
    local_label = "cpu"
    # No extension, no download command: a pure-Python build needs no toolchain
    # and is instant, so source-build is always the right fallback.
else:
    _point_cuda_home_at_conda()
    from torch.utils.cpp_extension import BuildExtension
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel

    local_label = get_local_label(torch)

    def _extensions_or_degrade():
        """Configure the _C extension, degrading to none on a recoverable build
        problem (e.g. missing MSVC) when SAM2_ALLOW_BUILD_ERRORS is set; the runtime
        JIT-compiles _C later. Otherwise re-raise as a fatal build error."""
        try:
            return get_extensions(torch)
        except _BuildError as e:
            if not ALLOW_BUILD_ERRORS:
                raise SystemExit(f"\nsam2 build error: {e}\n")
            warnings.warn(
                f"sam2: {e} Building a pure-Python wheel instead; sam2._C is "
                "JIT-compiled at runtime on first use "
                "(set SAM2_ALLOW_BUILD_ERRORS=0 to fail the build instead).",
                stacklevel=2,
            )
            return []

    if torch.cuda.is_available() or BUILD_CUDA == "1":
        ext_modules = _extensions_or_degrade()
    elif not FORCE_BUILD:
        # No CUDA and not forcing a local build: a download attempt (below) is
        # the only way to get _C; leave ext_modules empty so the fallback build
        # produces a pure-Python wheel rather than erroring on missing nvcc.
        pass
    else:
        ext_modules = _extensions_or_degrade()

    class _TolerantBuildExt(BuildExtension.with_options(no_python_abi_suffix=True)):
        """Compile _C; on a compile failure, degrade to a pure-Python wheel when
        SAM2_ALLOW_BUILD_ERRORS is set (runtime JIT-compiles _C later) instead of
        aborting the install."""

        def build_extensions(self):
            try:
                super().build_extensions()
            except Exception as e:
                if not ALLOW_BUILD_ERRORS:
                    raise
                warnings.warn(
                    f"sam2: compiling sam2._C failed ({e}). Installing a pure-Python "
                    "sam2; _C is JIT-compiled at runtime on first use "
                    "(set SAM2_ALLOW_BUILD_ERRORS=0 to fail the build instead).",
                    stacklevel=2,
                )
                self.extensions = []

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
        "build_ext": _TolerantBuildExt,
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
