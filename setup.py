# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Build backend for sam's CUDA extension (sam._C).

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
  SAM2_ALLOW_BUILD_ERRORS=1  on a _C build failure fall back to a pure-Python
                           wheel; _C is JIT-compiled at runtime on first use.
                           Defaults to 1 for an editable/dev install and 0 for a
                           distributed wheel, whose local label would otherwise
                           promise an extension the wheel does not contain.
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
from setuptools.errors import CompileError, ExecError, LinkError

# ---------------------------------------------------------------------------
# Static metadata
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).parent.resolve()
PACKAGE_NAME = "sam"
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
# Whether a failed _C compile degrades to a pure-Python wheel. Unset means
# "decide from the build kind" -- resolved once _DIST_WHEEL_BUILD is known.
_ALLOW_BUILD_ERRORS_ENV = os.getenv("SAM2_ALLOW_BUILD_ERRORS")


def _point_cuda_home_at_conda():
    """Make torch's build use the active pixi/conda nvcc, not a system one.

    torch.utils.cpp_extension resolves nvcc via CUDA_HOME (else PATH). A system
    CUDA on PATH that differs from the toolkit torch was built with triggers a
    fatal version-mismatch. The pixi env's nvcc matches its torch, so prefer it.
    Mirrors the runtime JIT logic in sam/utils/misc.py. Must run before
    torch.utils.cpp_extension is imported (it caches CUDA_HOME at import).
    """
    conda = os.environ.get("CONDA_PREFIX") or os.environ.get("MAMBA_ROOT_PREFIX")
    if not conda:
        return
    base = os.path.join(conda, "Library") if sys.platform == "win32" else conda
    nvcc = os.path.join(base, "bin", "nvcc.exe" if sys.platform == "win32" else "nvcc")
    if os.path.isfile(nvcc):
        os.environ["CUDA_HOME"] = base


def _sanitize_torch_cuda_arch_list():
    """Drop TORCH_CUDA_ARCH_LIST entries the installed torch rejects.

    conda-forge's cuda-nvcc activation exports a toolkit-wide default
    (...;10.0;10.1;12.0+PTX). torch's _get_cuda_arch_flags() raises on any entry
    it does not know -- torch 2.11 knows 10.0/10.3/12.0/12.1 but not 10.1 -- so
    an inherited list fails the _C compile on a fully supported GPU. Keeping only
    the accepted entries preserves an intentional CI cross-compile list while
    surviving a polluted one. Mirrors the runtime JIT logic in
    sam/utils/misc.py.

    An unset, empty or "native" list is left alone. If every entry is rejected
    the variable is cleared, which makes torch fall back to detecting the local
    device's arch.

    Returns:
        The rejected entries, so the caller can warn about them.
    """
    raw = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if not raw or raw.strip() == "native":
        return []

    try:
        from torch.utils.cpp_extension import _get_cuda_arch_flags
    except ImportError:
        # Private API; if a future torch drops it, leave the list untouched
        # rather than guessing which arches are valid.
        return []

    kept, dropped = [], []
    for entry in (e for e in raw.replace(" ", ";").split(";") if e):
        os.environ["TORCH_CUDA_ARCH_LIST"] = entry
        try:
            _get_cuda_arch_flags()
        except ValueError:
            dropped.append(entry)
        else:
            kept.append(entry)

    os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(kept)
    return dropped


def get_package_version():
    text = (THIS_DIR / "sam" / "version.py").read_text()
    return re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text).group(1)


# Non-torch runtime deps (torch/torchvision appended dynamically below).
BASE_DEPS = [
    "numpy>=1.24.4",
    "tqdm>=4.66.1",
    "hydra-core>=1.3.2",
    "iopath>=0.1.10",
    "pillow>=9.4.0",
    "opencv-python>=4.7.0",
    # SAM 3 text tokenizer (CLIP BPE): text normalisation + Unicode regex.
    "ftfy>=6.1.1",
    "regex>=2024.1.1",
    # SAM 3 det↔track association: Hungarian solver (Task 7).
    "scipy>=1.11.0",
]


def _minor_pin(name, ver):
    # "2.11.0+cu128" -> "name==2.11.*" : lock the ABI-critical minor, allow patches.
    major, minor = ver.split("+")[0].split(".")[:2]
    return f"{name}=={major}.{minor}.*"


def get_install_requires(torch, cpu=False):
    """Runtime deps. A DISTRIBUTED CUDA wheel pins torch (and torchvision) to the
    exact minor it was built against (ABI lock) so a mismatched torch errors at
    install. The editable/workspace build and the CPU wheel keep torch loose: the
    pixi environment (or the user) owns the torch version. See _DIST_WHEEL_BUILD."""
    if torch is None or cpu or not _DIST_WHEEL_BUILD:
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
# Failures that mean "this machine cannot compile _C": no toolchain, or nvcc /
# ninja returned an error. Deliberately not bare Exception -- a ValueError or
# TypeError here is a bug in this file or a torch API change, and quietly
# shipping a wheel without _C is exactly how a broken build reaches users.
_RECOVERABLE_BUILD_ERRORS = (CompileError, LinkError, ExecError, RuntimeError)


class _BuildError(Exception):
    """A recoverable _C build problem. Raised before/at compile so the tolerant
    build path (SAM2_ALLOW_BUILD_ERRORS) can degrade to a pure-Python wheel."""


def _require_msvc():
    """Check MSVC's cl.exe is on PATH (Windows); raise ``_BuildError`` otherwise.

    nvcc compiles the host side of sam._C with MSVC. Without an activated MSVC
    environment the build otherwise dies deep inside ninja with an opaque,
    swallowed error. Check up front and tell the user how to fix it.
    """
    if sys.platform != "win32":
        return
    import shutil

    if shutil.which("cl") is None:
        raise _BuildError(
            "MSVC compiler 'cl.exe' was not found on PATH. Building the sam._C "
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
            name="sam._C",
            sources=[
                "sam/csrc/connected_components.cu",
                "sam/csrc/connected_components_binding.cpp",
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

# The local version label (+cuXXXtorchYY) and the exact torch pin in
# install_requires are for DISTRIBUTED wheels only: they let an external
# `pip install sam-...+cu128torch28` refuse a mismatched torch. The in-workspace
# editable build (pixi `sam = { path = ".", editable = true }`) must NOT carry
# them — every pixi environment already pins torch via its cuNNN feature, so baking
# `torch==2.12.*` into the editable metadata makes sam unsatisfiable in any env on
# a different torch line and non-deterministic across solve-groups. Only a real
# wheel build (`bdist_wheel` / `build`) is a distribution; editable_wheel / develop
# / egg_info / dist_info are workspace steps that keep torch loose + no label.
_DIST_WHEEL_BUILD = "bdist_wheel" in sys.argv or "build" in sys.argv

# Degrade quietly only where the result cannot mislead: an editable/dev install
# carries no local label, so a pure-Python fallback is honest and the runtime JIT
# can still supply _C later. A distributed wheel is labelled +cuXXXtorchYY and
# gets cached and redistributed, so the same fallback there ships an artefact
# whose name promises an extension it does not contain -- fail loudly instead.
ALLOW_BUILD_ERRORS = (
    not _DIST_WHEEL_BUILD
    if _ALLOW_BUILD_ERRORS_ENV is None
    else _ALLOW_BUILD_ERRORS_ENV == "1"
)

try:
    import torch  # noqa: F401
except ImportError:
    if _is_building and BUILD_CUDA != "0":
        raise SystemExit(
            "\n"
            "sam build error: PyTorch is not importable in the build environment.\n"
            "\n"
            "sam inspects the installed torch (CUDA vs CPU, version, C++ ABI) to\n"
            "select/build the right native extension and to pin its torch dependency.\n"
            "It therefore must NOT be built with build isolation. Do this instead:\n"
            "\n"
            "  1. Install torch for your target platform first, e.g.\n"
            "       pip install torch torchvision\n"
            "     (or the CUDA build from https://pytorch.org for GPU kernels)\n"
            "  2. Reinstall sam with build isolation disabled:\n"
            "       pip install --no-build-isolation "
            "git+https://github.com/cjaverliat/sam2.git\n"
            "\n"
            "If torch reports a CPU-only build, sam produces a pure-Python (no _C)\n"
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
    # Label only on a distributed wheel; the editable workspace build stays plain
    # so its metadata matches the solve-time metadata across all environments.
    local_label = "cpu" if _DIST_WHEEL_BUILD else None
    # No extension, no download command: a pure-Python build needs no toolchain
    # and is instant, so source-build is always the right fallback.
else:
    _point_cuda_home_at_conda()
    from torch.utils.cpp_extension import BuildExtension
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel

    # Label only on a distributed wheel (it selects the prebuilt release asset and
    # ABI-tags the wheel). The editable workspace build stays plain so its metadata
    # is identical to the solve-time metadata and across every torch line.
    local_label = get_local_label(torch) if _DIST_WHEEL_BUILD else None

    def _refuse_mislabelled_wheel(reason):
        """Abort rather than emit a +cuXXXtorchYY wheel with no sam._C inside.

        Only a distributed wheel carries the CUDA local label; an editable
        install has ``local_label`` None and may degrade freely.
        """
        if _DIST_WHEEL_BUILD and local_label not in (None, "cpu"):
            raise SystemExit(
                f"\nsam build error: {reason}\n"
                f"Refusing to build {version}+{local_label} without sam._C: the "
                "label promises a CUDA extension this wheel would not contain. "
                "Use SAM2_BUILD_CUDA=0 for a correctly labelled pure-Python "
                "wheel, or fix the toolchain.\n"
            )

    def _extensions_or_degrade():
        """Configure the _C extension, degrading to none on a recoverable build
        problem (e.g. missing MSVC) when SAM2_ALLOW_BUILD_ERRORS is set; the runtime
        JIT-compiles _C later. Otherwise re-raise as a fatal build error."""
        try:
            return get_extensions(torch)
        except _BuildError as e:
            if not ALLOW_BUILD_ERRORS:
                raise SystemExit(f"\nsam build error: {e}\n")
            _refuse_mislabelled_wheel(e)
            warnings.warn(
                f"sam: {e} Building a pure-Python wheel instead; sam._C is "
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
            dropped = _sanitize_torch_cuda_arch_list()
            if dropped:
                warnings.warn(
                    "sam: ignoring TORCH_CUDA_ARCH_LIST entries this torch does "
                    f"not support ({', '.join(dropped)}); building for "
                    f"{os.environ['TORCH_CUDA_ARCH_LIST'] or 'the local device'}.",
                    stacklevel=2,
                )
            try:
                super().build_extensions()
            except _RECOVERABLE_BUILD_ERRORS as e:
                if not ALLOW_BUILD_ERRORS:
                    raise
                _refuse_mislabelled_wheel(f"compiling sam._C failed ({e}).")
                warnings.warn(
                    f"sam: compiling sam._C failed ({e}). Installing a pure-Python "
                    "sam; _C is JIT-compiled at runtime on first use "
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
                print(f"sam: fetching prebuilt wheel\n  {url}")
                urllib.request.urlretrieve(url, dest)
                print(f"sam: using prebuilt wheel -> {dest}")
            except (urllib.error.HTTPError, urllib.error.URLError) as e:
                print(
                    f"sam: no matching prebuilt wheel ({e}); building from source."
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
