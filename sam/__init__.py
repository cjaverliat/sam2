# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Import torch eagerly so the native extension (sam2._C) resolves its torch
# DLLs. On Windows, `import torch` calls os.add_dll_directory(<torch>/lib),
# registering c10.dll / torch_cpu.dll / c10_cuda.dll on the loader path;
# without it, loading sam2._C fails with "DLL load failed".
import torch  # noqa: F401

from hydra import initialize_config_module
from hydra.core.global_hydra import GlobalHydra

if not GlobalHydra.instance().is_initialized():
    initialize_config_module("sam", version_base="1.2")
