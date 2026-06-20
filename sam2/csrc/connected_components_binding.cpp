// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.

// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

// pybind binding for the connected-components op. Kept in a .cpp so
// <torch/extension.h> is compiled by plain MSVC, not nvcc (see the .cu).
#include <torch/extension.h>

#include <vector>

// Defined in connected_components.cu
std::vector<at::Tensor> get_connected_componnets(const at::Tensor& inputs);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "get_connected_componnets",
      &get_connected_componnets,
      "get_connected_componnets");
}
