cmake_minimum_required(VERSION 3.21...3.28)

# Force Release before find_package(Python). A Debug build links Python's debug
# import lib (pythonXX_d.lib), which standard CPython does not ship -> LNK1104.
# (scikit-build-core ignores -DCMAKE_BUILD_TYPE when -G Ninja is set.)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

project(sam2_ext LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES all)

if(DEFINED ENV{SAM2_BUILD_CUDA} AND "$ENV{SAM2_BUILD_CUDA}" STREQUAL "0")
    message(STATUS "SAM2_BUILD_CUDA is 0. Skipping CUDA extension build.")
    return()
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import torch.utils; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")
find_package(Torch REQUIRED)

if(NOT TORCH_INCLUDE_DIRS)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.__path__[0])"
        OUTPUT_VARIABLE TORCH_ROOT_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(TORCH_INCLUDE_DIRS "${TORCH_ROOT_PATH}/include")
endif()

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.__path__[0])"
    OUTPUT_VARIABLE TORCH_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

include_directories("${TORCH_PATH}/include")
include_directories("${TORCH_PATH}/include/torch/csrc")
include_directories("${TORCH_PATH}/include/torch/csrc/api/include")
include_directories("${TORCH_PATH}/include/torch/csrc/c10")

if(DEFINED ENV{CONDA_PREFIX})
    if(WIN32)
        include_directories("$ENV{CONDA_PREFIX}/Library/include")
    else()
        include_directories("$ENV{CONDA_PREFIX}/include")
    endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_compile_definitions(
    CUDA_HAS_FP16=1
    __CUDA_NO_HALF_OPERATORS__
    __CUDA_NO_HALF_CONVERSIONS__
    __CUDA_NO_HALF2_OPERATORS__
)

# Soft-fail opt-in: if SAM2_BUILD_ALLOW_ERRORS=1 and the extension can't be
# compiled on this machine, skip it and install pure-Python sam2 instead of
# aborting the whole install. A failed `cmake --build` cannot be made
# non-fatal, so probe at configure time with try_compile (STATIC_LIBRARY target
# type -> compile only, no torch link needed). Default (unset) keeps the hard
# failure with full compiler diagnostics. Runtime degrades gracefully: callers
# of get_connected_components() catch the missing _C (see sam2/utils/misc.py).
if(DEFINED ENV{SAM2_BUILD_ALLOW_ERRORS} AND "$ENV{SAM2_BUILD_ALLOW_ERRORS}" STREQUAL "1")
    set(_SAM2_TRYC_INCLUDES
        "${TORCH_PATH}/include"
        "${TORCH_PATH}/include/torch/csrc"
        "${TORCH_PATH}/include/torch/csrc/api/include"
        "${TORCH_PATH}/include/torch/csrc/c10"
        "${Python_INCLUDE_DIRS}"
    )
    if(DEFINED ENV{CONDA_PREFIX})
        if(WIN32)
            list(APPEND _SAM2_TRYC_INCLUDES "$ENV{CONDA_PREFIX}/Library/include")
        else()
            list(APPEND _SAM2_TRYC_INCLUDES "$ENV{CONDA_PREFIX}/include")
        endif()
    endif()

    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
    try_compile(_SAM2_C_COMPILES "${CMAKE_BINARY_DIR}/try_sam2c"
        SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/sam2/csrc/connected_components.cu"
            "${CMAKE_CURRENT_SOURCE_DIR}/sam2/csrc/connected_components_binding.cpp"
        CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${_SAM2_TRYC_INCLUDES}"
        COMPILE_DEFINITIONS
            -DTORCH_EXTENSION_NAME=_C
            -DCUDA_HAS_FP16=1
            -D__CUDA_NO_HALF_OPERATORS__
            -D__CUDA_NO_HALF_CONVERSIONS__
            -D__CUDA_NO_HALF2_OPERATORS__
        OUTPUT_VARIABLE _SAM2_C_BUILD_LOG
    )
    unset(CMAKE_TRY_COMPILE_TARGET_TYPE)

    if(NOT _SAM2_C_COMPILES)
        message(WARNING
            "SAM2_BUILD_ALLOW_ERRORS=1: the CUDA extension failed to compile, "
            "skipping it. sam2 will install without _C; postprocessing that "
            "needs the connected-components kernel is disabled.\n"
            "Build log:\n${_SAM2_C_BUILD_LOG}")
        return()
    endif()
endif()

python_add_library(_C MODULE
    sam2/csrc/connected_components.cu
    sam2/csrc/connected_components_binding.cpp
)

# Names the pybind module "_C" (torch's CUDAExtension injects this; a manual
# CMake build must define it itself).
target_compile_definitions(_C PRIVATE TORCH_EXTENSION_NAME=_C)

# at::Tensor <-> Python type casters (THPVariable_*) live in torch_python, which
# is not part of TORCH_LIBRARIES. Without it: LNK2019 unresolved external.
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_PATH}/lib" REQUIRED)

target_link_libraries(_C PRIVATE
    "${TORCH_LIBRARIES}"
    "${TORCH_PYTHON_LIBRARY}"
    c10_cuda
    c10
)

set_target_properties(_C PROPERTIES
    PREFIX ""
    SUFFIX ".pyd"
)

install(TARGETS _C DESTINATION sam2)