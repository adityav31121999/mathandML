
# CMakeLists.txt for Multilayer Perceptron
cmake_minimum_required(VERSION 3.30.0 FATAL_ERROR)
project(MLP CXX)

# "include" folder
include_directories(include)

# OpenCL setup
find_package(CUDAToolkit REQUIRED)
message(STATUS "Found CUDA version ${CUDAToolkit_VERSION}")
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# Ampere (RTX 30 series)
set(MY_CUDA_ARCH "86" CACHE STRING "Target CUDA Architecture(s) e.g., 75, 86, 90")
set(CMAKE_CUDA_ARCHITECTURES ${MY_CUDA_ARCH})
include_directories(${CUDAToolkit_INCLUDE_DIRS}) # Ensure CUDA headers are available

add_library(mlp STATIC
    # activation functions
    src/activations.cpp

    # mlp files
    src/mlp.cpp
    src/forprop.cpp
    src/backprop.cpp
    src/train.cpp
    src/weights.cpp
    src/loss.cpp
)
