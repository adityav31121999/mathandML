
# CMakeLists.txt for Multilayer Perceptron
cmake_minimum_required(VERSION 3.30.0 FATAL_ERROR)
project(MLP CXX)

# "include" folder
include_directories(include)

# OpenCL setup
set(OpenCL_CL_VERSION "300")
set(CL_TARGET_OPENCL_VERSION "300")
find_package(OpenCL REQUIRED)
message(STATUS "Found OpenCL version ${OpenCL_CL_VERSION}")
include_directories(${OpenCL_INCLUDE_DIRS})

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
