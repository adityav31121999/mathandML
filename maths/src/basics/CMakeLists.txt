
# src/maths/CMakeLists.txt
cmake_minimum_required(VERSION 3.30.0 FATAL_ERROR)
project(basics C CXX)

include_directories(include)

add_library(basics STATIC
    src/activations.cpp
    src/activationsder.cpp
)

target_link_libraries(basics
    PUBLIC # Important: Make OpenCL linking public
        ${OpenCL_LIBRARIES}
)
