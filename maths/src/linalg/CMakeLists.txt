
# src/maths/CMakeLists.txt
cmake_minimum_required(VERSION 3.30.0 FATAL_ERROR)
project(linalg C CXX)

include_directories(include)

add_library(linalg STATIC
    src/mat.cpp
    src/vec1.cpp
    src/vec2.cpp
)

target_link_libraries(linalg
    PUBLIC # Important: Make OpenCL linking public
        ${OpenCL_LIBRARIES}
)
