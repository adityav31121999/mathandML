
cmake_minimum_required(VERSION 3.30.0)
project(mockCLsetup)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find OpenCL
find_package(CUDAToolkit REQUIRED)
message(STATUS "Found CUDA version ${CUDAToolkit_VERSION}")
message(STATUS "Using CUDA for compilation.")
message(STATUS "Using CUDA for better efficiency on CUDA-based GPUs:)")
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# Ampere (RTX 30 series)
set(MY_CUDA_ARCH "86" CACHE STRING "Target CUDA Architecture(s) e.g., 75, 86, 90")
set(CMAKE_CUDA_ARCHITECTURES ${MY_CUDA_ARCH})
add_compile_definitions(USE_CUDA)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Check if OpenCL was found
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit Found. Version: ${CUDAToolkit_VERSION}")
    # include_directories(${CUDAToolkit_INCLUDE_DIRS}) # Already done above
    # add_compile_definitions(USE_CUDA) # Already done above
else()
    message(FATAL_ERROR "CUDA not found!")
endif()

# Add executable
add_executable(mockCUDAsetup main.cpp header.hpp def.cu kernel.cu)

# Link OpenCL library
target_link_libraries(mockCUDAsetup PRIVATE CUDA::cudart) # Link CUDA runtime library

# Install target
install(TARGETS mockCUDAsetup DESTINATION bin)
