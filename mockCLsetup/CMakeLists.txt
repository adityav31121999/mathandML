
cmake_minimum_required(VERSION 3.30.0)
project(mockCLsetup)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find OpenCL
set(OpenCL_CL_VERSION "3.0")
set(CL_TARGET_OPENCL_VERSION "3.0")
find_package(OpenCL REQUIRED)

# Check if OpenCL was found
if(OpenCL_FOUND)
    message(STATUS "Found OpenCL: ${OpenCL_LIBRARIES}")
    include_directories(${OpenCL_INCLUDE_DIRS})
    add_compile_definitions(USE_OPENCL)
else()
    message(FATAL_ERROR "OpenCL not found!")
endif()

# Add executable
add_executable(mockCLsetup main.cpp header.hpp def.cpp kernel.cl)

# Link OpenCL library
target_link_libraries(mockCLsetup PRIVATE OpenCL::OpenCL)

# Install target
install(TARGETS mockCLsetup DESTINATION bin)
