
# root CMakeLists.txt
cmake_minimum_required(VERSION 3.30.0 FATAL_ERROR)

# project and language
project(maths C CXX)
set(VERSION 0.0.0.1)

# language settings
enable_language(C CXX)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# if(MSVC)
#     add_compile_options(/std::c++latest)
#     set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
#     set(BUILD_SHARED_LIBS TRUE)
# endif()

# binaries for debug and release
set(OSbit 32)
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(OSbit 64)
endif()
set(FullOutputDir "${CMAKE_SOURCE_DIR}/bin/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}-${OSbit}/${CMAKE_BUILD_TYPE}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${FullOutputDir}/static_libs")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${FullOutputDir}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${FullOutputDir}")

# subprojects lists for subdirectories
set(subProjects
    src/basics
    src/fourier
    src/linalg
    src/plots
    src/poly
    src/stats
    src/tensor
)

# Find OpenCL package and include directories
set(OpenCL_CL_VERSION "300") 
set(CL_TARGET_OPENCL_VERSION "300")
add_definitions(-DCL_HPP_TARGET_OPENCL_VERSION=300)

# include each subdirectory
foreach(projects IN LISTS subProjects)
    include_directories(${projects})
    if(NOT TARGET ${projects})
        add_subdirectory(${projects})
    endif()
endforeach()
