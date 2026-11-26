cmake_minimum_required(VERSION 3.16)
project(cg_hybrid CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to an optimised build: benchmarks are meaningless at -O0.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Apple Clang has no bundled OpenMP runtime: point CMake at Homebrew's libomp.
if(APPLE AND NOT DEFINED OpenMP_ROOT)
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE BREW_RESULT
        ERROR_QUIET)
    if(BREW_RESULT EQUAL 0 AND EXISTS "${LIBOMP_PREFIX}")
        set(OpenMP_ROOT "${LIBOMP_PREFIX}")
    endif()
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(cg src/main.cpp
                src/csr.cpp
                src/spmv.cpp
                src/vec_ops.cpp
                src/cg.cpp
                src/partition.cpp
                src/halo.cpp)
target_include_directories(cg PRIVATE include)
target_compile_options(cg PRIVATE -Wall -Wextra)
target_link_libraries(cg PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)