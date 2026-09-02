cmake_minimum_required(VERSION 3.24)

project(
  Accelerateworld
  VERSION 0.2.0
  DESCRIPTION "Reproducible cross-generation RTX CUDA experiments and benchmarks"
  LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(
  ACCELERATEWORLD_CUDA_ARCHITECTURES
  "75;86;89;120"
  CACHE STRING
  "CUDA SM architectures to compile (RTX 20/30/40/50: 75/86/89/120)"
)
set(CMAKE_CUDA_ARCHITECTURES "${ACCELERATEWORLD_CUDA_ARCHITECTURES}")

option(ACCELERATEWORLD_ENABLE_TESTING "Enable CTest tests" ON)
option(ACCELERATEWORLD_ENABLE_FAST_MATH "Compile kernels with --use_fast_math" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ProjectOptions)

if(ACCELERATEWORLD_ENABLE_TESTING)
  include(CTest)
  enable_testing()
endif()

add_subdirectory(experiments)

message(STATUS "Accelerateworld ${PROJECT_VERSION}")
message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA toolkit: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
