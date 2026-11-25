cmake_minimum_required(VERSION 3.18)
project(avatarforcing_cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(AVATARFORCING_BUILD_TESTS "Build ctest targets" ON)

# -march=native (GGML_NATIVE) and tinyBLAS SGEMM (GGML_LLAMAFILE) are free
# speedups on the CPU backend. Use -DGGML_NATIVE=OFF for portable binaries.
if(NOT DEFINED GGML_NATIVE)
  set(GGML_NATIVE ON CACHE BOOL "ggml: optimize the build for the current system" FORCE)
endif()
if(NOT DEFINED GGML_LLAMAFILE)
  set(GGML_LLAMAFILE ON CACHE BOOL "ggml: use LLAMAFILE (tinyBLAS SGEMM)" FORCE)
endif()

add_subdirectory(third_party/ggml)

add_library(avatarforcing STATIC
    src/backend.cpp
    src/model_loader.cpp
    src/flow.cpp)
target_include_directories(avatarforcing PUBLIC src)
target_link_libraries(avatarforcing PUBLIC ggml)

if(AVATARFORCING_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
