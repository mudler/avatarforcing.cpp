cmake_minimum_required(VERSION 3.18)
project(avatarforcing_cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(AVATARFORCING_BUILD_TESTS "Build ctest targets" ON)
option(AVATARFORCING_BUILD_CLI "Build the avatarforcing-cli tool" ON)
option(AVATARFORCING_BUILD_SHARED "Build libavatarforcing, the shared library with the C API" ON)

# The shared library holds ggml, so that it loads with dlopen and nothing
# else: ggml is built as static libraries with position-independent code,
# unless BUILD_SHARED_LIBS is set. The library exports only the C API.
if(AVATARFORCING_BUILD_SHARED)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "build ggml as shared libraries")
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# -march=native (GGML_NATIVE) and tinyBLAS SGEMM (GGML_LLAMAFILE) are free
# speedups on the CPU backend. Use -DGGML_NATIVE=OFF for portable binaries.
if(NOT DEFINED GGML_NATIVE)
  set(GGML_NATIVE ON CACHE BOOL "ggml: optimize the build for the current system" FORCE)
endif()
if(NOT DEFINED GGML_LLAMAFILE)
  set(GGML_LLAMAFILE ON CACHE BOOL "ggml: use LLAMAFILE (tinyBLAS SGEMM)" FORCE)
endif()

# ggml is used with the patches in patches/ applied: a grid_sample operator
# with CPU and CUDA kernels, and a switch that keeps the CUDA GEMMs in full
# float32 (no TF32). The submodule stays clean; configure copies it to the
# build tree and applies the patches there.
set(AF_GGML_SRC ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml)
set(AF_GGML_PATCHED ${CMAKE_CURRENT_BINARY_DIR}/ggml-patched)
file(GLOB AF_GGML_PATCHES ${CMAKE_CURRENT_SOURCE_DIR}/patches/ggml-*.patch)
list(SORT AF_GGML_PATCHES)
set(AF_GGML_STAMP_VALUE "")
foreach(p ${AF_GGML_PATCHES})
  file(SHA256 ${p} h)
  string(APPEND AF_GGML_STAMP_VALUE "${h};")
endforeach()
execute_process(COMMAND git -C ${AF_GGML_SRC} rev-parse HEAD
                OUTPUT_VARIABLE AF_GGML_REV OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
string(APPEND AF_GGML_STAMP_VALUE "${AF_GGML_REV}")
set(AF_GGML_STAMP_FILE ${AF_GGML_PATCHED}/.af-patch-stamp)
set(AF_GGML_OLD_STAMP "")
if(EXISTS ${AF_GGML_STAMP_FILE})
  file(READ ${AF_GGML_STAMP_FILE} AF_GGML_OLD_STAMP)
endif()
if(NOT AF_GGML_OLD_STAMP STREQUAL AF_GGML_STAMP_VALUE)
  message(STATUS "avatarforcing: copying ggml and applying ${CMAKE_CURRENT_SOURCE_DIR}/patches")
  file(REMOVE_RECURSE ${AF_GGML_PATCHED})
  file(COPY ${AF_GGML_SRC}/ DESTINATION ${AF_GGML_PATCHED} PATTERN ".git" EXCLUDE)
  foreach(p ${AF_GGML_PATCHES})
    # The copy can sit inside another git work tree (the build tree of a
    # checkout); stop git from finding it, or git apply skips every file.
    execute_process(COMMAND ${CMAKE_COMMAND} -E env GIT_CEILING_DIRECTORIES=${CMAKE_CURRENT_BINARY_DIR}
                            git apply --whitespace=nowarn ${p}
                    WORKING_DIRECTORY ${AF_GGML_PATCHED} RESULT_VARIABLE rc)
    if(NOT rc EQUAL 0)
      message(FATAL_ERROR "avatarforcing: cannot apply ${p} to ggml")
    endif()
  endforeach()
  file(STRINGS ${AF_GGML_PATCHED}/include/ggml.h af_check REGEX "ggml_grid_sample")
  if(NOT af_check)
    message(FATAL_ERROR "avatarforcing: the ggml patches did not apply")
  endif()
  file(WRITE ${AF_GGML_STAMP_FILE} "${AF_GGML_STAMP_VALUE}")
endif()
# CUDA in strict float32 (see patches/): no -use_fast_math, no TF32 GEMMs,
# two-pass LayerNorm statistics, a race-free block reduction; plus the
# convolution kernels and fusions of the decoder, the op profiler and stream
# priorities (docs/gpu.md).
set(GGML_CUDA_NO_FAST_MATH ON)
add_subdirectory(${AF_GGML_PATCHED} ggml)
if(TARGET ggml-cuda)
  target_compile_definitions(ggml-cuda PRIVATE GGML_CUDA_NO_TF32 GGML_CUDA_NORM_TWO_PASS GGML_CUDA_CONV2D_IGEMM
                                                     GGML_CUDA_FUSE_CHAN_AFFINE GGML_CUDA_CONV2D_DW_TILED
                                                     GGML_CUDA_BLOCK_REDUCE_SYNC GGML_CUDA_CONV2D_TILE
                                                     GGML_CUDA_FUSE_DECODER GGML_CUDA_OP_PROFILE GGML_CUDA_STREAM_PRIORITY)
endif()

add_library(avatarforcing STATIC
    src/common.cpp
    src/backend.cpp
    src/model_loader.cpp
    src/flow.cpp
    src/sampler.cpp
    src/grid_sample.cpp
    src/motion.cpp
    src/wav2vec2.cpp
    src/audio_io.cpp
    src/pipeline.cpp
    src/live.cpp)
# grid_sample.cpp reproduces the rounding of the PyTorch CPU kernel with
# explicit fmaf calls; motion.cpp builds the base grids like numpy. Neither may
# get extra fused multiply-adds from the compiler.
set_source_files_properties(src/grid_sample.cpp src/motion.cpp PROPERTIES
    COMPILE_OPTIONS "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-ffp-contract=off>")
target_include_directories(avatarforcing PUBLIC src include)
target_include_directories(avatarforcing SYSTEM PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/stb
    ${PROJECT_SOURCE_DIR}/third_party/dr_libs)
target_link_libraries(avatarforcing PUBLIC ggml)
set_target_properties(avatarforcing PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)

# libavatarforcing: the C API (include/avatarforcing.h). See docs/capi.md.
if(AVATARFORCING_BUILD_SHARED)
  add_library(avatarforcing_shared SHARED src/capi.cpp)
  target_include_directories(avatarforcing_shared PUBLIC include)
  target_compile_definitions(avatarforcing_shared PRIVATE AVATARFORCING_BUILD)
  target_link_libraries(avatarforcing_shared PRIVATE avatarforcing)
  set_target_properties(avatarforcing_shared PROPERTIES
      OUTPUT_NAME avatarforcing
      CXX_VISIBILITY_PRESET hidden
      VISIBILITY_INLINES_HIDDEN ON)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Export only avatarforcing_*: the static ggml and the C++ code stay
    # local to the library.
    target_link_options(avatarforcing_shared PRIVATE
        "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/avatarforcing.map"
        "LINKER:--no-undefined")
    set_property(TARGET avatarforcing_shared APPEND PROPERTY LINK_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/avatarforcing.map)
  endif()
  install(TARGETS avatarforcing_shared LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
  install(FILES include/avatarforcing.h DESTINATION include)
endif()

if(AVATARFORCING_BUILD_CLI)
  add_subdirectory(examples/cli)
endif()

if(AVATARFORCING_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
