cmake_minimum_required(VERSION 3.20)
project(rl_snake LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(RL_BUILD_TESTS        "Build the test suite"                      ON)
option(RL_WARNINGS_AS_ERRORS "Treat warnings as errors"                 OFF)
option(RL_SANITIZE           "ASan + UBSan (GCC/Clang only)"             OFF)
option(RL_COVERAGE           "Coverage instrumentation (GCC/Clang only)" OFF)

# Global flags so every target (library, tests, exe) is instrumented consistently.
if (RL_SANITIZE AND NOT MSVC)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer
                      -fno-sanitize-recover=undefined)
  add_link_options(-fsanitize=address,undefined)
endif()
if (RL_COVERAGE AND NOT MSVC)
  add_compile_options(--coverage -O0 -g)
  add_link_options(--coverage)
endif()

function(rl_configure target)
  if (MSVC)
    target_compile_options(${target} PRIVATE /W4 /permissive- /Zc:__cplusplus /arch:AVX2)
    if (RL_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE /WX)
    endif()
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -mavx2 -mfma)
    if (RL_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE -Werror)
    endif()
  endif()
endfunction()

# ---- CUDA auto-detection -------------------------------------------------
option(RL_ENABLE_CUDA "Enable CUDA GPU acceleration" ON)
set(RL_CUDA_SOURCES "")

if (RL_ENABLE_CUDA)
  find_package(CUDAToolkit QUIET)
  find_program(CLANG_CUDA_COMPILER clang++.exe
    PATHS "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/Llvm/x64/bin"
  )
  if (CUDAToolkit_FOUND AND CLANG_CUDA_COMPILER)
    message(STATUS "CUDA acceleration enabled via CUDAToolkit ${CUDAToolkit_VERSION} & Clang CUDA compiler")
    set(CUDA_BACKEND_OBJ "${CMAKE_CURRENT_BINARY_DIR}/cuda_backend.obj")
    add_custom_command(
      OUTPUT "${CUDA_BACKEND_OBJ}"
      COMMAND "${CLANG_CUDA_COMPILER}"
              "-std=c++20"
              "-DRL_USE_CUDA"
              "-O3"
              "-fms-runtime-lib=dll"
              "--cuda-path=${CUDAToolkit_LIBRARY_ROOT}"
              "--cuda-gpu-arch=sm_86"
              "-I${CMAKE_CURRENT_SOURCE_DIR}/src"
              "-c" "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/cuda_backend.cu"
              "-o" "${CUDA_BACKEND_OBJ}"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/cuda_backend.cu"
              "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/cuda_backend.hpp"
      COMMENT "Compiling cuda_backend.cu to sm_86 object"
      VERBATIM
    )
    add_custom_target(build_cuda_backend DEPENDS "${CUDA_BACKEND_OBJ}")
    list(APPEND RL_CUDA_SOURCES "${CUDA_BACKEND_OBJ}")
    add_compile_definitions(RL_USE_CUDA)
  else()
    message(STATUS "CUDA Toolkit or Clang compiler not found; building in CPU-only mode.")
  endif()
endif()

# ---- library -------------------------------------------------------------
add_library(rl_core STATIC
  src/arena.cpp src/prng.cpp src/mat.cpp src/autograd.cpp
  src/env.cpp   src/model.cpp src/train.cpp src/optimizer.cpp
  src/nn.cpp    src/serialize.cpp src/device.cpp src/thread_pool.cpp
  src/matmul_ladder.cpp src/fused.cpp
  src/envs/bandit.cpp src/envs/gridworld.cpp src/envs/cartpole.cpp
  src/envs/mountain_car.cpp src/envs/pendulum.cpp src/envs/snake_v2.cpp
  src/vec_env.cpp src/eval.cpp
  ${RL_CUDA_SOURCES})
target_include_directories(rl_core PUBLIC src)
target_compile_features(rl_core PUBLIC cxx_std_20)
rl_configure(rl_core)

if (TARGET build_cuda_backend)
  add_dependencies(rl_core build_cuda_backend)
  target_link_libraries(rl_core PUBLIC CUDA::cudart)
  target_include_directories(rl_core SYSTEM PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
endif()

# ---- executable ----------------------------------------------------------
add_executable(rl_snake src/main.cpp)
target_link_libraries(rl_snake PRIVATE rl_core)
rl_configure(rl_snake)

# ---- benchmark -----------------------------------------------------------
add_executable(rl_bench bench/bench_matmul.cpp)
target_link_libraries(rl_bench PRIVATE rl_core)
rl_configure(rl_bench)

# ---- tests ---------------------------------------------------------------
if (RL_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()