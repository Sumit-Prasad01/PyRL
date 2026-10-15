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
    target_compile_options(${target} PRIVATE /W4 /permissive- /Zc:__cplusplus)
    if (RL_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE /WX)
    endif()
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    if (RL_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE -Werror)
    endif()
  endif()
endfunction()

# ---- library -------------------------------------------------------------
add_library(rl_core STATIC
  src/arena.cpp src/prng.cpp src/mat.cpp src/autograd.cpp
  src/env.cpp   src/model.cpp src/train.cpp)
target_include_directories(rl_core PUBLIC src)
target_compile_features(rl_core PUBLIC cxx_std_20)
rl_configure(rl_core)

# ---- executable ----------------------------------------------------------
add_executable(rl_snake src/main.cpp)
target_link_libraries(rl_snake PRIVATE rl_core)
rl_configure(rl_snake)

# ---- tests ---------------------------------------------------------------
if (RL_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()