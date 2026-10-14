cmake_minimum_required(VERSION 3.20)
project(rl_snake CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(rl_snake
    src/main.cpp src/env.cpp src/model.cpp src/train.cpp
    src/arena.cpp src/prng.cpp src/mat.cpp src/autograd.cpp)

if (MSVC)
  target_compile_options(rl_snake PRIVATE /W4 /permissive- /Zc:__cplusplus)
else()
  target_compile_options(rl_snake PRIVATE -Wall -Wextra -Wpedantic)
endif()