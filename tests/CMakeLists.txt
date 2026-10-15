include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.5.2)
FetchContent_MakeAvailable(Catch2)

find_package(Threads REQUIRED)

add_executable(rl_tests
  test_base.cpp
  test_prng.cpp
  test_arena.cpp
  test_mat.cpp
  test_autograd.cpp
  test_gradcheck.cpp
  test_env.cpp
  test_train.cpp)

target_link_libraries(rl_tests PRIVATE rl_core Catch2::Catch2WithMain Threads::Threads)
rl_configure(rl_tests)

add_test(NAME unit_tests COMMAND rl_tests "~[slow]")
set_tests_properties(unit_tests PROPERTIES LABELS fast)

add_test(NAME slow_tests COMMAND rl_tests "[slow]")
set_tests_properties(slow_tests PROPERTIES LABELS slow TIMEOUT 600)
