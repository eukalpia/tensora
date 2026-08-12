cmake_minimum_required(VERSION 3.20)

project(tensora_native VERSION 0.1.0 LANGUAGES CXX)

option(TENSORA_BUILD_TESTS "Build Tensora native tests" ON)
option(TENSORA_BUILD_BENCHMARKS "Build Tensora native benchmarks" ON)
option(TENSORA_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

add_library(tensora_native SHARED
  src/backends/cpu/cpu_backend.cc
  src/c_api.cc
  src/core/status.cc
  src/memory/cpu_storage.cc
  src/runtime/dispatcher.cc
  src/runtime/handle_registry.cc
  src/tensor/shape.cc
  src/tensor/tensor.cc
)

target_compile_definitions(tensora_native PRIVATE TENSORA_NATIVE_BUILD)
target_include_directories(tensora_native
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(tensora_native PRIVATE Threads::Threads)

if(MSVC)
  target_compile_options(tensora_native PRIVATE /W4 /WX /permissive-)
else()
  target_compile_options(tensora_native PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

if(TENSORA_ENABLE_SANITIZERS)
  if(MSVC)
    message(FATAL_ERROR "TENSORA_ENABLE_SANITIZERS is not supported by this CMake configuration on MSVC")
  endif()
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    message(FATAL_ERROR "TENSORA_ENABLE_SANITIZERS requires Clang or GCC")
  endif()

  set(TENSORA_SANITIZER_FLAGS -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_compile_options(tensora_native PRIVATE ${TENSORA_SANITIZER_FLAGS})
  target_link_options(tensora_native PRIVATE -fsanitize=address,undefined)
endif()

if(TENSORA_BUILD_TESTS)
  enable_testing()

  add_executable(tensora_native_tests tests/tensor_core_test.cc)
  target_include_directories(tensora_native_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  target_link_libraries(tensora_native_tests PRIVATE tensora_native Threads::Threads)

  if(MSVC)
    target_compile_options(tensora_native_tests PRIVATE /W4 /WX /permissive-)
  else()
    target_compile_options(tensora_native_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()

  if(TENSORA_ENABLE_SANITIZERS)
    target_compile_options(tensora_native_tests PRIVATE ${TENSORA_SANITIZER_FLAGS})
    target_link_options(tensora_native_tests PRIVATE -fsanitize=address,undefined)
  endif()

  add_test(NAME tensora_native_tests COMMAND tensora_native_tests)
endif()

if(TENSORA_BUILD_BENCHMARKS)
  add_executable(tensora_native_benchmark ../benchmarks/native_tensor_benchmark.cc)
  target_include_directories(tensora_native_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(tensora_native_benchmark PRIVATE tensora_native)

  if(MSVC)
    target_compile_options(tensora_native_benchmark PRIVATE /W4 /WX /permissive-)
  else()
    target_compile_options(tensora_native_benchmark PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()

  if(TENSORA_ENABLE_SANITIZERS)
    target_compile_options(tensora_native_benchmark PRIVATE ${TENSORA_SANITIZER_FLAGS})
    target_link_options(tensora_native_benchmark PRIVATE -fsanitize=address,undefined)
  endif()
endif()
