cmake_minimum_required(VERSION 3.20)

project(tensora_native VERSION 0.2.0 LANGUAGES C CXX)

option(TENSORA_BUILD_TESTS "Build Tensora native tests" ON)
option(TENSORA_BUILD_BENCHMARKS "Build Tensora native benchmarks" ON)
option(TENSORA_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(TENSORA_WITH_TORCH "Build the LibTorch training backend" OFF)
option(TENSORA_WITH_ONNXRUNTIME "Build the ONNX Runtime inference backend" OFF)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

set(TENSORA_NATIVE_SOURCES
  src/backends/cpu/cpu_backend.cc
  src/c_api.cc
  src/core/status.cc
  src/memory/cpu_storage.cc
  src/runtime/dispatcher.cc
  src/runtime/handle_registry.cc
  src/tensor/shape.cc
  src/tensor/tensor.cc
  src/training/training_c_api.cc
)

if(TENSORA_WITH_TORCH)
  find_package(Torch REQUIRED)
  list(APPEND TENSORA_NATIVE_SOURCES
    src/training/torch_backend.cc
    src/training/torch_storage.cc
    src/training/training_bridge_torch.cc
  )
else()
  list(APPEND TENSORA_NATIVE_SOURCES
    src/training/training_bridge_stub.cc
  )
endif()

add_library(tensora_native SHARED ${TENSORA_NATIVE_SOURCES})

target_compile_definitions(tensora_native PRIVATE TENSORA_NATIVE_BUILD)
if(TENSORA_WITH_TORCH)
  target_compile_definitions(tensora_native PRIVATE TENSORA_WITH_TORCH=1)
endif()

target_include_directories(tensora_native
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(tensora_native PRIVATE Threads::Threads)

if(TENSORA_WITH_TORCH)
  target_link_libraries(tensora_native PRIVATE ${TORCH_LIBRARIES})
  if(TORCH_CXX_FLAGS)
    separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(tensora_native PRIVATE ${TORCH_CXX_FLAGS_LIST})
  endif()
endif()

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
  if(TENSORA_WITH_TORCH)
    message(FATAL_ERROR "Run Tensora sanitizers in a core build; third-party LibTorch binaries are not instrumented by this configuration")
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

  add_executable(tensora_handle_registry_tests
    tests/handle_registry_test.cc
    src/core/status.cc
    src/runtime/handle_registry.cc
  )
  target_include_directories(tensora_handle_registry_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  target_link_libraries(tensora_handle_registry_tests PRIVATE Threads::Threads)

  add_executable(tensora_c_abi_tests tests/c_abi_test.c)
  target_include_directories(tensora_c_abi_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(tensora_c_abi_tests PRIVATE tensora_native)
  set_target_properties(tensora_c_abi_tests PROPERTIES LINKER_LANGUAGE CXX)

  if(TENSORA_WITH_TORCH)
    add_executable(tensora_training_tests tests/training_test.cc)
    target_include_directories(tensora_training_tests PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(tensora_training_tests PRIVATE tensora_native)
  endif()

  if(MSVC)
    target_compile_options(tensora_native_tests PRIVATE /W4 /WX /permissive-)
    target_compile_options(tensora_handle_registry_tests PRIVATE /W4 /WX /permissive-)
    target_compile_options(tensora_c_abi_tests PRIVATE /W4 /WX)
    if(TENSORA_WITH_TORCH)
      target_compile_options(tensora_training_tests PRIVATE /W4 /WX /permissive-)
    endif()
  else()
    target_compile_options(tensora_native_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(tensora_handle_registry_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(tensora_c_abi_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    if(TENSORA_WITH_TORCH)
      target_compile_options(tensora_training_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
  endif()

  if(TENSORA_ENABLE_SANITIZERS)
    target_compile_options(tensora_native_tests PRIVATE ${TENSORA_SANITIZER_FLAGS})
    target_link_options(tensora_native_tests PRIVATE -fsanitize=address,undefined)
    target_compile_options(tensora_handle_registry_tests PRIVATE ${TENSORA_SANITIZER_FLAGS})
    target_link_options(tensora_handle_registry_tests PRIVATE -fsanitize=address,undefined)
    target_compile_options(tensora_c_abi_tests PRIVATE ${TENSORA_SANITIZER_FLAGS})
    target_link_options(tensora_c_abi_tests PRIVATE -fsanitize=address,undefined)
  endif()

  add_test(NAME tensora_native_tests COMMAND tensora_native_tests)
  add_test(NAME tensora_handle_registry_tests COMMAND tensora_handle_registry_tests)
  add_test(NAME tensora_c_abi_tests COMMAND tensora_c_abi_tests)
  if(TENSORA_WITH_TORCH)
    add_test(NAME tensora_training_tests COMMAND tensora_training_tests)
  endif()
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
