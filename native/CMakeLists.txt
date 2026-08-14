cmake_minimum_required(VERSION 3.20)

project(tensora_native LANGUAGES C CXX)

option(TENSORA_BUILD_TESTS "Build Tensora native tests" ON)
option(TENSORA_BUILD_BENCHMARKS "Build Tensora native benchmarks" ON)
option(TENSORA_ENABLE_SANITIZERS "Enable ASan and UBSan" OFF)
option(TENSORA_ENABLE_THREAD_SANITIZER "Enable ThreadSanitizer" OFF)
option(TENSORA_WITH_TORCH "Enable LibTorch training support" OFF)
option(TENSORA_WITH_ONNXRUNTIME "Enable ONNX Runtime inference support" OFF)
option(TENSORA_WITH_ORT_DML "Enable DirectML ONNX provider factory" OFF)
option(TENSORA_WITH_ORT_COREML "Enable CoreML ONNX provider factory" OFF)
set(TENSORA_ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime distribution root")

if(TENSORA_ENABLE_SANITIZERS AND TENSORA_ENABLE_THREAD_SANITIZER)
  message(FATAL_ERROR "Address/UB sanitizers and ThreadSanitizer cannot be enabled together")
endif()
if((TENSORA_WITH_ORT_DML OR TENSORA_WITH_ORT_COREML) AND NOT TENSORA_WITH_ONNXRUNTIME)
  message(FATAL_ERROR "Platform ONNX provider factories require TENSORA_WITH_ONNXRUNTIME=ON")
endif()
if(TENSORA_WITH_ORT_DML AND NOT WIN32)
  message(FATAL_ERROR "DirectML ONNX provider factory is supported only on Windows")
endif()
if(TENSORA_WITH_ORT_COREML AND NOT APPLE)
  message(FATAL_ERROR "CoreML ONNX provider factory is supported only on Apple platforms")
endif()

find_package(Threads REQUIRED)
if(TENSORA_WITH_TORCH)
  find_package(Torch REQUIRED)
endif()
if(TENSORA_WITH_ONNXRUNTIME)
  if(NOT TENSORA_ONNXRUNTIME_ROOT)
    message(FATAL_ERROR "TENSORA_ONNXRUNTIME_ROOT is required when TENSORA_WITH_ONNXRUNTIME=ON")
  endif()
  find_path(TENSORA_ONNXRUNTIME_INCLUDE_DIR
    onnxruntime_cxx_api.h
    PATHS ${TENSORA_ONNXRUNTIME_ROOT}/include
    NO_DEFAULT_PATH
    REQUIRED
  )
  find_library(TENSORA_ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS ${TENSORA_ONNXRUNTIME_ROOT}/lib
    NO_DEFAULT_PATH
    REQUIRED
  )
endif()

set(TENSORA_NATIVE_SOURCES
  src/backends/cpu/cpu_backend.cc
  src/c_api.cc
  src/core/status.cc
  src/inference/inference_c_api.cc
  src/memory/cpu_storage.cc
  src/runtime/device_codec.cc
  src/runtime/dispatcher.cc
  src/runtime/handle_registry.cc
  src/tensor/shape.cc
  src/tensor/tensor.cc
  src/training/training_c_api.cc
)

if(TENSORA_WITH_TORCH)
  list(APPEND TENSORA_NATIVE_SOURCES
    src/training/torch_backend.cc
    src/training/torch_storage.cc
    src/training/training_bridge_torch.cc
  )
else()
  list(APPEND TENSORA_NATIVE_SOURCES src/training/training_bridge_stub.cc)
endif()

if(TENSORA_WITH_ONNXRUNTIME)
  list(APPEND TENSORA_NATIVE_SOURCES src/inference/inference_bridge_onnx.cc)
else()
  list(APPEND TENSORA_NATIVE_SOURCES src/inference/inference_bridge_stub.cc)
endif()

add_library(tensora_native SHARED ${TENSORA_NATIVE_SOURCES})

target_compile_features(tensora_native PUBLIC cxx_std_20)
target_compile_definitions(tensora_native PRIVATE TENSORA_NATIVE_BUILD=1)
if(TENSORA_WITH_TORCH)
  target_compile_definitions(tensora_native PRIVATE TENSORA_WITH_TORCH=1)
endif()
if(TENSORA_WITH_ONNXRUNTIME)
  target_compile_definitions(tensora_native PRIVATE TENSORA_WITH_ONNXRUNTIME=1)
endif()
if(TENSORA_WITH_ORT_DML)
  target_compile_definitions(tensora_native PRIVATE TENSORA_WITH_ORT_DML=1)
endif()
if(TENSORA_WITH_ORT_COREML)
  target_compile_definitions(tensora_native PRIVATE TENSORA_WITH_ORT_COREML=1)
endif()

target_include_directories(tensora_native
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
if(TENSORA_WITH_TORCH)
  target_include_directories(tensora_native SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
endif()
if(TENSORA_WITH_ONNXRUNTIME)
  target_include_directories(tensora_native PRIVATE ${TENSORA_ONNXRUNTIME_INCLUDE_DIR})
endif()

target_link_libraries(tensora_native PRIVATE Threads::Threads)
if(TENSORA_WITH_TORCH)
  target_link_libraries(tensora_native PRIVATE ${TORCH_LIBRARIES})
endif()
if(TENSORA_WITH_ONNXRUNTIME)
  target_link_libraries(tensora_native PRIVATE ${TENSORA_ONNXRUNTIME_LIBRARY})
endif()

if(MSVC)
  target_compile_options(tensora_native PRIVATE /W4 /WX /permissive-)
  if(TENSORA_WITH_TORCH)
    target_compile_options(tensora_native PRIVATE /external:anglebrackets /external:W0)
    # PyTorch 2.13 triggers C4267 and C4702 inside its own headers with MSVC
    # 19.51. Keep /WX for Tensora and suppress only those two diagnostics in
    # translation units that must compile the upstream LibTorch API.
    set_source_files_properties(
      src/runtime/dispatcher.cc
      src/training/torch_backend.cc
      src/training/torch_storage.cc
      src/training/training_bridge_torch.cc
      PROPERTIES COMPILE_OPTIONS "/wd4267;/wd4702"
    )
  endif()
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
  if(TENSORA_WITH_TORCH OR TENSORA_WITH_ONNXRUNTIME)
    message(FATAL_ERROR "Run Tensora sanitizers in a core build; third-party runtime binaries are not instrumented by this configuration")
  endif()

  set(TENSORA_SANITIZER_FLAGS -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_compile_options(tensora_native PRIVATE ${TENSORA_SANITIZER_FLAGS})
  target_link_options(tensora_native PRIVATE -fsanitize=address,undefined)
endif()

if(TENSORA_ENABLE_THREAD_SANITIZER)
  if(MSVC)
    message(FATAL_ERROR "TENSORA_ENABLE_THREAD_SANITIZER is not supported by this CMake configuration on MSVC")
  endif()
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    message(FATAL_ERROR "TENSORA_ENABLE_THREAD_SANITIZER requires Clang or GCC")
  endif()
  if(TENSORA_WITH_TORCH OR TENSORA_WITH_ONNXRUNTIME)
    message(FATAL_ERROR "Run Tensora ThreadSanitizer in a core build; third-party runtime binaries are not instrumented by this configuration")
  endif()

  set(TENSORA_THREAD_SANITIZER_FLAGS -fsanitize=thread -fno-omit-frame-pointer)
  target_compile_options(tensora_native PRIVATE ${TENSORA_THREAD_SANITIZER_FLAGS})
  target_link_options(tensora_native PRIVATE -fsanitize=thread)
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
    src/backends/cpu/cpu_backend.cc
    src/core/status.cc
    src/memory/cpu_storage.cc
    src/runtime/dispatcher.cc
    src/runtime/handle_registry.cc
    src/tensor/shape.cc
    src/tensor/tensor.cc
  )
  target_compile_features(tensora_handle_registry_tests PRIVATE cxx_std_20)
  target_include_directories(tensora_handle_registry_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  target_link_libraries(tensora_handle_registry_tests PRIVATE Threads::Threads)

  add_executable(tensora_device_codec_tests
    tests/device_codec_test.cc
    src/core/status.cc
    src/runtime/device_codec.cc
  )
  target_compile_features(tensora_device_codec_tests PRIVATE cxx_std_20)
  target_include_directories(tensora_device_codec_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  )

  add_executable(tensora_c_abi_tests tests/c_abi_test.c)
  target_include_directories(tensora_c_abi_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(tensora_c_abi_tests PRIVATE tensora_native)
  set_target_properties(tensora_c_abi_tests PROPERTIES LINKER_LANGUAGE CXX)
  # CPU training is now a core capability. This legacy test switch selects the
  # training-enabled ABI expectations regardless of whether LibTorch is built.
  target_compile_definitions(tensora_c_abi_tests PRIVATE TENSORA_WITH_TORCH=1)
  if(TENSORA_WITH_ONNXRUNTIME)
    target_compile_definitions(tensora_c_abi_tests PRIVATE TENSORA_WITH_ONNXRUNTIME=1)
  endif()

  add_executable(tensora_cpu_training_tests tests/cpu_training_engine_test.cc)
  target_include_directories(tensora_cpu_training_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(tensora_cpu_training_tests PRIVATE tensora_native)

  add_executable(tensora_autograd_finite_difference_tests
    tests/autograd_finite_difference_test.cc
  )
  target_include_directories(tensora_autograd_finite_difference_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(tensora_autograd_finite_difference_tests PRIVATE tensora_native)

  add_executable(tensora_tensor_view_tests tests/tensor_view_test.cc)
  target_include_directories(tensora_tensor_view_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(tensora_tensor_view_tests PRIVATE tensora_native)

  if(TENSORA_WITH_TORCH)
    add_executable(tensora_training_tests tests/training_test.cc)
    target_include_directories(tensora_training_tests PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(tensora_training_tests PRIVATE tensora_native)

    add_executable(tensora_training_device_tests tests/training_device_test.cc)
    target_include_directories(tensora_training_device_tests PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(tensora_training_device_tests PRIVATE tensora_native)

    add_executable(tensora_training_backend_internal_tests
      tests/training_backend_internal_test.cc
      src/core/status.cc
      src/memory/cpu_storage.cc
      src/tensor/shape.cc
      src/tensor/tensor.cc
      src/training/torch_backend.cc
      src/training/torch_storage.cc
    )
    target_compile_features(tensora_training_backend_internal_tests PRIVATE cxx_std_20)
    target_include_directories(tensora_training_backend_internal_tests PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_include_directories(tensora_training_backend_internal_tests SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(tensora_training_backend_internal_tests PRIVATE
      Threads::Threads
      ${TORCH_LIBRARIES}
    )
  endif()

  if(TENSORA_WITH_ONNXRUNTIME)
    add_executable(tensora_inference_tests tests/inference_test.cc)
    target_include_directories(tensora_inference_tests PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(tensora_inference_tests PRIVATE tensora_native Threads::Threads)
  endif()

  if(MSVC)
    target_compile_options(tensora_native_tests PRIVATE /W4 /WX /permissive-)
    target_compile_options(tensora_handle_registry_tests PRIVATE /W4 /WX /permissive-)
    target_compile_options(tensora_device_codec_tests PRIVATE /W4 /WX /permissive-)
    target_compile_options(tensora_c_abi_tests PRIVATE /W4 /WX)
    target_compile_options(tensora_cpu_training_tests PRIVATE /W4 /WX /permissive-)
    target_compile_options(tensora_autograd_finite_difference_tests PRIVATE /W4 /WX /permissive-)
    target_compile_options(tensora_tensor_view_tests PRIVATE /W4 /WX /permissive-)
    if(TENSORA_WITH_TORCH)
      target_compile_options(tensora_training_tests PRIVATE /W4 /WX /permissive-)
      target_compile_options(tensora_training_device_tests PRIVATE /W4 /WX /permissive-)
      target_compile_options(tensora_training_backend_internal_tests PRIVATE /W4 /WX /permissive- /external:anglebrackets /external:W0 /wd4267 /wd4702)
    endif()
    if(TENSORA_WITH_ONNXRUNTIME)
      target_compile_options(tensora_inference_tests PRIVATE /W4 /WX /permissive-)
    endif()
  else()
    target_compile_options(tensora_native_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(tensora_handle_registry_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(tensora_device_codec_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(tensora_c_abi_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(tensora_cpu_training_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(tensora_autograd_finite_difference_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(tensora_tensor_view_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    if(TENSORA_WITH_TORCH)
      target_compile_options(tensora_training_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
      target_compile_options(tensora_training_device_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
      target_compile_options(tensora_training_backend_internal_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    if(TENSORA_WITH_ONNXRUNTIME)
      target_compile_options(tensora_inference_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
  endif()

  if(TENSORA_ENABLE_SANITIZERS)
    target_compile_options(tensora_native_tests PRIVATE ${TENSORA_SANITIZER_FLAGS})
    target_link_options(tensora_native_tests PRIVATE -fsanitize=address,undefined)
    target_compile_options(tensora_handle_registry_tests PRIVATE ${TENSORA_SANITIZER_FLAGS})
    target_link_options(tensora_handle_registry_tests PRIVATE -fsanitize=address,undefined)
    target_compile_options(tensora_device_codec_tests PRIVATE ${TENSORA_SANITIZER_FLAGS})
    target_link_options(tensora_device_codec_tests PRIVATE -fsanitize=address,undefined)
    target_compile_options(tensora_c_abi_tests PRIVATE ${TENSORA_SANITIZER_FLAGS})
    target_link_options(tensora_c_abi_tests PRIVATE -fsanitize=address,undefined)
    target_compile_options(tensora_cpu_training_tests PRIVATE ${TENSORA_SANITIZER_FLAGS})
    target_link_options(tensora_cpu_training_tests PRIVATE -fsanitize=address,undefined)
    target_compile_options(tensora_autograd_finite_difference_tests PRIVATE ${TENSORA_SANITIZER_FLAGS})
    target_link_options(tensora_autograd_finite_difference_tests PRIVATE -fsanitize=address,undefined)
    target_compile_options(tensora_tensor_view_tests PRIVATE ${TENSORA_SANITIZER_FLAGS})
    target_link_options(tensora_tensor_view_tests PRIVATE -fsanitize=address,undefined)
  endif()

  if(TENSORA_ENABLE_THREAD_SANITIZER)
    target_compile_options(tensora_native_tests PRIVATE ${TENSORA_THREAD_SANITIZER_FLAGS})
    target_link_options(tensora_native_tests PRIVATE -fsanitize=thread)
    target_compile_options(tensora_handle_registry_tests PRIVATE ${TENSORA_THREAD_SANITIZER_FLAGS})
    target_link_options(tensora_handle_registry_tests PRIVATE -fsanitize=thread)
    target_compile_options(tensora_device_codec_tests PRIVATE ${TENSORA_THREAD_SANITIZER_FLAGS})
    target_link_options(tensora_device_codec_tests PRIVATE -fsanitize=thread)
    target_compile_options(tensora_c_abi_tests PRIVATE ${TENSORA_THREAD_SANITIZER_FLAGS})
    target_link_options(tensora_c_abi_tests PRIVATE -fsanitize=thread)
    target_compile_options(tensora_cpu_training_tests PRIVATE ${TENSORA_THREAD_SANITIZER_FLAGS})
    target_link_options(tensora_cpu_training_tests PRIVATE -fsanitize=thread)
    target_compile_options(tensora_autograd_finite_difference_tests PRIVATE ${TENSORA_THREAD_SANITIZER_FLAGS})
    target_link_options(tensora_autograd_finite_difference_tests PRIVATE -fsanitize=thread)
    target_compile_options(tensora_tensor_view_tests PRIVATE ${TENSORA_THREAD_SANITIZER_FLAGS})
    target_link_options(tensora_tensor_view_tests PRIVATE -fsanitize=thread)
  endif()

  add_test(NAME tensora_native_tests COMMAND tensora_native_tests)
  add_test(NAME tensora_handle_registry_tests COMMAND tensora_handle_registry_tests)
  add_test(NAME tensora_device_codec_tests COMMAND tensora_device_codec_tests)
  add_test(NAME tensora_c_abi_tests COMMAND tensora_c_abi_tests)
  add_test(NAME tensora_cpu_training_tests COMMAND tensora_cpu_training_tests)
  add_test(NAME tensora_autograd_finite_difference_tests COMMAND tensora_autograd_finite_difference_tests)
  add_test(NAME tensora_tensor_view_tests COMMAND tensora_tensor_view_tests)
  if(TENSORA_WITH_TORCH)
    add_test(NAME tensora_training_tests COMMAND tensora_training_tests)
    add_test(NAME tensora_training_device_tests COMMAND tensora_training_device_tests)
    add_test(NAME tensora_training_backend_internal_tests COMMAND tensora_training_backend_internal_tests)
  endif()
  if(TENSORA_WITH_ONNXRUNTIME)
    add_test(NAME tensora_inference_tests COMMAND tensora_inference_tests)
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
