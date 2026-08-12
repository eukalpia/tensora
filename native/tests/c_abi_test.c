#include "tensora.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define TS_STATIC_ASSERT(name, expression) \
  typedef char name[(expression) ? 1 : -1]

TS_STATIC_ASSERT(ts_status_is_exactly_32_bits, sizeof(ts_status_t) == 4);
TS_STATIC_ASSERT(ts_tensor_handle_is_exactly_64_bits, sizeof(ts_tensor_t) == 8);
TS_STATIC_ASSERT(ts_module_handle_is_exactly_64_bits, sizeof(ts_module_t) == 8);
TS_STATIC_ASSERT(ts_optimizer_handle_is_exactly_64_bits,
                 sizeof(ts_optimizer_t) == 8);
TS_STATIC_ASSERT(ts_onnx_session_handle_is_exactly_64_bits,
                 sizeof(ts_onnx_session_t) == 8);
TS_STATIC_ASSERT(ts_dtype_is_exactly_32_bits, sizeof(ts_dtype_t) == 4);
TS_STATIC_ASSERT(ts_device_is_exactly_32_bits, sizeof(ts_device_t) == 4);

int main(void) {
  const int64_t dims[2] = {2, 2};
  const float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  ts_tensor_t tensor = 0;
  ts_tensor_t copied = 0;
  ts_module_t module = 0;
  ts_optimizer_t optimizer = 0;
  uint64_t numel = 0;
  uint32_t device = 0;
  int32_t device_index = -1;
  uint32_t cuda_count = 0;
  uint8_t training_available = 0;
  uint8_t requires_grad = 1;
  float values[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  size_t written = 0;

  if (ts_abi_version() != TS_ABI_VERSION) return 1;
  if (ts_noop() != TS_OK) return 2;
  if (TS_DEVICE_CPU == TS_DEVICE_CUDA) return 3;
  if (ts_training_available(&training_available) != TS_OK) return 4;
#if defined(TENSORA_WITH_TORCH)
  if (training_available != 1) return 5;
#else
  if (training_available != 0) return 5;
#endif
  if (ts_training_available(NULL) != TS_INVALID_ARGUMENT) return 6;
  if (ts_runtime_cuda_device_count(&cuda_count) != TS_OK) return 7;
  if (ts_runtime_cuda_device_count(NULL) != TS_INVALID_ARGUMENT) return 8;
#if defined(TENSORA_WITH_TORCH)
  if (ts_manual_seed(42) != TS_OK) return 9;
#else
  if (cuda_count != 0) return 9;
  if (ts_manual_seed(42) != TS_UNSUPPORTED) return 10;
#endif

  if (ts_tensor_from_f32(input, 3, dims, 2, &tensor) != TS_INVALID_ARGUMENT)
    return 11;
  if (tensor != 0) return 12;
  if (ts_tensor_from_f32(input, 5, dims, 2, &tensor) != TS_INVALID_ARGUMENT)
    return 13;
  if (tensor != 0) return 14;
  if (ts_tensor_from_f32(input, 4, dims, 2, &tensor) != TS_OK) return 15;
  if (tensor == 0) return 16;
  if (ts_tensor_numel(tensor, &numel) != TS_OK || numel != 4) return 17;
  if (ts_tensor_device(tensor, &device) != TS_OK || device != TS_DEVICE_CPU)
    return 18;
  if (ts_tensor_device_index(tensor, &device_index) != TS_OK ||
      device_index != 0)
    return 19;
  if (ts_tensor_copy_to_host_f32(tensor, values, 4, &written) != TS_OK)
    return 20;
  if (written != 4) return 21;
  if (values[0] != 1.0f || values[3] != 4.0f) return 22;

  if (ts_tensor_requires_grad(tensor, &requires_grad) != TS_OK ||
      requires_grad != 0)
    return 23;
  if (ts_tensor_requires_grad(tensor, NULL) != TS_INVALID_ARGUMENT) return 24;
#if defined(TENSORA_WITH_TORCH)
  if (ts_tensor_with_requires_grad(tensor, 1, &copied) != TS_OK) return 25;
  if (copied == 0) return 26;
  if (ts_tensor_release(copied) != TS_OK) return 27;
  copied = 0;
  if (ts_tensor_relu(tensor, &copied) != TS_OK) return 28;
  if (copied == 0) return 29;
  if (ts_tensor_release(copied) != TS_OK) return 30;
  copied = 0;
#else
  if (ts_tensor_with_requires_grad(tensor, 1, &copied) != TS_UNSUPPORTED)
    return 25;
  if (copied != 0) return 26;
  if (ts_tensor_relu(tensor, &copied) != TS_UNSUPPORTED) return 27;
  if (copied != 0) return 28;
#endif

  if (ts_tensor_to_device(tensor, TS_DEVICE_CPU, 0, &copied) != TS_OK)
    return 31;
  if (copied == 0) return 32;
  if (ts_tensor_device(copied, &device) != TS_OK || device != TS_DEVICE_CPU)
    return 33;
  if (ts_tensor_device_index(copied, &device_index) != TS_OK ||
      device_index != 0)
    return 34;
  if (ts_tensor_release(copied) != TS_OK) return 35;
  copied = 0;

  if (ts_tensor_to_device(tensor, 999u, 0, &copied) != TS_UNSUPPORTED)
    return 36;
  if (copied != 0) return 37;
  if (ts_tensor_to_device(tensor, TS_DEVICE_CPU, 1, &copied) !=
      TS_INVALID_ARGUMENT)
    return 38;
  if (copied != 0) return 39;
  if (cuda_count == 0) {
    if (ts_tensor_to_device(tensor, TS_DEVICE_CUDA, 0, &copied) !=
        TS_UNSUPPORTED)
      return 40;
    if (copied != 0) return 41;
  }

#if defined(TENSORA_WITH_TORCH)
  if (ts_linear_create(1, 1, 1, &module) != TS_OK) return 42;
  if (module == 0) return 43;
  if (ts_sgd_create(module, 0.1, 0.0, 0.0, &optimizer) != TS_OK) return 44;
  if (optimizer == 0) return 45;
  if (ts_optimizer_release(optimizer) != TS_OK) return 46;
  if (ts_module_release(module) != TS_OK) return 47;
#else
  if (ts_linear_create(1, 1, 1, &module) != TS_UNSUPPORTED) return 42;
  if (module != 0) return 43;
  if (ts_sgd_create(module, 0.1, 0.0, 0.0, &optimizer) != TS_UNSUPPORTED)
    return 44;
  if (optimizer != 0) return 45;
#endif

  if (ts_tensor_release(tensor) != TS_OK) return 48;

  tensor = 0;
  if (ts_tensor_full_f32(dims, 2, 3.0f, &tensor) != TS_OK) return 49;
  if (ts_tensor_release(tensor) != TS_OK) return 50;

  if (strcmp(ts_status_name(TS_MODEL_ERROR), "MODEL_ERROR") != 0) return 51;
  if (strcmp(ts_status_name(999), "UNKNOWN_STATUS") != 0) return 52;

  return 0;
}
