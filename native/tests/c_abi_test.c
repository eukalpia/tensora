#include "tensora.h"

#include <stddef.h>
#include <stdint.h>

#define TS_STATIC_ASSERT(name, expression) \
  typedef char name[(expression) ? 1 : -1]

TS_STATIC_ASSERT(ts_status_is_exactly_32_bits, sizeof(ts_status_t) == 4);
TS_STATIC_ASSERT(ts_tensor_handle_is_exactly_64_bits, sizeof(ts_tensor_t) == 8);
TS_STATIC_ASSERT(ts_dtype_is_exactly_32_bits, sizeof(ts_dtype_t) == 4);
TS_STATIC_ASSERT(ts_device_is_exactly_32_bits, sizeof(ts_device_t) == 4);

int main(void) {
  const int64_t dims[2] = {2, 2};
  const float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  ts_tensor_t tensor = 0;
  ts_tensor_t copied = 0;
  uint64_t numel = 0;
  uint32_t device = 0;
  int32_t device_index = -1;
  uint32_t cuda_count = 0;
  float values[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  size_t written = 0;

  if (ts_abi_version() != TS_ABI_VERSION) return 1;
  if (ts_noop() != TS_OK) return 2;
  if (TS_DEVICE_CPU == TS_DEVICE_CUDA) return 3;
  if (ts_runtime_cuda_device_count(&cuda_count) != TS_OK) return 4;
  if (ts_runtime_cuda_device_count(NULL) != TS_INVALID_ARGUMENT) return 5;

  if (ts_tensor_from_f32(input, 3, dims, 2, &tensor) != TS_INVALID_ARGUMENT)
    return 6;
  if (tensor != 0) return 7;
  if (ts_tensor_from_f32(input, 5, dims, 2, &tensor) != TS_INVALID_ARGUMENT)
    return 8;
  if (tensor != 0) return 9;
  if (ts_tensor_from_f32(input, 4, dims, 2, &tensor) != TS_OK) return 10;
  if (tensor == 0) return 11;
  if (ts_tensor_numel(tensor, &numel) != TS_OK || numel != 4) return 12;
  if (ts_tensor_device(tensor, &device) != TS_OK || device != TS_DEVICE_CPU)
    return 13;
  if (ts_tensor_device_index(tensor, &device_index) != TS_OK ||
      device_index != 0)
    return 14;
  if (ts_tensor_copy_to_host_f32(tensor, values, 4, &written) != TS_OK)
    return 15;
  if (written != 4) return 16;
  if (values[0] != 1.0f || values[3] != 4.0f) return 17;

  if (ts_tensor_to_device(tensor, TS_DEVICE_CPU, 0, &copied) != TS_OK)
    return 18;
  if (copied == 0) return 19;
  if (ts_tensor_device(copied, &device) != TS_OK || device != TS_DEVICE_CPU)
    return 20;
  if (ts_tensor_device_index(copied, &device_index) != TS_OK ||
      device_index != 0)
    return 21;
  if (ts_tensor_release(copied) != TS_OK) return 22;
  copied = 0;

  if (ts_tensor_to_device(tensor, 999u, 0, &copied) != TS_UNSUPPORTED)
    return 23;
  if (copied != 0) return 24;
  if (ts_tensor_to_device(tensor, TS_DEVICE_CPU, 1, &copied) !=
      TS_INVALID_ARGUMENT)
    return 25;
  if (copied != 0) return 26;

  if (ts_tensor_release(tensor) != TS_OK) return 27;

  tensor = 0;
  if (ts_tensor_full_f32(dims, 2, 3.0f, &tensor) != TS_OK) return 28;
  if (ts_tensor_release(tensor) != TS_OK) return 29;

  return 0;
}
