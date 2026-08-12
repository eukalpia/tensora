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
  uint64_t numel = 0;
  float values[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  size_t written = 0;

  if (ts_abi_version() != TS_ABI_VERSION) return 1;
  if (ts_noop() != TS_OK) return 2;
  if (ts_tensor_from_f32(input, 3, dims, 2, &tensor) != TS_INVALID_ARGUMENT)
    return 3;
  if (tensor != 0) return 4;
  if (ts_tensor_from_f32(input, 5, dims, 2, &tensor) != TS_INVALID_ARGUMENT)
    return 5;
  if (tensor != 0) return 6;
  if (ts_tensor_from_f32(input, 4, dims, 2, &tensor) != TS_OK) return 7;
  if (tensor == 0) return 8;
  if (ts_tensor_numel(tensor, &numel) != TS_OK || numel != 4) return 9;
  if (ts_tensor_copy_to_host_f32(tensor, values, 4, &written) != TS_OK)
    return 10;
  if (written != 4) return 11;
  if (values[0] != 1.0f || values[3] != 4.0f) return 12;
  if (ts_tensor_release(tensor) != TS_OK) return 13;

  tensor = 0;
  if (ts_tensor_full_f32(dims, 2, 3.0f, &tensor) != TS_OK) return 14;
  if (ts_tensor_release(tensor) != TS_OK) return 15;

  return 0;
}
