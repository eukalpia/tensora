#include <cstdint>

#include "tensora.h"

namespace {

int CheckCount(uint32_t device, uint32_t expected_minimum) {
  uint32_t count = 999;
  if (ts_runtime_device_count(device, &count) != TS_OK) return 1;
  if (count < expected_minimum) return 2;
  return 0;
}

int CheckAcceleratorTransfer(ts_tensor_t input, uint32_t device) {
  uint32_t count = 0;
  if (ts_runtime_device_count(device, &count) != TS_OK) return 1;

  ts_tensor_t output = 123;
  const ts_status_t status = ts_tensor_to_device(input, device, 0, &output);
  if (count == 0) {
    if (status != TS_UNSUPPORTED) return 2;
    if (output != 0) return 3;
    return 0;
  }

  if (status != TS_OK || output == 0) return 4;
  uint32_t actual_device = 0;
  int32_t actual_index = -1;
  if (ts_tensor_device(output, &actual_device) != TS_OK) return 5;
  if (ts_tensor_device_index(output, &actual_index) != TS_OK) return 6;
  if (actual_device != device || actual_index != 0) return 7;
  if (ts_tensor_release(output) != TS_OK) return 8;
  return 0;
}

}  // namespace

int main() {
  if (CheckCount(TS_DEVICE_CPU, 1) != 0) return 1;
  if (CheckCount(TS_DEVICE_CUDA, 0) != 0) return 2;
  if (CheckCount(TS_DEVICE_MPS, 0) != 0) return 3;
  if (CheckCount(TS_DEVICE_XPU, 0) != 0) return 4;
  if (CheckCount(TS_DEVICE_HIP, 0) != 0) return 5;

  if (ts_runtime_device_count(TS_DEVICE_CPU, nullptr) != TS_INVALID_ARGUMENT)
    return 6;
  uint32_t unknown_count = 99;
  if (ts_runtime_device_count(0xffffffffu, &unknown_count) != TS_UNSUPPORTED)
    return 7;
  if (unknown_count != 0) return 8;

  const int64_t dims[1] = {1};
  const float value[1] = {2.0f};
  ts_tensor_t input = 0;
  if (ts_tensor_from_f32(value, 1, dims, 1, &input) != TS_OK || input == 0)
    return 10;

  ts_tensor_t cpu_copy = 0;
  if (ts_tensor_to_device(input, TS_DEVICE_CPU, 0, &cpu_copy) != TS_OK ||
      cpu_copy == 0)
    return 11;
  uint32_t cpu_device = 0;
  if (ts_tensor_device(cpu_copy, &cpu_device) != TS_OK ||
      cpu_device != TS_DEVICE_CPU)
    return 12;
  if (ts_tensor_release(cpu_copy) != TS_OK) return 13;

  if (CheckAcceleratorTransfer(input, TS_DEVICE_CUDA) != 0) return 20;
  if (CheckAcceleratorTransfer(input, TS_DEVICE_MPS) != 0) return 21;
  if (CheckAcceleratorTransfer(input, TS_DEVICE_XPU) != 0) return 22;
  if (CheckAcceleratorTransfer(input, TS_DEVICE_HIP) != 0) return 23;

  if (ts_tensor_release(input) != TS_OK) return 30;

  uint64_t live_tensors = 1;
  if (ts_runtime_live_tensor_count(&live_tensors) != TS_OK || live_tensors != 0)
    return 31;

  return 0;
}
