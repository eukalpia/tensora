#include "runtime/device_codec.h"

#include "tensora.h"

namespace {

int Check(uint32_t code, tensora::Device expected) {
  tensora::Device actual = tensora::Device::kCpu;
  const tensora::Status status = tensora::DeviceFromCode(code, &actual);
  if (!status.ok()) return 1;
  return actual == expected ? 0 : 2;
}

}  // namespace

int main() {
  if (Check(TS_DEVICE_CPU, tensora::Device::kCpu) != 0) return 1;
  if (Check(TS_DEVICE_CUDA, tensora::Device::kCuda) != 0) return 2;
  if (Check(TS_DEVICE_MPS, tensora::Device::kMps) != 0) return 3;
  if (Check(TS_DEVICE_XPU, tensora::Device::kXpu) != 0) return 4;
  if (Check(TS_DEVICE_HIP, tensora::Device::kHip) != 0) return 5;

  tensora::Device value = tensora::Device::kCpu;
  if (tensora::DeviceFromCode(0xffffffffu, &value).code() !=
      TS_UNSUPPORTED) {
    return 6;
  }
  if (tensora::DeviceFromCode(TS_DEVICE_CPU, nullptr).code() !=
      TS_INVALID_ARGUMENT) {
    return 7;
  }
  return 0;
}
