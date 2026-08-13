#include "runtime/device_codec.h"

#include "tensora.h"

namespace tensora {

Status DeviceFromCode(uint32_t code, Device* out_device) {
  if (out_device == nullptr) {
    return InvalidArgument("device: output pointer is null");
  }
  switch (code) {
    case TS_DEVICE_CPU:
      *out_device = Device::kCpu;
      return Status::Ok();
    case TS_DEVICE_CUDA:
      *out_device = Device::kCuda;
      return Status::Ok();
    case TS_DEVICE_MPS:
      *out_device = Device::kMps;
      return Status::Ok();
    case TS_DEVICE_XPU:
      *out_device = Device::kXpu;
      return Status::Ok();
    case TS_DEVICE_HIP:
      *out_device = Device::kHip;
      return Status::Ok();
    default:
      return Unsupported("device: unknown device kind");
  }
}

}  // namespace tensora
