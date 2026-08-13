#include "training/training_bridge.h"

namespace tensora::training {

Status DeviceCount(Device device, uint32_t* out_count) {
  if (out_count == nullptr) {
    return InvalidArgument("device_count: output pointer is null");
  }
  *out_count = device == Device::kCpu ? 1u : 0u;
  return Status::Ok();
}

}  // namespace tensora::training
