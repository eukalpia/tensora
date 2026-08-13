#include "training/training_bridge.h"

#include <limits>
#include <optional>

#include <ATen/DeviceAccelerator.h>

namespace tensora::training {
namespace {

bool MatchesAccelerator(Device device, c10::DeviceType accelerator) {
  switch (device) {
    case Device::kCuda:
      return accelerator == c10::DeviceType::CUDA;
    case Device::kMps:
      return accelerator == c10::DeviceType::MPS;
    case Device::kXpu:
      return accelerator == c10::DeviceType::XPU;
    case Device::kHip:
      return accelerator == c10::DeviceType::HIP;
    case Device::kCpu:
      return false;
  }
  return false;
}

}  // namespace

Status DeviceCount(Device device, uint32_t* out_count) {
  if (out_count == nullptr) {
    return InvalidArgument("device_count: output pointer is null");
  }
  *out_count = 0;

  if (device == Device::kCpu) {
    *out_count = 1;
    return Status::Ok();
  }

  try {
    const std::optional<c10::DeviceType> accelerator =
        at::accelerator::getAccelerator(false);
    if (!accelerator.has_value() ||
        !MatchesAccelerator(device, *accelerator)) {
      return Status::Ok();
    }

    const c10::DeviceIndex count = at::accelerator::deviceCount();
    if (count < 0) {
      return InternalError("device_count: accelerator returned a negative count");
    }
    if (static_cast<uint64_t>(count) >
        std::numeric_limits<uint32_t>::max()) {
      return InternalError("device_count: accelerator count exceeds ABI range");
    }
    *out_count = static_cast<uint32_t>(count);
    return Status::Ok();
  } catch (const c10::Error& error) {
    return InternalError(std::string("device_count: ") + error.what());
  }
}

}  // namespace tensora::training
