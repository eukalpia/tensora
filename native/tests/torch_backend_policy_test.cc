#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <string>

#include <torch/torch.h>

#include "core/status.h"
#include "training/torch_backend.h"

namespace {

bool ExpectStatus(const tensora::Status& status,
                  ts_status_t expected,
                  const char* operation) {
  if (status.code() == expected) return true;
  std::cerr << operation << " expected status " << expected << ", got "
            << status.code() << ": " << status.message() << "\n";
  return false;
}

int CheckAcceleratorMatching() {
  using tensora::Device;
  using tensora::training::internal::MatchesAccelerator;

  if (!MatchesAccelerator(Device::kCuda, c10::DeviceType::CUDA)) return 1;
  if (!MatchesAccelerator(Device::kMps, c10::DeviceType::MPS)) return 2;
  if (!MatchesAccelerator(Device::kXpu, c10::DeviceType::XPU)) return 3;
  if (!MatchesAccelerator(Device::kHip, c10::DeviceType::HIP)) return 4;
  if (MatchesAccelerator(Device::kCpu, c10::DeviceType::CPU)) return 5;
  if (MatchesAccelerator(Device::kCuda, c10::DeviceType::MPS)) return 6;
  if (MatchesAccelerator(static_cast<Device>(999), c10::DeviceType::CUDA)) {
    return 7;
  }
  return 0;
}

int CheckDeviceCountPolicy() {
  using tensora::Device;
  using tensora::training::internal::DeviceCountFromSnapshot;

  uint32_t count = 99;
  if (!ExpectStatus(
          DeviceCountFromSnapshot(Device::kCpu, std::nullopt, 0, nullptr),
          TS_INVALID_ARGUMENT, "count null"))
    return 10;
  if (!ExpectStatus(
          DeviceCountFromSnapshot(Device::kCpu, std::nullopt, 0, &count),
          TS_OK, "count CPU") ||
      count != 1)
    return 11;

  count = 99;
  if (!ExpectStatus(
          DeviceCountFromSnapshot(Device::kCuda, std::nullopt, 0, &count),
          TS_OK, "count no accelerator") ||
      count != 0)
    return 12;

  count = 99;
  if (!ExpectStatus(DeviceCountFromSnapshot(
                        Device::kCuda, c10::DeviceType::MPS, 4, &count),
                    TS_OK, "count mismatched accelerator") ||
      count != 0)
    return 13;

  if (!ExpectStatus(DeviceCountFromSnapshot(
                        Device::kCuda, c10::DeviceType::CUDA, -1, &count),
                    TS_INTERNAL_ERROR, "count negative"))
    return 14;

  if (!ExpectStatus(DeviceCountFromSnapshot(
                        Device::kCuda, c10::DeviceType::CUDA, 2, &count),
                    TS_OK, "count CUDA") ||
      count != 2)
    return 15;

  return 0;
}

int CheckTorchDevicePolicy() {
  using tensora::Device;
  using tensora::training::internal::TorchDeviceFromSnapshot;

  torch::Device output(torch::kCPU);
  if (!ExpectStatus(TorchDeviceFromSnapshot(Device::kCpu, 0, std::nullopt, 1,
                                             nullptr),
                    TS_INVALID_ARGUMENT, "device null"))
    return 20;
  if (!ExpectStatus(TorchDeviceFromSnapshot(Device::kCpu, 1, std::nullopt, 1,
                                             &output),
                    TS_INVALID_ARGUMENT, "CPU index"))
    return 21;
  if (!ExpectStatus(TorchDeviceFromSnapshot(Device::kCpu, 0, std::nullopt, 1,
                                             &output),
                    TS_OK, "CPU") ||
      !output.is_cpu())
    return 22;
  if (!ExpectStatus(TorchDeviceFromSnapshot(Device::kCuda, -1,
                                             c10::DeviceType::CUDA, 1, &output),
                    TS_INVALID_ARGUMENT, "negative accelerator index"))
    return 23;
  if (!ExpectStatus(TorchDeviceFromSnapshot(Device::kMps, 1,
                                             c10::DeviceType::MPS, 1, &output),
                    TS_INVALID_ARGUMENT, "MPS index"))
    return 24;
  const int32_t too_large_index =
      static_cast<int32_t>(std::numeric_limits<c10::DeviceIndex>::max()) + 1;
  if (!ExpectStatus(TorchDeviceFromSnapshot(
                        Device::kCuda, too_large_index,
                        c10::DeviceType::CUDA, 1, &output),
                    TS_INVALID_ARGUMENT, "accelerator index range"))
    return 25;
  if (!ExpectStatus(TorchDeviceFromSnapshot(Device::kCuda, 0, std::nullopt, 0,
                                             &output),
                    TS_UNSUPPORTED, "unavailable CUDA"))
    return 26;
  if (!ExpectStatus(TorchDeviceFromSnapshot(Device::kCuda, 0,
                                             c10::DeviceType::MPS, 1, &output),
                    TS_UNSUPPORTED, "mismatched CUDA"))
    return 27;

  if (!ExpectStatus(TorchDeviceFromSnapshot(Device::kCuda, 1,
                                             c10::DeviceType::CUDA, 2, &output),
                    TS_OK, "CUDA") ||
      output.type() != c10::DeviceType::CUDA || output.index() != 1)
    return 28;
  if (!ExpectStatus(TorchDeviceFromSnapshot(Device::kMps, 0,
                                             c10::DeviceType::MPS, 1, &output),
                    TS_OK, "MPS") ||
      output.type() != c10::DeviceType::MPS)
    return 29;
  if (!ExpectStatus(TorchDeviceFromSnapshot(Device::kXpu, 0,
                                             c10::DeviceType::XPU, 1, &output),
                    TS_OK, "XPU") ||
      output.type() != c10::DeviceType::XPU)
    return 30;
  if (!ExpectStatus(TorchDeviceFromSnapshot(Device::kHip, 0,
                                             c10::DeviceType::HIP, 1, &output),
                    TS_OK, "HIP") ||
      output.type() != c10::DeviceType::CUDA || output.index() != 0)
    return 31;
  if (!ExpectStatus(TorchDeviceFromSnapshot(static_cast<Device>(999), 0,
                                             c10::DeviceType::CUDA, 1, &output),
                    TS_UNSUPPORTED, "unknown device"))
    return 32;
  return 0;
}

int CheckTorchDeviceMapping() {
  using tensora::Device;
  using tensora::training::internal::MapTorchDevice;

  Device device = Device::kCpu;
  int32_t index = -99;
  if (!ExpectStatus(MapTorchDevice(c10::DeviceType::CPU, -1, std::nullopt,
                                   nullptr, &index),
                    TS_INVALID_ARGUMENT, "map null device"))
    return 40;
  if (!ExpectStatus(MapTorchDevice(c10::DeviceType::CPU, -1, std::nullopt,
                                   &device, nullptr),
                    TS_INVALID_ARGUMENT, "map null index"))
    return 41;
  if (!ExpectStatus(MapTorchDevice(c10::DeviceType::CPU, -1, std::nullopt,
                                   &device, &index),
                    TS_OK, "map CPU") ||
      device != Device::kCpu || index != 0)
    return 42;
  if (!ExpectStatus(MapTorchDevice(c10::DeviceType::MPS, -1, std::nullopt,
                                   &device, &index),
                    TS_OK, "map MPS") ||
      device != Device::kMps || index != 0)
    return 43;
  if (!ExpectStatus(MapTorchDevice(c10::DeviceType::XPU, 2, std::nullopt,
                                   &device, &index),
                    TS_OK, "map XPU") ||
      device != Device::kXpu || index != 2)
    return 44;
  if (!ExpectStatus(MapTorchDevice(c10::DeviceType::HIP, 3, std::nullopt,
                                   &device, &index),
                    TS_OK, "map HIP") ||
      device != Device::kHip || index != 3)
    return 45;
  if (!ExpectStatus(MapTorchDevice(c10::DeviceType::CUDA, 1, std::nullopt,
                                   &device, &index),
                    TS_OK, "map CUDA") ||
      device != Device::kCuda || index != 1)
    return 46;
  if (!ExpectStatus(MapTorchDevice(c10::DeviceType::CUDA, 1,
                                   c10::DeviceType::HIP, &device, &index),
                    TS_OK, "map ROCm CUDA facade") ||
      device != Device::kHip || index != 1)
    return 47;
  if (!ExpectStatus(MapTorchDevice(c10::DeviceType::CUDA, -1, std::nullopt,
                                   &device, &index),
                    TS_INTERNAL_ERROR, "map missing CUDA index"))
    return 48;
  if (!ExpectStatus(MapTorchDevice(c10::DeviceType::Meta, -1, std::nullopt,
                                   &device, &index),
                    TS_UNSUPPORTED, "map unsupported"))
    return 49;
  return 0;
}

int CheckGuards() {
  using tensora::Status;
  using tensora::training::internal::GuardAllocation;
  using tensora::training::internal::GuardTorch;

  if (!ExpectStatus(
          GuardTorch("torch success", [] { return Status::Ok(); }), TS_OK,
          "guard success"))
    return 50;

  const Status torch_failure = GuardTorch("forced torch failure", [] {
    TORCH_CHECK(false, "deterministic torch error");
    return Status::Ok();
  });
  if (!ExpectStatus(torch_failure, TS_INTERNAL_ERROR, "guard torch failure") ||
      torch_failure.message().find("forced torch failure") == std::string::npos)
    return 51;

  if (!ExpectStatus(
          GuardAllocation("allocation success", [] { return Status::Ok(); }),
          TS_OK, "allocation success"))
    return 52;
  const Status allocation_failure = GuardAllocation("forced allocation", [] {
    throw std::bad_alloc();
    return Status::Ok();
  });
  if (!ExpectStatus(allocation_failure, TS_OUT_OF_MEMORY,
                    "allocation failure") ||
      allocation_failure.message().find("forced allocation") ==
          std::string::npos)
    return 53;
  return 0;
}

}  // namespace

int main() {
  if (const int code = CheckAcceleratorMatching(); code != 0) return code;
  if (const int code = CheckDeviceCountPolicy(); code != 0) return code;
  if (const int code = CheckTorchDevicePolicy(); code != 0) return code;
  if (const int code = CheckTorchDeviceMapping(); code != 0) return code;
  if (const int code = CheckGuards(); code != 0) return code;
  return 0;
}
