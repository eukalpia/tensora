#include <cstdint>

#include <torch/torch.h>

#include "tensor/tensor.h"
#include "training/torch_backend.h"
#include "training/training_bridge.h"

namespace {

int CheckDeviceCount(tensora::Device device) {
  uint32_t count = 999;
  const tensora::Status status = tensora::training::DeviceCount(device, &count);
  if (!status.ok()) return 1;
  if (device == tensora::Device::kCpu && count != 1) return 2;
  return 0;
}

int CheckUnavailableMapping(tensora::Device device, uint32_t count) {
  if (count != 0) return 0;
  torch::Device mapped(torch::kCPU);
  const tensora::Status status =
      tensora::training::TorchDevice(device, 0, &mapped);
  if (status.code() != TS_UNSUPPORTED) return 1;
  return 0;
}

}  // namespace

int main() {
  for (const tensora::Device device : {
           tensora::Device::kCpu,
           tensora::Device::kCuda,
           tensora::Device::kMps,
           tensora::Device::kXpu,
           tensora::Device::kHip,
       }) {
    const int result = CheckDeviceCount(device);
    if (result != 0) return result;
  }

  uint32_t cuda_count = 0;
  uint32_t mps_count = 0;
  uint32_t xpu_count = 0;
  uint32_t hip_count = 0;
  if (!tensora::training::DeviceCount(tensora::Device::kCuda, &cuda_count).ok())
    return 10;
  if (!tensora::training::DeviceCount(tensora::Device::kMps, &mps_count).ok())
    return 11;
  if (!tensora::training::DeviceCount(tensora::Device::kXpu, &xpu_count).ok())
    return 12;
  if (!tensora::training::DeviceCount(tensora::Device::kHip, &hip_count).ok())
    return 13;

  if (CheckUnavailableMapping(tensora::Device::kCuda, cuda_count) != 0)
    return 20;
  if (CheckUnavailableMapping(tensora::Device::kMps, mps_count) != 0)
    return 21;
  if (CheckUnavailableMapping(tensora::Device::kXpu, xpu_count) != 0)
    return 22;
  if (CheckUnavailableMapping(tensora::Device::kHip, hip_count) != 0)
    return 23;

  if (tensora::training::DeviceCount(tensora::Device::kCpu, nullptr).code() !=
      TS_INVALID_ARGUMENT)
    return 30;

  torch::Device mapped(torch::kCPU);
  if (!tensora::training::TorchDevice(tensora::Device::kCpu, 0, &mapped).ok())
    return 31;
  if (!mapped.is_cpu()) return 32;

  return 0;
}
