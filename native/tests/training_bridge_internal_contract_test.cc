#include <cstdint>
#include <iostream>
#include <memory>

#include "memory/cpu_storage.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"
#include "training/training_bridge.h"

namespace {

using tensora::CpuStorage;
using tensora::Device;
using tensora::ShapeInfo;
using tensora::Status;
using tensora::Tensor;

bool ExpectStatus(const Status& status,
                  ts_status_t expected,
                  const char* operation) {
  if (status.code() == expected) return true;
  std::cerr << operation << " expected status " << expected << ", got "
            << status.code() << ": " << status.message() << "\n";
  return false;
}

std::shared_ptr<Tensor> MakeTensor(Device device = Device::kCpu,
                                   int32_t device_index = 0) {
  const int64_t dims[2] = {1, 2};
  const float values[2] = {1.0f, 2.0f};
  ShapeInfo shape;
  if (!tensora::ValidateShape(dims, 2, &shape).ok()) return nullptr;
  std::shared_ptr<CpuStorage> storage;
  if (!CpuStorage::FromData(values, 2, &storage).ok()) return nullptr;
  return std::make_shared<Tensor>(std::move(shape), std::move(storage),
                                  tensora::DType::kFloat32, device,
                                  device_index);
}

int CheckDirectBridgeFailures() {
  auto cpu = MakeTensor();
  auto fake_cuda = MakeTensor(Device::kCuda, 0);
  if (!cpu || !fake_cuda) return 1;

  uint32_t count = 99;
  if (!ExpectStatus(tensora::training::CudaDeviceCount(nullptr),
                    TS_INVALID_ARGUMENT, "cuda count null"))
    return 10;
  if (!ExpectStatus(tensora::training::CudaDeviceCount(&count), TS_OK,
                    "cuda count") ||
      count != 0)
    return 11;

  if (!ExpectStatus(tensora::training::Transfer(
                        *cpu, Device::kCpu, 0, nullptr),
                    TS_INVALID_ARGUMENT, "transfer null"))
    return 12;
  if (!ExpectStatus(tensora::training::WithRequiresGrad(*cpu, true, nullptr),
                    TS_INVALID_ARGUMENT, "requires-grad null"))
    return 13;

  uint8_t requires_grad = 9;
  if (!ExpectStatus(tensora::training::RequiresGrad(*cpu, &requires_grad),
                    TS_OK, "plain CPU requires-grad") ||
      requires_grad != 0)
    return 14;
  if (!ExpectStatus(tensora::training::Gradient(*cpu, nullptr),
                    TS_INVALID_ARGUMENT, "gradient null"))
    return 15;

  std::shared_ptr<Tensor> output;
  if (!ExpectStatus(tensora::training::MseLoss(*cpu, *fake_cuda, &output),
                    TS_INVALID_ARGUMENT, "MSE mixed device"))
    return 16;
  if (!ExpectStatus(
          tensora::training::CrossEntropyLoss(*cpu, *fake_cuda, &output),
          TS_INVALID_ARGUMENT, "cross entropy mixed device"))
    return 17;

  if (!ExpectStatus(tensora::training::LinearCreate(2, 2, true, nullptr),
                    TS_INVALID_ARGUMENT, "linear null"))
    return 18;

  uint64_t module = 0;
  if (!ExpectStatus(tensora::training::LinearCreate(2, 2, true, &module),
                    TS_OK, "linear create"))
    return 19;
  if (module == 0) return 20;

  if (!ExpectStatus(tensora::training::ModuleForward(module, *cpu, nullptr),
                    TS_INVALID_ARGUMENT, "module forward null"))
    return 21;
  if (!ExpectStatus(
          tensora::training::ModuleToDevice(module, Device::kCpu, 0), TS_OK,
          "module to CPU"))
    return 22;
  if (!ExpectStatus(
          tensora::training::ModuleParameterAt(module, 0, nullptr),
          TS_INVALID_ARGUMENT, "parameter null"))
    return 23;
  if (!ExpectStatus(tensora::training::ModuleBufferAt(module, 0, nullptr),
                    TS_INVALID_ARGUMENT, "buffer null"))
    return 24;

  if (!ExpectStatus(tensora::training::ModuleRelease(module), TS_OK,
                    "module release"))
    return 25;
  return 0;
}

}  // namespace

int main() { return CheckDirectBridgeFailures(); }
