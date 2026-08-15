#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "memory/cpu_storage.h"
#include "runtime/dispatcher.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"
#include "training/training_bridge.h"

namespace {

using tensora::Backend;
using tensora::CpuStorage;
using tensora::Device;
using tensora::Dispatcher;
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

int CheckDispatcherContracts(const Tensor& cpu,
                             const Tensor& fake_cuda,
                             bool expect_accelerators) {
  const Backend* backend = nullptr;
  const ts_status_t expected = expect_accelerators ? TS_OK : TS_UNSUPPORTED;
  for (const Device device : {Device::kCuda, Device::kMps, Device::kXpu,
                              Device::kHip}) {
    const Status status = Dispatcher::For(device, &backend);
    if (!ExpectStatus(status, expected, "dispatcher accelerator")) {
      return 30;
    }
    if (expect_accelerators && backend == nullptr) return 31;
    if (!expect_accelerators && backend != nullptr) return 32;
    backend = nullptr;
  }

  if (!ExpectStatus(Dispatcher::ForTensors(cpu, fake_cuda, &backend),
                    TS_INVALID_ARGUMENT, "dispatcher mixed devices")) {
    return 33;
  }
  return 0;
}

int CheckDirectBridgeFailures(bool expect_accelerators) {
  auto cpu = MakeTensor();
  auto fake_cuda = MakeTensor(Device::kCuda, 0);
  if (!cpu || !fake_cuda) return 1;

  if (const int code =
          CheckDispatcherContracts(*cpu, *fake_cuda, expect_accelerators);
      code != 0) {
    return code;
  }

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
  if (!ExpectStatus(tensora::training::RequiresGrad(*cpu, nullptr),
                    TS_INVALID_ARGUMENT, "requires-grad output null"))
    return 14;

  uint8_t requires_grad = 9;
  if (!ExpectStatus(tensora::training::RequiresGrad(*cpu, &requires_grad),
                    TS_OK, "plain CPU requires-grad") ||
      requires_grad != 0)
    return 15;
  if (!ExpectStatus(tensora::training::Gradient(*cpu, nullptr),
                    TS_INVALID_ARGUMENT, "gradient null"))
    return 16;

  std::shared_ptr<Tensor> output;
  if (!ExpectStatus(tensora::training::MseLoss(*cpu, *fake_cuda, &output),
                    TS_INVALID_ARGUMENT, "MSE mixed device"))
    return 17;
  if (!ExpectStatus(
          tensora::training::CrossEntropyLoss(*cpu, *fake_cuda, &output),
          TS_INVALID_ARGUMENT, "cross entropy mixed device"))
    return 18;

  if (!ExpectStatus(tensora::training::LinearCreate(2, 2, true, nullptr),
                    TS_INVALID_ARGUMENT, "linear null"))
    return 19;

  uint64_t module = 0;
  if (!ExpectStatus(tensora::training::LinearCreate(2, 2, true, &module),
                    TS_OK, "linear create"))
    return 20;
  if (module == 0) return 21;

  if (!ExpectStatus(tensora::training::ModuleForward(module, *cpu, nullptr),
                    TS_INVALID_ARGUMENT, "module forward null"))
    return 22;
  if (!ExpectStatus(
          tensora::training::ModuleToDevice(module, Device::kCpu, 0), TS_OK,
          "module to CPU"))
    return 23;
  if (!ExpectStatus(
          tensora::training::ModuleParameterAt(module, 0, nullptr),
          TS_INVALID_ARGUMENT, "parameter null"))
    return 24;
  if (!ExpectStatus(tensora::training::ModuleBufferAt(module, 0, nullptr),
                    TS_INVALID_ARGUMENT, "buffer null"))
    return 25;
  if (!ExpectStatus(tensora::training::ModuleBufferAt(module, 0, &output),
                    TS_INVALID_ARGUMENT, "linear buffer out of range"))
    return 26;

  if (!ExpectStatus(tensora::training::ModuleRelease(module), TS_OK,
                    "module release"))
    return 27;
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const bool expect_accelerators =
      argc > 1 && std::string(argv[1]) == "accelerators";
  return CheckDirectBridgeFailures(expect_accelerators);
}
