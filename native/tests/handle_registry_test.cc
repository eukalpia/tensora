#include "runtime/handle_registry.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "backends/backend.h"
#include "backends/cpu/cpu_backend.h"
#include "core/abi_guard.h"
#include "core/status.h"
#include "memory/cpu_storage.h"
#include "runtime/dispatcher.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

namespace {

using tensora::CpuStorage;
using tensora::HandleRegistry;
using tensora::HandleType;
using tensora::ShapeInfo;
using tensora::Status;
using tensora::Tensor;

bool ExpectCode(const Status& status,
                int32_t expected,
                const char* operation) {
  if (status.code() == expected) return true;
  std::cerr << operation << ": expected status " << expected << ", got "
            << status.code() << "\n";
  return false;
}


int TestAbiGuardContracts() {
  using tensora::AbiGuard;
  using tensora::InvalidArgument;
  using tensora::Status;

  if (AbiGuard("abi_ok", [] { return Status::Ok(); }) != TS_OK) return 201;
  if (AbiGuard("abi_empty", [] { return Status(TS_INVALID_ARGUMENT, ""); }) !=
      TS_INVALID_ARGUMENT)
    return 202;
  if (std::string(tensora::LastErrorMessage()) != "abi_empty: failed") return 203;
  if (AbiGuard("abi_error", [] { return InvalidArgument("bad input"); }) !=
      TS_INVALID_ARGUMENT)
    return 204;
  if (std::string(tensora::LastErrorMessage()) != "bad input") return 205;
  if (AbiGuard("abi_alloc", []() -> Status { throw std::bad_alloc(); }) !=
      TS_OUT_OF_MEMORY)
    return 206;
  if (std::string(tensora::LastErrorMessage()).find("native allocation failed") ==
      std::string::npos)
    return 207;
  if (AbiGuard("abi_std", []() -> Status { throw std::runtime_error("boom"); }) !=
      TS_INTERNAL_ERROR)
    return 208;
  if (std::string(tensora::LastErrorMessage()).find("boom") ==
      std::string::npos)
    return 209;
  if (AbiGuard("abi_unknown", []() -> Status { throw 7; }) !=
      TS_INTERNAL_ERROR)
    return 210;
  if (std::string(tensora::LastErrorMessage()).find("unknown native exception") ==
      std::string::npos)
    return 211;
  return 0;
}

int TestRegistryContracts() {
  auto& registry = HandleRegistry::Instance();
  const uint64_t baseline_tensors = registry.Count(HandleType::kTensor);
  const uint64_t baseline_modules = registry.Count(HandleType::kModule);

  if (!ExpectCode(registry.Insert(HandleType::kTensor, nullptr, nullptr),
                  TS_INVALID_ARGUMENT, "insert null output"))
    return 10;

  uint64_t handle = 777;
  if (!ExpectCode(registry.Insert(HandleType::kTensor, nullptr, &handle),
                  TS_INVALID_ARGUMENT, "insert null object") ||
      handle != 0)
    return 11;

  if (!ExpectCode(registry.Lookup<Tensor>(0, HandleType::kTensor, nullptr),
                  TS_INVALID_ARGUMENT, "lookup null output"))
    return 12;

  std::shared_ptr<Tensor> tensor;
  if (!ExpectCode(registry.Lookup<Tensor>(0, HandleType::kTensor, &tensor),
                  TS_INVALID_HANDLE, "lookup zero"))
    return 13;
  if (!ExpectCode(
          registry.Lookup<Tensor>(UINT64_C(999999999), HandleType::kTensor,
                                  &tensor),
          TS_INVALID_HANDLE, "lookup unknown"))
    return 14;
  if (!ExpectCode(registry.Retain(0, HandleType::kTensor), TS_INVALID_HANDLE,
                  "retain zero"))
    return 15;
  if (!ExpectCode(registry.Release(0, HandleType::kTensor), TS_INVALID_HANDLE,
                  "release zero"))
    return 16;
  if (!ExpectCode(registry.Retain(UINT64_C(999999999), HandleType::kTensor),
                  TS_INVALID_HANDLE, "retain unknown"))
    return 17;
  if (!ExpectCode(registry.Release(UINT64_C(999999999), HandleType::kTensor),
                  TS_INVALID_HANDLE, "release unknown"))
    return 18;

  auto object = std::make_shared<int>(42);
  if (!registry.Insert(HandleType::kTensor, object, &handle).ok() || handle == 0)
    return 19;
  if (registry.Count(HandleType::kTensor) != baseline_tensors + 1) return 20;
  if (!ExpectCode(registry.Lookup<int>(handle, HandleType::kModule, &object),
                  TS_INVALID_HANDLE, "lookup wrong type"))
    return 21;
  if (!ExpectCode(registry.Retain(handle, HandleType::kModule),
                  TS_INVALID_HANDLE, "retain wrong type"))
    return 22;
  if (!ExpectCode(registry.Release(handle, HandleType::kModule),
                  TS_INVALID_HANDLE, "release wrong type"))
    return 23;
  if (!registry.Retain(handle, HandleType::kTensor).ok()) return 24;
  if (!registry.Release(handle, HandleType::kTensor).ok()) return 25;
  if (registry.Count(HandleType::kTensor) != baseline_tensors + 1) return 26;
  if (!registry.Release(handle, HandleType::kTensor).ok()) return 27;
  if (registry.Count(HandleType::kTensor) != baseline_tensors) return 28;
  if (!ExpectCode(registry.Release(handle, HandleType::kTensor),
                  TS_INVALID_HANDLE, "double release"))
    return 29;
  if (registry.Count(HandleType::kModule) != baseline_modules) return 30;
  return 0;
}

int TestCoreContracts() {
  Status status = Status::Ok();
  if (!status.ok() || status.code() != TS_OK || !status.message().empty())
    return 100;
  status = tensora::InvalidArgument("invalid argument");
  if (status.ok() || status.code() != TS_INVALID_ARGUMENT ||
      status.message() != "invalid argument")
    return 101;
  status = tensora::InvalidShape("invalid shape");
  if (status.code() != TS_INVALID_SHAPE) return 102;
  status = tensora::OutOfMemory("oom");
  if (status.code() != TS_OUT_OF_MEMORY) return 103;
  status = tensora::Unsupported("unsupported");
  if (status.code() != TS_UNSUPPORTED) return 104;
  status = tensora::InvalidHandle("invalid handle");
  if (status.code() != TS_INVALID_HANDLE) return 105;
  status = tensora::InternalError("internal");
  if (status.code() != TS_INTERNAL_ERROR) return 106;

  ShapeInfo shape;
  const int64_t zero_dim[1] = {0};
  if (!ExpectCode(tensora::ValidateShape(zero_dim, 1, &shape),
                  TS_INVALID_SHAPE, "shape zero dimension"))
    return 107;
  const int64_t negative_dim[1] = {-1};
  if (!ExpectCode(tensora::ValidateShape(negative_dim, 1, &shape),
                  TS_INVALID_SHAPE, "shape negative dimension"))
    return 108;
  const int64_t numel_overflow[2] = {
      std::numeric_limits<int64_t>::max(), 2};
  if (!ExpectCode(tensora::ValidateShape(numel_overflow, 2, &shape),
                  TS_INVALID_SHAPE, "shape numel overflow"))
    return 109;
  const int64_t byte_overflow[1] = {
      static_cast<int64_t>(std::numeric_limits<size_t>::max() / sizeof(float)) +
      1};
  if (!ExpectCode(tensora::ValidateShape(byte_overflow, 1, &shape),
                  TS_INVALID_SHAPE, "shape byte overflow"))
    return 110;

  const int64_t dims[2] = {2, 2};
  if (!tensora::ValidateShape(dims, 2, &shape).ok() ||
      shape.dimensions.size() != 2 || shape.numel != 4 ||
      shape.byte_size != 16 || shape.strides.size() != 2 ||
      shape.strides[0] != 2 || shape.strides[1] != 1) {
    return 111;
  }
  ShapeInfo same_shape;
  if (!tensora::ValidateShape(dims, 2, &same_shape).ok() ||
      !tensora::SameShape(shape, same_shape))
    return 112;
  const int64_t flat_dims[1] = {4};
  ShapeInfo flat_shape;
  if (!tensora::ValidateShape(flat_dims, 1, &flat_shape).ok() ||
      tensora::SameShape(shape, flat_shape))
    return 113;

  const uint64_t baseline_bytes = CpuStorage::LiveBytes();
  std::shared_ptr<CpuStorage> storage;
  if (!ExpectCode(CpuStorage::Filled(4, 2.0f, nullptr), TS_INVALID_ARGUMENT,
                  "CpuStorage filled null output"))
    return 114;
  if (!ExpectCode(CpuStorage::FromData(nullptr, 4, &storage),
                  TS_INVALID_ARGUMENT, "CpuStorage from null data"))
    return 115;
  if (!CpuStorage::Filled(4, 2.0f, &storage).ok() || !storage) return 116;
  if (storage->kind() != tensora::StorageKind::kCpu || storage->numel() != 4 ||
      storage->byte_size() != 16 || CpuStorage::LiveBytes() != baseline_bytes + 16)
    return 117;
  size_t written = 777;
  if (!ExpectCode(storage->CopyToHostF32(nullptr, 4, &written),
                  TS_INVALID_ARGUMENT, "CpuStorage copy null output"))
    return 118;
  float too_small[3] = {};
  if (!ExpectCode(storage->CopyToHostF32(too_small, 3, &written),
                  TS_INVALID_ARGUMENT, "CpuStorage small copy"))
    return 119;
  float values[4] = {};
  if (!storage->CopyToHostF32(values, 4, &written).ok() || written != 4 ||
      values[0] != 2.0f || values[3] != 2.0f)
    return 120;

  std::shared_ptr<Tensor> tensor = std::make_shared<Tensor>(shape, storage);
  if (tensor->numel() != 4 || tensor->dtype() != tensora::DType::kFloat32 ||
      tensor->device() != tensora::Device::kCpu || tensor->device_index() != 0)
    return 121;

  const tensora::Backend* backend = nullptr;
  if (!tensora::Dispatcher::For(tensora::Device::kCpu, &backend).ok() ||
      backend == nullptr)
    return 122;
  if (!ExpectCode(tensora::Dispatcher::For(tensora::Device::kCuda, &backend),
                  TS_UNSUPPORTED, "dispatcher cuda"))
    return 123;
  if (!ExpectCode(tensora::Dispatcher::For(tensora::Device::kCpu, nullptr),
                  TS_INVALID_ARGUMENT, "dispatcher null output"))
    return 124;

  std::shared_ptr<Tensor> full;
  tensora::CpuBackend cpu;
  if (!cpu.Full(shape, 3.0f, &full).ok() || !full) return 125;
  std::shared_ptr<Tensor> reshaped;
  const int64_t flat[1] = {4};
  ShapeInfo reshaped_shape;
  if (!tensora::ValidateShape(flat, 1, &reshaped_shape).ok() ||
      !cpu.Reshape(*full, reshaped_shape, &reshaped).ok() || !reshaped)
    return 126;
  std::shared_ptr<Tensor> transposed;
  if (!cpu.Transpose2D(*full, &transposed).ok() || !transposed) return 127;
  std::shared_ptr<Tensor> added;
  if (!cpu.Add(*full, *full, &added).ok() || !added) return 128;
  std::shared_ptr<Tensor> multiplied;
  if (!cpu.Multiply(*full, *full, &multiplied).ok() || !multiplied)
    return 129;
  std::shared_ptr<Tensor> summed;
  if (!cpu.Sum(*full, &summed).ok() || !summed || summed->numel() != 1)
    return 130;
  std::shared_ptr<Tensor> product;
  if (!cpu.Matmul(*full, *full, &product).ok() || !product) return 131;

  product.reset();
  summed.reset();
  multiplied.reset();
  added.reset();
  transposed.reset();
  reshaped.reset();
  full.reset();
  tensor.reset();
  storage.reset();
  if (CpuStorage::LiveBytes() != baseline_bytes) return 132;
  return 0;
}

}  // namespace

int main() {
  const int abi_guard = TestAbiGuardContracts();
  if (abi_guard != 0) return abi_guard;
  const int registry = TestRegistryContracts();
  if (registry != 0) return registry;
  return TestCoreContracts();
}
