#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>

#include "backends/backend.h"
#include "core/status.h"
#include "memory/cpu_storage.h"
#include "runtime/dispatcher.h"
#include "runtime/handle_registry.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

namespace {

bool ExpectCode(const tensora::Status& status,
                ts_status_t expected,
                const char* message) {
  if (status.code() == expected) return true;
  std::cerr << message << ": expected status " << expected << ", got "
            << status.code() << "\n";
  return false;
}

int TestCoreContracts() {
  using tensora::Backend;
  using tensora::CpuStorage;
  using tensora::Device;
  using tensora::Dispatcher;
  using tensora::ShapeInfo;
  using tensora::Tensor;

  if (!ExpectCode(tensora::InvalidArgument("invalid"), TS_INVALID_ARGUMENT,
                  "InvalidArgument status") ||
      !ExpectCode(tensora::InvalidShape("shape"), TS_INVALID_SHAPE,
                  "InvalidShape status") ||
      !ExpectCode(tensora::OutOfMemory("oom"), TS_OUT_OF_MEMORY,
                  "OutOfMemory status") ||
      !ExpectCode(tensora::Unsupported("unsupported"), TS_UNSUPPORTED,
                  "Unsupported status") ||
      !ExpectCode(tensora::InvalidHandle("handle"), TS_INVALID_HANDLE,
                  "InvalidHandle status") ||
      !ExpectCode(tensora::InternalError("internal"), TS_INTERNAL_ERROR,
                  "InternalError status") ||
      !ExpectCode(tensora::ModelError("model"), TS_MODEL_ERROR,
                  "ModelError status")) {
    return 100;
  }

  tensora::ClearLastError();
  if (std::strlen(tensora::LastErrorMessage()) != 0) return 101;
  tensora::SetLastError(tensora::InternalError("internal"));
  if (std::strcmp(tensora::LastErrorMessage(), "internal") != 0) return 102;
  tensora::SetLastError(tensora::Status::Ok());
  if (std::strlen(tensora::LastErrorMessage()) != 0) return 103;

  ShapeInfo shape;
  if (!ExpectCode(tensora::ValidateShape(nullptr, 0, nullptr),
                  TS_INVALID_ARGUMENT, "shape null output"))
    return 104;
  if (!ExpectCode(tensora::ValidateShape(nullptr, 9, &shape), TS_INVALID_SHAPE,
                  "shape excessive rank"))
    return 105;
  if (!ExpectCode(tensora::ValidateShape(nullptr, 1, &shape),
                  TS_INVALID_ARGUMENT, "shape null dimensions"))
    return 106;

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
  if (!tensora::ValidateShape(dims, 2, &shape).ok() || shape.rank() != 2 ||
      shape.numel != 4 || shape.byte_size != 16 || shape.strides.size() != 2 ||
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
                  TS_INVALID_ARGUMENT, "CpuStorage null input"))
    return 115;
  const float values[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  if (!CpuStorage::FromData(values, 4, &storage).ok() || !storage ||
      CpuStorage::LiveBytes() != baseline_bytes + 16)
    return 116;

  size_t written = 99;
  float output[4] = {0, 0, 0, 0};
  if (!ExpectCode(storage->CopyToHostF32(output, 4, nullptr),
                  TS_INVALID_ARGUMENT, "CpuStorage null written"))
    return 117;
  if (!ExpectCode(storage->CopyToHostF32(output, 3, &written),
                  TS_INVALID_ARGUMENT, "CpuStorage short output") ||
      written != 0)
    return 118;
  if (!ExpectCode(storage->CopyToHostF32(nullptr, 4, &written),
                  TS_INVALID_ARGUMENT, "CpuStorage null output") ||
      written != 0)
    return 119;
  if (!storage->CopyToHostF32(output, 4, &written).ok() || written != 4 ||
      output[0] != 1.0f || output[3] != 4.0f)
    return 120;

  auto tensor = std::make_shared<Tensor>(shape, storage);
  const Backend* backend = nullptr;
  if (!ExpectCode(Dispatcher::For(Device::kCpu, nullptr), TS_INVALID_ARGUMENT,
                  "Dispatcher null backend"))
    return 121;
  if (!Dispatcher::For(Device::kCpu, &backend).ok() || backend == nullptr)
    return 122;
  backend = reinterpret_cast<const Backend*>(1);
  if (!ExpectCode(Dispatcher::For(static_cast<Device>(999), &backend),
                  TS_UNSUPPORTED, "Dispatcher unknown device") ||
      backend != nullptr)
    return 123;
  if (!ExpectCode(Dispatcher::ForTensor(*tensor, nullptr), TS_INVALID_ARGUMENT,
                  "Dispatcher tensor null backend"))
    return 124;
  if (!Dispatcher::ForTensor(*tensor, &backend).ok() || backend == nullptr)
    return 125;
  if (!ExpectCode(Dispatcher::ForTensors(*tensor, *tensor, nullptr),
                  TS_INVALID_ARGUMENT, "Dispatcher tensors null backend"))
    return 126;

  auto accelerator_tensor = std::make_shared<Tensor>(
      shape, storage, tensora::DType::kFloat32, Device::kCuda, 0);
  if (!ExpectCode(
          Dispatcher::ForTensors(*tensor, *accelerator_tensor, &backend),
          TS_INVALID_ARGUMENT, "Dispatcher mismatched devices"))
    return 127;
  if (!Dispatcher::ForTensors(*tensor, *tensor, &backend).ok() ||
      backend == nullptr)
    return 128;

  accelerator_tensor.reset();
  tensor.reset();
  storage.reset();
  if (CpuStorage::LiveBytes() != baseline_bytes) return 129;
  return 0;
}

}  // namespace

int main() {
  auto& registry = tensora::HandleRegistry::Instance();
  const auto payload = std::make_shared<int>(42);

  uint64_t handle = 123;
  if (!ExpectCode(registry.Insert(tensora::HandleType::kRuntimeReserved,
                                  payload, nullptr),
                  TS_INVALID_ARGUMENT,
                  "insert accepted null output pointer")) {
    return 1;
  }
  if (!ExpectCode(registry.Insert(tensora::HandleType::kRuntimeReserved,
                                  std::shared_ptr<void>(), &handle),
                  TS_INVALID_ARGUMENT, "insert accepted null object")) {
    return 2;
  }
  if (handle != 123) {
    std::cerr << "failed insert unexpectedly mutated output handle\n";
    return 3;
  }

  uint64_t wrong_type_handle = 0;
  const tensora::Status insert_status = registry.Insert(
      tensora::HandleType::kRuntimeReserved, payload, &wrong_type_handle);
  if (!insert_status.ok() || wrong_type_handle == 0) {
    std::cerr << "failed to create internal wrong-type test handle\n";
    return 4;
  }
  if (registry.Count(tensora::HandleType::kRuntimeReserved) != 1 ||
      registry.Count(tensora::HandleType::kTensor) != 0) {
    std::cerr << "registry count did not reflect inserted object type\n";
    return 5;
  }

  std::shared_ptr<tensora::Tensor> tensor;
  if (!ExpectCode(registry.Lookup<tensora::Tensor>(
                      wrong_type_handle, tensora::HandleType::kTensor, nullptr),
                  TS_INVALID_ARGUMENT,
                  "lookup accepted null output pointer")) {
    return 6;
  }
  if (!ExpectCode(registry.Lookup<tensora::Tensor>(
                      0, tensora::HandleType::kTensor, &tensor),
                  TS_INVALID_HANDLE, "lookup accepted handle zero")) {
    return 7;
  }
  if (!ExpectCode(registry.Lookup<tensora::Tensor>(
                      UINT64_C(999999999), tensora::HandleType::kTensor,
                      &tensor),
                  TS_INVALID_HANDLE, "lookup accepted unknown handle")) {
    return 8;
  }

  const tensora::Status lookup_status = registry.Lookup<tensora::Tensor>(
      wrong_type_handle, tensora::HandleType::kTensor, &tensor);
  if (lookup_status.code() != TS_INVALID_HANDLE || tensor) {
    std::cerr << "handle registry accepted a handle with the wrong object type\n";
    return 9;
  }

  std::shared_ptr<int> retained_payload;
  const tensora::Status good_lookup = registry.Lookup<int>(
      wrong_type_handle, tensora::HandleType::kRuntimeReserved,
      &retained_payload);
  if (!good_lookup.ok() || !retained_payload || *retained_payload != 42) {
    std::cerr << "handle registry failed a valid typed lookup\n";
    return 10;
  }

  if (!ExpectCode(registry.Retain(0, tensora::HandleType::kRuntimeReserved),
                  TS_INVALID_HANDLE, "retain accepted handle zero")) {
    return 11;
  }
  if (!ExpectCode(registry.Retain(UINT64_C(999999999),
                                  tensora::HandleType::kRuntimeReserved),
                  TS_INVALID_HANDLE, "retain accepted unknown handle")) {
    return 12;
  }
  if (!ExpectCode(registry.Retain(wrong_type_handle,
                                  tensora::HandleType::kTensor),
                  TS_INVALID_HANDLE, "retain accepted wrong handle type")) {
    return 13;
  }
  if (!registry.Retain(wrong_type_handle,
                       tensora::HandleType::kRuntimeReserved)
           .ok()) {
    std::cerr << "valid retain failed\n";
    return 14;
  }

  if (!ExpectCode(registry.Release(0, tensora::HandleType::kRuntimeReserved),
                  TS_INVALID_HANDLE, "release accepted handle zero")) {
    return 15;
  }
  if (!ExpectCode(registry.Release(UINT64_C(999999999),
                                   tensora::HandleType::kRuntimeReserved),
                  TS_INVALID_HANDLE, "release accepted unknown handle")) {
    return 16;
  }
  if (!ExpectCode(registry.Release(wrong_type_handle,
                                   tensora::HandleType::kTensor),
                  TS_INVALID_HANDLE, "release accepted wrong handle type")) {
    return 17;
  }

  if (!registry.Release(wrong_type_handle,
                        tensora::HandleType::kRuntimeReserved)
           .ok()) {
    std::cerr << "release after retain failed\n";
    return 18;
  }
  if (registry.Count(tensora::HandleType::kRuntimeReserved) != 1) {
    std::cerr << "release after retain prematurely retired object\n";
    return 19;
  }

  const tensora::Status release_status = registry.Release(
      wrong_type_handle, tensora::HandleType::kRuntimeReserved);
  if (!release_status.ok()) {
    std::cerr << "failed to release internal wrong-type test handle\n";
    return 20;
  }
  if (registry.Count(tensora::HandleType::kRuntimeReserved) != 0) {
    std::cerr << "final release did not retire object\n";
    return 21;
  }

  const tensora::Status stale_status = registry.Lookup<tensora::Tensor>(
      wrong_type_handle, tensora::HandleType::kTensor, &tensor);
  if (stale_status.code() != TS_INVALID_HANDLE) {
    std::cerr << "handle registry accepted a stale handle\n";
    return 22;
  }
  if (!ExpectCode(registry.Release(wrong_type_handle,
                                   tensora::HandleType::kRuntimeReserved),
                  TS_INVALID_HANDLE, "double release accepted stale handle")) {
    return 23;
  }

  const int core_contracts = TestCoreContracts();
  if (core_contracts != 0) return core_contracts;
  return 0;
}
