#include "training/torch_storage.h"

#include <cstring>
#include <limits>
#include <utility>

#include "training/torch_backend.h"

namespace tensora::training {

namespace internal {

Status ValidateTorchStorageSize(int64_t numel, uint64_t* out_bytes) {
  if (out_bytes == nullptr) {
    return InvalidArgument("torch storage: byte-size output pointer is null");
  }
  *out_bytes = 0;
  if (numel < 0) {
    return InternalError("torch storage: tensor reported a negative element count");
  }
  const auto unsigned_numel = static_cast<uint64_t>(numel);
  if (unsigned_numel >
      std::numeric_limits<uint64_t>::max() / sizeof(float)) {
    return OutOfMemory("torch storage: byte size overflow");
  }
  *out_bytes = unsigned_numel * sizeof(float);
  return Status::Ok();
}

Status ValidateHostCopyMetadata(int64_t expected_numel,
                                torch::ScalarType actual_type,
                                int64_t actual_numel) {
  if (actual_type != torch::kFloat32) {
    return Unsupported("torch storage: host copy is not float32");
  }
  if (actual_numel != expected_numel) {
    return InternalError("torch storage: host copy changed element count");
  }
  return Status::Ok();
}

}  // namespace internal

std::atomic<uint64_t> TorchStorage::live_bytes_{0};

TorchStorage::TorchStorage(torch::Tensor tensor)
    : tensor_(std::move(tensor)),
      byte_size_(static_cast<uint64_t>(tensor_.numel()) * sizeof(float)) {
  live_bytes_.fetch_add(byte_size_, std::memory_order_relaxed);
}

TorchStorage::~TorchStorage() {
  live_bytes_.fetch_sub(byte_size_, std::memory_order_relaxed);
}

Status TorchStorage::FromTensor(torch::Tensor tensor,
                                std::shared_ptr<TorchStorage>* out) {
  if (out == nullptr) {
    return InvalidArgument("torch storage: output pointer is null");
  }
  *out = nullptr;
  if (!tensor.defined()) {
    return InvalidArgument("torch storage: tensor is undefined");
  }
  if (tensor.scalar_type() != torch::kFloat32) {
    return Unsupported("torch storage: only float32 is supported");
  }

  uint64_t byte_size = 0;
  Status status = internal::ValidateTorchStorageSize(tensor.numel(), &byte_size);
  if (!status.ok()) return status;
  (void)byte_size;

  return internal::GuardAllocation("torch storage", [&]() {
    return internal::GuardTorch("torch storage", [&]() {
      *out = std::shared_ptr<TorchStorage>(
          new TorchStorage(std::move(tensor)));
      return Status::Ok();
    });
  });
}

Status TorchStorage::CopyToHostF32(float* out_values,
                                   size_t capacity,
                                   size_t* out_written) const {
  if (out_written == nullptr) {
    return InvalidArgument("torch storage: output count pointer is null");
  }
  *out_written = 0;

  const auto numel = static_cast<size_t>(tensor_.numel());
  if (capacity < numel) {
    return InvalidArgument("torch storage: output capacity is too small");
  }
  if (numel > 0 && out_values == nullptr) {
    return InvalidArgument("torch storage: output values pointer is null");
  }

  return internal::GuardTorch("torch storage host copy", [&]() {
    torch::NoGradGuard guard;
    const torch::Tensor host = tensor_.detach().to(torch::kCPU).contiguous();
    Status status = internal::ValidateHostCopyMetadata(
        tensor_.numel(), host.scalar_type(), host.numel());
    if (!status.ok()) return status;
    if (numel > 0) {
      std::memcpy(out_values, host.data_ptr<float>(), numel * sizeof(float));
    }
    *out_written = numel;
    return Status::Ok();
  });
}

uint64_t TorchStorage::LiveBytes() {
  return live_bytes_.load(std::memory_order_relaxed);
}

}  // namespace tensora::training
