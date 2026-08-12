#include "training/torch_storage.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>

namespace tensora::training {

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
  if (tensor.numel() < 0) {
    return InternalError("torch storage: tensor reported a negative element count");
  }
  const auto numel = static_cast<uint64_t>(tensor.numel());
  if (numel > std::numeric_limits<uint64_t>::max() / sizeof(float)) {
    return OutOfMemory("torch storage: byte size overflow");
  }

  try {
    *out = std::shared_ptr<TorchStorage>(
        new TorchStorage(std::move(tensor)));
    return Status::Ok();
  } catch (const c10::Error& error) {
    return InternalError(std::string("torch storage: ") + error.what());
  } catch (const std::bad_alloc&) {
    return OutOfMemory("torch storage: allocation failed");
  }
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

  try {
    torch::NoGradGuard guard;
    const torch::Tensor host =
        tensor_.detach().to(torch::kCPU).contiguous();
    if (host.scalar_type() != torch::kFloat32) {
      return Unsupported("torch storage: host copy is not float32");
    }
    if (host.numel() != tensor_.numel()) {
      return InternalError("torch storage: host copy changed element count");
    }
    if (numel > 0) {
      std::memcpy(out_values, host.data_ptr<float>(), numel * sizeof(float));
    }
    *out_written = numel;
    return Status::Ok();
  } catch (const c10::Error& error) {
    return InternalError(std::string("torch storage host copy: ") + error.what());
  }
}

uint64_t TorchStorage::LiveBytes() {
  return live_bytes_.load(std::memory_order_relaxed);
}

}  // namespace tensora::training
