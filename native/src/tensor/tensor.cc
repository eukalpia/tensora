#include "tensor/tensor.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "memory/cpu_storage.h"

namespace tensora {
namespace {

std::atomic<uint64_t> next_tensor_identity{1};

}  // namespace

std::shared_ptr<TensorIdentityAnchor> NewTensorIdentityAnchor() {
  const uint64_t identity =
      next_tensor_identity.fetch_add(1, std::memory_order_relaxed);
  return std::make_shared<TensorIdentityAnchor>(identity);
}

TensorVersionCounter::TensorVersionCounter(
    std::shared_ptr<TensorIdentityAnchor> identity_anchor)
    : identity(identity_anchor ? std::move(identity_anchor)
                               : NewTensorIdentityAnchor()) {}

Tensor::Tensor(ShapeInfo shape,
               std::shared_ptr<TensorStorage> storage,
               DType dtype,
               Device device,
               int32_t device_index,
               uint64_t storage_offset,
               std::shared_ptr<TensorVersionCounter> version_counter,
               std::shared_ptr<TensorIdentityAnchor> identity_anchor)
    : shape_(std::move(shape)),
      storage_(std::move(storage)),
      dtype_(dtype),
      device_(device),
      device_index_(device_index),
      storage_offset_(storage_offset),
      version_counter_(
          version_counter
              ? std::move(version_counter)
              : std::make_shared<TensorVersionCounter>(std::move(identity_anchor))) {}

bool Tensor::is_contiguous() const {
  uint64_t expected_stride = 1;
  for (size_t index = shape_.dimensions.size(); index > 0; --index) {
    const size_t dimension_index = index - 1;
    if (shape_.strides[dimension_index] != expected_stride) return false;
    expected_stride *=
        static_cast<uint64_t>(shape_.dimensions[dimension_index]);
  }
  return true;
}

uint64_t Tensor::logical_storage_index(uint64_t logical_index) const {
  uint64_t remaining = logical_index;
  uint64_t storage_index = storage_offset_;
  for (size_t index = shape_.dimensions.size(); index > 0; --index) {
    const size_t dimension_index = index - 1;
    const uint64_t dimension =
        static_cast<uint64_t>(shape_.dimensions[dimension_index]);
    const uint64_t coordinate = remaining % dimension;
    remaining /= dimension;
    storage_index += coordinate * shape_.strides[dimension_index];
  }
  return storage_index;
}

uint64_t Tensor::version() const {
  return version_counter_->value.load(std::memory_order_acquire);
}

void Tensor::increment_version() {
  version_counter_->value.fetch_add(1, std::memory_order_acq_rel);
}

Status Tensor::CopyToHostF32(float* out_values,
                             size_t capacity,
                             size_t* out_written) const {
  if (out_written == nullptr) {
    return InvalidArgument("tensor copy: output count pointer is null");
  }
  *out_written = 0;
  if (dtype_ != DType::kFloat32) {
    return Unsupported("tensor copy: tensor dtype is not float32");
  }
  if (capacity < numel()) {
    return InvalidArgument("tensor copy: output capacity is smaller than tensor");
  }
  if (numel() > 0 && out_values == nullptr) {
    return InvalidArgument("tensor copy: output values pointer is null");
  }

  if (is_contiguous() && storage_offset_ == 0 &&
      storage_->byte_size() == shape_.byte_size) {
    Status status = storage_->CopyToHostF32(out_values, capacity, out_written);
    if (!status.ok()) return status;
    if (*out_written != numel()) {
      *out_written = 0;
      return InternalError(
          "tensor copy: storage returned an inconsistent element count");
    }
    return Status::Ok();
  }

  if (device_ != Device::kCpu || device_index_ != 0) {
    return Unsupported(
        "tensor copy: non-contiguous accelerator views are not qualified yet");
  }
  auto storage = std::dynamic_pointer_cast<CpuStorage>(storage_);
  if (!storage) {
    return Unsupported("tensor copy: tensor is not backed by CPU storage");
  }
  const auto& values = storage->values();
  for (uint64_t logical_index = 0; logical_index < numel(); ++logical_index) {
    const uint64_t storage_index = logical_storage_index(logical_index);
    if (storage_index >= values.size()) {
      *out_written = 0;
      return InternalError("tensor copy: view layout exceeds backing storage");
    }
    out_values[static_cast<size_t>(logical_index)] =
        values[static_cast<size_t>(storage_index)];
  }
  *out_written = static_cast<size_t>(numel());
  return Status::Ok();
}

}  // namespace tensora
