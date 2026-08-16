#include "tensor/tensor.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "core/abi_guard.h"
#include "core/integer_math.h"
#include "memory/cpu_storage.h"
#include "runtime/handle_registry.h"
#include "tensora.h"

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

Status Tensor::CopyToHostRaw(void* out_data,
                             size_t capacity_bytes,
                             size_t* out_written_bytes) const {
  if (out_written_bytes == nullptr) {
    return InvalidArgument("tensor copy: output byte count pointer is null");
  }
  *out_written_bytes = 0;

  size_t expected_bytes = 0;
  Status status = CheckedByteSize(numel(), DTypeByteWidth(dtype_),
                                  "tensor copy", &expected_bytes);
  if (!status.ok()) return status;
  if (capacity_bytes < expected_bytes) {
    return InvalidArgument("tensor copy: output byte capacity is too small");
  }
  if (expected_bytes > 0 && out_data == nullptr) {
    return InvalidArgument("tensor copy: output data pointer is null");
  }

  if (device_ != Device::kCpu || device_index_ != 0) {
    return Unsupported(
        "tensor copy: generic host copy is qualified for CPU tensors only");
  }
  auto storage = std::dynamic_pointer_cast<CpuStorage>(storage_);
  if (!storage) {
    return Unsupported("tensor copy: tensor is not backed by CPU storage");
  }
  if (storage->dtype() != dtype_) {
    return InternalError("tensor copy: tensor and storage dtype disagree");
  }

  if (is_contiguous() && storage_offset_ == 0 && storage->numel() == numel()) {
    status =
        storage->CopyToHostRaw(out_data, capacity_bytes, out_written_bytes);
    if (!status.ok()) return status;
    if (*out_written_bytes != expected_bytes) {
      *out_written_bytes = 0;
      return InternalError(
          "tensor copy: storage returned an inconsistent byte count");
    }
    return Status::Ok();
  }

  auto* output = static_cast<uint8_t*>(out_data);
  const size_t element_bytes = DTypeByteWidth(dtype_);
  for (uint64_t logical_index = 0; logical_index < numel(); ++logical_index) {
    const uint64_t storage_index = logical_storage_index(logical_index);
    status = storage->CopyElementTo(
        storage_index,
        output + static_cast<size_t>(logical_index) * element_bytes,
        element_bytes);
    if (!status.ok()) {
      *out_written_bytes = 0;
      return status;
    }
  }
  *out_written_bytes = expected_bytes;
  return Status::Ok();
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

  if (device_ == Device::kCpu && device_index_ == 0) {
    size_t written_bytes = 0;
    Status status = CopyToHostRaw(
        out_values, capacity * sizeof(float), &written_bytes);
    if (!status.ok()) return status;
    if (written_bytes % sizeof(float) != 0) {
      return InternalError("tensor copy: float32 byte count is inconsistent");
    }
    *out_written = written_bytes / sizeof(float);
    return Status::Ok();
  }

  if (is_contiguous() && storage_offset_ == 0 &&
      storage_->byte_size() == numel() * sizeof(float)) {
    Status status = storage_->CopyToHostF32(out_values, capacity, out_written);
    if (!status.ok()) return status;
    if (*out_written != numel()) {
      *out_written = 0;
      return InternalError(
          "tensor copy: storage returned an inconsistent element count");
    }
    return Status::Ok();
  }

  return Unsupported(
      "tensor copy: non-contiguous accelerator views are not qualified yet");
}

namespace typed_tensor_abi {

Status Lookup(ts_tensor_t handle, std::shared_ptr<Tensor>* out) {
  return HandleRegistry::Instance().Lookup<Tensor>(
      handle, HandleType::kTensor, out);
}

Status Insert(std::shared_ptr<Tensor> tensor, ts_tensor_t* out_handle) {
  if (out_handle == nullptr) {
    return InvalidArgument("typed tensor ABI: output handle pointer is null");
  }
  *out_handle = 0;
  return HandleRegistry::Instance().Insert(
      HandleType::kTensor, std::move(tensor), out_handle);
}

Status MakeCpuTensor(ShapeInfo shape,
                     DType dtype,
                     std::shared_ptr<CpuStorage> storage,
                     std::shared_ptr<Tensor>* out) {
  if (out == nullptr) {
    return InvalidArgument("typed tensor ABI: output tensor pointer is null");
  }
  *out = nullptr;
  if (!storage || storage->dtype() != dtype ||
      storage->numel() != shape.numel) {
    return InternalError(
        "typed tensor ABI: storage metadata does not match tensor metadata");
  }
  return AllocationGuard("typed tensor ABI", [&]() -> Status {
    *out = std::make_shared<Tensor>(std::move(shape), std::move(storage), dtype);
    return Status::Ok();
  });
}

}  // namespace typed_tensor_abi

}  // namespace tensora

extern "C" {

ts_status_t ts_tensor_from_host(const void* data,
                                size_t data_bytes,
                                uint32_t dtype_code,
                                const int64_t* dims,
                                size_t rank,
                                ts_tensor_t* out_tensor) {
  return tensora::AbiGuard("tensor_from_host", [&]() -> tensora::Status {
    if (out_tensor == nullptr) {
      return tensora::InvalidArgument(
          "tensor_from_host: output handle pointer is null");
    }
    *out_tensor = 0;

    tensora::DType dtype = tensora::DType::kFloat32;
    tensora::Status status = tensora::DTypeFromCode(dtype_code, &dtype);
    if (!status.ok()) return status;

    tensora::ShapeInfo shape;
    status = tensora::ValidateShape(dims, rank, &shape);
    if (!status.ok()) return status;

    size_t expected_bytes = 0;
    status = tensora::CheckedByteSize(
        shape.numel, tensora::DTypeByteWidth(dtype), "tensor_from_host",
        &expected_bytes);
    if (!status.ok()) return status;
    if (data_bytes != expected_bytes) {
      return tensora::InvalidArgument(
          "tensor_from_host: byte count must equal shape elements times dtype width");
    }
    if (expected_bytes > 0 && data == nullptr) {
      return tensora::InvalidArgument(
          "tensor_from_host: input data pointer is null");
    }

    std::shared_ptr<tensora::CpuStorage> storage;
    status = tensora::CpuStorage::FromRaw(
        data, data_bytes, shape.numel, dtype, &storage);
    if (!status.ok()) return status;

    std::shared_ptr<tensora::Tensor> tensor;
    status = tensora::typed_tensor_abi::MakeCpuTensor(
        std::move(shape), dtype, std::move(storage), &tensor);
    if (!status.ok()) return status;
    return tensora::typed_tensor_abi::Insert(std::move(tensor), out_tensor);
  });
}

ts_status_t ts_tensor_full(const void* scalar,
                           size_t scalar_bytes,
                           uint32_t dtype_code,
                           const int64_t* dims,
                           size_t rank,
                           ts_tensor_t* out_tensor) {
  return tensora::AbiGuard("tensor_full", [&]() -> tensora::Status {
    if (out_tensor == nullptr) {
      return tensora::InvalidArgument(
          "tensor_full: output handle pointer is null");
    }
    *out_tensor = 0;

    tensora::DType dtype = tensora::DType::kFloat32;
    tensora::Status status = tensora::DTypeFromCode(dtype_code, &dtype);
    if (!status.ok()) return status;
    if (scalar_bytes != tensora::DTypeByteWidth(dtype)) {
      return tensora::InvalidArgument(
          "tensor_full: scalar byte count does not match dtype");
    }
    if (scalar == nullptr) {
      return tensora::InvalidArgument("tensor_full: scalar pointer is null");
    }

    tensora::ShapeInfo shape;
    status = tensora::ValidateShape(dims, rank, &shape);
    if (!status.ok()) return status;

    std::shared_ptr<tensora::CpuStorage> storage;
    status = tensora::CpuStorage::Full(
        scalar, scalar_bytes, shape.numel, dtype, &storage);
    if (!status.ok()) return status;

    std::shared_ptr<tensora::Tensor> tensor;
    status = tensora::typed_tensor_abi::MakeCpuTensor(
        std::move(shape), dtype, std::move(storage), &tensor);
    if (!status.ok()) return status;
    return tensora::typed_tensor_abi::Insert(std::move(tensor), out_tensor);
  });
}

ts_status_t ts_tensor_cast(ts_tensor_t tensor,
                           uint32_t target_dtype_code,
                           ts_tensor_t* out_tensor) {
  return tensora::AbiGuard("tensor_cast", [&]() -> tensora::Status {
    if (out_tensor == nullptr) {
      return tensora::InvalidArgument(
          "tensor_cast: output handle pointer is null");
    }
    *out_tensor = 0;

    tensora::DType target_dtype = tensora::DType::kFloat32;
    tensora::Status status =
        tensora::DTypeFromCode(target_dtype_code, &target_dtype);
    if (!status.ok()) return status;

    std::shared_ptr<tensora::Tensor> source;
    status = tensora::typed_tensor_abi::Lookup(tensor, &source);
    if (!status.ok()) return status;
    if (source->device() != tensora::Device::kCpu ||
        source->device_index() != 0) {
      return tensora::Unsupported(
          "tensor_cast: P1A cast is qualified for CPU tensors only");
    }
    auto source_storage =
        std::dynamic_pointer_cast<tensora::CpuStorage>(source->storage());
    if (!source_storage || source_storage->dtype() != source->dtype()) {
      return tensora::InternalError(
          "tensor_cast: CPU tensor storage metadata is inconsistent");
    }

    std::shared_ptr<tensora::CpuStorage> target_storage;
    status = tensora::CpuStorage::Cast(
        *source_storage, target_dtype, &target_storage);
    if (!status.ok()) return status;

    std::shared_ptr<tensora::Tensor> result;
    status = tensora::typed_tensor_abi::MakeCpuTensor(
        source->shape(), target_dtype, std::move(target_storage), &result);
    if (!status.ok()) return status;
    return tensora::typed_tensor_abi::Insert(std::move(result), out_tensor);
  });
}

ts_status_t ts_tensor_copy_to_host(ts_tensor_t tensor,
                                   void* out_data,
                                   size_t capacity_bytes,
                                   size_t* out_written_bytes) {
  return tensora::AbiGuard("tensor_copy_to_host", [&]() -> tensora::Status {
    if (out_written_bytes == nullptr) {
      return tensora::InvalidArgument(
          "tensor_copy_to_host: output byte count pointer is null");
    }
    *out_written_bytes = 0;

    std::shared_ptr<tensora::Tensor> object;
    tensora::Status status =
        tensora::typed_tensor_abi::Lookup(tensor, &object);
    if (!status.ok()) return status;
    return object->CopyToHostRaw(
        out_data, capacity_bytes, out_written_bytes);
  });
}

}  // extern "C"
