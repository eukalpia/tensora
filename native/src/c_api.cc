#include "tensora.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <string>

#include "backends/backend.h"
#include "core/status.h"
#include "memory/cpu_storage.h"
#include "runtime/dispatcher.h"
#include "runtime/handle_registry.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"
#include "training/training_bridge.h"

namespace tensora {
namespace {

template <typename Function>
ts_status_t GuardedAbiCall(const char* operation, Function&& function) noexcept {
  ClearLastError();
  try {
    Status status = function();
    if (!status.ok()) {
      if (status.message().empty()) {
        status = Status(status.code(), std::string(operation) + ": failed");
      }
      SetLastError(status);
    }
    return status.code();
  } catch (const std::bad_alloc&) {
    const Status status =
        OutOfMemory(std::string(operation) + ": native allocation failed");
    SetLastError(status);
    return status.code();
  } catch (const std::exception& error) {
    const Status status =
        InternalError(std::string(operation) + ": " + error.what());
    SetLastError(status);
    return status.code();
  } catch (...) {
    const Status status =
        InternalError(std::string(operation) + ": unknown native exception");
    SetLastError(status);
    return status.code();
  }
}

Status LookupTensor(ts_tensor_t handle, std::shared_ptr<Tensor>* out) {
  return HandleRegistry::Instance().Lookup<Tensor>(
      handle, HandleType::kTensor, out);
}

Status InsertTensor(std::shared_ptr<Tensor> tensor, ts_tensor_t* out_handle) {
  if (out_handle == nullptr) {
    return InvalidArgument("tensor: output handle pointer is null");
  }
  *out_handle = 0;
  return HandleRegistry::Instance().Insert(
      HandleType::kTensor, std::move(tensor), out_handle);
}

Status BackendFor(const Tensor& tensor, const Backend** out_backend) {
  return Dispatcher::ForTensor(tensor, out_backend);
}

Status DeviceFromCode(uint32_t code, Device* out_device) {
  if (out_device == nullptr) {
    return InvalidArgument("device: output pointer is null");
  }
  switch (code) {
    case TS_DEVICE_CPU:
      *out_device = Device::kCpu;
      return Status::Ok();
    case TS_DEVICE_CUDA:
      *out_device = Device::kCuda;
      return Status::Ok();
    case TS_DEVICE_MPS:
      *out_device = Device::kMps;
      return Status::Ok();
    case TS_DEVICE_XPU:
      *out_device = Device::kXpu;
      return Status::Ok();
    case TS_DEVICE_HIP:
      *out_device = Device::kHip;
      return Status::Ok();
    default:
      return Unsupported("device: unknown device kind");
  }
}

using BinaryOperation = Status (Backend::*)(
    const Tensor&, const Tensor&, std::shared_ptr<Tensor>*) const;

Status RunBinaryOperation(ts_tensor_t left,
                          ts_tensor_t right,
                          ts_tensor_t* out_tensor,
                          const char* operation,
                          BinaryOperation function) {
  if (out_tensor == nullptr) {
    return InvalidArgument(std::string(operation) +
                           ": output handle pointer is null");
  }
  *out_tensor = 0;

  std::shared_ptr<Tensor> left_object;
  std::shared_ptr<Tensor> right_object;
  Status status = LookupTensor(left, &left_object);
  if (!status.ok()) return status;
  status = LookupTensor(right, &right_object);
  if (!status.ok()) return status;

  const Backend* backend = nullptr;
  status = Dispatcher::ForTensors(*left_object, *right_object, &backend);
  if (!status.ok()) return status;

  std::shared_ptr<Tensor> result;
  status = (backend->*function)(*left_object, *right_object, &result);
  if (!status.ok()) return status;
  return InsertTensor(std::move(result), out_tensor);
}

}  // namespace
}  // namespace tensora

extern "C" {

uint32_t ts_abi_version(void) { return TS_ABI_VERSION; }

const char* ts_last_error_message(void) { return tensora::LastErrorMessage(); }

const char* ts_status_name(int32_t status) {
  switch (status) {
    case TS_OK:
      return "OK";
    case TS_INVALID_ARGUMENT:
      return "INVALID_ARGUMENT";
    case TS_INVALID_SHAPE:
      return "INVALID_SHAPE";
    case TS_OUT_OF_MEMORY:
      return "OUT_OF_MEMORY";
    case TS_UNSUPPORTED:
      return "UNSUPPORTED";
    case TS_INVALID_HANDLE:
      return "INVALID_HANDLE";
    case TS_INTERNAL_ERROR:
      return "INTERNAL_ERROR";
    case TS_MODEL_ERROR:
      return "MODEL_ERROR";
  }
  return "UNKNOWN_STATUS";
}

ts_status_t ts_noop(void) {
  return tensora::GuardedAbiCall("noop", [] { return tensora::Status::Ok(); });
}

ts_status_t ts_runtime_device_count(uint32_t device, uint32_t* out_count) {
  return tensora::GuardedAbiCall("runtime_device_count", [&] {
    if (out_count == nullptr) {
      return tensora::InvalidArgument(
          "runtime_device_count: output pointer is null");
    }
    *out_count = 0;
    switch (device) {
      case TS_DEVICE_CPU:
        *out_count = 1;
        return tensora::Status::Ok();
      case TS_DEVICE_CUDA:
        return tensora::training::CudaDeviceCount(out_count);
      case TS_DEVICE_MPS:
      case TS_DEVICE_XPU:
      case TS_DEVICE_HIP:
        return tensora::Status::Ok();
      default:
        return tensora::Unsupported("runtime_device_count: unknown device kind");
    }
  });
}

ts_status_t ts_runtime_cuda_device_count(uint32_t* out_count) {
  return ts_runtime_device_count(TS_DEVICE_CUDA, out_count);
}

ts_status_t ts_tensor_from_f32(const float* data,
                               size_t data_length,
                               const int64_t* dims,
                               size_t rank,
                               ts_tensor_t* out_tensor) {
  return tensora::GuardedAbiCall("tensor_from_f32", [&] {
    if (out_tensor == nullptr) {
      return tensora::InvalidArgument(
          "tensor_from_f32: output handle pointer is null");
    }
    *out_tensor = 0;

    tensora::ShapeInfo shape;
    tensora::Status status = tensora::ValidateShape(dims, rank, &shape);
    if (!status.ok()) return status;

    if (shape.numel != static_cast<uint64_t>(data_length)) {
      return tensora::InvalidArgument(
          "tensor_from_f32: data length must equal validated shape element count");
    }
    if (data_length > 0 && data == nullptr) {
      return tensora::InvalidArgument(
          "tensor_from_f32: input data pointer is null");
    }

    const tensora::Backend* backend = nullptr;
    status = tensora::Dispatcher::For(tensora::Device::kCpu, &backend);
    if (!status.ok()) return status;

    std::shared_ptr<tensora::Tensor> tensor;
    status = backend->FromData(shape, data, &tensor);
    if (!status.ok()) return status;
    return tensora::InsertTensor(std::move(tensor), out_tensor);
  });
}

ts_status_t ts_tensor_full_f32(const int64_t* dims,
                               size_t rank,
                               float value,
                               ts_tensor_t* out_tensor) {
  return tensora::GuardedAbiCall("tensor_full_f32", [&] {
    if (out_tensor == nullptr) {
      return tensora::InvalidArgument(
          "tensor_full_f32: output handle pointer is null");
    }
    *out_tensor = 0;

    tensora::ShapeInfo shape;
    tensora::Status status = tensora::ValidateShape(dims, rank, &shape);
    if (!status.ok()) return status;

    const tensora::Backend* backend = nullptr;
    status = tensora::Dispatcher::For(tensora::Device::kCpu, &backend);
    if (!status.ok()) return status;

    std::shared_ptr<tensora::Tensor> tensor;
    status = backend->Full(shape, value, &tensor);
    if (!status.ok()) return status;
    return tensora::InsertTensor(std::move(tensor), out_tensor);
  });
}

ts_status_t ts_tensor_rank(ts_tensor_t tensor, size_t* out_rank) {
  return tensora::GuardedAbiCall("tensor_rank", [&] {
    if (out_rank == nullptr) {
      return tensora::InvalidArgument("tensor_rank: output rank pointer is null");
    }

    std::shared_ptr<tensora::Tensor> object;
    tensora::Status status = tensora::LookupTensor(tensor, &object);
    if (!status.ok()) return status;
    *out_rank = object->shape().dimensions.size();
    return tensora::Status::Ok();
  });
}

ts_status_t ts_tensor_shape(ts_tensor_t tensor,
                            int64_t* out_dims,
                            size_t capacity,
                            size_t* out_rank) {
  return tensora::GuardedAbiCall("tensor_shape", [&] {
    if (out_rank == nullptr) {
      return tensora::InvalidArgument(
          "tensor_shape: output rank pointer is null");
    }

    std::shared_ptr<tensora::Tensor> object;
    tensora::Status status = tensora::LookupTensor(tensor, &object);
    if (!status.ok()) return status;

    const size_t rank = object->shape().dimensions.size();
    *out_rank = rank;
    if (capacity < rank) {
      return tensora::InvalidArgument(
          "tensor_shape: output dimension capacity is smaller than rank");
    }
    if (rank > 0 && out_dims == nullptr) {
      return tensora::InvalidArgument(
          "tensor_shape: output dimensions pointer is null");
    }

    if (rank > 0) {
      std::copy(object->shape().dimensions.begin(),
                object->shape().dimensions.end(), out_dims);
    }
    return tensora::Status::Ok();
  });
}

ts_status_t ts_tensor_dtype(ts_tensor_t tensor, uint32_t* out_dtype) {
  return tensora::GuardedAbiCall("tensor_dtype", [&] {
    if (out_dtype == nullptr) {
      return tensora::InvalidArgument(
          "tensor_dtype: output dtype pointer is null");
    }

    std::shared_ptr<tensora::Tensor> object;
    tensora::Status status = tensora::LookupTensor(tensor, &object);
    if (!status.ok()) return status;
    *out_dtype = static_cast<uint32_t>(object->dtype());
    return tensora::Status::Ok();
  });
}

ts_status_t ts_tensor_device(ts_tensor_t tensor, uint32_t* out_device) {
  return tensora::GuardedAbiCall("tensor_device", [&] {
    if (out_device == nullptr) {
      return tensora::InvalidArgument(
          "tensor_device: output device pointer is null");
    }

    std::shared_ptr<tensora::Tensor> object;
    tensora::Status status = tensora::LookupTensor(tensor, &object);
    if (!status.ok()) return status;
    *out_device = static_cast<uint32_t>(object->device());
    return tensora::Status::Ok();
  });
}

ts_status_t ts_tensor_device_index(ts_tensor_t tensor,
                                   int32_t* out_device_index) {
  return tensora::GuardedAbiCall("tensor_device_index", [&] {
    if (out_device_index == nullptr) {
      return tensora::InvalidArgument(
          "tensor_device_index: output device index pointer is null");
    }

    std::shared_ptr<tensora::Tensor> object;
    tensora::Status status = tensora::LookupTensor(tensor, &object);
    if (!status.ok()) return status;
    *out_device_index = object->device_index();
    return tensora::Status::Ok();
  });
}

ts_status_t ts_tensor_numel(ts_tensor_t tensor, uint64_t* out_numel) {
  return tensora::GuardedAbiCall("tensor_numel", [&] {
    if (out_numel == nullptr) {
      return tensora::InvalidArgument(
          "tensor_numel: output numel pointer is null");
    }

    std::shared_ptr<tensora::Tensor> object;
    tensora::Status status = tensora::LookupTensor(tensor, &object);
    if (!status.ok()) return status;
    *out_numel = object->numel();
    return tensora::Status::Ok();
  });
}

ts_status_t ts_tensor_to_device(ts_tensor_t tensor,
                                uint32_t device,
                                int32_t device_index,
                                ts_tensor_t* out_tensor) {
  return tensora::GuardedAbiCall("tensor_to_device", [&] {
    if (out_tensor == nullptr) {
      return tensora::InvalidArgument(
          "tensor_to_device: output handle pointer is null");
    }
    *out_tensor = 0;

    tensora::Device target = tensora::Device::kCpu;
    tensora::Status status = tensora::DeviceFromCode(device, &target);
    if (!status.ok()) return status;

    std::shared_ptr<tensora::Tensor> object;
    status = tensora::LookupTensor(tensor, &object);
    if (!status.ok()) return status;

    std::shared_ptr<tensora::Tensor> result;
    status = tensora::training::Transfer(
        *object, target, device_index, &result);
    if (!status.ok()) return status;
    return tensora::InsertTensor(std::move(result), out_tensor);
  });
}

ts_status_t ts_tensor_reshape(ts_tensor_t tensor,
                              const int64_t* dims,
                              size_t rank,
                              ts_tensor_t* out_tensor) {
  return tensora::GuardedAbiCall("tensor_reshape", [&] {
    if (out_tensor == nullptr) {
      return tensora::InvalidArgument(
          "tensor_reshape: output handle pointer is null");
    }
    *out_tensor = 0;

    tensora::ShapeInfo shape;
    tensora::Status status = tensora::ValidateShape(dims, rank, &shape);
    if (!status.ok()) return status;

    std::shared_ptr<tensora::Tensor> object;
    status = tensora::LookupTensor(tensor, &object);
    if (!status.ok()) return status;

    const tensora::Backend* backend = nullptr;
    status = tensora::BackendFor(*object, &backend);
    if (!status.ok()) return status;

    std::shared_ptr<tensora::Tensor> result;
    status = backend->Reshape(*object, shape, &result);
    if (!status.ok()) return status;
    return tensora::InsertTensor(std::move(result), out_tensor);
  });
}

ts_status_t ts_tensor_transpose2d(ts_tensor_t tensor,
                                  ts_tensor_t* out_tensor) {
  return tensora::GuardedAbiCall("tensor_transpose2d", [&] {
    if (out_tensor == nullptr) {
      return tensora::InvalidArgument(
          "tensor_transpose2d: output handle pointer is null");
    }
    *out_tensor = 0;

    std::shared_ptr<tensora::Tensor> object;
    tensora::Status status = tensora::LookupTensor(tensor, &object);
    if (!status.ok()) return status;

    const tensora::Backend* backend = nullptr;
    status = tensora::BackendFor(*object, &backend);
    if (!status.ok()) return status;

    std::shared_ptr<tensora::Tensor> result;
    status = backend->Transpose2D(*object, &result);
    if (!status.ok()) return status;
    return tensora::InsertTensor(std::move(result), out_tensor);
  });
}

ts_status_t ts_tensor_add(ts_tensor_t left,
                          ts_tensor_t right,
                          ts_tensor_t* out_tensor) {
  return tensora::GuardedAbiCall("tensor_add", [&] {
    return tensora::RunBinaryOperation(
        left, right, out_tensor, "tensor_add", &tensora::Backend::Add);
  });
}

ts_status_t ts_tensor_multiply(ts_tensor_t left,
                               ts_tensor_t right,
                               ts_tensor_t* out_tensor) {
  return tensora::GuardedAbiCall("tensor_multiply", [&] {
    return tensora::RunBinaryOperation(left, right, out_tensor,
                                       "tensor_multiply",
                                       &tensora::Backend::Multiply);
  });
}

ts_status_t ts_tensor_sum(ts_tensor_t tensor, ts_tensor_t* out_tensor) {
  return tensora::GuardedAbiCall("tensor_sum", [&] {
    if (out_tensor == nullptr) {
      return tensora::InvalidArgument(
          "tensor_sum: output handle pointer is null");
    }
    *out_tensor = 0;

    std::shared_ptr<tensora::Tensor> object;
    tensora::Status status = tensora::LookupTensor(tensor, &object);
    if (!status.ok()) return status;

    const tensora::Backend* backend = nullptr;
    status = tensora::BackendFor(*object, &backend);
    if (!status.ok()) return status;

    std::shared_ptr<tensora::Tensor> result;
    status = backend->Sum(*object, &result);
    if (!status.ok()) return status;
    return tensora::InsertTensor(std::move(result), out_tensor);
  });
}

ts_status_t ts_tensor_matmul(ts_tensor_t left,
                             ts_tensor_t right,
                             ts_tensor_t* out_tensor) {
  return tensora::GuardedAbiCall("tensor_matmul", [&] {
    return tensora::RunBinaryOperation(
        left, right, out_tensor, "tensor_matmul", &tensora::Backend::Matmul);
  });
}

ts_status_t ts_tensor_copy_to_host_f32(ts_tensor_t tensor,
                                       float* out_values,
                                       size_t capacity,
                                       size_t* out_written) {
  return tensora::GuardedAbiCall("tensor_copy_to_host_f32", [&] {
    if (out_written == nullptr) {
      return tensora::InvalidArgument(
          "tensor_copy_to_host_f32: output count pointer is null");
    }
    *out_written = 0;

    std::shared_ptr<tensora::Tensor> object;
    tensora::Status status = tensora::LookupTensor(tensor, &object);
    if (!status.ok()) return status;
    if (object->dtype() != tensora::DType::kFloat32) {
      return tensora::Unsupported(
          "tensor_copy_to_host_f32: tensor dtype is not float32");
    }
    if (capacity < object->numel()) {
      return tensora::InvalidArgument(
          "tensor_copy_to_host_f32: output capacity is smaller than tensor");
    }
    if (object->numel() > 0 && out_values == nullptr) {
      return tensora::InvalidArgument(
          "tensor_copy_to_host_f32: output values pointer is null");
    }

    status = object->storage()->CopyToHostF32(
        out_values, capacity, out_written);
    if (!status.ok()) return status;
    if (*out_written != object->numel()) {
      *out_written = 0;
      return tensora::InternalError(
          "tensor_copy_to_host_f32: storage returned an inconsistent element count");
    }
    return tensora::Status::Ok();
  });
}

ts_status_t ts_tensor_retain(ts_tensor_t tensor) {
  return tensora::GuardedAbiCall("tensor_retain", [&] {
    return tensora::HandleRegistry::Instance().Retain(
        tensor, tensora::HandleType::kTensor);
  });
}

ts_status_t ts_tensor_release(ts_tensor_t tensor) {
  return tensora::GuardedAbiCall("tensor_release", [&] {
    return tensora::HandleRegistry::Instance().Release(
        tensor, tensora::HandleType::kTensor);
  });
}

ts_status_t ts_runtime_live_tensor_count(uint64_t* out_count) {
  return tensora::GuardedAbiCall("runtime_live_tensor_count", [&] {
    if (out_count == nullptr) {
      return tensora::InvalidArgument(
          "runtime_live_tensor_count: output pointer is null");
    }
    *out_count = tensora::HandleRegistry::Instance().Count(
        tensora::HandleType::kTensor);
    return tensora::Status::Ok();
  });
}

ts_status_t ts_runtime_live_storage_bytes(uint64_t* out_bytes) {
  return tensora::GuardedAbiCall("runtime_live_storage_bytes", [&] {
    if (out_bytes == nullptr) {
      return tensora::InvalidArgument(
          "runtime_live_storage_bytes: output pointer is null");
    }
    const uint64_t core_bytes = tensora::CpuStorage::LiveBytes();
    const uint64_t training_bytes = tensora::training::LiveStorageBytes();
    if (training_bytes >
        std::numeric_limits<uint64_t>::max() - core_bytes) {
      return tensora::InternalError(
          "runtime_live_storage_bytes: counter overflow");
    }
    *out_bytes = core_bytes + training_bytes;
    return tensora::Status::Ok();
  });
}

}  // extern "C"