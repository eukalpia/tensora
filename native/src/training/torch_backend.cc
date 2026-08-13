#include "training/torch_backend.h"

#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "training/torch_storage.h"

namespace tensora::training {
namespace {

Status TorchFailure(const char* operation, const c10::Error& error) {
  return InternalError(std::string(operation) + ": " + error.what());
}

Status EnsureFloat32(const Tensor& tensor, const char* operation) {
  if (tensor.dtype() != DType::kFloat32) {
    return Unsupported(std::string(operation) + ": only float32 is supported");
  }
  return Status::Ok();
}

}  // namespace

Status TorchDevice(Device device, int32_t device_index, torch::Device* out) {
  if (out == nullptr) {
    return InvalidArgument("torch device: output pointer is null");
  }
  if (device == Device::kCpu) {
    if (device_index != 0) {
      return InvalidArgument("torch device: CPU device index must be zero");
    }
    *out = torch::Device(torch::kCPU);
    return Status::Ok();
  }
  if (device != Device::kCuda) {
    return Unsupported("torch device: unknown device kind");
  }
  if (device_index < 0) {
    return InvalidArgument("torch device: CUDA device index cannot be negative");
  }

  try {
    const auto count = torch::cuda::device_count();
    if (!torch::cuda::is_available() ||
        static_cast<int64_t>(device_index) >= count) {
      return Unsupported("torch device: requested CUDA device is not available");
    }
    *out = torch::Device(torch::kCUDA, device_index);
    return Status::Ok();
  } catch (const c10::Error& error) {
    return TorchFailure("torch device discovery", error);
  }
}

Status TensorToTorch(const Tensor& tensor, torch::Tensor* out) {
  if (out == nullptr) {
    return InvalidArgument("tensor to torch: output pointer is null");
  }
  Status status = EnsureFloat32(tensor, "tensor to torch");
  if (!status.ok()) return status;

  if (tensor.storage()->kind() == StorageKind::kTorch) {
    const auto storage =
        std::dynamic_pointer_cast<TorchStorage>(tensor.storage());
    if (!storage) {
      return InternalError("tensor to torch: invalid Torch storage type");
    }
    *out = storage->tensor();
    return Status::Ok();
  }

  std::vector<float> values(static_cast<size_t>(tensor.numel()));
  size_t written = 0;
  status = tensor.storage()->CopyToHostF32(
      values.data(), values.size(), &written);
  if (!status.ok()) return status;
  if (written != values.size()) {
    return InternalError(
        "tensor to torch: storage returned an inconsistent element count");
  }

  try {
    const auto options =
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    *out = torch::from_blob(values.data(), tensor.shape().dimensions, options)
               .clone();
    return Status::Ok();
  } catch (const c10::Error& error) {
    return TorchFailure("tensor to torch", error);
  }
}

Status WrapTorchTensor(torch::Tensor value, std::shared_ptr<Tensor>* out) {
  if (out == nullptr) {
    return InvalidArgument("wrap torch tensor: output pointer is null");
  }
  *out = nullptr;
  if (!value.defined()) {
    return InvalidArgument("wrap torch tensor: tensor is undefined");
  }
  if (value.scalar_type() != torch::kFloat32) {
    return Unsupported("wrap torch tensor: only float32 is supported");
  }

  ShapeInfo shape;
  const std::vector<int64_t> dimensions(value.sizes().begin(), value.sizes().end());
  const int64_t* dims = dimensions.empty() ? nullptr : dimensions.data();
  Status status = ValidateShape(dims, dimensions.size(), &shape);
  if (!status.ok()) return status;

  Device device = Device::kCpu;
  int32_t device_index = 0;
  if (value.device().is_cuda()) {
    device = Device::kCuda;
    device_index = static_cast<int32_t>(value.device().index());
  } else if (!value.device().is_cpu()) {
    return Unsupported("wrap torch tensor: unsupported Torch device");
  }

  std::shared_ptr<TorchStorage> storage;
  status = TorchStorage::FromTensor(std::move(value), &storage);
  if (!status.ok()) return status;

  try {
    *out = std::make_shared<Tensor>(
        std::move(shape), std::move(storage), DType::kFloat32, device,
        device_index);
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return OutOfMemory("wrap torch tensor: tensor object allocation failed");
  }
}

Status TorchBackend::FromData(const ShapeInfo& shape,
                              const float* data,
                              std::shared_ptr<Tensor>* out) const {
  if (shape.numel > 0 && data == nullptr) {
    return InvalidArgument("torch from data: input pointer is null");
  }
  try {
    const auto options =
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor value =
        torch::from_blob(const_cast<float*>(data), shape.dimensions, options)
            .clone();
    return WrapTorchTensor(std::move(value), out);
  } catch (const c10::Error& error) {
    return TorchFailure("torch from data", error);
  }
}

Status TorchBackend::Full(const ShapeInfo& shape,
                          float value,
                          std::shared_ptr<Tensor>* out) const {
  try {
    const auto options =
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    return WrapTorchTensor(torch::full(shape.dimensions, value, options), out);
  } catch (const c10::Error& error) {
    return TorchFailure("torch full", error);
  }
}

Status TorchBackend::Reshape(const Tensor& tensor,
                             const ShapeInfo& shape,
                             std::shared_ptr<Tensor>* out) const {
  if (tensor.numel() != shape.numel) {
    return InvalidShape("reshape: target shape must preserve element count");
  }
  torch::Tensor value;
  Status status = TensorToTorch(tensor, &value);
  if (!status.ok()) return status;
  try {
    return WrapTorchTensor(value.reshape(shape.dimensions).clone(), out);
  } catch (const c10::Error& error) {
    return TorchFailure("torch reshape", error);
  }
}

Status TorchBackend::Transpose2D(const Tensor& tensor,
                                 std::shared_ptr<Tensor>* out) const {
  if (tensor.shape().dimensions.size() != 2) {
    return InvalidShape("transpose: requires a rank-2 tensor");
  }
  torch::Tensor value;
  Status status = TensorToTorch(tensor, &value);
  if (!status.ok()) return status;
  try {
    return WrapTorchTensor(value.transpose(0, 1).contiguous(), out);
  } catch (const c10::Error& error) {
    return TorchFailure("torch transpose", error);
  }
}

Status TorchBackend::Add(const Tensor& left,
                         const Tensor& right,
                         std::shared_ptr<Tensor>* out) const {
  if (!SameShape(left.shape(), right.shape())) {
    return InvalidShape("add: input shapes must match");
  }
  torch::Tensor a;
  torch::Tensor b;
  Status status = TensorToTorch(left, &a);
  if (!status.ok()) return status;
  status = TensorToTorch(right, &b);
  if (!status.ok()) return status;
  try {
    return WrapTorchTensor(a + b, out);
  } catch (const c10::Error& error) {
    return TorchFailure("torch add", error);
  }
}

Status TorchBackend::Multiply(const Tensor& left,
                              const Tensor& right,
                              std::shared_ptr<Tensor>* out) const {
  if (!SameShape(left.shape(), right.shape())) {
    return InvalidShape("multiply: input shapes must match");
  }
  torch::Tensor a;
  torch::Tensor b;
  Status status = TensorToTorch(left, &a);
  if (!status.ok()) return status;
  status = TensorToTorch(right, &b);
  if (!status.ok()) return status;
  try {
    return WrapTorchTensor(a * b, out);
  } catch (const c10::Error& error) {
    return TorchFailure("torch multiply", error);
  }
}

Status TorchBackend::Sum(const Tensor& tensor,
                         std::shared_ptr<Tensor>* out) const {
  torch::Tensor value;
  Status status = TensorToTorch(tensor, &value);
  if (!status.ok()) return status;
  try {
    return WrapTorchTensor(value.sum(), out);
  } catch (const c10::Error& error) {
    return TorchFailure("torch sum", error);
  }
}

Status TorchBackend::Matmul(const Tensor& left,
                            const Tensor& right,
                            std::shared_ptr<Tensor>* out) const {
  if (left.shape().dimensions.size() != 2 ||
      right.shape().dimensions.size() != 2) {
    return InvalidShape("matmul: requires rank-2 tensors");
  }
  if (left.shape().dimensions[1] != right.shape().dimensions[0]) {
    return InvalidShape("matmul: inner dimensions must match");
  }
  torch::Tensor a;
  torch::Tensor b;
  Status status = TensorToTorch(left, &a);
  if (!status.ok()) return status;
  status = TensorToTorch(right, &b);
  if (!status.ok()) return status;
  try {
    return WrapTorchTensor(torch::matmul(a, b), out);
  } catch (const c10::Error& error) {
    return TorchFailure("torch matmul", error);
  }
}

}  // namespace tensora::training
