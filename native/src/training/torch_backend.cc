#include "training/torch_backend.h"

#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <ATen/DeviceAccelerator.h>

#include "training/torch_storage.h"
#include "training/training_bridge.h"

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

bool MatchesAccelerator(Device device, c10::DeviceType accelerator) {
  switch (device) {
    case Device::kCuda:
      return accelerator == c10::DeviceType::CUDA;
    case Device::kMps:
      return accelerator == c10::DeviceType::MPS;
    case Device::kXpu:
      return accelerator == c10::DeviceType::XPU;
    case Device::kHip:
      return accelerator == c10::DeviceType::HIP;
    case Device::kCpu:
      return false;
  }
  return false;
}

Status AcceleratorType(Device device, c10::DeviceType* out) {
  if (out == nullptr) {
    return InvalidArgument("torch device: accelerator output pointer is null");
  }

  try {
    const std::optional<c10::DeviceType> accelerator =
        at::accelerator::getAccelerator(false);
    if (!accelerator.has_value() ||
        !MatchesAccelerator(device, *accelerator)) {
      return Unsupported("torch device: requested accelerator is not available");
    }
    *out = *accelerator;
    return Status::Ok();
  } catch (const c10::Error& error) {
    return TorchFailure("torch accelerator discovery", error);
  }
}

Status DeviceIndex(const torch::Tensor& value, int32_t* out_index) {
  if (out_index == nullptr) {
    return InvalidArgument("wrap torch tensor: device index output is null");
  }
  const c10::DeviceIndex index = value.device().index();
  if (index < 0) {
    return InternalError("wrap torch tensor: accelerator has no device index");
  }
  *out_index = static_cast<int32_t>(index);
  return Status::Ok();
}

}  // namespace

Status DeviceCount(Device device, uint32_t* out_count) {
  if (out_count == nullptr) {
    return InvalidArgument("device_count: output pointer is null");
  }
  *out_count = 0;

  if (device == Device::kCpu) {
    *out_count = 1;
    return Status::Ok();
  }

  try {
    const std::optional<c10::DeviceType> accelerator =
        at::accelerator::getAccelerator(false);
    if (!accelerator.has_value() ||
        !MatchesAccelerator(device, *accelerator)) {
      return Status::Ok();
    }

    const c10::DeviceIndex count = at::accelerator::deviceCount();
    if (count < 0) {
      return InternalError("device_count: accelerator returned a negative count");
    }
    if (static_cast<uint64_t>(count) >
        std::numeric_limits<uint32_t>::max()) {
      return InternalError("device_count: accelerator count exceeds ABI range");
    }
    *out_count = static_cast<uint32_t>(count);
    return Status::Ok();
  } catch (const c10::Error& error) {
    return TorchFailure("device_count", error);
  }
}

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
  if (device_index < 0) {
    return InvalidArgument("torch device: accelerator index cannot be negative");
  }
  if (device == Device::kMps && device_index != 0) {
    return InvalidArgument("torch device: MPS device index must be zero");
  }
  if (static_cast<uint64_t>(device_index) >
      static_cast<uint64_t>(std::numeric_limits<c10::DeviceIndex>::max())) {
    return InvalidArgument("torch device: accelerator index exceeds LibTorch range");
  }

  uint32_t count = 0;
  Status status = DeviceCount(device, &count);
  if (!status.ok()) return status;
  if (static_cast<uint32_t>(device_index) >= count) {
    return Unsupported("torch device: requested accelerator is not available");
  }

  c10::DeviceType accelerator = c10::DeviceType::CPU;
  status = AcceleratorType(device, &accelerator);
  if (!status.ok()) return status;
  const c10::DeviceIndex torch_device_index =
      static_cast<c10::DeviceIndex>(device_index);

  switch (device) {
    case Device::kCuda:
      *out = torch::Device(c10::DeviceType::CUDA, torch_device_index);
      return Status::Ok();
    case Device::kMps:
      *out = torch::Device(c10::DeviceType::MPS);
      return Status::Ok();
    case Device::kXpu:
      *out = torch::Device(c10::DeviceType::XPU, torch_device_index);
      return Status::Ok();
    case Device::kHip:
      if (accelerator != c10::DeviceType::HIP) {
        return Unsupported("torch device: HIP accelerator is not available");
      }
      // ROCm PyTorch intentionally exposes tensors through CUDA device
      // semantics even though accelerator discovery identifies the HIP build.
      *out = torch::Device(c10::DeviceType::CUDA, torch_device_index);
      return Status::Ok();
    case Device::kCpu:
      break;
  }
  return Unsupported("torch device: unknown device kind");
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
  const c10::DeviceType type = value.device().type();
  if (value.device().is_cpu()) {
    device = Device::kCpu;
  } else if (type == c10::DeviceType::MPS) {
    device = Device::kMps;
  } else if (type == c10::DeviceType::XPU) {
    device = Device::kXpu;
    status = DeviceIndex(value, &device_index);
    if (!status.ok()) return status;
  } else if (type == c10::DeviceType::HIP) {
    device = Device::kHip;
    status = DeviceIndex(value, &device_index);
    if (!status.ok()) return status;
  } else if (value.device().is_cuda()) {
    try {
      const std::optional<c10::DeviceType> accelerator =
          at::accelerator::getAccelerator(false);
      device = accelerator.has_value() &&
               *accelerator == c10::DeviceType::HIP
                   ? Device::kHip
                   : Device::kCuda;
    } catch (const c10::Error& error) {
      return TorchFailure("wrap torch tensor accelerator discovery", error);
    }
    status = DeviceIndex(value, &device_index);
    if (!status.ok()) return status;
  } else {
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
