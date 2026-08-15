#include "training/torch_backend.h"

#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <ATen/DeviceAccelerator.h>

#include "training/torch_storage.h"
#include "training/training_bridge.h"

namespace tensora::training {
namespace {

Status EnsureFloat32(const Tensor& tensor, const char* operation) {
  if (tensor.dtype() != DType::kFloat32) {
    return Unsupported(std::string(operation) + ": only float32 is supported");
  }
  return Status::Ok();
}

}  // namespace

namespace internal {

Status TorchFailure(const char* operation, const c10::Error& error) {
  return InternalError(std::string(operation) + ": " + error.what());
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

Status DeviceCountFromSnapshot(
    Device device,
    std::optional<c10::DeviceType> accelerator,
    c10::DeviceIndex accelerator_count,
    uint32_t* out_count) {
  if (out_count == nullptr) {
    return InvalidArgument("device_count: output pointer is null");
  }
  *out_count = 0;

  if (device == Device::kCpu) {
    *out_count = 1;
    return Status::Ok();
  }
  if (!accelerator.has_value() ||
      !MatchesAccelerator(device, *accelerator)) {
    return Status::Ok();
  }
  if (accelerator_count < 0) {
    return InternalError("device_count: accelerator returned a negative count");
  }

  // c10::DeviceIndex is narrower than the ABI's uint32_t count field, so a
  // non-negative value is representable without a second unreachable range
  // branch.
  *out_count = static_cast<uint32_t>(accelerator_count);
  return Status::Ok();
}

Status TorchDeviceFromSnapshot(
    Device device,
    int32_t device_index,
    std::optional<c10::DeviceType> accelerator,
    c10::DeviceIndex accelerator_count,
    torch::Device* out) {
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
  Status status = DeviceCountFromSnapshot(
      device, accelerator, accelerator_count, &count);
  if (!status.ok()) return status;
  if (static_cast<uint32_t>(device_index) >= count) {
    return Unsupported("torch device: requested accelerator is not available");
  }

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
      if (!accelerator.has_value() ||
          *accelerator != c10::DeviceType::HIP) {
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

Status MapTorchDevice(c10::DeviceType torch_type,
                      c10::DeviceIndex torch_index,
                      std::optional<c10::DeviceType> accelerator,
                      Device* out_device,
                      int32_t* out_index) {
  if (out_device == nullptr || out_index == nullptr) {
    return InvalidArgument("wrap torch tensor: device output pointer is null");
  }
  *out_device = Device::kCpu;
  *out_index = 0;

  if (torch_type == c10::DeviceType::CPU) {
    return Status::Ok();
  }
  if (torch_type == c10::DeviceType::MPS) {
    *out_device = Device::kMps;
    return Status::Ok();
  }
  if (torch_type == c10::DeviceType::XPU ||
      torch_type == c10::DeviceType::HIP ||
      torch_type == c10::DeviceType::CUDA) {
    if (torch_index < 0) {
      return InternalError("wrap torch tensor: accelerator has no device index");
    }
    *out_index = static_cast<int32_t>(torch_index);
    if (torch_type == c10::DeviceType::XPU) {
      *out_device = Device::kXpu;
    } else if (torch_type == c10::DeviceType::HIP) {
      *out_device = Device::kHip;
    } else {
      *out_device = accelerator.has_value() &&
                            *accelerator == c10::DeviceType::HIP
                        ? Device::kHip
                        : Device::kCuda;
    }
    return Status::Ok();
  }
  return Unsupported("wrap torch tensor: unsupported Torch device");
}

}  // namespace internal

Status DeviceCount(Device device, uint32_t* out_count) {
  if (out_count == nullptr || device == Device::kCpu) {
    return internal::DeviceCountFromSnapshot(
        device, std::nullopt, 0, out_count);
  }

  return internal::GuardTorch("device_count", [&]() {
    const std::optional<c10::DeviceType> accelerator =
        at::accelerator::getAccelerator(false);
    c10::DeviceIndex count = 0;
    if (accelerator.has_value() &&
        internal::MatchesAccelerator(device, *accelerator)) {
      count = at::accelerator::deviceCount();
    }
    return internal::DeviceCountFromSnapshot(
        device, accelerator, count, out_count);
  });
}

Status TorchDevice(Device device, int32_t device_index, torch::Device* out) {
  if (out == nullptr || device == Device::kCpu || device_index < 0 ||
      (device == Device::kMps && device_index != 0) ||
      static_cast<uint64_t>(device_index) >
          static_cast<uint64_t>(std::numeric_limits<c10::DeviceIndex>::max())) {
    return internal::TorchDeviceFromSnapshot(
        device, device_index, std::nullopt, 0, out);
  }

  return internal::GuardTorch("torch device", [&]() {
    const std::optional<c10::DeviceType> accelerator =
        at::accelerator::getAccelerator(false);
    c10::DeviceIndex count = 0;
    if (accelerator.has_value() &&
        internal::MatchesAccelerator(device, *accelerator)) {
      count = at::accelerator::deviceCount();
    }
    return internal::TorchDeviceFromSnapshot(
        device, device_index, accelerator, count, out);
  });
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

  return internal::GuardTorch("tensor to torch", [&]() {
    const auto options =
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    *out = torch::from_blob(values.data(), tensor.shape().dimensions, options)
               .clone();
    return Status::Ok();
  });
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

  std::optional<c10::DeviceType> accelerator;
  const c10::DeviceType type = value.device().type();
  if (type == c10::DeviceType::CUDA) {
    status = internal::GuardTorch(
        "wrap torch tensor accelerator discovery", [&]() {
          accelerator = at::accelerator::getAccelerator(false);
          return Status::Ok();
        });
    if (!status.ok()) return status;
  }

  Device device = Device::kCpu;
  int32_t device_index = 0;
  status = internal::MapTorchDevice(
      type, value.device().index(), accelerator, &device, &device_index);
  if (!status.ok()) return status;

  std::shared_ptr<TorchStorage> storage;
  status = TorchStorage::FromTensor(std::move(value), &storage);
  if (!status.ok()) return status;

  return internal::GuardAllocation("wrap torch tensor", [&]() {
    *out = std::make_shared<Tensor>(
        std::move(shape), std::move(storage), DType::kFloat32, device,
        device_index);
    return Status::Ok();
  });
}

Status TorchBackend::FromData(const ShapeInfo& shape,
                              const float* data,
                              std::shared_ptr<Tensor>* out) const {
  if (shape.numel > 0 && data == nullptr) {
    return InvalidArgument("torch from data: input pointer is null");
  }
  return internal::GuardTorch("torch from data", [&]() {
    const auto options =
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor value =
        torch::from_blob(const_cast<float*>(data), shape.dimensions, options)
            .clone();
    return WrapTorchTensor(std::move(value), out);
  });
}

Status TorchBackend::Full(const ShapeInfo& shape,
                          float value,
                          std::shared_ptr<Tensor>* out) const {
  return internal::GuardTorch("torch full", [&]() {
    const auto options =
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    return WrapTorchTensor(torch::full(shape.dimensions, value, options), out);
  });
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
  return internal::GuardTorch("torch reshape", [&]() {
    return WrapTorchTensor(value.reshape(shape.dimensions).clone(), out);
  });
}

Status TorchBackend::Transpose2D(const Tensor& tensor,
                                 std::shared_ptr<Tensor>* out) const {
  if (tensor.shape().dimensions.size() != 2) {
    return InvalidShape("transpose: requires a rank-2 tensor");
  }
  torch::Tensor value;
  Status status = TensorToTorch(tensor, &value);
  if (!status.ok()) return status;
  return internal::GuardTorch("torch transpose", [&]() {
    return WrapTorchTensor(value.transpose(0, 1).contiguous(), out);
  });
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
  return internal::GuardTorch("torch add", [&]() {
    return WrapTorchTensor(a + b, out);
  });
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
  return internal::GuardTorch("torch multiply", [&]() {
    return WrapTorchTensor(a * b, out);
  });
}

Status TorchBackend::Sum(const Tensor& tensor,
                         std::shared_ptr<Tensor>* out) const {
  torch::Tensor value;
  Status status = TensorToTorch(tensor, &value);
  if (!status.ok()) return status;
  return internal::GuardTorch("torch sum", [&]() {
    return WrapTorchTensor(value.sum(), out);
  });
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
  return internal::GuardTorch("torch matmul", [&]() {
    return WrapTorchTensor(torch::matmul(a, b), out);
  });
}

}  // namespace tensora::training
