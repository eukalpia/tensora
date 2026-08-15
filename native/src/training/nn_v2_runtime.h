#ifndef TENSORA_TRAINING_NN_V2_RUNTIME_H_
#define TENSORA_TRAINING_NN_V2_RUNTIME_H_

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "autograd/autograd.h"
#include "core/allocation_guard.h"
#include "core/status.h"
#include "memory/tensor_storage.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#if defined(TENSORA_WITH_TORCH)
#include <torch/torch.h>

#include "training/torch_backend.h"
#endif

namespace tensora::training::nn_v2 {
namespace internal {

inline Status ValidateOutput(std::shared_ptr<Tensor>* out,
                             const char* operation) {
  if (out == nullptr) {
    return InvalidArgument(std::string(operation) +
                           ": output tensor pointer is null");
  }
  *out = nullptr;
  return Status::Ok();
}

inline Status ValidateCpuFloat32(const Tensor& tensor,
                                 const char* operation) {
  return autograd::EnsureCpuFloat32(tensor, operation);
}

inline Status CpuUnary(const Tensor& tensor,
                       autograd::Operation operation,
                       const char* operation_name,
                       float (*forward)(float),
                       std::shared_ptr<Tensor>* out) {
  Status status = ValidateOutput(out, operation_name);
  if (!status.ok()) return status;
  status = ValidateCpuFloat32(tensor, operation_name);
  if (!status.ok()) return status;

  std::vector<float> input;
  status = autograd::ReadLogicalCpuValues(tensor, operation_name, &input);
  if (!status.ok()) return status;
  std::vector<float> values;
  status = AllocationGuard(operation_name, [&]() -> Status {
    values.resize(input.size());
    return Status::Ok();
  });
  if (!status.ok()) return status;
  for (size_t index = 0; index < input.size(); ++index) {
    values[index] = forward(input[index]);
  }
  status = autograd::MakeCpuTensor(tensor.shape(), values, out);
  if (!status.ok()) return status;
  return autograd::RecordUnary(operation, tensor, *out);
}

inline float GeluForward(float x) {
  constexpr float kInvSqrt2 = 0.7071067811865475244f;
  return 0.5f * x * (1.0f + std::erf(x * kInvSqrt2));
}

inline float SiluForward(float x) {
  const float sigmoid = 1.0f / (1.0f + std::exp(-x));
  return x * sigmoid;
}

inline Status CpuSwiGlu(const Tensor& tensor, std::shared_ptr<Tensor>* out) {
  Status status = ValidateOutput(out, "tensor_swiglu");
  if (!status.ok()) return status;
  status = ValidateCpuFloat32(tensor, "tensor_swiglu");
  if (!status.ok()) return status;
  if (tensor.shape().dimensions.empty()) {
    return InvalidShape("tensor_swiglu: input must have rank at least one");
  }
  const int64_t width = tensor.shape().dimensions.back();
  if (width <= 0 || (width % 2) != 0) {
    return InvalidShape(
        "tensor_swiglu: final dimension must be positive and even");
  }
  const int64_t half = width / 2;
  const int64_t rows = tensor.numel() / width;

  ShapeInfo output_shape;
  std::vector<int64_t> dimensions = tensor.shape().dimensions;
  dimensions.back() = half;
  status = ValidateShape(dimensions.data(), dimensions.size(), &output_shape);
  if (!status.ok()) return status;

  std::vector<float> input;
  status = autograd::ReadLogicalCpuValues(tensor, "tensor_swiglu", &input);
  if (!status.ok()) return status;
  std::vector<float> values;
  status = AllocationGuard("tensor_swiglu", [&]() -> Status {
    values.assign(static_cast<size_t>(output_shape.numel), 0.0f);
    return Status::Ok();
  });
  if (!status.ok()) return status;

  for (int64_t row = 0; row < rows; ++row) {
    const size_t input_base = static_cast<size_t>(row * width);
    const size_t output_base = static_cast<size_t>(row * half);
    for (int64_t col = 0; col < half; ++col) {
      const float a = input[input_base + static_cast<size_t>(col)];
      const float b =
          input[input_base + static_cast<size_t>(half + col)];
      values[output_base + static_cast<size_t>(col)] = SiluForward(a) * b;
    }
  }

  status = autograd::MakeCpuTensor(output_shape, values, out);
  if (!status.ok()) return status;
  return autograd::RecordUnary(autograd::Operation::kSwiGlu, tensor, *out);
}

#if defined(TENSORA_WITH_TORCH)
inline Status TorchUnary(const Tensor& tensor,
                         const char* operation_name,
                         const std::function<torch::Tensor(const torch::Tensor&)>&
                             forward,
                         std::shared_ptr<Tensor>* out) {
  Status status = ValidateOutput(out, operation_name);
  if (!status.ok()) return status;
  torch::Tensor value;
  status = TensorToTorch(tensor, &value);
  if (!status.ok()) return status;
  return training::internal::GuardAllocation(operation_name, [&]() {
    return training::internal::GuardTorch(operation_name, [&]() {
      return WrapTorchTensor(forward(value), out);
    });
  });
}

inline Status TorchSwiGlu(const Tensor& tensor,
                          std::shared_ptr<Tensor>* out) {
  Status status = ValidateOutput(out, "tensor_swiglu");
  if (!status.ok()) return status;
  torch::Tensor value;
  status = TensorToTorch(tensor, &value);
  if (!status.ok()) return status;
  if (value.dim() == 0) {
    return InvalidShape("tensor_swiglu: input must have rank at least one");
  }
  const int64_t width = value.size(-1);
  if (width <= 0 || (width % 2) != 0) {
    return InvalidShape(
        "tensor_swiglu: final dimension must be positive and even");
  }
  const int64_t half = width / 2;
  return training::internal::GuardAllocation("tensor_swiglu", [&]() {
    return training::internal::GuardTorch("tensor_swiglu", [&]() {
      const torch::Tensor a = value.slice(-1, 0, half);
      const torch::Tensor b = value.slice(-1, half, width);
      return WrapTorchTensor(torch::silu(a) * b, out);
    });
  });
}
#endif

}  // namespace internal

inline Status Gelu(const Tensor& tensor, std::shared_ptr<Tensor>* out) {
#if defined(TENSORA_WITH_TORCH)
  if (tensor.storage()->kind() == StorageKind::kTorch) {
    return internal::TorchUnary(
        tensor, "tensor_gelu",
        [](const torch::Tensor& value) { return torch::gelu(value, "none"); },
        out);
  }
#endif
  return internal::CpuUnary(tensor, autograd::Operation::kGelu, "tensor_gelu",
                            internal::GeluForward, out);
}

inline Status Silu(const Tensor& tensor, std::shared_ptr<Tensor>* out) {
#if defined(TENSORA_WITH_TORCH)
  if (tensor.storage()->kind() == StorageKind::kTorch) {
    return internal::TorchUnary(
        tensor, "tensor_silu",
        [](const torch::Tensor& value) { return torch::silu(value); }, out);
  }
#endif
  return internal::CpuUnary(tensor, autograd::Operation::kSilu, "tensor_silu",
                            internal::SiluForward, out);
}

inline Status SwiGlu(const Tensor& tensor, std::shared_ptr<Tensor>* out) {
#if defined(TENSORA_WITH_TORCH)
  if (tensor.storage()->kind() == StorageKind::kTorch) {
    return internal::TorchSwiGlu(tensor, out);
  }
#endif
  return internal::CpuSwiGlu(tensor, out);
}

}  // namespace tensora::training::nn_v2

#endif  // TENSORA_TRAINING_NN_V2_RUNTIME_H_
