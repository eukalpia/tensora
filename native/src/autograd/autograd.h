#ifndef TENSORA_AUTOGRAD_AUTOGRAD_H_
#define TENSORA_AUTOGRAD_AUTOGRAD_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/allocation_guard.h"
#include "core/status.h"
#include "memory/cpu_storage.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

namespace tensora::autograd {

enum class Operation : uint8_t {
  kIdentity,
  kReshape,
  kTranspose2D,
  kAdd,
  kMultiply,
  kSum,
  kMatmul,
  kRelu,
  kSigmoid,
  kTanh,
  kGelu,
  kSilu,
  kSwiGlu,
  kMse,
  kCrossEntropy,
  kBiasAdd,
};

struct GradNode;

}  // namespace tensora::autograd

namespace tensora {

struct AutogradMeta {
  bool requires_grad = false;
  bool is_leaf = true;
  std::shared_ptr<Tensor> gradient;
  std::shared_ptr<autograd::GradNode> grad_fn;
  mutable std::mutex mutex;
};

}  // namespace tensora

namespace tensora::autograd {

struct GradientContribution {
  std::shared_ptr<Tensor> parent;
  std::shared_ptr<Tensor> gradient;
};

struct GradNode {
  Operation operation = Operation::kIdentity;
  std::vector<std::shared_ptr<Tensor>> parents;
  std::vector<uint64_t> parent_versions;
};

inline Status EnsureCpuFloat32(const Tensor& tensor, const char* operation) {
  if (tensor.device() != Device::kCpu || tensor.device_index() != 0) {
    return Unsupported(std::string(operation) + ": CPU autograd requires cpu:0");
  }
  if (tensor.dtype() != DType::kFloat32) {
    return Unsupported(std::string(operation) +
                       ": CPU autograd currently supports float32 only");
  }
  return Status::Ok();
}

// Direct storage access is intentionally limited to canonical tensors. This is
// used by parameter/checkpoint mutation code; graph evaluation uses logical
// reads so that non-contiguous views preserve tensor order.
inline Status CpuValues(const Tensor& tensor,
                        const char* operation,
                        const std::vector<float>** out) {
  if (out == nullptr) {
    return InvalidArgument(std::string(operation) +
                           ": output values pointer is null");
  }
  *out = nullptr;
  Status status = EnsureCpuFloat32(tensor, operation);
  if (!status.ok()) return status;
  if (!tensor.is_contiguous() || tensor.storage_offset() != 0) {
    return InvalidArgument(std::string(operation) +
                           ": direct storage access requires a contiguous tensor");
  }
  auto storage = std::dynamic_pointer_cast<CpuStorage>(tensor.storage());
  if (!storage) {
    return Unsupported(std::string(operation) +
                       ": tensor is not backed by CPU storage");
  }
  if (storage->numel() != tensor.numel()) {
    return InvalidArgument(std::string(operation) +
                           ": direct access requires full logical storage extent");
  }
  *out = &storage->values();
  return Status::Ok();
}

inline Status MutableCpuValues(Tensor& tensor,
                               const char* operation,
                               std::vector<float>** out) {
  if (out == nullptr) {
    return InvalidArgument(std::string(operation) +
                           ": output values pointer is null");
  }
  *out = nullptr;
  Status status = EnsureCpuFloat32(tensor, operation);
  if (!status.ok()) return status;
  if (!tensor.is_contiguous() || tensor.storage_offset() != 0) {
    return InvalidArgument(std::string(operation) +
                           ": mutable storage access requires a contiguous tensor");
  }
  auto storage = std::dynamic_pointer_cast<CpuStorage>(tensor.storage());
  if (!storage) {
    return Unsupported(std::string(operation) +
                       ": tensor is not backed by CPU storage");
  }
  if (storage->numel() != tensor.numel()) {
    return InvalidArgument(std::string(operation) +
                           ": mutable access to aliased sub-storage is not supported");
  }
  *out = &storage->mutable_values();
  return Status::Ok();
}

inline Status ReadLogicalCpuValues(const Tensor& tensor,
                                   const char* operation,
                                   std::vector<float>* out) {
  if (out == nullptr) {
    return InvalidArgument(std::string(operation) +
                           ": logical values output pointer is null");
  }
  Status status = EnsureCpuFloat32(tensor, operation);
  if (!status.ok()) return status;
  status = AllocationGuard(operation, [&]() -> Status {
    out->assign(static_cast<size_t>(tensor.numel()), 0.0f);
    return Status::Ok();
  });
  if (!status.ok()) return status;
  size_t written = 0;
  status = tensor.CopyToHostF32(out->data(), out->size(), &written);
  if (!status.ok()) return status;
  return Status::Ok();
}

inline Status MakeCpuTensor(const ShapeInfo& shape,
                            const std::vector<float>& values,
                            std::shared_ptr<Tensor>* out) {
  if (out == nullptr) {
    return InvalidArgument("autograd: output tensor pointer is null");
  }
  *out = nullptr;
  if (values.size() != static_cast<size_t>(shape.numel)) {
    return InternalError("autograd: gradient value count does not match shape");
  }

  ShapeInfo materialized_shape;
  const int64_t* dimensions =
      shape.dimensions.empty() ? nullptr : shape.dimensions.data();
  Status status =
      ValidateShape(dimensions, shape.dimensions.size(), &materialized_shape);
  if (!status.ok() || materialized_shape.numel != shape.numel) {
    return InternalError("autograd: could not canonicalize materialized gradient shape");
  }

  std::shared_ptr<CpuStorage> storage;
  status = CpuStorage::FromData(values.data(), shape.numel, &storage);
  if (!status.ok()) return status;
  return AllocationGuard("autograd tensor", [&]() -> Status {
    *out = std::make_shared<Tensor>(std::move(materialized_shape),
                                    std::move(storage));
    return Status::Ok();
  });
}

inline Status MakeCpuFull(const ShapeInfo& shape,
                          float value,
                          std::shared_ptr<Tensor>* out) {
  if (out == nullptr) {
    return InvalidArgument("autograd: output tensor pointer is null");
  }
  *out = nullptr;
  std::shared_ptr<CpuStorage> storage;
  Status status = CpuStorage::Filled(shape.numel, value, &storage);
  if (!status.ok()) return status;
  return AllocationGuard("autograd tensor", [&]() -> Status {
    *out = std::make_shared<Tensor>(shape, std::move(storage));
    return Status::Ok();
  });
}

inline Status CloneDetached(const Tensor& source,
                            std::shared_ptr<Tensor>* out) {
  std::vector<float> values;
  Status status = ReadLogicalCpuValues(source, "autograd_clone", &values);
  if (!status.ok()) return status;
  return MakeCpuTensor(source.shape(), values, out);
}

inline std::shared_ptr<AutogradMeta> Meta(const Tensor& tensor) {
  return tensor.autograd_meta();
}

inline bool RequiresGrad(const Tensor& tensor) {
  const auto meta = Meta(tensor);
  if (!meta) return false;
  std::lock_guard<std::mutex> lock(meta->mutex);
  return meta->requires_grad;
}

inline uint64_t Version(const Tensor& tensor) { return tensor.version(); }

inline void IncrementVersion(Tensor& tensor) { tensor.increment_version(); }

inline Status CloneAsLeaf(const Tensor& source,
                          bool requires_grad,
                          std::shared_ptr<Tensor>* out) {
  Status status = CloneDetached(source, out);
  if (!status.ok()) return status;
  auto meta = std::make_shared<AutogradMeta>();
  meta->requires_grad = requires_grad;
  meta->is_leaf = true;
  (*out)->set_autograd_meta(std::move(meta));
  return Status::Ok();
}

inline Status Share(const Tensor& tensor, std::shared_ptr<Tensor>* out) {
  if (out == nullptr) {
    return InvalidArgument("autograd: shared tensor output pointer is null");
  }
  *out = nullptr;
  try {
    *out = std::const_pointer_cast<Tensor>(tensor.shared_from_this());
    return Status::Ok();
  } catch (const std::bad_weak_ptr&) {
    return InternalError("autograd: differentiable tensor is not managed by shared ownership");
  }
}

inline Status AttachNode(std::shared_ptr<Tensor> result,
                         Operation operation,
                         std::vector<std::shared_ptr<Tensor>> parents) {
  if (!result) return InvalidArgument("autograd: result tensor is null");

  bool needs_grad = false;
  for (const auto& parent : parents) {
    if (parent && RequiresGrad(*parent)) {
      needs_grad = true;
      break;
    }
  }
  if (!needs_grad) return Status::Ok();

  auto node = std::make_shared<GradNode>();
  node->operation = operation;
  node->parents = std::move(parents);
  node->parent_versions.reserve(node->parents.size());
  for (const auto& parent : node->parents) {
    node->parent_versions.push_back(parent ? Version(*parent) : 0);
  }

  auto meta = std::make_shared<AutogradMeta>();
  meta->requires_grad = true;
  meta->is_leaf = false;
  meta->grad_fn = std::move(node);
  result->set_autograd_meta(std::move(meta));
  return Status::Ok();
}

inline Status RecordUnary(Operation operation,
                          const Tensor& input,
                          const std::shared_ptr<Tensor>& result) {
  if (!RequiresGrad(input)) return Status::Ok();
  std::shared_ptr<Tensor> parent;
  Status status = Share(input, &parent);
  if (!status.ok()) return status;
  return AttachNode(result, operation, {std::move(parent)});
}

inline Status RecordBinary(Operation operation,
                           const Tensor& left,
                           const Tensor& right,
                           const std::shared_ptr<Tensor>& result) {
  if (!RequiresGrad(left) && !RequiresGrad(right)) return Status::Ok();
  std::shared_ptr<Tensor> left_parent;
  std::shared_ptr<Tensor> right_parent;
  Status status = Share(left, &left_parent);
  if (!status.ok()) return status;
  status = Share(right, &right_parent);
  if (!status.ok()) return status;
  return AttachNode(result, operation,
                    {std::move(left_parent), std::move(right_parent)});
}

inline Status RecordBiasAdd(const Tensor& matrix,
                            const Tensor& bias,
                            const std::shared_ptr<Tensor>& result) {
  return RecordBinary(Operation::kBiasAdd, matrix, bias, result);
}

inline Status CopyGradient(const Tensor& gradient,
                           std::shared_ptr<Tensor>* out) {
  return CloneDetached(gradient, out);
}

inline Status AddInPlace(Tensor& destination, const Tensor& source) {
  if (!SameShape(destination.shape(), source.shape())) {
    return InternalError("autograd: gradient accumulation shape mismatch");
  }
  std::vector<float>* destination_values = nullptr;
  Status status =
      MutableCpuValues(destination, "autograd_accumulate", &destination_values);
  if (!status.ok()) return status;
  std::vector<float> source_values;
  status = ReadLogicalCpuValues(source, "autograd_accumulate", &source_values);
  if (!status.ok()) return status;
  for (size_t index = 0; index < destination_values->size(); ++index) {
    (*destination_values)[index] += source_values[index];
  }
  return Status::Ok();
}

inline Status Accumulate(std::shared_ptr<Tensor>* destination,
                         const Tensor& contribution) {
  if (destination == nullptr) {
    return InvalidArgument("autograd: accumulation destination is null");
  }
  if (!*destination) return CopyGradient(contribution, destination);
  return AddInPlace(**destination, contribution);
}

inline Status ValidateSavedVersions(const GradNode& node) {
  if (node.parents.size() != node.parent_versions.size()) {
    return InternalError("autograd: corrupt saved-version metadata");
  }
  for (size_t index = 0; index < node.parents.size(); ++index) {
    const auto& parent = node.parents[index];
    if (parent && Version(*parent) != node.parent_versions[index]) {
      return InvalidArgument("autograd: a saved tensor alias was modified before backward");
    }
  }
  return Status::Ok();
}

inline Status CloneWithShape(const Tensor& source,
                             const ShapeInfo& shape,
                             std::shared_ptr<Tensor>* out) {
  if (source.numel() != shape.numel) {
    return InternalError("autograd: reshape gradient element count mismatch");
  }
  std::vector<float> values;
  Status status = ReadLogicalCpuValues(source, "autograd_reshape", &values);
  if (!status.ok()) return status;
  return MakeCpuTensor(shape, values, out);
}

inline Status TransposeGradient(const Tensor& source,
                                std::shared_ptr<Tensor>* out) {
  if (source.shape().dimensions.size() != 2) {
    return InternalError("autograd: transpose gradient requires rank 2");
  }
  const int64_t rows = source.shape().dimensions[0];
  const int64_t cols = source.shape().dimensions[1];
  const int64_t dims[2] = {cols, rows};
  ShapeInfo shape;
  Status status = ValidateShape(dims, 2, &shape);
  if (!status.ok()) return status;

  std::vector<float> input;
  status = ReadLogicalCpuValues(source, "autograd_transpose", &input);
  if (!status.ok()) return status;
  std::vector<float> output(static_cast<size_t>(shape.numel), 0.0f);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < cols; ++col) {
      output[static_cast<size_t>(col * rows + row)] =
          input[static_cast<size_t>(row * cols + col)];
    }
  }
  return MakeCpuTensor(shape, output, out);
}

inline Status ApplyNode(const GradNode& node,
                        const Tensor& upstream,
                        std::vector<GradientContribution>* out) {
  if (out == nullptr) {
    return InvalidArgument("autograd: contribution output is null");
  }
  out->clear();
  Status status = ValidateSavedVersions(node);
  if (!status.ok()) return status;

  std::vector<float> grad;
  status = ReadLogicalCpuValues(upstream, "autograd_backward", &grad);
  if (!status.ok()) return status;

  auto emit = [&](const std::shared_ptr<Tensor>& parent,
                  std::shared_ptr<Tensor> contribution) {
    if (parent && RequiresGrad(*parent)) {
      out->push_back({parent, std::move(contribution)});
    }
  };

  switch (node.operation) {
    case Operation::kIdentity: {
      std::shared_ptr<Tensor> contribution;
      status = CopyGradient(upstream, &contribution);
      if (!status.ok()) return status;
      emit(node.parents.at(0), std::move(contribution));
      return Status::Ok();
    }
    case Operation::kReshape: {
      std::shared_ptr<Tensor> contribution;
      status = CloneWithShape(upstream, node.parents.at(0)->shape(),
                              &contribution);
      if (!status.ok()) return status;
      emit(node.parents.at(0), std::move(contribution));
      return Status::Ok();
    }
    case Operation::kTranspose2D: {
      std::shared_ptr<Tensor> contribution;
      status = TransposeGradient(upstream, &contribution);
      if (!status.ok()) return status;
      emit(node.parents.at(0), std::move(contribution));
      return Status::Ok();
    }
    case Operation::kAdd: {
      for (const auto& parent : node.parents) {
        if (!parent || !RequiresGrad(*parent)) continue;
        std::shared_ptr<Tensor> contribution;
        status = CopyGradient(upstream, &contribution);
        if (!status.ok()) return status;
        emit(parent, std::move(contribution));
      }
      return Status::Ok();
    }
    case Operation::kMultiply: {
      if (node.parents.size() != 2) {
        return InternalError("autograd: multiply node parent count mismatch");
      }
      std::vector<float> left;
      std::vector<float> right;
      status =
          ReadLogicalCpuValues(*node.parents[0], "autograd_multiply", &left);
      if (!status.ok()) return status;
      status =
          ReadLogicalCpuValues(*node.parents[1], "autograd_multiply", &right);
      if (!status.ok()) return status;
      if (grad.size() != left.size() || grad.size() != right.size()) {
        return InternalError("autograd: multiply gradient size mismatch");
      }

      if (RequiresGrad(*node.parents[0])) {
        std::vector<float> values(grad.size());
        for (size_t index = 0; index < values.size(); ++index) {
          values[index] = grad[index] * right[index];
        }
        std::shared_ptr<Tensor> contribution;
        status = MakeCpuTensor(node.parents[0]->shape(), values, &contribution);
        if (!status.ok()) return status;
        emit(node.parents[0], std::move(contribution));
      }
      if (RequiresGrad(*node.parents[1])) {
        std::vector<float> values(grad.size());
        for (size_t index = 0; index < values.size(); ++index) {
          values[index] = grad[index] * left[index];
        }
        std::shared_ptr<Tensor> contribution;
        status = MakeCpuTensor(node.parents[1]->shape(), values, &contribution);
        if (!status.ok()) return status;
        emit(node.parents[1], std::move(contribution));
      }
      return Status::Ok();
    }
    case Operation::kSum: {
      if (grad.size() != 1) {
        return InternalError("autograd: sum upstream gradient must be scalar");
      }
      std::shared_ptr<Tensor> contribution;
      status = MakeCpuFull(node.parents.at(0)->shape(), grad[0], &contribution);
      if (!status.ok()) return status;
      emit(node.parents.at(0), std::move(contribution));
      return Status::Ok();
    }
    case Operation::kMatmul: {
      if (node.parents.size() != 2) {
        return InternalError("autograd: matmul node parent count mismatch");
      }
      const Tensor& left_tensor = *node.parents[0];
      const Tensor& right_tensor = *node.parents[1];
      if (left_tensor.shape().dimensions.size() != 2 ||
          right_tensor.shape().dimensions.size() != 2 ||
          upstream.shape().dimensions.size() != 2) {
        return InternalError("autograd: matmul gradients require rank 2");
      }
      const int64_t m = left_tensor.shape().dimensions[0];
      const int64_t k = left_tensor.shape().dimensions[1];
      const int64_t n = right_tensor.shape().dimensions[1];
      std::vector<float> left;
      std::vector<float> right;
      status = ReadLogicalCpuValues(left_tensor, "autograd_matmul", &left);
      if (!status.ok()) return status;
      status = ReadLogicalCpuValues(right_tensor, "autograd_matmul", &right);
      if (!status.ok()) return status;

      if (RequiresGrad(left_tensor)) {
        std::vector<float> values(static_cast<size_t>(m * k), 0.0f);
        for (int64_t row = 0; row < m; ++row) {
          for (int64_t inner = 0; inner < k; ++inner) {
            float value = 0.0f;
            for (int64_t col = 0; col < n; ++col) {
              value += grad[static_cast<size_t>(row * n + col)] *
                       right[static_cast<size_t>(inner * n + col)];
            }
            values[static_cast<size_t>(row * k + inner)] = value;
          }
        }
        std::shared_ptr<Tensor> contribution;
        status = MakeCpuTensor(left_tensor.shape(), values, &contribution);
        if (!status.ok()) return status;
        emit(node.parents[0], std::move(contribution));
      }

      if (RequiresGrad(right_tensor)) {
        std::vector<float> values(static_cast<size_t>(k * n), 0.0f);
        for (int64_t inner = 0; inner < k; ++inner) {
          for (int64_t col = 0; col < n; ++col) {
            float value = 0.0f;
            for (int64_t row = 0; row < m; ++row) {
              value += left[static_cast<size_t>(row * k + inner)] *
                       grad[static_cast<size_t>(row * n + col)];
            }
            values[static_cast<size_t>(inner * n + col)] = value;
          }
        }
        std::shared_ptr<Tensor> contribution;
        status = MakeCpuTensor(right_tensor.shape(), values, &contribution);
        if (!status.ok()) return status;
        emit(node.parents[1], std::move(contribution));
      }
      return Status::Ok();
    }
    case Operation::kRelu:
    case Operation::kSigmoid:
    case Operation::kTanh:
    case Operation::kGelu:
    case Operation::kSilu: {
      const Tensor& parent = *node.parents.at(0);
      std::vector<float> input;
      status = ReadLogicalCpuValues(parent, "autograd_activation", &input);
      if (!status.ok()) return status;
      if (input.size() != grad.size()) {
        return InternalError("autograd: activation gradient size mismatch");
      }
      std::vector<float> values(grad.size());
      constexpr float kInvSqrt2 = 0.7071067811865475244f;
      constexpr float kInvSqrt2Pi = 0.39894228040143267794f;
      for (size_t index = 0; index < values.size(); ++index) {
        float local = 0.0f;
        if (node.operation == Operation::kRelu) {
          local = input[index] > 0.0f ? 1.0f : 0.0f;
        } else if (node.operation == Operation::kSigmoid) {
          const float sigmoid = 1.0f / (1.0f + std::exp(-input[index]));
          local = sigmoid * (1.0f - sigmoid);
        } else if (node.operation == Operation::kTanh) {
          const float hyperbolic = std::tanh(input[index]);
          local = 1.0f - hyperbolic * hyperbolic;
        } else if (node.operation == Operation::kGelu) {
          const float x = input[index];
          local = 0.5f * (1.0f + std::erf(x * kInvSqrt2)) +
                  x * std::exp(-0.5f * x * x) * kInvSqrt2Pi;
        } else {
          const float x = input[index];
          const float sigmoid = 1.0f / (1.0f + std::exp(-x));
          local = sigmoid * (1.0f + x * (1.0f - sigmoid));
        }
        values[index] = grad[index] * local;
      }
      std::shared_ptr<Tensor> contribution;
      status = MakeCpuTensor(parent.shape(), values, &contribution);
      if (!status.ok()) return status;
      emit(node.parents.at(0), std::move(contribution));
      return Status::Ok();
    }
    case Operation::kSwiGlu: {
      if (node.parents.size() != 1) {
        return InternalError("autograd: SwiGLU node parent count mismatch");
      }
      const Tensor& parent = *node.parents[0];
      if (parent.shape().dimensions.empty()) {
        return InternalError("autograd: SwiGLU parent rank is invalid");
      }
      const int64_t width = parent.shape().dimensions.back();
      if (width <= 0 || (width % 2) != 0) {
        return InternalError("autograd: SwiGLU parent width is invalid");
      }
      const int64_t half = width / 2;
      const int64_t rows = parent.numel() / width;
      if (grad.size() != static_cast<size_t>(rows * half)) {
        return InternalError("autograd: SwiGLU gradient shape mismatch");
      }
      std::vector<float> input;
      status = ReadLogicalCpuValues(parent, "autograd_swiglu", &input);
      if (!status.ok()) return status;
      std::vector<float> values(input.size(), 0.0f);
      for (int64_t row = 0; row < rows; ++row) {
        const size_t input_base = static_cast<size_t>(row * width);
        const size_t grad_base = static_cast<size_t>(row * half);
        for (int64_t col = 0; col < half; ++col) {
          const size_t left_index = input_base + static_cast<size_t>(col);
          const size_t right_index =
              input_base + static_cast<size_t>(half + col);
          const float a = input[left_index];
          const float b = input[right_index];
          const float sigmoid = 1.0f / (1.0f + std::exp(-a));
          const float silu = a * sigmoid;
          const float silu_prime =
              sigmoid * (1.0f + a * (1.0f - sigmoid));
          const float upstream_value =
              grad[grad_base + static_cast<size_t>(col)];
          values[left_index] = upstream_value * b * silu_prime;
          values[right_index] = upstream_value * silu;
        }
      }
      std::shared_ptr<Tensor> contribution;
      status = MakeCpuTensor(parent.shape(), values, &contribution);
      if (!status.ok()) return status;
      emit(node.parents[0], std::move(contribution));
      return Status::Ok();
    }
    case Operation::kMse: {
      if (node.parents.size() != 2 || grad.size() != 1) {
        return InternalError("autograd: invalid MSE node metadata");
      }
      const Tensor& prediction = *node.parents[0];
      const Tensor& target = *node.parents[1];
      std::vector<float> prediction_values;
      std::vector<float> target_values;
      status =
          ReadLogicalCpuValues(prediction, "autograd_mse", &prediction_values);
      if (!status.ok()) return status;
      status = ReadLogicalCpuValues(target, "autograd_mse", &target_values);
      if (!status.ok()) return status;
      const float scale =
          2.0f * grad[0] / static_cast<float>(prediction_values.size());

      if (RequiresGrad(prediction)) {
        std::vector<float> values(prediction_values.size());
        for (size_t index = 0; index < values.size(); ++index) {
          values[index] =
              scale * (prediction_values[index] - target_values[index]);
        }
        std::shared_ptr<Tensor> contribution;
        status = MakeCpuTensor(prediction.shape(), values, &contribution);
        if (!status.ok()) return status;
        emit(node.parents[0], std::move(contribution));
      }
      if (RequiresGrad(target)) {
        std::vector<float> values(target_values.size());
        for (size_t index = 0; index < values.size(); ++index) {
          values[index] =
              -scale * (prediction_values[index] - target_values[index]);
        }
        std::shared_ptr<Tensor> contribution;
        status = MakeCpuTensor(target.shape(), values, &contribution);
        if (!status.ok()) return status;
        emit(node.parents[1], std::move(contribution));
      }
      return Status::Ok();
    }
    case Operation::kCrossEntropy: {
      if (node.parents.size() != 2 || grad.size() != 1) {
        return InternalError("autograd: invalid cross entropy metadata");
      }
      const Tensor& logits = *node.parents[0];
      const Tensor& target = *node.parents[1];
      if (logits.shape().dimensions.size() != 2 ||
          !SameShape(logits.shape(), target.shape())) {
        return InternalError("autograd: invalid cross entropy shapes");
      }
      const int64_t batch = logits.shape().dimensions[0];
      const int64_t classes = logits.shape().dimensions[1];
      std::vector<float> logits_values;
      std::vector<float> target_values;
      status = ReadLogicalCpuValues(logits, "autograd_cross_entropy",
                                    &logits_values);
      if (!status.ok()) return status;
      status = ReadLogicalCpuValues(target, "autograd_cross_entropy",
                                    &target_values);
      if (!status.ok()) return status;
      const float scale = grad[0] / static_cast<float>(batch);

      if (RequiresGrad(logits)) {
        std::vector<float> values(logits_values.size(), 0.0f);
        for (int64_t row = 0; row < batch; ++row) {
          float max_logit = logits_values[static_cast<size_t>(row * classes)];
          for (int64_t col = 1; col < classes; ++col) {
            max_logit = std::max(
                max_logit,
                logits_values[static_cast<size_t>(row * classes + col)]);
          }
          float denominator = 0.0f;
          for (int64_t col = 0; col < classes; ++col) {
            denominator += std::exp(
                logits_values[static_cast<size_t>(row * classes + col)] -
                max_logit);
          }
          for (int64_t col = 0; col < classes; ++col) {
            const size_t index = static_cast<size_t>(row * classes + col);
            const float probability =
                std::exp(logits_values[index] - max_logit) / denominator;
            values[index] = scale * (probability - target_values[index]);
          }
        }
        std::shared_ptr<Tensor> contribution;
        status = MakeCpuTensor(logits.shape(), values, &contribution);
        if (!status.ok()) return status;
        emit(node.parents[0], std::move(contribution));
      }

      if (RequiresGrad(target)) {
        std::vector<float> values(target_values.size(), 0.0f);
        for (int64_t row = 0; row < batch; ++row) {
          float max_logit = logits_values[static_cast<size_t>(row * classes)];
          for (int64_t col = 1; col < classes; ++col) {
            max_logit = std::max(
                max_logit,
                logits_values[static_cast<size_t>(row * classes + col)]);
          }
          float denominator = 0.0f;
          for (int64_t col = 0; col < classes; ++col) {
            denominator += std::exp(
                logits_values[static_cast<size_t>(row * classes + col)] -
                max_logit);
          }
          const float log_denominator = std::log(denominator) + max_logit;
          for (int64_t col = 0; col < classes; ++col) {
            const size_t index = static_cast<size_t>(row * classes + col);
            values[index] =
                -scale * (logits_values[index] - log_denominator);
          }
        }
        std::shared_ptr<Tensor> contribution;
        status = MakeCpuTensor(target.shape(), values, &contribution);
        if (!status.ok()) return status;
        emit(node.parents[1], std::move(contribution));
      }
      return Status::Ok();
    }
    case Operation::kBiasAdd: {
      if (node.parents.size() != 2) {
        return InternalError("autograd: bias-add node parent count mismatch");
      }
      const Tensor& matrix = *node.parents[0];
      const Tensor& bias = *node.parents[1];
      if (matrix.shape().dimensions.size() != 2 ||
          bias.shape().dimensions.size() != 1) {
        return InternalError("autograd: bias-add shape metadata is invalid");
      }
      const int64_t rows = matrix.shape().dimensions[0];
      const int64_t cols = matrix.shape().dimensions[1];
      if (bias.shape().dimensions[0] != cols ||
          grad.size() != static_cast<size_t>(rows * cols)) {
        return InternalError("autograd: bias-add gradient shape mismatch");
      }

      if (RequiresGrad(matrix)) {
        std::shared_ptr<Tensor> contribution;
        status = CopyGradient(upstream, &contribution);
        if (!status.ok()) return status;
        emit(node.parents[0], std::move(contribution));
      }
      if (RequiresGrad(bias)) {
        std::vector<float> values(static_cast<size_t>(cols), 0.0f);
        for (int64_t row = 0; row < rows; ++row) {
          for (int64_t col = 0; col < cols; ++col) {
            values[static_cast<size_t>(col)] +=
                grad[static_cast<size_t>(row * cols + col)];
          }
        }
        std::shared_ptr<Tensor> contribution;
        status = MakeCpuTensor(bias.shape(), values, &contribution);
        if (!status.ok()) return status;
        emit(node.parents[1], std::move(contribution));
      }
      return Status::Ok();
    }
  }
  return InternalError("autograd: unknown operation");
}

inline void BuildTopology(const std::shared_ptr<Tensor>& value,
                          std::unordered_set<Tensor*>* visited,
                          std::vector<std::shared_ptr<Tensor>>* topology) {
  if (!value || visited == nullptr || topology == nullptr) return;
  if (!visited->insert(value.get()).second) return;
  const auto meta = value->autograd_meta();
  if (meta) {
    std::shared_ptr<GradNode> node;
    {
      std::lock_guard<std::mutex> lock(meta->mutex);
      node = meta->grad_fn;
    }
    if (node) {
      for (const auto& parent : node->parents) {
        BuildTopology(parent, visited, topology);
      }
    }
  }
  topology->push_back(value);
}

inline Status AccumulateLeaf(AutogradMeta& meta, const Tensor& gradient) {
  std::lock_guard<std::mutex> lock(meta.mutex);
  return Accumulate(&meta.gradient, gradient);
}

inline Status Backward(const Tensor& root) {
  if (root.numel() != 1) {
    return InvalidShape("tensor_backward: loss tensor must contain one value");
  }
  if (!RequiresGrad(root)) {
    return InvalidArgument("tensor_backward: tensor does not require gradients");
  }

  std::shared_ptr<Tensor> root_ptr;
  Status status = Share(root, &root_ptr);
  if (!status.ok()) return status;

  std::unordered_set<Tensor*> visited;
  std::vector<std::shared_ptr<Tensor>> topology;
  BuildTopology(root_ptr, &visited, &topology);

  // Validate all aliases before publishing any leaf gradient. A stale view
  // therefore fails transactionally instead of leaving a partially updated
  // gradient set behind.
  for (const auto& value : topology) {
    const auto meta = value->autograd_meta();
    if (!meta) continue;
    std::shared_ptr<GradNode> node;
    {
      std::lock_guard<std::mutex> lock(meta->mutex);
      node = meta->grad_fn;
    }
    if (node) {
      status = ValidateSavedVersions(*node);
      if (!status.ok()) return status;
    }
  }

  std::unordered_map<Tensor*, std::shared_ptr<Tensor>> gradients;
  std::shared_ptr<Tensor> root_gradient;
  status = MakeCpuFull(root.shape(), 1.0f, &root_gradient);
  if (!status.ok()) return status;
  gradients.emplace(root_ptr.get(), std::move(root_gradient));

  for (auto iterator = topology.rbegin(); iterator != topology.rend();
       ++iterator) {
    const auto& value = *iterator;
    const auto gradient_it = gradients.find(value.get());
    if (gradient_it == gradients.end()) continue;
    const std::shared_ptr<Tensor>& upstream = gradient_it->second;

    const auto meta = value->autograd_meta();
    if (!meta) continue;

    bool is_leaf = false;
    bool requires_grad = false;
    std::shared_ptr<GradNode> node;
    {
      std::lock_guard<std::mutex> lock(meta->mutex);
      is_leaf = meta->is_leaf;
      requires_grad = meta->requires_grad;
      node = meta->grad_fn;
    }

    if (requires_grad && is_leaf) {
      status = AccumulateLeaf(*meta, *upstream);
      if (!status.ok()) return status;
    }
    if (!node) continue;

    std::vector<GradientContribution> contributions;
    status = ApplyNode(*node, *upstream, &contributions);
    if (!status.ok()) return status;
    for (auto& contribution : contributions) {
      auto& accumulated = gradients[contribution.parent.get()];
      status = Accumulate(&accumulated, *contribution.gradient);
      if (!status.ok()) return status;
    }
  }
  return Status::Ok();
}

inline Status GradientSnapshot(const Tensor& tensor,
                               std::shared_ptr<Tensor>* out) {
  if (out == nullptr) {
    return InvalidArgument("tensor_grad: output tensor pointer is null");
  }
  *out = nullptr;
  const auto meta = tensor.autograd_meta();
  if (!meta) return InvalidArgument("tensor_grad: gradient is not available");
  std::lock_guard<std::mutex> lock(meta->mutex);
  if (!meta->gradient) {
    return InvalidArgument("tensor_grad: gradient is not available");
  }
  return CopyGradient(*meta->gradient, out);
}

inline void ClearGradient(Tensor& tensor) {
  const auto meta = tensor.autograd_meta();
  if (!meta) return;
  std::lock_guard<std::mutex> lock(meta->mutex);
  meta->gradient.reset();
}

}  // namespace tensora::autograd

#endif  // TENSORA_AUTOGRAD_AUTOGRAD_H_
