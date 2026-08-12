#include "backends/cpu/cpu_backend.h"

#include <algorithm>
#include <memory>

namespace tensora {
namespace {

Status MakeTensor(ShapeInfo shape,
                  std::shared_ptr<CpuStorage> storage,
                  std::shared_ptr<Tensor>* out) {
  if (out == nullptr) {
    return InvalidArgument("cpu backend: output tensor pointer is null");
  }
  try {
    *out = std::make_shared<Tensor>(std::move(shape), std::move(storage));
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return OutOfMemory("cpu backend: tensor object allocation failed");
  }
}

Status EnsureCpuFloat32(const Tensor& tensor, const char* operation) {
  if (tensor.device() != Device::kCpu) {
    return Unsupported(std::string(operation) + ": only CPU is supported");
  }
  if (tensor.dtype() != DType::kFloat32) {
    return Unsupported(std::string(operation) + ": only float32 is supported");
  }
  return Status::Ok();
}

}  // namespace

Status CpuBackend::FromData(const ShapeInfo& shape,
                            const float* data,
                            std::shared_ptr<Tensor>* out) const {
  std::shared_ptr<CpuStorage> storage;
  Status status = CpuStorage::FromData(data, shape.numel, &storage);
  if (!status.ok()) return status;
  return MakeTensor(shape, std::move(storage), out);
}

Status CpuBackend::Full(const ShapeInfo& shape,
                        float value,
                        std::shared_ptr<Tensor>* out) const {
  std::shared_ptr<CpuStorage> storage;
  Status status = CpuStorage::Filled(shape.numel, value, &storage);
  if (!status.ok()) return status;
  return MakeTensor(shape, std::move(storage), out);
}

Status CpuBackend::Reshape(const Tensor& tensor,
                           const ShapeInfo& shape,
                           std::shared_ptr<Tensor>* out) const {
  Status status = EnsureCpuFloat32(tensor, "reshape");
  if (!status.ok()) return status;
  if (tensor.numel() != shape.numel) {
    return InvalidShape("reshape: target shape must preserve element count");
  }

  std::shared_ptr<CpuStorage> storage;
  const auto& values = tensor.storage()->values();
  status = CpuStorage::FromData(values.data(), tensor.numel(), &storage);
  if (!status.ok()) return status;
  return MakeTensor(shape, std::move(storage), out);
}

Status CpuBackend::Transpose2D(const Tensor& tensor,
                               std::shared_ptr<Tensor>* out) const {
  Status status = EnsureCpuFloat32(tensor, "transpose");
  if (!status.ok()) return status;
  if (tensor.shape().dimensions.size() != 2) {
    return InvalidShape("transpose: Milestone 1 requires a rank-2 tensor");
  }

  const int64_t rows = tensor.shape().dimensions[0];
  const int64_t cols = tensor.shape().dimensions[1];
  const int64_t output_dims[2] = {cols, rows};
  ShapeInfo output_shape;
  status = ValidateShape(output_dims, 2, &output_shape);
  if (!status.ok()) return status;

  std::shared_ptr<CpuStorage> storage;
  status = CpuStorage::Filled(output_shape.numel, 0.0f, &storage);
  if (!status.ok()) return status;

  const auto& input = tensor.storage()->values();
  auto& output = storage->mutable_values();
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < cols; ++col) {
      output[static_cast<size_t>(col * rows + row)] =
          input[static_cast<size_t>(row * cols + col)];
    }
  }

  return MakeTensor(std::move(output_shape), std::move(storage), out);
}

Status CpuBackend::Add(const Tensor& left,
                       const Tensor& right,
                       std::shared_ptr<Tensor>* out) const {
  Status status = EnsureCpuFloat32(left, "add");
  if (!status.ok()) return status;
  status = EnsureCpuFloat32(right, "add");
  if (!status.ok()) return status;
  if (!SameShape(left.shape(), right.shape())) {
    return InvalidShape(
        "add: Milestone 1 elementwise operations require equal shapes");
  }

  std::shared_ptr<CpuStorage> storage;
  status = CpuStorage::Filled(left.numel(), 0.0f, &storage);
  if (!status.ok()) return status;

  const auto& a = left.storage()->values();
  const auto& b = right.storage()->values();
  auto& result = storage->mutable_values();
  for (size_t i = 0; i < result.size(); ++i) {
    result[i] = a[i] + b[i];
  }

  return MakeTensor(left.shape(), std::move(storage), out);
}

Status CpuBackend::Multiply(const Tensor& left,
                            const Tensor& right,
                            std::shared_ptr<Tensor>* out) const {
  Status status = EnsureCpuFloat32(left, "multiply");
  if (!status.ok()) return status;
  status = EnsureCpuFloat32(right, "multiply");
  if (!status.ok()) return status;
  if (!SameShape(left.shape(), right.shape())) {
    return InvalidShape(
        "multiply: Milestone 1 elementwise operations require equal shapes");
  }

  std::shared_ptr<CpuStorage> storage;
  status = CpuStorage::Filled(left.numel(), 0.0f, &storage);
  if (!status.ok()) return status;

  const auto& a = left.storage()->values();
  const auto& b = right.storage()->values();
  auto& result = storage->mutable_values();
  for (size_t i = 0; i < result.size(); ++i) {
    result[i] = a[i] * b[i];
  }

  return MakeTensor(left.shape(), std::move(storage), out);
}

Status CpuBackend::Sum(const Tensor& tensor,
                       std::shared_ptr<Tensor>* out) const {
  Status status = EnsureCpuFloat32(tensor, "sum");
  if (!status.ok()) return status;

  ShapeInfo scalar_shape;
  status = ValidateShape(nullptr, 0, &scalar_shape);
  if (!status.ok()) return status;

  float value = 0.0f;
  for (float item : tensor.storage()->values()) {
    value += item;
  }

  std::shared_ptr<CpuStorage> storage;
  status = CpuStorage::Filled(1, value, &storage);
  if (!status.ok()) return status;
  return MakeTensor(std::move(scalar_shape), std::move(storage), out);
}

Status CpuBackend::Matmul(const Tensor& left,
                          const Tensor& right,
                          std::shared_ptr<Tensor>* out) const {
  Status status = EnsureCpuFloat32(left, "matmul");
  if (!status.ok()) return status;
  status = EnsureCpuFloat32(right, "matmul");
  if (!status.ok()) return status;

  if (left.shape().dimensions.size() != 2 ||
      right.shape().dimensions.size() != 2) {
    return InvalidShape("matmul: Milestone 1 requires rank-2 tensors");
  }

  const int64_t m = left.shape().dimensions[0];
  const int64_t k = left.shape().dimensions[1];
  const int64_t right_k = right.shape().dimensions[0];
  const int64_t n = right.shape().dimensions[1];
  if (k != right_k) {
    return InvalidShape("matmul: inner dimensions must match");
  }

  const int64_t output_dims[2] = {m, n};
  ShapeInfo output_shape;
  status = ValidateShape(output_dims, 2, &output_shape);
  if (!status.ok()) return status;

  std::shared_ptr<CpuStorage> storage;
  status = CpuStorage::Filled(output_shape.numel, 0.0f, &storage);
  if (!status.ok()) return status;

  const auto& a = left.storage()->values();
  const auto& b = right.storage()->values();
  auto& c = storage->mutable_values();

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t inner = 0; inner < k; ++inner) {
      const float a_value = a[static_cast<size_t>(row * k + inner)];
      for (int64_t col = 0; col < n; ++col) {
        c[static_cast<size_t>(row * n + col)] +=
            a_value * b[static_cast<size_t>(inner * n + col)];
      }
    }
  }

  return MakeTensor(std::move(output_shape), std::move(storage), out);
}

}  // namespace tensora
