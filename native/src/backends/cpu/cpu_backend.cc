#include "backends/cpu/cpu_backend.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "autograd/autograd.h"
#include "core/allocation_guard.h"

namespace tensora {
namespace {

Status MakeTensor(ShapeInfo shape,
                  std::shared_ptr<CpuStorage> storage,
                  std::shared_ptr<Tensor>* out) {
  if (out == nullptr) {
    return InvalidArgument("cpu backend: output tensor pointer is null");
  }
  if (!storage) {
    return InvalidArgument("cpu backend: materialized tensor storage is null");
  }

  ShapeInfo materialized_shape;
  const int64_t* dimensions =
      shape.dimensions.empty() ? nullptr : shape.dimensions.data();
  Status status =
      ValidateShape(dimensions, shape.dimensions.size(), &materialized_shape);
  if (!status.ok() || materialized_shape.numel != shape.numel ||
      storage->numel() != shape.numel) {
    return InternalError(
        "cpu backend: materialized tensor shape/storage contract is invalid");
  }

  return AllocationGuard("cpu backend tensor", [&]() -> Status {
    *out = std::make_shared<Tensor>(std::move(materialized_shape),
                                    std::move(storage));
    return Status::Ok();
  });
}

Status MakeView(ShapeInfo shape,
                const Tensor& source,
                std::shared_ptr<Tensor>* out) {
  if (out == nullptr) {
    return InvalidArgument("cpu backend: output tensor pointer is null");
  }
  *out = nullptr;
  return AllocationGuard("cpu backend view", [&]() -> Status {
    *out = std::make_shared<Tensor>(
        std::move(shape), source.storage(), source.dtype(), source.device(),
        source.device_index(), source.storage_offset(), source.version_counter());
    return Status::Ok();
  });
}

Status EnsureCpuFloat32(const Tensor& tensor, const char* operation) {
  if (tensor.device() != Device::kCpu || tensor.device_index() != 0) {
    return Unsupported(std::string(operation) + ": CPU backend requires cpu:0");
  }
  if (tensor.dtype() != DType::kFloat32) {
    return Unsupported(std::string(operation) + ": only float32 is supported");
  }
  return Status::Ok();
}

Status LogicalValues(const Tensor& tensor,
                     const char* operation,
                     std::vector<float>* out) {
  if (out == nullptr) {
    return InvalidArgument(std::string(operation) +
                           ": output values pointer is null");
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
  if (written != out->size()) {
    return InternalError(std::string(operation) +
                         ": logical copy returned inconsistent size");
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

  if (tensor.is_contiguous()) {
    status = MakeView(shape, tensor, out);
  } else {
    std::vector<float> values;
    status = LogicalValues(tensor, "reshape", &values);
    if (!status.ok()) return status;
    std::shared_ptr<CpuStorage> storage;
    status = CpuStorage::FromData(values.data(), tensor.numel(), &storage);
    if (!status.ok()) return status;
    status = MakeTensor(shape, std::move(storage), out);
  }
  if (!status.ok()) return status;
  return autograd::RecordUnary(autograd::Operation::kReshape, tensor, *out);
}

Status CpuBackend::Transpose2D(const Tensor& tensor,
                               std::shared_ptr<Tensor>* out) const {
  Status status = EnsureCpuFloat32(tensor, "transpose");
  if (!status.ok()) return status;
  if (tensor.shape().dimensions.size() != 2) {
    return InvalidShape("transpose: Milestone 1 requires a rank-2 tensor");
  }

  ShapeInfo output_shape = tensor.shape();
  std::swap(output_shape.dimensions[0], output_shape.dimensions[1]);
  std::swap(output_shape.strides[0], output_shape.strides[1]);
  status = MakeView(std::move(output_shape), tensor, out);
  if (!status.ok()) return status;
  return autograd::RecordUnary(autograd::Operation::kTranspose2D, tensor, *out);
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

  std::vector<float> left_values;
  std::vector<float> right_values;
  status = LogicalValues(left, "add", &left_values);
  if (!status.ok()) return status;
  status = LogicalValues(right, "add", &right_values);
  if (!status.ok()) return status;

  std::shared_ptr<CpuStorage> storage;
  status = CpuStorage::Filled(left.numel(), 0.0f, &storage);
  if (!status.ok()) return status;
  auto& result = storage->mutable_values();
  for (size_t index = 0; index < result.size(); ++index) {
    result[index] = left_values[index] + right_values[index];
  }

  status = MakeTensor(left.shape(), std::move(storage), out);
  if (!status.ok()) return status;
  return autograd::RecordBinary(autograd::Operation::kAdd, left, right, *out);
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

  std::vector<float> left_values;
  std::vector<float> right_values;
  status = LogicalValues(left, "multiply", &left_values);
  if (!status.ok()) return status;
  status = LogicalValues(right, "multiply", &right_values);
  if (!status.ok()) return status;

  std::shared_ptr<CpuStorage> storage;
  status = CpuStorage::Filled(left.numel(), 0.0f, &storage);
  if (!status.ok()) return status;
  auto& result = storage->mutable_values();
  for (size_t index = 0; index < result.size(); ++index) {
    result[index] = left_values[index] * right_values[index];
  }

  status = MakeTensor(left.shape(), std::move(storage), out);
  if (!status.ok()) return status;
  return autograd::RecordBinary(autograd::Operation::kMultiply, left, right,
                                *out);
}

Status CpuBackend::Sum(const Tensor& tensor,
                       std::shared_ptr<Tensor>* out) const {
  Status status = EnsureCpuFloat32(tensor, "sum");
  if (!status.ok()) return status;

  std::vector<float> input;
  status = LogicalValues(tensor, "sum", &input);
  if (!status.ok()) return status;

  ShapeInfo scalar_shape;
  status = ValidateShape(nullptr, 0, &scalar_shape);
  if (!status.ok()) return status;

  float value = 0.0f;
  for (float item : input) value += item;

  std::shared_ptr<CpuStorage> storage;
  status = CpuStorage::Filled(1, value, &storage);
  if (!status.ok()) return status;
  status = MakeTensor(std::move(scalar_shape), std::move(storage), out);
  if (!status.ok()) return status;
  return autograd::RecordUnary(autograd::Operation::kSum, tensor, *out);
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

  std::vector<float> left_values;
  std::vector<float> right_values;
  status = LogicalValues(left, "matmul", &left_values);
  if (!status.ok()) return status;
  status = LogicalValues(right, "matmul", &right_values);
  if (!status.ok()) return status;

  const int64_t output_dims[2] = {m, n};
  ShapeInfo output_shape;
  status = ValidateShape(output_dims, 2, &output_shape);
  if (!status.ok()) return status;

  std::shared_ptr<CpuStorage> storage;
  status = CpuStorage::Filled(output_shape.numel, 0.0f, &storage);
  if (!status.ok()) return status;
  auto& output = storage->mutable_values();

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t inner = 0; inner < k; ++inner) {
      const float left_value =
          left_values[static_cast<size_t>(row * k + inner)];
      for (int64_t col = 0; col < n; ++col) {
        output[static_cast<size_t>(row * n + col)] +=
            left_value * right_values[static_cast<size_t>(inner * n + col)];
      }
    }
  }

  status = MakeTensor(std::move(output_shape), std::move(storage), out);
  if (!status.ok()) return status;
  return autograd::RecordBinary(autograd::Operation::kMatmul, left, right,
                                *out);
}

}  // namespace tensora
