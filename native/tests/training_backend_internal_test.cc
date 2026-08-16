#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include <torch/torch.h>

#include "core/status.h"
#include "memory/cpu_storage.h"
#include "memory/tensor_storage.h"
#include "tensor/shape.h"
#include "training/nn_v2_parameter_control.h"
#include "training/nn_v2_state.h"
#include "training/torch_backend.h"
#include "training/training_bridge.h"
#include "training/torch_storage.h"

namespace {

bool ExpectStatus(const tensora::Status& status,
                  ts_status_t expected,
                  const char* operation) {
  if (status.code() == expected) return true;
  std::cerr << operation << " expected status " << expected << ", got "
            << status.code() << ": " << status.message() << "\n";
  return false;
}

class FakeStorage final : public tensora::TensorStorage {
 public:
  FakeStorage(tensora::StorageKind kind, uint64_t byte_size, size_t written)
      : kind_(kind), byte_size_(byte_size), written_(written) {}

  tensora::StorageKind kind() const override { return kind_; }
  tensora::Status CopyToHostF32(float* out_values,
                                size_t capacity,
                                size_t* out_written) const override {
    if (out_written == nullptr) {
      return tensora::InvalidArgument("fake storage: written pointer is null");
    }
    *out_written = 0;
    if (capacity < written_) {
      return tensora::InvalidArgument("fake storage: capacity is too small");
    }
    if (written_ > 0 && out_values == nullptr) {
      return tensora::InvalidArgument("fake storage: values pointer is null");
    }
    for (size_t index = 0; index < written_; ++index) out_values[index] = 1.0f;
    *out_written = written_;
    return tensora::Status::Ok();
  }
  uint64_t byte_size() const override { return byte_size_; }

 private:
  tensora::StorageKind kind_;
  uint64_t byte_size_;
  size_t written_;
};

int CheckDirectTorchBackend() {
  using tensora::Device;
  using tensora::ShapeInfo;
  using tensora::Tensor;
  using tensora::training::TensorToTorch;
  using tensora::training::TorchBackend;
  using tensora::training::TorchDevice;
  using tensora::training::TorchStorage;
  using tensora::training::WrapTorchTensor;

  TorchBackend backend;

  uint32_t count = 999;
  if (!ExpectStatus(tensora::training::DeviceCount(Device::kCpu, nullptr),
                    TS_INVALID_ARGUMENT, "DeviceCount null"))
    return 100;
  if (!ExpectStatus(tensora::training::DeviceCount(Device::kCpu, &count),
                    TS_OK, "DeviceCount CPU") ||
      count != 1)
    return 101;

  torch::Device torch_device(torch::kCPU);
  if (!ExpectStatus(TorchDevice(Device::kCpu, 0, nullptr), TS_INVALID_ARGUMENT,
                    "TorchDevice null"))
    return 102;
  if (!ExpectStatus(TorchDevice(Device::kCpu, 1, &torch_device),
                    TS_INVALID_ARGUMENT, "TorchDevice CPU index"))
    return 103;
  if (!ExpectStatus(TorchDevice(Device::kCpu, 0, &torch_device), TS_OK,
                    "TorchDevice CPU") ||
      !torch_device.is_cpu())
    return 104;
  if (!ExpectStatus(TorchDevice(Device::kCuda, -1, &torch_device),
                    TS_INVALID_ARGUMENT, "TorchDevice negative CUDA"))
    return 105;
  if (!ExpectStatus(TorchDevice(Device::kMps, 1, &torch_device),
                    TS_INVALID_ARGUMENT, "TorchDevice MPS index"))
    return 106;

  for (const Device device :
       {Device::kCuda, Device::kMps, Device::kXpu, Device::kHip}) {
    count = 999;
    const tensora::Status count_status =
        tensora::training::DeviceCount(device, &count);
    if (!count_status.ok()) return 107;
    if (count == 0 &&
        !ExpectStatus(TorchDevice(device, 0, &torch_device), TS_UNSUPPORTED,
                      "TorchDevice unavailable accelerator"))
      return 108;
  }

  ShapeInfo matrix_shape;
  const int64_t matrix_dims[2] = {2, 2};
  if (!tensora::ValidateShape(matrix_dims, 2, &matrix_shape).ok()) return 109;
  ShapeInfo vector_shape;
  const int64_t vector_dims[1] = {2};
  if (!tensora::ValidateShape(vector_dims, 1, &vector_shape).ok()) return 110;
  ShapeInfo other_shape;
  const int64_t other_dims[2] = {2, 3};
  if (!tensora::ValidateShape(other_dims, 2, &other_shape).ok()) return 111;
  ShapeInfo transposed_shape;
  const int64_t transposed_dims[2] = {3, 2};
  if (!tensora::ValidateShape(transposed_dims, 2, &transposed_shape).ok())
    return 112;

  const float matrix_values[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  std::shared_ptr<Tensor> matrix;
  if (!ExpectStatus(backend.FromData(matrix_shape, nullptr, &matrix),
                    TS_INVALID_ARGUMENT, "TorchBackend FromData null data"))
    return 113;
  if (!ExpectStatus(backend.FromData(matrix_shape, matrix_values, nullptr),
                    TS_INVALID_ARGUMENT, "TorchBackend FromData null output"))
    return 114;
  if (!ExpectStatus(backend.FromData(matrix_shape, matrix_values, &matrix),
                    TS_OK, "TorchBackend FromData") ||
      !matrix)
    return 115;

  torch::Tensor converted;
  if (!ExpectStatus(TensorToTorch(*matrix, nullptr), TS_INVALID_ARGUMENT,
                    "TensorToTorch null output"))
    return 116;
  if (!ExpectStatus(TensorToTorch(*matrix, &converted), TS_OK,
                    "TensorToTorch TorchStorage") ||
      !converted.defined() || converted.scalar_type() != torch::kFloat32)
    return 117;

  Tensor unsupported_dtype(matrix_shape, matrix->storage(),
                           static_cast<tensora::DType>(999), Device::kCpu, 0);
  if (!ExpectStatus(TensorToTorch(unsupported_dtype, &converted), TS_UNSUPPORTED,
                    "TensorToTorch dtype"))
    return 148;

  const auto fake_torch_storage = std::make_shared<FakeStorage>(
      tensora::StorageKind::kTorch, matrix_shape.byte_size, 4);
  Tensor fake_torch_tensor(matrix_shape, fake_torch_storage);
  if (!ExpectStatus(TensorToTorch(fake_torch_tensor, &converted),
                    TS_INTERNAL_ERROR, "TensorToTorch fake Torch storage"))
    return 149;

  std::shared_ptr<tensora::CpuStorage> cpu_storage;
  if (!ExpectStatus(tensora::CpuStorage::FromData(matrix_values, 4, &cpu_storage),
                    TS_OK, "CpuStorage FromData for TensorToTorch") ||
      !cpu_storage)
    return 150;
  Tensor cpu_tensor(matrix_shape, cpu_storage);
  if (!ExpectStatus(TensorToTorch(cpu_tensor, &converted), TS_OK,
                    "TensorToTorch CPU storage") ||
      !converted.defined() || converted.device().type() != c10::DeviceType::CPU)
    return 151;

  const auto inconsistent_storage = std::make_shared<FakeStorage>(
      tensora::StorageKind::kCpu, matrix_shape.byte_size, 0);
  Tensor inconsistent_tensor(matrix_shape, inconsistent_storage);
  if (!ExpectStatus(TensorToTorch(inconsistent_tensor, &converted),
                    TS_INTERNAL_ERROR, "TensorToTorch inconsistent storage"))
    return 152;

  std::shared_ptr<Tensor> filled;
  if (!ExpectStatus(backend.Full(matrix_shape, 3.0f, &filled), TS_OK,
                    "TorchBackend Full") ||
      !filled)
    return 118;

  if (!ExpectStatus(backend.Full(matrix_shape, 3.0f, nullptr),
                    TS_INVALID_ARGUMENT, "TorchBackend Full null output"))
    return 154;

  std::shared_ptr<Tensor> wrapped;
  if (!ExpectStatus(WrapTorchTensor(torch::Tensor(), &wrapped),
                    TS_INVALID_ARGUMENT, "WrapTorchTensor undefined"))
    return 119;
  if (!ExpectStatus(WrapTorchTensor(torch::ones({1}, torch::kFloat64), &wrapped),
                    TS_UNSUPPORTED, "WrapTorchTensor float64"))
    return 120;
  if (!ExpectStatus(WrapTorchTensor(torch::ones({1}, torch::kFloat32), nullptr),
                    TS_INVALID_ARGUMENT, "WrapTorchTensor null output"))
    return 121;
  if (!ExpectStatus(WrapTorchTensor(torch::tensor(5.0f), &wrapped), TS_OK,
                    "WrapTorchTensor scalar") ||
      !wrapped || wrapped->numel() != 1)
    return 122;
  wrapped.reset();

  try {
    const auto meta = torch::empty(
        {1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kMeta));
    if (!ExpectStatus(WrapTorchTensor(meta, &wrapped), TS_UNSUPPORTED,
                      "WrapTorchTensor meta device"))
      return 153;
  } catch (const c10::Error&) {
    // Older LibTorch builds may reject construction before Tensora sees it.
  }
  wrapped.reset();

  const uint64_t before_storage_bytes = TorchStorage::LiveBytes();
  std::shared_ptr<TorchStorage> storage;
  if (!ExpectStatus(TorchStorage::FromTensor(torch::Tensor(), &storage),
                    TS_INVALID_ARGUMENT, "TorchStorage undefined"))
    return 123;
  if (!ExpectStatus(
          TorchStorage::FromTensor(torch::ones({2}, torch::kFloat64), &storage),
          TS_UNSUPPORTED, "TorchStorage float64"))
    return 124;
  if (!ExpectStatus(
          TorchStorage::FromTensor(torch::ones({2}, torch::kFloat32), nullptr),
          TS_INVALID_ARGUMENT, "TorchStorage null output"))
    return 125;
  if (!ExpectStatus(
          TorchStorage::FromTensor(torch::tensor({7.0f, 8.0f}), &storage),
          TS_OK, "TorchStorage FromTensor") ||
      !storage || TorchStorage::LiveBytes() != before_storage_bytes + 8)
    return 126;

  size_t written = 999;
  float copied[2] = {0.0f, 0.0f};
  if (!ExpectStatus(storage->CopyToHostF32(copied, 2, nullptr),
                    TS_INVALID_ARGUMENT, "TorchStorage null written"))
    return 127;
  if (!ExpectStatus(storage->CopyToHostF32(copied, 1, &written),
                    TS_INVALID_ARGUMENT, "TorchStorage short capacity") ||
      written != 0)
    return 128;
  if (!ExpectStatus(storage->CopyToHostF32(nullptr, 2, &written),
                    TS_INVALID_ARGUMENT, "TorchStorage null values") ||
      written != 0)
    return 129;
  if (!ExpectStatus(storage->CopyToHostF32(copied, 2, &written), TS_OK,
                    "TorchStorage copy") ||
      written != 2 || copied[0] != 7.0f || copied[1] != 8.0f)
    return 130;
  storage.reset();
  if (TorchStorage::LiveBytes() != before_storage_bytes) return 131;

  std::shared_ptr<Tensor> result;
  ShapeInfo wrong_reshape;
  const int64_t wrong_reshape_dims[1] = {3};
  if (!tensora::ValidateShape(wrong_reshape_dims, 1, &wrong_reshape).ok())
    return 132;
  if (!ExpectStatus(backend.Reshape(*matrix, wrong_reshape, &result),
                    TS_INVALID_SHAPE, "TorchBackend bad reshape"))
    return 133;
  ShapeInfo flat_shape;
  const int64_t flat_dims[1] = {4};
  if (!tensora::ValidateShape(flat_dims, 1, &flat_shape).ok()) return 134;
  if (!ExpectStatus(backend.Reshape(*matrix, flat_shape, &result), TS_OK,
                    "TorchBackend reshape") ||
      !result || result->shape().dimensions != flat_shape.dimensions)
    return 135;

  if (!ExpectStatus(backend.Transpose2D(*result, &wrapped), TS_INVALID_SHAPE,
                    "TorchBackend transpose rank"))
    return 136;
  if (!ExpectStatus(backend.Transpose2D(*matrix, &wrapped), TS_OK,
                    "TorchBackend transpose") ||
      !wrapped || wrapped->shape().dimensions != matrix_shape.dimensions)
    return 137;

  const float other_values[6] = {1, 2, 3, 4, 5, 6};
  std::shared_ptr<Tensor> other;
  if (!ExpectStatus(backend.FromData(other_shape, other_values, &other), TS_OK,
                    "TorchBackend other") ||
      !other)
    return 138;

  if (!ExpectStatus(backend.Add(*matrix, *other, &result), TS_INVALID_SHAPE,
                    "TorchBackend add shape"))
    return 139;
  if (!ExpectStatus(backend.Add(*matrix, *filled, &result), TS_OK,
                    "TorchBackend add") ||
      !result)
    return 140;
  if (!ExpectStatus(backend.Multiply(*matrix, *other, &result),
                    TS_INVALID_SHAPE, "TorchBackend multiply shape"))
    return 141;
  if (!ExpectStatus(backend.Multiply(*matrix, *filled, &result), TS_OK,
                    "TorchBackend multiply") ||
      !result)
    return 142;
  if (!ExpectStatus(backend.Sum(*matrix, &result), TS_OK,
                    "TorchBackend sum") ||
      !result || result->numel() != 1)
    return 143;

  if (!ExpectStatus(backend.Matmul(*result, *matrix, &wrapped),
                    TS_INVALID_SHAPE, "TorchBackend matmul rank"))
    return 144;
  if (!ExpectStatus(backend.Matmul(*other, *other, &wrapped), TS_INVALID_SHAPE,
                    "TorchBackend matmul inner"))
    return 145;

  std::shared_ptr<Tensor> rhs;
  if (!ExpectStatus(backend.FromData(transposed_shape, other_values, &rhs),
                    TS_OK, "TorchBackend matmul rhs") ||
      !rhs)
    return 146;
  if (!ExpectStatus(backend.Matmul(*other, *rhs, &result), TS_OK,
                    "TorchBackend matmul") ||
      !result || result->shape().dimensions != matrix_shape.dimensions)
    return 147;

  if (!ExpectStatus(backend.Reshape(*matrix, flat_shape, nullptr),
                    TS_INVALID_ARGUMENT, "TorchBackend reshape null output"))
    return 155;
  if (!ExpectStatus(backend.Transpose2D(*matrix, nullptr), TS_INVALID_ARGUMENT,
                    "TorchBackend transpose null output"))
    return 156;
  if (!ExpectStatus(backend.Add(*matrix, *filled, nullptr), TS_INVALID_ARGUMENT,
                    "TorchBackend add null output"))
    return 157;
  if (!ExpectStatus(backend.Multiply(*matrix, *filled, nullptr),
                    TS_INVALID_ARGUMENT, "TorchBackend multiply null output"))
    return 158;
  if (!ExpectStatus(backend.Sum(*matrix, nullptr), TS_INVALID_ARGUMENT,
                    "TorchBackend sum null output"))
    return 159;
  if (!ExpectStatus(backend.Matmul(*other, *rhs, nullptr), TS_INVALID_ARGUMENT,
                    "TorchBackend matmul null output"))
    return 160;

  return 0;
}

int CheckTorchIdentityLifetime() {
  using tensora::Tensor;
  using tensora::training::WrapTorchTensor;
  using tensora::training::nn_v2_state::TensorIdentity;
  using tensora::training::nn_v2_state::internal::TorchIdentityCacheSizeForTesting;

  if (TorchIdentityCacheSizeForTesting() != 0) return 210;

  uint64_t first_identity = 0;
  {
    torch::Tensor leaf = torch::tensor(
        {3.0f}, torch::TensorOptions().dtype(torch::kFloat32));
    std::shared_ptr<Tensor> first;
    std::shared_ptr<Tensor> alias;
    if (!ExpectStatus(WrapTorchTensor(leaf, &first), TS_OK,
                      "wrap Torch identity leaf") ||
        !first)
      return 211;
    if (!ExpectStatus(WrapTorchTensor(leaf, &alias), TS_OK,
                      "wrap Torch identity alias") ||
        !alias)
      return 212;

    uint64_t alias_identity = 0;
    if (!ExpectStatus(TensorIdentity(*first, &first_identity), TS_OK,
                      "Torch identity first") ||
        first_identity == 0)
      return 213;
    if (!ExpectStatus(TensorIdentity(*alias, &alias_identity), TS_OK,
                      "Torch identity alias") ||
        alias_identity != first_identity)
      return 214;
    if (TorchIdentityCacheSizeForTesting() != 1) return 215;
  }

  if (TorchIdentityCacheSizeForTesting() != 0) return 216;

  uint64_t second_identity = 0;
  {
    torch::Tensor leaf = torch::tensor(
        {4.0f}, torch::TensorOptions().dtype(torch::kFloat32));
    std::shared_ptr<Tensor> wrapped;
    if (!ExpectStatus(WrapTorchTensor(leaf, &wrapped), TS_OK,
                      "wrap second Torch identity leaf") ||
        !wrapped)
      return 217;
    if (!ExpectStatus(TensorIdentity(*wrapped, &second_identity), TS_OK,
                      "Torch identity second") ||
        second_identity == 0 || second_identity == first_identity)
      return 218;
  }

  if (TorchIdentityCacheSizeForTesting() != 0) return 219;
  return 0;
}

int CheckTorchParameterControl() {
  using tensora::Tensor;
  using tensora::training::WrapTorchTensor;
  using tensora::training::nn_v2_parameter_control::SetRequiresGrad;

  torch::Tensor leaf = torch::tensor(
      {2.0f}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
  std::shared_ptr<Tensor> wrapped;
  if (!ExpectStatus(WrapTorchTensor(leaf, &wrapped), TS_OK,
                    "wrap trainable Torch leaf") ||
      !wrapped)
    return 200;

  if (!ExpectStatus(SetRequiresGrad(*wrapped, true), TS_OK,
                    "Torch unfreeze leaf"))
    return 201;
  leaf.sum().backward();
  if (!leaf.grad().defined()) return 202;

  if (!ExpectStatus(SetRequiresGrad(*wrapped, false), TS_OK,
                    "Torch freeze leaf"))
    return 203;
  if (leaf.requires_grad() || leaf.grad().defined()) return 204;

  if (!ExpectStatus(SetRequiresGrad(*wrapped, true), TS_OK,
                    "Torch re-unfreeze leaf"))
    return 205;
  if (!leaf.requires_grad()) return 206;

  torch::Tensor non_leaf = leaf + leaf;
  std::shared_ptr<Tensor> wrapped_non_leaf;
  if (!ExpectStatus(WrapTorchTensor(non_leaf, &wrapped_non_leaf), TS_OK,
                    "wrap Torch non-leaf") ||
      !wrapped_non_leaf)
    return 207;
  if (!ExpectStatus(SetRequiresGrad(*wrapped_non_leaf, false),
                    TS_INVALID_ARGUMENT, "Torch non-leaf freeze rejected"))
    return 208;
  if (!ExpectStatus(SetRequiresGrad(*wrapped_non_leaf, true),
                    TS_INVALID_ARGUMENT, "Torch non-leaf unfreeze rejected"))
    return 209;

  return 0;
}

}  // namespace

int main() {
  const int backend = CheckDirectTorchBackend();
  if (backend != 0) return backend;
  const int identity = CheckTorchIdentityLifetime();
  if (identity != 0) return identity;
  return CheckTorchParameterControl();
}
