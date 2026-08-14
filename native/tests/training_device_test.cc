#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include <torch/torch.h>

#include "tensora.h"
#include "tensor/shape.h"
#include "training/torch_backend.h"
#include "training/torch_storage.h"
#include "training/training_bridge.h"

namespace {

bool ExpectStatus(const tensora::Status& status,
                  ts_status_t expected,
                  const char* operation) {
  if (status.code() == expected) return true;
  std::cerr << operation << " expected status " << expected << ", got "
            << status.code() << ": " << status.message() << "\n";
  return false;
}

int CheckCount(uint32_t device, uint32_t expected_minimum) {
  uint32_t count = 999;
  if (ts_runtime_device_count(device, &count) != TS_OK) return 1;
  if (count < expected_minimum) return 2;
  return 0;
}

int CheckAcceleratorTransfer(ts_tensor_t input, uint32_t device) {
  uint32_t count = 0;
  if (ts_runtime_device_count(device, &count) != TS_OK) return 1;

  ts_tensor_t output = 123;
  const ts_status_t status = ts_tensor_to_device(input, device, 0, &output);
  if (count == 0) {
    if (status != TS_UNSUPPORTED) return 2;
    if (output != 0) return 3;
    return 0;
  }

  if (status != TS_OK || output == 0) return 4;
  uint32_t actual_device = 0;
  int32_t actual_index = -1;
  if (ts_tensor_device(output, &actual_device) != TS_OK) return 5;
  if (ts_tensor_device_index(output, &actual_index) != TS_OK) return 6;
  if (actual_device != device || actual_index != 0) return 7;
  if (ts_tensor_release(output) != TS_OK) return 8;
  return 0;
}

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

  std::shared_ptr<Tensor> filled;
  if (!ExpectStatus(backend.Full(matrix_shape, 3.0f, &filled), TS_OK,
                    "TorchBackend Full") ||
      !filled)
    return 118;

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

  return 0;
}

}  // namespace

int main() {
  if (CheckCount(TS_DEVICE_CPU, 1) != 0) return 1;
  if (CheckCount(TS_DEVICE_CUDA, 0) != 0) return 2;
  if (CheckCount(TS_DEVICE_MPS, 0) != 0) return 3;
  if (CheckCount(TS_DEVICE_XPU, 0) != 0) return 4;
  if (CheckCount(TS_DEVICE_HIP, 0) != 0) return 5;

  if (ts_runtime_device_count(TS_DEVICE_CPU, nullptr) != TS_INVALID_ARGUMENT)
    return 6;
  uint32_t unknown_count = 99;
  if (ts_runtime_device_count(0xffffffffu, &unknown_count) != TS_UNSUPPORTED)
    return 7;
  if (unknown_count != 0) return 8;

  const int64_t dims[1] = {1};
  const float value[1] = {2.0f};
  ts_tensor_t input = 0;
  if (ts_tensor_from_f32(value, 1, dims, 1, &input) != TS_OK || input == 0)
    return 10;

  ts_tensor_t cpu_copy = 0;
  if (ts_tensor_to_device(input, TS_DEVICE_CPU, 0, &cpu_copy) != TS_OK ||
      cpu_copy == 0)
    return 11;
  uint32_t cpu_device = 0;
  if (ts_tensor_device(cpu_copy, &cpu_device) != TS_OK ||
      cpu_device != TS_DEVICE_CPU)
    return 12;
  if (ts_tensor_release(cpu_copy) != TS_OK) return 13;

  if (CheckAcceleratorTransfer(input, TS_DEVICE_CUDA) != 0) return 20;
  if (CheckAcceleratorTransfer(input, TS_DEVICE_MPS) != 0) return 21;
  if (CheckAcceleratorTransfer(input, TS_DEVICE_XPU) != 0) return 22;
  if (CheckAcceleratorTransfer(input, TS_DEVICE_HIP) != 0) return 23;

  if (ts_tensor_release(input) != TS_OK) return 30;

  uint64_t live_tensors = 1;
  if (ts_runtime_live_tensor_count(&live_tensors) != TS_OK || live_tensors != 0)
    return 31;

  const int direct_backend = CheckDirectTorchBackend();
  if (direct_backend != 0) return direct_backend;

  return 0;
}
