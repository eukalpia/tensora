#ifndef TENSORA_TENSOR_TENSOR_H_
#define TENSORA_TENSOR_TENSOR_H_

#include <cstdint>
#include <memory>

#include "memory/tensor_storage.h"
#include "tensor/shape.h"

namespace tensora {

enum class DType : uint32_t {
  kFloat32 = TS_DTYPE_FLOAT32,
};

enum class Device : uint32_t {
  kCpu = TS_DEVICE_CPU,
  kCuda = TS_DEVICE_CUDA,
  kMps = TS_DEVICE_MPS,
  kXpu = TS_DEVICE_XPU,
  kHip = TS_DEVICE_HIP,
};

class Tensor {
 public:
  Tensor(ShapeInfo shape,
         std::shared_ptr<TensorStorage> storage,
         DType dtype = DType::kFloat32,
         Device device = Device::kCpu,
         int32_t device_index = 0);

  const ShapeInfo& shape() const { return shape_; }
  const std::shared_ptr<TensorStorage>& storage() const { return storage_; }
  DType dtype() const { return dtype_; }
  Device device() const { return device_; }
  int32_t device_index() const { return device_index_; }
  uint64_t numel() const { return shape_.numel; }

 private:
  ShapeInfo shape_;
  std::shared_ptr<TensorStorage> storage_;
  DType dtype_;
  Device device_;
  int32_t device_index_;
};

}  // namespace tensora

#endif  // TENSORA_TENSOR_TENSOR_H_