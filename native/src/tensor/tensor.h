#ifndef TENSORA_TENSOR_TENSOR_H_
#define TENSORA_TENSOR_TENSOR_H_

#include <memory>

#include "memory/cpu_storage.h"
#include "tensor/shape.h"

namespace tensora {

enum class DType : uint32_t {
  kFloat32 = TS_DTYPE_FLOAT32,
};

enum class Device : uint32_t {
  kCpu = TS_DEVICE_CPU,
};

class Tensor {
 public:
  Tensor(ShapeInfo shape,
         std::shared_ptr<CpuStorage> storage,
         DType dtype = DType::kFloat32,
         Device device = Device::kCpu);

  const ShapeInfo& shape() const { return shape_; }
  const std::shared_ptr<CpuStorage>& storage() const { return storage_; }
  DType dtype() const { return dtype_; }
  Device device() const { return device_; }
  uint64_t numel() const { return shape_.numel; }

 private:
  ShapeInfo shape_;
  std::shared_ptr<CpuStorage> storage_;
  DType dtype_;
  Device device_;
};

}  // namespace tensora

#endif  // TENSORA_TENSOR_TENSOR_H_
