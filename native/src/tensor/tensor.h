#ifndef TENSORA_TENSOR_TENSOR_H_
#define TENSORA_TENSOR_TENSOR_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "core/status.h"
#include "memory/tensor_storage.h"
#include "tensor/shape.h"

namespace tensora {

struct AutogradMeta;

struct TensorVersionCounter {
  std::atomic<uint64_t> value{0};
};

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

class Tensor : public std::enable_shared_from_this<Tensor> {
 public:
  Tensor(ShapeInfo shape,
         std::shared_ptr<TensorStorage> storage,
         DType dtype = DType::kFloat32,
         Device device = Device::kCpu,
         int32_t device_index = 0,
         uint64_t storage_offset = 0,
         std::shared_ptr<TensorVersionCounter> version_counter = nullptr);

  const ShapeInfo& shape() const { return shape_; }
  const std::shared_ptr<TensorStorage>& storage() const { return storage_; }
  DType dtype() const { return dtype_; }
  Device device() const { return device_; }
  int32_t device_index() const { return device_index_; }
  uint64_t numel() const { return shape_.numel; }
  uint64_t storage_offset() const { return storage_offset_; }

  bool is_contiguous() const;
  uint64_t logical_storage_index(uint64_t logical_index) const;

  const std::shared_ptr<TensorVersionCounter>& version_counter() const {
    return version_counter_;
  }
  uint64_t version() const;
  void increment_version();

  Status CopyToHostF32(float* out_values,
                       size_t capacity,
                       size_t* out_written) const;

  const std::shared_ptr<AutogradMeta>& autograd_meta() const {
    return autograd_meta_;
  }
  void set_autograd_meta(std::shared_ptr<AutogradMeta> meta) {
    autograd_meta_ = std::move(meta);
  }

 private:
  ShapeInfo shape_;
  std::shared_ptr<TensorStorage> storage_;
  DType dtype_;
  Device device_;
  int32_t device_index_;
  uint64_t storage_offset_;
  std::shared_ptr<TensorVersionCounter> version_counter_;
  std::shared_ptr<AutogradMeta> autograd_meta_;
};

}  // namespace tensora

#endif  // TENSORA_TENSOR_TENSOR_H_
