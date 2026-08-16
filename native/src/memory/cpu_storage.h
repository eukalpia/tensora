#ifndef TENSORA_MEMORY_CPU_STORAGE_H_
#define TENSORA_MEMORY_CPU_STORAGE_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

#include "memory/tensor_storage.h"
#include "tensor/dtype.h"

namespace tensora {

class CpuStorage final : public TensorStorage {
 public:
  using Data = std::variant<std::vector<uint16_t>,
                            std::vector<float>,
                            std::vector<double>,
                            std::vector<int8_t>,
                            std::vector<uint8_t>,
                            std::vector<int16_t>,
                            std::vector<int32_t>,
                            std::vector<int64_t>>;

  static Status Filled(uint64_t numel,
                       float value,
                       std::shared_ptr<CpuStorage>* out);
  static Status FromData(const float* data,
                         uint64_t numel,
                         std::shared_ptr<CpuStorage>* out);

  static Status FromRaw(const void* data,
                        size_t data_bytes,
                        uint64_t numel,
                        DType dtype,
                        std::shared_ptr<CpuStorage>* out);
  static Status Full(const void* scalar,
                     size_t scalar_bytes,
                     uint64_t numel,
                     DType dtype,
                     std::shared_ptr<CpuStorage>* out);
  static Status Cast(const CpuStorage& source,
                     DType target_dtype,
                     std::shared_ptr<CpuStorage>* out);

  ~CpuStorage() override;

  CpuStorage(const CpuStorage&) = delete;
  CpuStorage& operator=(const CpuStorage&) = delete;

  StorageKind kind() const override { return StorageKind::kCpu; }
  DType dtype() const { return dtype_; }

  // Compatibility accessors for the existing float32 autograd/training path.
  // Callers must validate dtype() == DType::kFloat32 before using them.
  const std::vector<float>& values() const {
    return std::get<std::vector<float>>(data_);
  }
  std::vector<float>& mutable_values() {
    return std::get<std::vector<float>>(data_);
  }

  uint64_t numel() const { return numel_; }
  uint64_t byte_size() const override { return byte_size_; }

  Status CopyToHostRaw(void* out_data,
                       size_t capacity_bytes,
                       size_t* out_written_bytes) const;
  Status CopyElementTo(uint64_t storage_index,
                       void* out_data,
                       size_t element_bytes) const;
  Status CopyToHostF32(float* out_values,
                       size_t capacity,
                       size_t* out_written) const override;

  static uint64_t LiveBytes();

 private:
  CpuStorage(DType dtype, uint64_t numel, Data data, uint64_t byte_size);

  DType dtype_;
  uint64_t numel_;
  Data data_;
  uint64_t byte_size_;
  static std::atomic<uint64_t> live_bytes_;
};

}  // namespace tensora

#endif  // TENSORA_MEMORY_CPU_STORAGE_H_
