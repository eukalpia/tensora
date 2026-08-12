#ifndef TENSORA_MEMORY_CPU_STORAGE_H_
#define TENSORA_MEMORY_CPU_STORAGE_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "memory/tensor_storage.h"

namespace tensora {

class CpuStorage final : public TensorStorage {
 public:
  static Status Filled(uint64_t numel,
                       float value,
                       std::shared_ptr<CpuStorage>* out);
  static Status FromData(const float* data,
                         uint64_t numel,
                         std::shared_ptr<CpuStorage>* out);

  ~CpuStorage() override;

  CpuStorage(const CpuStorage&) = delete;
  CpuStorage& operator=(const CpuStorage&) = delete;

  const std::vector<float>& values() const { return values_; }
  std::vector<float>& mutable_values() { return values_; }
  uint64_t numel() const { return static_cast<uint64_t>(values_.size()); }
  uint64_t byte_size() const override {
    return static_cast<uint64_t>(values_.size()) * sizeof(float);
  }

  Status CopyToHostF32(float* out_values,
                       size_t capacity,
                       size_t* out_written) const override;

  static uint64_t LiveBytes();

 private:
  explicit CpuStorage(std::vector<float> values);

  std::vector<float> values_;
  static std::atomic<uint64_t> live_bytes_;
};

}  // namespace tensora

#endif  // TENSORA_MEMORY_CPU_STORAGE_H_
