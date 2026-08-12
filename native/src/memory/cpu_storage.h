#ifndef TENSORA_MEMORY_CPU_STORAGE_H_
#define TENSORA_MEMORY_CPU_STORAGE_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/status.h"

namespace tensora {

class CpuStorage {
 public:
  static Status Filled(uint64_t numel,
                       float value,
                       std::shared_ptr<CpuStorage>* out);
  static Status FromData(const float* data,
                         uint64_t numel,
                         std::shared_ptr<CpuStorage>* out);

  ~CpuStorage();

  CpuStorage(const CpuStorage&) = delete;
  CpuStorage& operator=(const CpuStorage&) = delete;

  const std::vector<float>& values() const { return values_; }
  std::vector<float>& mutable_values() { return values_; }
  uint64_t numel() const { return static_cast<uint64_t>(values_.size()); }
  uint64_t byte_size() const {
    return static_cast<uint64_t>(values_.size()) * sizeof(float);
  }

  static uint64_t LiveBytes();

 private:
  explicit CpuStorage(std::vector<float> values);

  std::vector<float> values_;
  static std::atomic<uint64_t> live_bytes_;
};

}  // namespace tensora

#endif  // TENSORA_MEMORY_CPU_STORAGE_H_
