#include "memory/cpu_storage.h"

#include <algorithm>
#include <new>

namespace tensora {

std::atomic<uint64_t> CpuStorage::live_bytes_{0};

CpuStorage::CpuStorage(std::vector<float> values) : values_(std::move(values)) {
  live_bytes_.fetch_add(byte_size(), std::memory_order_relaxed);
}

CpuStorage::~CpuStorage() {
  live_bytes_.fetch_sub(byte_size(), std::memory_order_relaxed);
}

Status CpuStorage::Filled(uint64_t numel,
                          float value,
                          std::shared_ptr<CpuStorage>* out) {
  if (out == nullptr) {
    return InvalidArgument("cpu storage: output pointer is null");
  }
  try {
    std::vector<float> values(static_cast<size_t>(numel), value);
    *out = std::shared_ptr<CpuStorage>(new CpuStorage(std::move(values)));
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return OutOfMemory("cpu storage: allocation failed");
  }
}

Status CpuStorage::FromData(const float* data,
                            uint64_t numel,
                            std::shared_ptr<CpuStorage>* out) {
  if (out == nullptr) {
    return InvalidArgument("cpu storage: output pointer is null");
  }
  if (numel > 0 && data == nullptr) {
    return InvalidArgument("cpu storage: input data pointer is null");
  }
  try {
    std::vector<float> values(static_cast<size_t>(numel));
    std::copy_n(data, static_cast<size_t>(numel), values.begin());
    *out = std::shared_ptr<CpuStorage>(new CpuStorage(std::move(values)));
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return OutOfMemory("cpu storage: allocation failed");
  }
}

uint64_t CpuStorage::LiveBytes() {
  return live_bytes_.load(std::memory_order_relaxed);
}

}  // namespace tensora
