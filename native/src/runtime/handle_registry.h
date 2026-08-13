#ifndef TENSORA_RUNTIME_HANDLE_REGISTRY_H_
#define TENSORA_RUNTIME_HANDLE_REGISTRY_H_

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "core/status.h"

namespace tensora {

enum class HandleType : uint8_t {
  kTensor = 1,
  kRuntimeReserved = 2,
  kModule = 3,
  kOptimizer = 4,
  kInferenceSession = 5,
};

class HandleRegistry {
 public:
  static HandleRegistry& Instance();

  Status Insert(HandleType type,
                std::shared_ptr<void> object,
                uint64_t* out_handle);
  Status Retain(uint64_t handle, HandleType expected_type);
  Status Release(uint64_t handle, HandleType expected_type);

  template <typename T>
  Status Lookup(uint64_t handle,
                HandleType expected_type,
                std::shared_ptr<T>* out) {
    if (out == nullptr) {
      return InvalidArgument("handle registry: lookup output pointer is null");
    }
    if (handle == 0) {
      return InvalidHandle("handle registry: handle 0 is invalid");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = entries_.find(handle);
    if (it == entries_.end()) {
      return InvalidHandle("handle registry: unknown or released handle");
    }
    if (it->second.type != expected_type) {
      return InvalidHandle("handle registry: handle type mismatch");
    }

    *out = std::shared_ptr<T>(it->second.object,
                              static_cast<T*>(it->second.object.get()));
    return Status::Ok();
  }

  uint64_t Count(HandleType type) const;

 private:
  struct Entry {
    HandleType type;
    uint64_t refs;
    std::shared_ptr<void> object;
  };

  HandleRegistry() = default;

  mutable std::mutex mutex_;
  std::unordered_map<uint64_t, Entry> entries_;
  uint64_t next_handle_ = 1;
};

}  // namespace tensora

#endif  // TENSORA_RUNTIME_HANDLE_REGISTRY_H_
