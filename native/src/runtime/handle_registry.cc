#include "runtime/handle_registry.h"

#include <limits>

namespace tensora {

HandleRegistry& HandleRegistry::Instance() {
  static HandleRegistry registry;
  return registry;
}

Status HandleRegistry::Insert(HandleType type,
                              std::shared_ptr<void> object,
                              uint64_t* out_handle) {
  if (out_handle == nullptr) {
    return InvalidArgument("handle registry: output handle pointer is null");
  }
  *out_handle = 0;
  if (!object) {
    return InvalidArgument("handle registry: cannot register null object");
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (next_handle_ == 0 ||
      next_handle_ == std::numeric_limits<uint64_t>::max()) {
    return InternalError("handle registry: handle identifier space exhausted");
  }

  const uint64_t handle = next_handle_++;
  const auto [_, inserted] =
      entries_.emplace(handle, Entry{type, 1, std::move(object)});
  if (!inserted) {
    return InternalError("handle registry: duplicate handle identifier");
  }

  *out_handle = handle;
  return Status::Ok();
}

Status HandleRegistry::Retain(uint64_t handle, HandleType expected_type) {
  if (handle == 0) {
    return InvalidHandle("handle registry: handle 0 is invalid");
  }

  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = entries_.find(handle);
  if (it == entries_.end()) {
    return InvalidHandle("handle registry: cannot retain unknown/released handle");
  }
  if (it->second.type != expected_type) {
    return InvalidHandle("handle registry: handle type mismatch");
  }
  if (it->second.refs == std::numeric_limits<uint64_t>::max()) {
    return InternalError("handle registry: reference count overflow");
  }
  ++it->second.refs;
  return Status::Ok();
}

Status HandleRegistry::Release(uint64_t handle, HandleType expected_type) {
  if (handle == 0) {
    return InvalidHandle("handle registry: handle 0 is invalid");
  }

  std::shared_ptr<void> retired;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = entries_.find(handle);
    if (it == entries_.end()) {
      return InvalidHandle(
          "handle registry: cannot release unknown/released handle");
    }
    if (it->second.type != expected_type) {
      return InvalidHandle("handle registry: handle type mismatch");
    }
    if (it->second.refs == 0) {
      return InternalError("handle registry: corrupted zero reference count");
    }

    --it->second.refs;
    if (it->second.refs == 0) {
      retired = std::move(it->second.object);
      entries_.erase(it);
    }
  }

  return Status::Ok();
}

uint64_t HandleRegistry::Count(HandleType type) const {
  std::lock_guard<std::mutex> lock(mutex_);
  uint64_t count = 0;
  for (const auto& [_, entry] : entries_) {
    if (entry.type == type) {
      ++count;
    }
  }
  return count;
}

}  // namespace tensora
