#include "runtime/handle_registry.h"

#include "runtime/handle_registry_internal.h"

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
  Status status = handle_registry_internal::ValidateNextHandle(next_handle_);
  if (!status.ok()) return status;

  const uint64_t handle = next_handle_++;
  const auto [_, inserted] =
      entries_.emplace(handle, Entry{type, 1, std::move(object)});
  status = handle_registry_internal::ValidateInsertion(inserted);
  if (!status.ok()) return status;

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
  return handle_registry_internal::IncrementReferenceCount(it->second.refs);
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
    Status status =
        handle_registry_internal::DecrementReferenceCount(it->second.refs);
    if (!status.ok()) return status;
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
