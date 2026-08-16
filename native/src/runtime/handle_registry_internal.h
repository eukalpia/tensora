#ifndef TENSORA_RUNTIME_HANDLE_REGISTRY_INTERNAL_H_
#define TENSORA_RUNTIME_HANDLE_REGISTRY_INTERNAL_H_

#include <cstdint>
#include <limits>

#include "core/status.h"

namespace tensora::handle_registry_internal {

inline Status ValidateNextHandle(uint64_t next_handle) {
  if (next_handle == 0 ||
      next_handle == std::numeric_limits<uint64_t>::max()) {
    return InternalError("handle registry: handle identifier space exhausted");
  }
  return Status::Ok();
}

inline Status ValidateInsertion(bool inserted) {
  if (!inserted) {
    return InternalError("handle registry: duplicate handle identifier");
  }
  return Status::Ok();
}

inline Status IncrementReferenceCount(uint64_t& refs) {
  if (refs == std::numeric_limits<uint64_t>::max()) {
    return InternalError("handle registry: reference count overflow");
  }
  ++refs;
  return Status::Ok();
}

inline Status DecrementReferenceCount(uint64_t& refs) {
  if (refs == 0) {
    return InternalError("handle registry: corrupted zero reference count");
  }
  --refs;
  return Status::Ok();
}

}  // namespace tensora::handle_registry_internal

#endif  // TENSORA_RUNTIME_HANDLE_REGISTRY_INTERNAL_H_
