#ifndef TENSORA_CORE_ALLOCATION_GUARD_H_
#define TENSORA_CORE_ALLOCATION_GUARD_H_

#include <new>
#include <stdexcept>
#include <string>
#include <utility>

#include "core/status.h"

namespace tensora {

template <typename Function>
Status AllocationGuard(const char* operation, Function&& function) noexcept {
  try {
    return std::forward<Function>(function)();
  } catch (const std::bad_alloc&) {
    return OutOfMemory(std::string(operation) + ": allocation failed");
  } catch (const std::length_error&) {
    return OutOfMemory(std::string(operation) + ": allocation size is too large");
  }
}

}  // namespace tensora

#endif  // TENSORA_CORE_ALLOCATION_GUARD_H_
