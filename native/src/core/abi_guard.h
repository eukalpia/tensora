#ifndef TENSORA_CORE_ABI_GUARD_H_
#define TENSORA_CORE_ABI_GUARD_H_

#include <exception>
#include <new>
#include <string>
#include <utility>

#include "core/status.h"

namespace tensora {

template <typename Function>
ts_status_t AbiGuard(const char* operation, Function&& function) noexcept {
  ClearLastError();
  try {
    Status status = std::forward<Function>(function)();
    if (!status.ok()) {
      if (status.message().empty()) {
        status = Status(status.code(), std::string(operation) + ": failed");
      }
      SetLastError(status);
    }
    return status.code();
  } catch (const std::bad_alloc&) {
    const Status status =
        OutOfMemory(std::string(operation) + ": native allocation failed");
    SetLastError(status);
    return status.code();
  } catch (const std::exception& error) {
    const Status status =
        InternalError(std::string(operation) + ": " + error.what());
    SetLastError(status);
    return status.code();
  } catch (...) {
    const Status status =
        InternalError(std::string(operation) + ": unknown native exception");
    SetLastError(status);
    return status.code();
  }
}

}  // namespace tensora

#endif  // TENSORA_CORE_ABI_GUARD_H_
