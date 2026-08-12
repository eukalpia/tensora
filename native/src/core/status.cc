#include "core/status.h"

namespace tensora {
namespace {
thread_local std::string g_last_error;
}  // namespace

Status InvalidArgument(std::string message) {
  return Status(TS_INVALID_ARGUMENT, std::move(message));
}

Status InvalidShape(std::string message) {
  return Status(TS_INVALID_SHAPE, std::move(message));
}

Status OutOfMemory(std::string message) {
  return Status(TS_OUT_OF_MEMORY, std::move(message));
}

Status Unsupported(std::string message) {
  return Status(TS_UNSUPPORTED, std::move(message));
}

Status InvalidHandle(std::string message) {
  return Status(TS_INVALID_HANDLE, std::move(message));
}

Status InternalError(std::string message) {
  return Status(TS_INTERNAL_ERROR, std::move(message));
}

void ClearLastError() { g_last_error.clear(); }

void SetLastError(const Status& status) {
  g_last_error = status.ok() ? std::string() : status.message();
}

const char* LastErrorMessage() { return g_last_error.c_str(); }

}  // namespace tensora
