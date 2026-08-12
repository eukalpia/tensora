#include "runtime/dispatcher.h"

#include "backends/cpu/cpu_backend.h"

namespace tensora {

Status Dispatcher::For(Device device, const Backend** out) {
  if (out == nullptr) {
    return InvalidArgument("dispatcher: output backend pointer is null");
  }

  static const CpuBackend cpu_backend;
  if (device == Device::kCpu) {
    *out = &cpu_backend;
    return Status::Ok();
  }

  return Unsupported("dispatcher: requested device is not supported");
}

}  // namespace tensora
