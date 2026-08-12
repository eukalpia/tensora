#include "tensora.h"

#include <cstdint>
#include <iostream>
#include <memory>

#include "runtime/handle_registry.h"

int main() {
  uint64_t wrong_type_handle = 0;
  const auto payload = std::make_shared<int>(42);

  const tensora::Status insert_status =
      tensora::HandleRegistry::Instance().Insert(
          tensora::HandleType::kRuntimeReserved, payload, &wrong_type_handle);
  if (!insert_status.ok() || wrong_type_handle == 0) {
    std::cerr << "failed to create internal wrong-type test handle\n";
    return 1;
  }

  uint64_t numel = 0;
  if (ts_tensor_numel(wrong_type_handle, &numel) != TS_INVALID_HANDLE) {
    std::cerr << "tensor ABI accepted a handle with the wrong object type\n";
    return 2;
  }

  const tensora::Status release_status =
      tensora::HandleRegistry::Instance().Release(
          wrong_type_handle, tensora::HandleType::kRuntimeReserved);
  if (!release_status.ok()) {
    std::cerr << "failed to release internal wrong-type test handle\n";
    return 3;
  }

  if (ts_tensor_numel(wrong_type_handle, &numel) != TS_INVALID_HANDLE) {
    std::cerr << "tensor ABI accepted a stale wrong-type handle\n";
    return 4;
  }

  return 0;
}
