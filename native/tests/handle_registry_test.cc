#include <cstdint>
#include <iostream>
#include <memory>

#include "runtime/handle_registry.h"
#include "tensor/tensor.h"

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

  std::shared_ptr<tensora::Tensor> tensor;
  const tensora::Status lookup_status =
      tensora::HandleRegistry::Instance().Lookup<tensora::Tensor>(
          wrong_type_handle, tensora::HandleType::kTensor, &tensor);
  if (lookup_status.code() != TS_INVALID_HANDLE || tensor) {
    std::cerr << "handle registry accepted a handle with the wrong object type\n";
    return 2;
  }

  const tensora::Status release_status =
      tensora::HandleRegistry::Instance().Release(
          wrong_type_handle, tensora::HandleType::kRuntimeReserved);
  if (!release_status.ok()) {
    std::cerr << "failed to release internal wrong-type test handle\n";
    return 3;
  }

  const tensora::Status stale_status =
      tensora::HandleRegistry::Instance().Lookup<tensora::Tensor>(
          wrong_type_handle, tensora::HandleType::kTensor, &tensor);
  if (stale_status.code() != TS_INVALID_HANDLE) {
    std::cerr << "handle registry accepted a stale handle\n";
    return 4;
  }

  return 0;
}
