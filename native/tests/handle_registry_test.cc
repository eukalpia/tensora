#include <cstdint>
#include <iostream>
#include <memory>

#include "runtime/handle_registry.h"
#include "tensor/tensor.h"

namespace {

bool ExpectCode(const tensora::Status& status,
                ts_status_t expected,
                const char* message) {
  if (status.code() == expected) return true;
  std::cerr << message << ": expected status " << expected << ", got "
            << status.code() << "\n";
  return false;
}

}  // namespace

int main() {
  auto& registry = tensora::HandleRegistry::Instance();
  const auto payload = std::make_shared<int>(42);

  uint64_t handle = 123;
  if (!ExpectCode(registry.Insert(tensora::HandleType::kRuntimeReserved,
                                  payload, nullptr),
                  TS_INVALID_ARGUMENT,
                  "insert accepted null output pointer")) {
    return 1;
  }
  if (!ExpectCode(registry.Insert(tensora::HandleType::kRuntimeReserved,
                                  std::shared_ptr<void>(), &handle),
                  TS_INVALID_ARGUMENT, "insert accepted null object")) {
    return 2;
  }
  if (handle != 123) {
    std::cerr << "failed insert unexpectedly mutated output handle\n";
    return 3;
  }

  uint64_t wrong_type_handle = 0;
  const tensora::Status insert_status = registry.Insert(
      tensora::HandleType::kRuntimeReserved, payload, &wrong_type_handle);
  if (!insert_status.ok() || wrong_type_handle == 0) {
    std::cerr << "failed to create internal wrong-type test handle\n";
    return 4;
  }
  if (registry.Count(tensora::HandleType::kRuntimeReserved) != 1 ||
      registry.Count(tensora::HandleType::kTensor) != 0) {
    std::cerr << "registry count did not reflect inserted object type\n";
    return 5;
  }

  std::shared_ptr<tensora::Tensor> tensor;
  if (!ExpectCode(registry.Lookup<tensora::Tensor>(
                      wrong_type_handle, tensora::HandleType::kTensor, nullptr),
                  TS_INVALID_ARGUMENT,
                  "lookup accepted null output pointer")) {
    return 6;
  }
  if (!ExpectCode(registry.Lookup<tensora::Tensor>(
                      0, tensora::HandleType::kTensor, &tensor),
                  TS_INVALID_HANDLE, "lookup accepted handle zero")) {
    return 7;
  }
  if (!ExpectCode(registry.Lookup<tensora::Tensor>(
                      UINT64_C(999999999), tensora::HandleType::kTensor,
                      &tensor),
                  TS_INVALID_HANDLE, "lookup accepted unknown handle")) {
    return 8;
  }

  const tensora::Status lookup_status = registry.Lookup<tensora::Tensor>(
      wrong_type_handle, tensora::HandleType::kTensor, &tensor);
  if (lookup_status.code() != TS_INVALID_HANDLE || tensor) {
    std::cerr << "handle registry accepted a handle with the wrong object type\n";
    return 9;
  }

  std::shared_ptr<int> retained_payload;
  const tensora::Status good_lookup = registry.Lookup<int>(
      wrong_type_handle, tensora::HandleType::kRuntimeReserved,
      &retained_payload);
  if (!good_lookup.ok() || !retained_payload || *retained_payload != 42) {
    std::cerr << "handle registry failed a valid typed lookup\n";
    return 10;
  }

  if (!ExpectCode(registry.Retain(0, tensora::HandleType::kRuntimeReserved),
                  TS_INVALID_HANDLE, "retain accepted handle zero")) {
    return 11;
  }
  if (!ExpectCode(registry.Retain(UINT64_C(999999999),
                                  tensora::HandleType::kRuntimeReserved),
                  TS_INVALID_HANDLE, "retain accepted unknown handle")) {
    return 12;
  }
  if (!ExpectCode(registry.Retain(wrong_type_handle,
                                  tensora::HandleType::kTensor),
                  TS_INVALID_HANDLE, "retain accepted wrong handle type")) {
    return 13;
  }
  if (!registry.Retain(wrong_type_handle,
                       tensora::HandleType::kRuntimeReserved)
           .ok()) {
    std::cerr << "valid retain failed\n";
    return 14;
  }

  if (!ExpectCode(registry.Release(0, tensora::HandleType::kRuntimeReserved),
                  TS_INVALID_HANDLE, "release accepted handle zero")) {
    return 15;
  }
  if (!ExpectCode(registry.Release(UINT64_C(999999999),
                                   tensora::HandleType::kRuntimeReserved),
                  TS_INVALID_HANDLE, "release accepted unknown handle")) {
    return 16;
  }
  if (!ExpectCode(registry.Release(wrong_type_handle,
                                   tensora::HandleType::kTensor),
                  TS_INVALID_HANDLE, "release accepted wrong handle type")) {
    return 17;
  }

  if (!registry.Release(wrong_type_handle,
                        tensora::HandleType::kRuntimeReserved)
           .ok()) {
    std::cerr << "release after retain failed\n";
    return 18;
  }
  if (registry.Count(tensora::HandleType::kRuntimeReserved) != 1) {
    std::cerr << "release after retain prematurely retired object\n";
    return 19;
  }

  const tensora::Status release_status = registry.Release(
      wrong_type_handle, tensora::HandleType::kRuntimeReserved);
  if (!release_status.ok()) {
    std::cerr << "failed to release internal wrong-type test handle\n";
    return 20;
  }
  if (registry.Count(tensora::HandleType::kRuntimeReserved) != 0) {
    std::cerr << "final release did not retire object\n";
    return 21;
  }

  const tensora::Status stale_status = registry.Lookup<tensora::Tensor>(
      wrong_type_handle, tensora::HandleType::kTensor, &tensor);
  if (stale_status.code() != TS_INVALID_HANDLE) {
    std::cerr << "handle registry accepted a stale handle\n";
    return 22;
  }
  if (!ExpectCode(registry.Release(wrong_type_handle,
                                   tensora::HandleType::kRuntimeReserved),
                  TS_INVALID_HANDLE, "double release accepted stale handle")) {
    return 23;
  }

  return 0;
}
