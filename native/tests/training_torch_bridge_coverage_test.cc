#include <cstdint>
#include <iostream>
#include <memory>
#include <utility>

#include <torch/torch.h>

#include "core/status.h"
#include "runtime/handle_registry.h"
#include "tensor/tensor.h"
#include "training/training_bridge.h"

#include "../src/training/training_bridge_torch.cc"

namespace tensora::training {
namespace {

bool ExpectStatus(const Status& status,
                  ts_status_t expected,
                  const char* operation) {
  if (status.code() == expected) return true;
  std::cerr << operation << " expected status " << expected << ", got "
            << status.code() << ": " << status.message() << '\n';
  return false;
}

}  // namespace

int RunBridgeCoverageContracts() {
  if (!ExpectStatus(CreateOptimizerState(nullptr, nullptr, nullptr),
                    TS_INVALID_ARGUMENT, "optimizer null output")) {
    return 1;
  }

  auto state = std::make_shared<LinearState>(1, 1, true);
  state->module->register_buffer(
      "coverage_buffer",
      torch::ones({1}, torch::TensorOptions().dtype(torch::kFloat32)));

  uint64_t module = 0;
  if (!ExpectStatus(HandleRegistry::Instance().Insert(
                        HandleType::kModule, state, &module),
                    TS_OK, "module registry insert") ||
      module == 0) {
    return 2;
  }

  std::shared_ptr<Tensor> buffer;
  if (!ExpectStatus(ModuleBufferAt(module, 0, &buffer), TS_OK,
                    "module buffer conversion") ||
      !buffer || buffer->numel() != 1) {
    HandleRegistry::Instance().Release(module, HandleType::kModule);
    return 3;
  }

  if (!ExpectStatus(HandleRegistry::Instance().Release(
                        module, HandleType::kModule),
                    TS_OK, "module registry release")) {
    return 4;
  }
  return 0;
}

}  // namespace tensora::training

int main() { return tensora::training::RunBridgeCoverageContracts(); }
