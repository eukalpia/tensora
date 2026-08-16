#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "core/status.h"
#include "runtime/handle_registry.h"
#include "tensor/tensor.h"
#include "training/nn_v2_optimizer.h"
#include "training/nn_v2_runtime.h"
#include "training/nn_v2_state.h"
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

bool Contains(const Status& status, const char* fragment) {
  return status.message().find(fragment) != std::string::npos;
}

int CheckBridgeErrorPaths() {
  std::shared_ptr<Tensor> non_scalar;
  if (!ExpectStatus(
          WrapTorchTensor(
              torch::ones(
                  {2}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true)),
              &non_scalar),
          TS_OK, "wrap non-scalar backward tensor") ||
      !non_scalar) {
    return 10;
  }
  if (!ExpectStatus(Backward(*non_scalar), TS_INVALID_SHAPE,
                    "Torch backward rejects non-scalar")) {
    return 11;
  }

  std::shared_ptr<Tensor> frozen_scalar;
  if (!ExpectStatus(
          WrapTorchTensor(torch::tensor(1.0f, torch::kFloat32), &frozen_scalar),
          TS_OK, "wrap frozen scalar") ||
      !frozen_scalar) {
    return 12;
  }
  if (!ExpectStatus(Backward(*frozen_scalar), TS_INVALID_ARGUMENT,
                    "Torch backward rejects frozen scalar")) {
    return 13;
  }
  std::shared_ptr<Tensor> gradient;
  if (!ExpectStatus(Gradient(*frozen_scalar, &gradient), TS_INVALID_ARGUMENT,
                    "Torch grad rejects missing gradient")) {
    return 14;
  }

  std::shared_ptr<Tensor> scalar_output;
  if (!ExpectStatus(nn_v2::SwiGlu(*frozen_scalar, &scalar_output),
                    TS_INVALID_SHAPE, "Torch SwiGLU rejects rank zero")) {
    return 15;
  }
  std::shared_ptr<Tensor> odd_width;
  if (!ExpectStatus(
          WrapTorchTensor(torch::ones({3}, torch::kFloat32), &odd_width), TS_OK,
          "wrap odd-width SwiGLU tensor") ||
      !odd_width) {
    return 16;
  }
  if (!ExpectStatus(nn_v2::SwiGlu(*odd_width, &scalar_output), TS_INVALID_SHAPE,
                    "Torch SwiGLU rejects odd final dimension")) {
    return 17;
  }

  if (!ExpectStatus(nn_v2_optimizer::InitializeTorch(nullptr, "coverage"),
                    TS_INVALID_ARGUMENT, "Torch optimizer rejects null state")) {
    return 18;
  }
  return 0;
}

int CheckTorchIdentityExhaustion() {
  using nn_v2_state::TensorIdentity;
  using nn_v2_state::internal::NextTorchIdentity;
  using nn_v2_state::internal::TorchIdentityCacheSizeForTesting;

  if (TorchIdentityCacheSizeForTesting() != 0) return 20;
  std::shared_ptr<Tensor> wrapped;
  if (!ExpectStatus(WrapTorchTensor(torch::ones({1}, torch::kFloat32), &wrapped),
                    TS_OK, "wrap identity exhaustion tensor") ||
      !wrapped) {
    return 21;
  }

  uint64_t& next = NextTorchIdentity();
  const uint64_t saved = next;
  next = 0;
  uint64_t identity = 99;
  const Status status = TensorIdentity(*wrapped, &identity);
  next = saved;
  if (!ExpectStatus(status, TS_INTERNAL_ERROR, "Torch identity exhaustion") ||
      identity != 0) {
    return 22;
  }
  if (TorchIdentityCacheSizeForTesting() != 0) return 23;
  return 0;
}

int CheckTorchTransactionHelpers() {
  using nn_v2_state::internal::CommitTorchAssignments;
  using nn_v2_state::internal::RollbackTorch;

  std::vector<torch::Tensor> rollback_targets = {torch::zeros({1})};
  const std::vector<torch::Tensor> rollback_backups = {torch::ones({1})};
  if (!RollbackTorch(rollback_targets, rollback_backups, 1) ||
      rollback_targets[0].item<float>() != 1.0f) {
    return 30;
  }
  const std::vector<torch::Tensor> invalid_targets = {torch::Tensor()};
  if (RollbackTorch(invalid_targets, rollback_backups, 1)) return 31;

  const std::vector<torch::Tensor> sources = {torch::ones({1}),
                                               torch::full({1}, 2.0f)};
  const std::vector<torch::Tensor> backups = {torch::full({1}, 7.0f),
                                               torch::full({1}, 8.0f)};

  std::vector<torch::Tensor> success_targets = {torch::zeros({1}),
                                                 torch::zeros({1})};
  Status status = CommitTorchAssignments(
      success_targets, sources, backups,
      [](torch::Tensor& target, const torch::Tensor& source, size_t) {
        target.copy_(source);
      });
  if (!ExpectStatus(status, TS_OK, "Torch transaction success") ||
      success_targets[0].item<float>() != 1.0f ||
      success_targets[1].item<float>() != 2.0f) {
    return 32;
  }

  std::vector<torch::Tensor> provider_targets = {torch::zeros({1}),
                                                  torch::zeros({1})};
  status = CommitTorchAssignments(
      provider_targets, sources, backups,
      [](torch::Tensor& target, const torch::Tensor& source, size_t index) {
        if (index == 1) TORCH_CHECK(false, "injected provider failure");
        target.copy_(source);
      });
  if (!ExpectStatus(status, TS_INTERNAL_ERROR,
                    "Torch provider mutation rollback") ||
      provider_targets[0].item<float>() != 7.0f) {
    return 33;
  }

  std::vector<torch::Tensor> provider_rollback_failure = {
      torch::zeros({1}), torch::zeros({1})};
  status = CommitTorchAssignments(
      provider_rollback_failure, sources, backups,
      [&](torch::Tensor& target, const torch::Tensor& source, size_t index) {
        if (index == 1) {
          provider_rollback_failure[0] = torch::Tensor();
          TORCH_CHECK(false, "injected provider rollback failure");
        }
        target.copy_(source);
      });
  if (!ExpectStatus(status, TS_INTERNAL_ERROR,
                    "Torch provider rollback failure") ||
      !Contains(status, "rollback failed")) {
    return 34;
  }

  std::vector<torch::Tensor> allocation_targets = {torch::zeros({1}),
                                                    torch::zeros({1})};
  status = CommitTorchAssignments(
      allocation_targets, sources, backups,
      [](torch::Tensor& target, const torch::Tensor& source, size_t index) {
        if (index == 1) throw std::bad_alloc();
        target.copy_(source);
      });
  if (!ExpectStatus(status, TS_OUT_OF_MEMORY,
                    "Torch allocation mutation rollback") ||
      allocation_targets[0].item<float>() != 7.0f) {
    return 35;
  }

  std::vector<torch::Tensor> allocation_rollback_failure = {
      torch::zeros({1}), torch::zeros({1})};
  status = CommitTorchAssignments(
      allocation_rollback_failure, sources, backups,
      [&](torch::Tensor& target, const torch::Tensor& source, size_t index) {
        if (index == 1) {
          allocation_rollback_failure[0] = torch::Tensor();
          throw std::bad_alloc();
        }
        target.copy_(source);
      });
  if (!ExpectStatus(status, TS_INTERNAL_ERROR,
                    "Torch allocation rollback failure") ||
      !Contains(status, "rollback failed")) {
    return 36;
  }

  return 0;
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
  if (const int code = CheckBridgeErrorPaths(); code != 0) return code;
  if (const int code = CheckTorchIdentityExhaustion(); code != 0) return code;
  if (const int code = CheckTorchTransactionHelpers(); code != 0) return code;
  return 0;
}

}  // namespace tensora::training

int main() { return tensora::training::RunBridgeCoverageContracts(); }
