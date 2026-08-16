#ifndef TENSORA_TRAINING_NN_V2_PARAMETER_CONTROL_H_
#define TENSORA_TRAINING_NN_V2_PARAMETER_CONTROL_H_

#include <memory>
#include <mutex>

#include "autograd/autograd.h"
#include "core/allocation_guard.h"
#include "core/status.h"
#include "memory/tensor_storage.h"
#include "tensor/tensor.h"

#if defined(TENSORA_WITH_TORCH)
#include <torch/torch.h>

#include "training/torch_backend.h"
#endif

namespace tensora::training::nn_v2_parameter_control {
namespace internal {

#if defined(TENSORA_WITH_TORCH)
inline Status SetTorchRequiresGrad(Tensor& tensor, bool requires_grad) {
  torch::Tensor value;
  Status status = TensorToTorch(tensor, &value);
  if (!status.ok()) return status;
  return training::internal::GuardTorch("tensor_set_requires_grad", [&]() {
    if (!value.is_leaf()) {
      return InvalidArgument(
          "tensor_set_requires_grad: only leaf tensors may change requiresGrad");
    }
    value.set_requires_grad(requires_grad);
    if (!requires_grad && value.grad().defined()) {
      value.mutable_grad() = torch::Tensor();
    }
    return Status::Ok();
  });
}
#endif

inline Status SetCoreRequiresGrad(Tensor& tensor, bool requires_grad) {
  auto meta = tensor.autograd_meta();
  if (!meta) {
    if (!requires_grad) return Status::Ok();
    return AllocationGuard("tensor_set_requires_grad", [&]() -> Status {
      auto created = std::make_shared<AutogradMeta>();
      created->requires_grad = true;
      created->is_leaf = true;
      tensor.set_autograd_meta(std::move(created));
      return Status::Ok();
    });
  }

  std::lock_guard<std::mutex> lock(meta->mutex);
  if (!meta->is_leaf) {
    return InvalidArgument(
        "tensor_set_requires_grad: only leaf tensors may change requiresGrad");
  }
  meta->requires_grad = requires_grad;
  if (!requires_grad) {
    meta->gradient.reset();
  }
  return Status::Ok();
}

}  // namespace internal

inline Status SetRequiresGrad(Tensor& tensor, bool requires_grad) {
#if defined(TENSORA_WITH_TORCH)
  if (tensor.storage()->kind() == StorageKind::kTorch) {
    return internal::SetTorchRequiresGrad(tensor, requires_grad);
  }
#endif
  return internal::SetCoreRequiresGrad(tensor, requires_grad);
}

}  // namespace tensora::training::nn_v2_parameter_control

#endif  // TENSORA_TRAINING_NN_V2_PARAMETER_CONTROL_H_
