#ifndef TENSORA_TRAINING_NN_V2_STATE_H_
#define TENSORA_TRAINING_NN_V2_STATE_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "autograd/autograd.h"
#include "core/allocation_guard.h"
#include "core/status.h"
#include "memory/tensor_storage.h"
#include "runtime/handle_registry.h"
#include "tensor/tensor.h"

#if defined(TENSORA_WITH_TORCH)
#include <torch/torch.h>

#include "training/torch_backend.h"
#endif

namespace tensora::training::nn_v2_state {
namespace internal {

#if defined(TENSORA_WITH_TORCH)
inline std::mutex& TorchIdentityMutex() {
  static std::mutex mutex;
  return mutex;
}

inline std::unordered_map<uintptr_t, uint64_t>& TorchIdentityMap() {
  static std::unordered_map<uintptr_t, uint64_t> identities;
  return identities;
}

inline uint64_t& NextTorchIdentity() {
  static uint64_t next = UINT64_C(1) << 63U;
  return next;
}

inline Status TorchIdentity(const Tensor& tensor, uint64_t* out_identity) {
  torch::Tensor value;
  Status status = TensorToTorch(tensor, &value);
  if (!status.ok()) return status;
  const uintptr_t key =
      reinterpret_cast<uintptr_t>(value.unsafeGetTensorImpl());
  if (key == 0) {
    return InternalError("tensor_identity: provider returned null tensor identity");
  }

  std::lock_guard<std::mutex> lock(TorchIdentityMutex());
  auto& identities = TorchIdentityMap();
  const auto existing = identities.find(key);
  if (existing != identities.end()) {
    *out_identity = existing->second;
    return Status::Ok();
  }
  uint64_t& next = NextTorchIdentity();
  if (next == 0) {
    return InternalError("tensor_identity: opaque identity space exhausted");
  }
  const uint64_t assigned = next++;
  try {
    identities.emplace(key, assigned);
  } catch (const std::bad_alloc&) {
    return OutOfMemory("tensor_identity: identity registry allocation failed");
  }
  *out_identity = assigned;
  return Status::Ok();
}
#endif

inline Status ValidatePair(const Tensor& target,
                           const Tensor& source,
                           const char* operation) {
  if (!SameShape(target.shape(), source.shape())) {
    return InvalidShape(std::string(operation) + ": tensor shape mismatch");
  }
  if (target.dtype() != source.dtype()) {
    return InvalidArgument(std::string(operation) + ": tensor dtype mismatch");
  }
  if (target.device() != source.device() ||
      target.device_index() != source.device_index()) {
    return InvalidArgument(std::string(operation) + ": tensor device mismatch");
  }
  if (target.storage()->kind() != source.storage()->kind()) {
    return InvalidArgument(std::string(operation) +
                           ": tensor storage backend mismatch");
  }
  return Status::Ok();
}

#if defined(TENSORA_WITH_TORCH)
inline bool RollbackTorch(const std::vector<torch::Tensor>& targets,
                          const std::vector<torch::Tensor>& backups,
                          size_t committed) {
  try {
    torch::NoGradGuard no_grad;
    for (size_t index = 0; index < committed; ++index) {
      targets[index].copy_(backups[index]);
    }
    return true;
  } catch (...) {
    return false;
  }
}
#endif

}  // namespace internal

inline Status TensorIdentity(const Tensor& tensor, uint64_t* out_identity) {
  if (out_identity == nullptr) {
    return InvalidArgument("tensor_identity: output pointer is null");
  }
  *out_identity = 0;
#if defined(TENSORA_WITH_TORCH)
  if (tensor.storage()->kind() == StorageKind::kTorch) {
    return internal::TorchIdentity(tensor, out_identity);
  }
#endif
  *out_identity = tensor.identity();
  if (*out_identity == 0) {
    return InternalError("tensor_identity: opaque identity is zero");
  }
  return Status::Ok();
}

inline Status CloneDetached(const Tensor& tensor, std::shared_ptr<Tensor>* out) {
  if (out == nullptr) {
    return InvalidArgument("tensor_clone_detached: output tensor pointer is null");
  }
  *out = nullptr;
#if defined(TENSORA_WITH_TORCH)
  if (tensor.storage()->kind() == StorageKind::kTorch) {
    torch::Tensor value;
    Status status = TensorToTorch(tensor, &value);
    if (!status.ok()) return status;
    return training::internal::GuardAllocation("tensor_clone_detached", [&]() {
      return training::internal::GuardTorch("tensor_clone_detached", [&]() {
        return WrapTorchTensor(value.detach().clone(), out);
      });
    });
  }
#endif
  return autograd::CloneDetached(tensor, out);
}

inline Status AssignMany(const uint64_t* target_handles,
                         const uint64_t* source_handles,
                         size_t count) {
  if (count == 0) return Status::Ok();
  if (target_handles == nullptr || source_handles == nullptr) {
    return InvalidArgument("tensor_assign_many: tensor arrays are null");
  }

  std::vector<std::shared_ptr<Tensor>> targets;
  std::vector<std::shared_ptr<Tensor>> sources;
  Status status = AllocationGuard("tensor_assign_many", [&]() -> Status {
    targets.reserve(count);
    sources.reserve(count);
    return Status::Ok();
  });
  if (!status.ok()) return status;

  std::unordered_set<uint64_t> target_identities;
  bool have_kind = false;
  StorageKind batch_kind = StorageKind::kCpu;
  for (size_t index = 0; index < count; ++index) {
    std::shared_ptr<Tensor> target;
    std::shared_ptr<Tensor> source;
    status = HandleRegistry::Instance().Lookup<Tensor>(
        target_handles[index], HandleType::kTensor, &target);
    if (!status.ok()) return status;
    status = HandleRegistry::Instance().Lookup<Tensor>(
        source_handles[index], HandleType::kTensor, &source);
    if (!status.ok()) return status;
    status = internal::ValidatePair(*target, *source, "tensor_assign_many");
    if (!status.ok()) return status;

    uint64_t target_identity = 0;
    status = TensorIdentity(*target, &target_identity);
    if (!status.ok()) return status;
    try {
      if (!target_identities.insert(target_identity).second) {
        return InvalidArgument("tensor_assign_many: duplicate target tensor");
      }
    } catch (const std::bad_alloc&) {
      return OutOfMemory("tensor_assign_many: target registry allocation failed");
    }

    const StorageKind kind = target->storage()->kind();
    if (!have_kind) {
      batch_kind = kind;
      have_kind = true;
    } else if (kind != batch_kind) {
      return InvalidArgument(
          "tensor_assign_many: one transaction cannot span storage backends");
    }
    targets.push_back(std::move(target));
    sources.push_back(std::move(source));
  }

  if (batch_kind == StorageKind::kCpu) {
    std::vector<std::vector<float>> staged_sources;
    std::vector<std::vector<float>*> mutable_targets;
    status = AllocationGuard("tensor_assign_many", [&]() -> Status {
      staged_sources.resize(count);
      mutable_targets.resize(count, nullptr);
      return Status::Ok();
    });
    if (!status.ok()) return status;

    for (size_t index = 0; index < count; ++index) {
      status = autograd::ReadLogicalCpuValues(
          *sources[index], "tensor_assign_many", &staged_sources[index]);
      if (!status.ok()) return status;
      status = autograd::MutableCpuValues(
          *targets[index], "tensor_assign_many", &mutable_targets[index]);
      if (!status.ok()) return status;
      if (mutable_targets[index]->size() != staged_sources[index].size()) {
        return InternalError(
            "tensor_assign_many: validated CPU tensor size changed");
      }
    }

    for (size_t index = 0; index < count; ++index) {
      std::copy(staged_sources[index].begin(), staged_sources[index].end(),
                mutable_targets[index]->begin());
      autograd::IncrementVersion(*targets[index]);
      autograd::ClearGradient(*targets[index]);
    }
    return Status::Ok();
  }

#if defined(TENSORA_WITH_TORCH)
  if (batch_kind == StorageKind::kTorch) {
    std::vector<torch::Tensor> torch_targets;
    std::vector<torch::Tensor> staged_sources;
    std::vector<torch::Tensor> backups;
    status = training::internal::GuardAllocation("tensor_assign_many", [&]() {
      torch_targets.reserve(count);
      staged_sources.reserve(count);
      backups.reserve(count);
      for (size_t index = 0; index < count; ++index) {
        torch::Tensor target;
        torch::Tensor source;
        Status conversion = TensorToTorch(*targets[index], &target);
        if (!conversion.ok()) return conversion;
        conversion = TensorToTorch(*sources[index], &source);
        if (!conversion.ok()) return conversion;
        torch_targets.push_back(target);
        Status clone_status = training::internal::GuardTorch(
            "tensor_assign_many", [&]() -> Status {
              staged_sources.push_back(source.detach().clone());
              backups.push_back(target.detach().clone());
              return Status::Ok();
            });
        if (!clone_status.ok()) return clone_status;
      }
      return Status::Ok();
    });
    if (!status.ok()) return status;

    size_t committed = 0;
    try {
      torch::NoGradGuard no_grad;
      for (; committed < count; ++committed) {
        torch_targets[committed].copy_(staged_sources[committed]);
      }
    } catch (const c10::Error& error) {
      const bool rolled_back =
          internal::RollbackTorch(torch_targets, backups, committed);
      if (!rolled_back) {
        return InternalError(
            "tensor_assign_many: provider mutation failed and rollback failed");
      }
      return training::internal::TorchFailure("tensor_assign_many", error);
    } catch (const std::bad_alloc&) {
      const bool rolled_back =
          internal::RollbackTorch(torch_targets, backups, committed);
      if (!rolled_back) {
        return InternalError(
            "tensor_assign_many: allocation failed and rollback failed");
      }
      return OutOfMemory("tensor_assign_many: provider allocation failed");
    }

    for (const auto& target : targets) {
      autograd::IncrementVersion(*target);
    }
    return Status::Ok();
  }
#endif

  return Unsupported("tensor_assign_many: storage backend is unsupported");
}

}  // namespace tensora::training::nn_v2_state

#endif  // TENSORA_TRAINING_NN_V2_STATE_H_
