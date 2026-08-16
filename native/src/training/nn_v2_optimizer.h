#ifndef TENSORA_TRAINING_NN_V2_OPTIMIZER_H_
#define TENSORA_TRAINING_NN_V2_OPTIMIZER_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "autograd/autograd.h"
#include "core/allocation_guard.h"
#include "core/status.h"
#include "memory/tensor_storage.h"
#include "runtime/handle_registry.h"
#include "tensor/tensor.h"
#include "training/training_bridge.h"

#if defined(TENSORA_WITH_TORCH)
#include <torch/torch.h>

#include "training/torch_backend.h"
#endif

namespace tensora::training::nn_v2_optimizer {

enum class Kind : uint8_t { kSgd, kAdam, kAdamW };

struct State {
  Kind kind = Kind::kSgd;
  std::vector<std::shared_ptr<Tensor>> parameters;
  StorageKind storage_kind = StorageKind::kCpu;
  double learning_rate = 0.0;
  double momentum = 0.0;
  double beta1 = 0.0;
  double beta2 = 0.0;
  double epsilon = 0.0;
  double weight_decay = 0.0;
  uint64_t step = 0;
  std::vector<std::vector<float>> first_moment;
  std::vector<std::vector<float>> second_moment;
#if defined(TENSORA_WITH_TORCH)
  std::vector<torch::Tensor> torch_parameters;
  std::unique_ptr<torch::optim::Optimizer> torch_optimizer;
#endif
  std::mutex mutex;
};

inline Status ValidateLearningRate(double value, const char* operation) {
  if (!std::isfinite(value) || value <= 0.0) {
    return InvalidArgument(std::string(operation) +
                           ": learning rate must be finite and positive");
  }
  return Status::Ok();
}

inline Status ValidateWeightDecay(double value, const char* operation) {
  if (!std::isfinite(value) || value < 0.0) {
    return InvalidArgument(std::string(operation) +
                           ": weight decay must be finite and non-negative");
  }
  return Status::Ok();
}

inline Status ValidateMomentum(double value, const char* operation) {
  if (!std::isfinite(value) || value < 0.0) {
    return InvalidArgument(std::string(operation) +
                           ": momentum must be finite and non-negative");
  }
  return Status::Ok();
}

inline Status ValidateBetas(double beta1,
                            double beta2,
                            const char* operation) {
  if (!std::isfinite(beta1) || beta1 < 0.0 || beta1 >= 1.0 ||
      !std::isfinite(beta2) || beta2 < 0.0 || beta2 >= 1.0) {
    return InvalidArgument(std::string(operation) +
                           ": beta values must be finite in [0, 1)");
  }
  return Status::Ok();
}

inline Status ValidateEpsilon(double value, const char* operation) {
  if (!std::isfinite(value) || value <= 0.0) {
    return InvalidArgument(std::string(operation) +
                           ": epsilon must be finite and positive");
  }
  return Status::Ok();
}

inline Status Lookup(uint64_t optimizer, std::shared_ptr<State>* out) {
  return HandleRegistry::Instance().Lookup<State>(
      optimizer, HandleType::kParameterOptimizer, out);
}

inline Status CollectParameters(const uint64_t* handles,
                                size_t count,
                                const char* operation,
                                std::vector<std::shared_ptr<Tensor>>* out,
                                StorageKind* out_kind) {
  if (out == nullptr || out_kind == nullptr) {
    return InvalidArgument(std::string(operation) +
                           ": internal output pointer is null");
  }
  out->clear();
  if (handles == nullptr || count == 0) {
    return InvalidArgument(std::string(operation) +
                           ": parameter collection must not be empty");
  }

  std::unordered_set<Tensor*> seen;
  bool have_kind = false;
  StorageKind kind = StorageKind::kCpu;
  for (size_t index = 0; index < count; ++index) {
    std::shared_ptr<Tensor> parameter;
    Status status = HandleRegistry::Instance().Lookup<Tensor>(
        handles[index], HandleType::kTensor, &parameter);
    if (!status.ok()) return status;
    if (!seen.insert(parameter.get()).second) {
      return InvalidArgument(std::string(operation) +
                             ": duplicate parameter tensor");
    }

    uint8_t requires_grad = 0;
    status = training::RequiresGrad(*parameter, &requires_grad);
    if (!status.ok()) return status;
    if (requires_grad == 0) continue;

    const StorageKind parameter_kind = parameter->storage()->kind();
    if (!have_kind) {
      kind = parameter_kind;
      have_kind = true;
    } else if (parameter_kind != kind) {
      return InvalidArgument(std::string(operation) +
                             ": all trainable parameters must use one backend");
    }
    out->push_back(std::move(parameter));
  }

  if (out->empty()) {
    return InvalidArgument(std::string(operation) +
                           ": no trainable parameters remain after filtering");
  }
  *out_kind = kind;
  return Status::Ok();
}

inline Status InitializeCpuMoments(State* state, const char* operation) {
  if (state == nullptr) {
    return InvalidArgument(std::string(operation) + ": state is null");
  }
  return AllocationGuard(operation, [&]() -> Status {
    state->first_moment.clear();
    state->second_moment.clear();
    state->first_moment.reserve(state->parameters.size());
    state->second_moment.reserve(state->parameters.size());
    for (const auto& parameter : state->parameters) {
      std::vector<float>* values = nullptr;
      Status status =
          autograd::MutableCpuValues(*parameter, operation, &values);
      if (!status.ok()) return status;
      state->first_moment.emplace_back(values->size(), 0.0f);
      state->second_moment.emplace_back(values->size(), 0.0f);
    }
    return Status::Ok();
  });
}

#if defined(TENSORA_WITH_TORCH)
inline Status InitializeTorch(State* state, const char* operation) {
  if (state == nullptr) {
    return InvalidArgument(std::string(operation) + ": state is null");
  }
  return training::internal::GuardAllocation(operation, [&]() {
    state->torch_parameters.clear();
    state->torch_parameters.reserve(state->parameters.size());
    for (const auto& parameter : state->parameters) {
      torch::Tensor value;
      Status status = TensorToTorch(*parameter, &value);
      if (!status.ok()) return status;
      state->torch_parameters.push_back(std::move(value));
    }

    return training::internal::GuardTorch(operation, [&]() {
      if (state->kind == Kind::kSgd) {
        auto options = torch::optim::SGDOptions(state->learning_rate)
                           .momentum(state->momentum)
                           .weight_decay(state->weight_decay);
        state->torch_optimizer = std::make_unique<torch::optim::SGD>(
            state->torch_parameters, options);
      } else if (state->kind == Kind::kAdam) {
        auto options = torch::optim::AdamOptions(state->learning_rate)
                           .betas(std::make_tuple(state->beta1, state->beta2))
                           .eps(state->epsilon)
                           .weight_decay(state->weight_decay);
        state->torch_optimizer = std::make_unique<torch::optim::Adam>(
            state->torch_parameters, options);
      } else {
        auto options = torch::optim::AdamWOptions(state->learning_rate)
                           .betas(std::make_tuple(state->beta1, state->beta2))
                           .eps(state->epsilon)
                           .weight_decay(state->weight_decay);
        state->torch_optimizer = std::make_unique<torch::optim::AdamW>(
            state->torch_parameters, options);
      }
      return Status::Ok();
    });
  });
}
#endif

inline Status Create(const uint64_t* handles,
                     size_t count,
                     Kind kind,
                     double learning_rate,
                     double momentum,
                     double beta1,
                     double beta2,
                     double epsilon,
                     double weight_decay,
                     uint64_t* out_optimizer,
                     const char* operation) {
  if (out_optimizer == nullptr) {
    return InvalidArgument(std::string(operation) +
                           ": output handle pointer is null");
  }
  *out_optimizer = 0;
  Status status = ValidateLearningRate(learning_rate, operation);
  if (!status.ok()) return status;
  status = ValidateWeightDecay(weight_decay, operation);
  if (!status.ok()) return status;
  if (kind == Kind::kSgd) {
    status = ValidateMomentum(momentum, operation);
  } else {
    status = ValidateBetas(beta1, beta2, operation);
    if (status.ok()) status = ValidateEpsilon(epsilon, operation);
  }
  if (!status.ok()) return status;

  std::vector<std::shared_ptr<Tensor>> parameters;
  StorageKind storage_kind = StorageKind::kCpu;
  status = CollectParameters(handles, count, operation, &parameters,
                             &storage_kind);
  if (!status.ok()) return status;

  std::shared_ptr<State> state;
  status = AllocationGuard(operation, [&]() -> Status {
    state = std::make_shared<State>();
    state->kind = kind;
    state->parameters = std::move(parameters);
    state->storage_kind = storage_kind;
    state->learning_rate = learning_rate;
    state->momentum = momentum;
    state->beta1 = beta1;
    state->beta2 = beta2;
    state->epsilon = epsilon;
    state->weight_decay = weight_decay;
    return Status::Ok();
  });
  if (!status.ok()) return status;

  if (storage_kind == StorageKind::kCpu) {
    status = InitializeCpuMoments(state.get(), operation);
#if defined(TENSORA_WITH_TORCH)
  } else if (storage_kind == StorageKind::kTorch) {
    status = InitializeTorch(state.get(), operation);
#endif
  } else {
    status = Unsupported(std::string(operation) +
                         ": parameter storage backend is unsupported");
  }
  if (!status.ok()) return status;

  return HandleRegistry::Instance().Insert(
      HandleType::kParameterOptimizer, std::move(state), out_optimizer);
}

inline Status SgdCreate(const uint64_t* handles,
                        size_t count,
                        double learning_rate,
                        double momentum,
                        double weight_decay,
                        uint64_t* out_optimizer) {
  return Create(handles, count, Kind::kSgd, learning_rate, momentum, 0.0, 0.0,
                0.0, weight_decay, out_optimizer,
                "sgd_create_for_tensors");
}

inline Status AdamCreate(const uint64_t* handles,
                         size_t count,
                         double learning_rate,
                         double beta1,
                         double beta2,
                         double epsilon,
                         double weight_decay,
                         uint64_t* out_optimizer) {
  return Create(handles, count, Kind::kAdam, learning_rate, 0.0, beta1, beta2,
                epsilon, weight_decay, out_optimizer,
                "adam_create_for_tensors");
}

inline Status AdamWCreate(const uint64_t* handles,
                          size_t count,
                          double learning_rate,
                          double beta1,
                          double beta2,
                          double epsilon,
                          double weight_decay,
                          uint64_t* out_optimizer) {
  return Create(handles, count, Kind::kAdamW, learning_rate, 0.0, beta1,
                beta2, epsilon, weight_decay, out_optimizer,
                "adamw_create_for_tensors");
}

inline Status ZeroGrad(uint64_t optimizer) {
  std::shared_ptr<State> state;
  Status status = Lookup(optimizer, &state);
  if (!status.ok()) return status;
  std::lock_guard<std::mutex> lock(state->mutex);
#if defined(TENSORA_WITH_TORCH)
  if (state->storage_kind == StorageKind::kTorch) {
    return training::internal::GuardTorch("parameter_optimizer_zero_grad", [&]() {
      state->torch_optimizer->zero_grad();
      return Status::Ok();
    });
  }
#endif
  for (const auto& parameter : state->parameters) {
    autograd::ClearGradient(*parameter);
  }
  return Status::Ok();
}

inline Status CpuStep(State* state) {
  ++state->step;
  for (size_t parameter_index = 0;
       parameter_index < state->parameters.size(); ++parameter_index) {
    const auto& parameter = state->parameters[parameter_index];
    const auto meta = parameter->autograd_meta();
    if (!meta) continue;

    std::shared_ptr<Tensor> gradient;
    {
      std::lock_guard<std::mutex> gradient_lock(meta->mutex);
      gradient = meta->gradient;
    }
    if (!gradient) continue;

    std::vector<float>* values = nullptr;
    std::vector<float> gradient_values;
    Status status = autograd::MutableCpuValues(
        *parameter, "parameter_optimizer_step", &values);
    if (!status.ok()) return status;
    status = autograd::ReadLogicalCpuValues(
        *gradient, "parameter_optimizer_step", &gradient_values);
    if (!status.ok()) return status;
    if (values->size() != gradient_values.size()) {
      return InternalError("parameter_optimizer_step: gradient size mismatch");
    }

    auto& first = state->first_moment[parameter_index];
    if (state->kind == Kind::kSgd) {
      for (size_t index = 0; index < values->size(); ++index) {
        double grad_value = static_cast<double>(gradient_values[index]) +
                            state->weight_decay *
                                static_cast<double>((*values)[index]);
        if (state->momentum > 0.0) {
          first[index] = static_cast<float>(
              state->momentum * static_cast<double>(first[index]) + grad_value);
          grad_value = first[index];
        }
        (*values)[index] = static_cast<float>(
            static_cast<double>((*values)[index]) -
            state->learning_rate * grad_value);
      }
    } else {
      auto& second = state->second_moment[parameter_index];
      const double beta1_power =
          std::pow(state->beta1, static_cast<double>(state->step));
      const double beta2_power =
          std::pow(state->beta2, static_cast<double>(state->step));
      for (size_t index = 0; index < values->size(); ++index) {
        double parameter_value = static_cast<double>((*values)[index]);
        double grad_value = static_cast<double>(gradient_values[index]);
        if (state->kind == Kind::kAdam) {
          grad_value += state->weight_decay * parameter_value;
        } else if (state->weight_decay > 0.0) {
          parameter_value *=
              1.0 - state->learning_rate * state->weight_decay;
        }

        const double first_value =
            state->beta1 * static_cast<double>(first[index]) +
            (1.0 - state->beta1) * grad_value;
        const double second_value =
            state->beta2 * static_cast<double>(second[index]) +
            (1.0 - state->beta2) * grad_value * grad_value;
        first[index] = static_cast<float>(first_value);
        second[index] = static_cast<float>(second_value);

        const double first_hat = first_value / (1.0 - beta1_power);
        const double second_hat = second_value / (1.0 - beta2_power);
        parameter_value -=
            state->learning_rate * first_hat /
            (std::sqrt(second_hat) + state->epsilon);
        (*values)[index] = static_cast<float>(parameter_value);
      }
    }
    autograd::IncrementVersion(*parameter);
  }
  return Status::Ok();
}

inline Status Step(uint64_t optimizer) {
  std::shared_ptr<State> state;
  Status status = Lookup(optimizer, &state);
  if (!status.ok()) return status;
  std::lock_guard<std::mutex> lock(state->mutex);
#if defined(TENSORA_WITH_TORCH)
  if (state->storage_kind == StorageKind::kTorch) {
    status = training::internal::GuardTorch("parameter_optimizer_step", [&]() {
      state->torch_optimizer->step();
      return Status::Ok();
    });
    if (!status.ok()) return status;
    for (size_t index = 0; index < state->parameters.size(); ++index) {
      if (state->torch_parameters[index].grad().defined()) {
        autograd::IncrementVersion(*state->parameters[index]);
      }
    }
    return Status::Ok();
  }
#endif
  return CpuStep(state.get());
}

inline Status Release(uint64_t optimizer) {
  return HandleRegistry::Instance().Release(
      optimizer, HandleType::kParameterOptimizer);
}

}  // namespace tensora::training::nn_v2_optimizer

#endif  // TENSORA_TRAINING_NN_V2_OPTIMIZER_H_
