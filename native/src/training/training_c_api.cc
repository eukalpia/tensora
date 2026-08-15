#include "tensora.h"

#include <exception>
#include <memory>
#include <new>
#include <string>

#include "core/abi_guard.h"
#include "core/status.h"
#include "runtime/device_codec.h"
#include "runtime/handle_registry.h"
#include "tensor/tensor.h"
#include "training/nn_v2_optimizer.h"
#include "training/nn_v2_runtime.h"
#include "training/nn_v2_state.h"
#include "training/training_bridge.h"

namespace tensora {
namespace {

Status LookupTrainingTensor(ts_tensor_t handle, std::shared_ptr<Tensor>* out) {
  return HandleRegistry::Instance().Lookup<Tensor>(
      handle, HandleType::kTensor, out);
}

Status InsertTrainingTensor(std::shared_ptr<Tensor> tensor,
                            ts_tensor_t* out_handle) {
  if (out_handle == nullptr) {
    return InvalidArgument("training tensor: output handle pointer is null");
  }
  *out_handle = 0;
  return HandleRegistry::Instance().Insert(
      HandleType::kTensor, std::move(tensor), out_handle);
}

template <typename Fetch>
Status FetchAndInsertTrainingTensor(ts_tensor_t* out_tensor,
                                    const char* operation,
                                    Fetch&& fetch) {
  if (out_tensor == nullptr) {
    return InvalidArgument(std::string(operation) +
                           ": output handle pointer is null");
  }
  *out_tensor = 0;
  std::shared_ptr<Tensor> result;
  Status status = fetch(&result);
  if (!status.ok()) return status;
  return InsertTrainingTensor(std::move(result), out_tensor);
}

template <typename Unary>
Status RunTrainingUnary(ts_tensor_t tensor,
                        ts_tensor_t* out_tensor,
                        Unary&& operation) {
  if (out_tensor == nullptr) {
    return InvalidArgument("training unary: output handle pointer is null");
  }
  *out_tensor = 0;
  std::shared_ptr<Tensor> object;
  Status status = LookupTrainingTensor(tensor, &object);
  if (!status.ok()) return status;
  std::shared_ptr<Tensor> result;
  status = operation(*object, &result);
  if (!status.ok()) return status;
  return InsertTrainingTensor(std::move(result), out_tensor);
}

template <typename Binary>
Status RunTrainingBinary(ts_tensor_t left,
                         ts_tensor_t right,
                         ts_tensor_t* out_tensor,
                         Binary&& operation) {
  if (out_tensor == nullptr) {
    return InvalidArgument("training binary: output handle pointer is null");
  }
  *out_tensor = 0;
  std::shared_ptr<Tensor> left_object;
  std::shared_ptr<Tensor> right_object;
  Status status = LookupTrainingTensor(left, &left_object);
  if (!status.ok()) return status;
  status = LookupTrainingTensor(right, &right_object);
  if (!status.ok()) return status;
  std::shared_ptr<Tensor> result;
  status = operation(*left_object, *right_object, &result);
  if (!status.ok()) return status;
  return InsertTrainingTensor(std::move(result), out_tensor);
}

}  // namespace
}  // namespace tensora

extern "C" {

ts_status_t ts_training_available(uint8_t* out_available) {
  return tensora::AbiGuard("training_available", [&] {
    return tensora::training::IsAvailable(out_available);
  });
}

ts_status_t ts_manual_seed(uint64_t seed) {
  return tensora::AbiGuard("manual_seed", [&] {
    return tensora::training::ManualSeed(seed);
  });
}

ts_status_t ts_tensor_identity(ts_tensor_t tensor, uint64_t* out_identity) {
  return tensora::AbiGuard("tensor_identity", [&] {
    if (out_identity == nullptr) {
      return tensora::InvalidArgument(
          "tensor_identity: output identity pointer is null");
    }
    *out_identity = 0;
    std::shared_ptr<tensora::Tensor> object;
    tensora::Status status = tensora::LookupTrainingTensor(tensor, &object);
    if (!status.ok()) return status;
    return tensora::training::nn_v2_state::TensorIdentity(
        *object, out_identity);
  });
}

ts_status_t ts_tensor_clone_detached(ts_tensor_t tensor,
                                     ts_tensor_t* out_tensor) {
  return tensora::AbiGuard("tensor_clone_detached", [&] {
    return tensora::RunTrainingUnary(
        tensor, out_tensor,
        [](const tensora::Tensor& value,
           std::shared_ptr<tensora::Tensor>* out) {
          return tensora::training::nn_v2_state::CloneDetached(value, out);
        });
  });
}

ts_status_t ts_tensor_assign_many(const ts_tensor_t* targets,
                                  const ts_tensor_t* sources,
                                  size_t count) {
  return tensora::AbiGuard("tensor_assign_many", [&] {
    return tensora::training::nn_v2_state::AssignMany(targets, sources, count);
  });
}

ts_status_t ts_tensor_with_requires_grad(ts_tensor_t tensor,
                                         uint8_t requires_grad,
                                         ts_tensor_t* out_tensor) {
  return tensora::AbiGuard("tensor_with_requires_grad", [&] {
    return tensora::RunTrainingUnary(
        tensor, out_tensor,
        [&](const tensora::Tensor& value,
            std::shared_ptr<tensora::Tensor>* out) {
          return tensora::training::WithRequiresGrad(
              value, requires_grad != 0, out);
        });
  });
}

ts_status_t ts_tensor_requires_grad(ts_tensor_t tensor,
                                    uint8_t* out_requires_grad) {
  return tensora::AbiGuard("tensor_requires_grad", [&] {
    if (out_requires_grad == nullptr) {
      return tensora::InvalidArgument(
          "tensor_requires_grad: output pointer is null");
    }
    *out_requires_grad = 0;
    std::shared_ptr<tensora::Tensor> object;
    tensora::Status status = tensora::LookupTrainingTensor(tensor, &object);
    if (!status.ok()) return status;
    return tensora::training::RequiresGrad(*object, out_requires_grad);
  });
}

ts_status_t ts_tensor_backward(ts_tensor_t tensor) {
  return tensora::AbiGuard("tensor_backward", [&] {
    std::shared_ptr<tensora::Tensor> object;
    tensora::Status status = tensora::LookupTrainingTensor(tensor, &object);
    if (!status.ok()) return status;
    return tensora::training::Backward(*object);
  });
}

ts_status_t ts_tensor_grad(ts_tensor_t tensor, ts_tensor_t* out_tensor) {
  return tensora::AbiGuard("tensor_grad", [&] {
    return tensora::RunTrainingUnary(
        tensor, out_tensor,
        [](const tensora::Tensor& value,
           std::shared_ptr<tensora::Tensor>* out) {
          return tensora::training::Gradient(value, out);
        });
  });
}

ts_status_t ts_tensor_relu(ts_tensor_t tensor, ts_tensor_t* out_tensor) {
  return tensora::AbiGuard("tensor_relu", [&] {
    return tensora::RunTrainingUnary(
        tensor, out_tensor,
        [](const tensora::Tensor& value,
           std::shared_ptr<tensora::Tensor>* out) {
          return tensora::training::Relu(value, out);
        });
  });
}

ts_status_t ts_tensor_sigmoid(ts_tensor_t tensor, ts_tensor_t* out_tensor) {
  return tensora::AbiGuard("tensor_sigmoid", [&] {
    return tensora::RunTrainingUnary(
        tensor, out_tensor,
        [](const tensora::Tensor& value,
           std::shared_ptr<tensora::Tensor>* out) {
          return tensora::training::Sigmoid(value, out);
        });
  });
}

ts_status_t ts_tensor_tanh(ts_tensor_t tensor, ts_tensor_t* out_tensor) {
  return tensora::AbiGuard("tensor_tanh", [&] {
    return tensora::RunTrainingUnary(
        tensor, out_tensor,
        [](const tensora::Tensor& value,
           std::shared_ptr<tensora::Tensor>* out) {
          return tensora::training::Tanh(value, out);
        });
  });
}

ts_status_t ts_tensor_gelu(ts_tensor_t tensor, ts_tensor_t* out_tensor) {
  return tensora::AbiGuard("tensor_gelu", [&] {
    return tensora::RunTrainingUnary(
        tensor, out_tensor,
        [](const tensora::Tensor& value,
           std::shared_ptr<tensora::Tensor>* out) {
          return tensora::training::nn_v2::Gelu(value, out);
        });
  });
}

ts_status_t ts_tensor_silu(ts_tensor_t tensor, ts_tensor_t* out_tensor) {
  return tensora::AbiGuard("tensor_silu", [&] {
    return tensora::RunTrainingUnary(
        tensor, out_tensor,
        [](const tensora::Tensor& value,
           std::shared_ptr<tensora::Tensor>* out) {
          return tensora::training::nn_v2::Silu(value, out);
        });
  });
}

ts_status_t ts_tensor_swiglu(ts_tensor_t tensor, ts_tensor_t* out_tensor) {
  return tensora::AbiGuard("tensor_swiglu", [&] {
    return tensora::RunTrainingUnary(
        tensor, out_tensor,
        [](const tensora::Tensor& value,
           std::shared_ptr<tensora::Tensor>* out) {
          return tensora::training::nn_v2::SwiGlu(value, out);
        });
  });
}

ts_status_t ts_mse_loss(ts_tensor_t prediction,
                        ts_tensor_t target,
                        ts_tensor_t* out_tensor) {
  return tensora::AbiGuard("mse_loss", [&] {
    return tensora::RunTrainingBinary(
        prediction, target, out_tensor,
        [](const tensora::Tensor& left,
           const tensora::Tensor& right,
           std::shared_ptr<tensora::Tensor>* out) {
          return tensora::training::MseLoss(left, right, out);
        });
  });
}

ts_status_t ts_cross_entropy_loss(ts_tensor_t logits,
                                  ts_tensor_t one_hot_target,
                                  ts_tensor_t* out_tensor) {
  return tensora::AbiGuard("cross_entropy_loss", [&] {
    return tensora::RunTrainingBinary(
        logits, one_hot_target, out_tensor,
        [](const tensora::Tensor& left,
           const tensora::Tensor& right,
           std::shared_ptr<tensora::Tensor>* out) {
          return tensora::training::CrossEntropyLoss(left, right, out);
        });
  });
}

ts_status_t ts_linear_create(int64_t in_features,
                             int64_t out_features,
                             uint8_t use_bias,
                             ts_module_t* out_module) {
  return tensora::AbiGuard("linear_create", [&] {
    if (out_module == nullptr) {
      return tensora::InvalidArgument(
          "linear_create: output handle pointer is null");
    }
    *out_module = 0;
    return tensora::training::LinearCreate(
        in_features, out_features, use_bias != 0, out_module);
  });
}

ts_status_t ts_module_forward(ts_module_t module,
                              ts_tensor_t input,
                              ts_tensor_t* out_tensor) {
  return tensora::AbiGuard("module_forward", [&] {
    if (out_tensor == nullptr) {
      return tensora::InvalidArgument(
          "module_forward: output handle pointer is null");
    }
    *out_tensor = 0;
    std::shared_ptr<tensora::Tensor> input_object;
    tensora::Status status =
        tensora::LookupTrainingTensor(input, &input_object);
    if (!status.ok()) return status;
    std::shared_ptr<tensora::Tensor> result;
    status = tensora::training::ModuleForward(module, *input_object, &result);
    if (!status.ok()) return status;
    return tensora::InsertTrainingTensor(std::move(result), out_tensor);
  });
}

ts_status_t ts_module_set_training(ts_module_t module, uint8_t training) {
  return tensora::AbiGuard("module_set_training", [&] {
    return tensora::training::ModuleSetTraining(module, training != 0);
  });
}

ts_status_t ts_module_to_device(ts_module_t module,
                                uint32_t device,
                                int32_t device_index) {
  return tensora::AbiGuard("module_to_device", [&] {
    tensora::Device target = tensora::Device::kCpu;
    tensora::Status status = tensora::DeviceFromCode(device, &target);
    if (!status.ok()) return status;
    return tensora::training::ModuleToDevice(module, target, device_index);
  });
}

ts_status_t ts_module_parameter_count(ts_module_t module, size_t* out_count) {
  return tensora::AbiGuard("module_parameter_count", [&] {
    return tensora::training::ModuleParameterCount(module, out_count);
  });
}

ts_status_t ts_module_parameter_at(ts_module_t module,
                                   size_t index,
                                   ts_tensor_t* out_tensor) {
  return tensora::AbiGuard("module_parameter_at", [&] {
    return tensora::FetchAndInsertTrainingTensor(
        out_tensor, "module_parameter_at",
        [&](std::shared_ptr<tensora::Tensor>* out) {
          return tensora::training::ModuleParameterAt(module, index, out);
        });
  });
}

ts_status_t ts_module_buffer_count(ts_module_t module, size_t* out_count) {
  return tensora::AbiGuard("module_buffer_count", [&] {
    return tensora::training::ModuleBufferCount(module, out_count);
  });
}

ts_status_t ts_module_buffer_at(ts_module_t module,
                                size_t index,
                                ts_tensor_t* out_tensor) {
  return tensora::AbiGuard("module_buffer_at", [&] {
    return tensora::FetchAndInsertTrainingTensor(
        out_tensor, "module_buffer_at",
        [&](std::shared_ptr<tensora::Tensor>* out) {
          return tensora::training::ModuleBufferAt(module, index, out);
        });
  });
}

ts_status_t ts_module_save(ts_module_t module, const char* path) {
  return tensora::AbiGuard("module_save", [&] {
    if (path == nullptr) {
      return tensora::InvalidArgument("module_save: path pointer is null");
    }
    return tensora::training::ModuleSave(module, std::string(path));
  });
}

ts_status_t ts_module_load(ts_module_t module, const char* path) {
  return tensora::AbiGuard("module_load", [&] {
    if (path == nullptr) {
      return tensora::InvalidArgument("module_load: path pointer is null");
    }
    return tensora::training::ModuleLoad(module, std::string(path));
  });
}

ts_status_t ts_module_release(ts_module_t module) {
  return tensora::AbiGuard("module_release", [&] {
    return tensora::training::ModuleRelease(module);
  });
}

ts_status_t ts_sgd_create(ts_module_t module,
                          double learning_rate,
                          double momentum,
                          double weight_decay,
                          ts_optimizer_t* out_optimizer) {
  return tensora::AbiGuard("sgd_create", [&] {
    return tensora::training::SgdCreate(
        module, learning_rate, momentum, weight_decay, out_optimizer);
  });
}

ts_status_t ts_adam_create(ts_module_t module,
                           double learning_rate,
                           double beta1,
                           double beta2,
                           double epsilon,
                           double weight_decay,
                           ts_optimizer_t* out_optimizer) {
  return tensora::AbiGuard("adam_create", [&] {
    return tensora::training::AdamCreate(module, learning_rate, beta1, beta2,
                                         epsilon, weight_decay, out_optimizer);
  });
}

ts_status_t ts_adamw_create(ts_module_t module,
                            double learning_rate,
                            double beta1,
                            double beta2,
                            double epsilon,
                            double weight_decay,
                            ts_optimizer_t* out_optimizer) {
  return tensora::AbiGuard("adamw_create", [&] {
    return tensora::training::AdamWCreate(module, learning_rate, beta1, beta2,
                                          epsilon, weight_decay,
                                          out_optimizer);
  });
}

ts_status_t ts_optimizer_zero_grad(ts_optimizer_t optimizer) {
  return tensora::AbiGuard("optimizer_zero_grad", [&] {
    return tensora::training::OptimizerZeroGrad(optimizer);
  });
}

ts_status_t ts_optimizer_step(ts_optimizer_t optimizer) {
  return tensora::AbiGuard("optimizer_step", [&] {
    return tensora::training::OptimizerStep(optimizer);
  });
}

ts_status_t ts_optimizer_release(ts_optimizer_t optimizer) {
  return tensora::AbiGuard("optimizer_release", [&] {
    return tensora::training::OptimizerRelease(optimizer);
  });
}

ts_status_t ts_sgd_create_for_tensors(const ts_tensor_t* parameters,
                                      size_t count,
                                      double learning_rate,
                                      double momentum,
                                      double weight_decay,
                                      ts_optimizer_t* out_optimizer) {
  return tensora::AbiGuard("sgd_create_for_tensors", [&] {
    return tensora::training::nn_v2_optimizer::SgdCreate(
        parameters, count, learning_rate, momentum, weight_decay,
        out_optimizer);
  });
}

ts_status_t ts_adam_create_for_tensors(const ts_tensor_t* parameters,
                                       size_t count,
                                       double learning_rate,
                                       double beta1,
                                       double beta2,
                                       double epsilon,
                                       double weight_decay,
                                       ts_optimizer_t* out_optimizer) {
  return tensora::AbiGuard("adam_create_for_tensors", [&] {
    return tensora::training::nn_v2_optimizer::AdamCreate(
        parameters, count, learning_rate, beta1, beta2, epsilon, weight_decay,
        out_optimizer);
  });
}

ts_status_t ts_adamw_create_for_tensors(const ts_tensor_t* parameters,
                                        size_t count,
                                        double learning_rate,
                                        double beta1,
                                        double beta2,
                                        double epsilon,
                                        double weight_decay,
                                        ts_optimizer_t* out_optimizer) {
  return tensora::AbiGuard("adamw_create_for_tensors", [&] {
    return tensora::training::nn_v2_optimizer::AdamWCreate(
        parameters, count, learning_rate, beta1, beta2, epsilon, weight_decay,
        out_optimizer);
  });
}

ts_status_t ts_parameter_optimizer_zero_grad(ts_optimizer_t optimizer) {
  return tensora::AbiGuard("parameter_optimizer_zero_grad", [&] {
    return tensora::training::nn_v2_optimizer::ZeroGrad(optimizer);
  });
}

ts_status_t ts_parameter_optimizer_step(ts_optimizer_t optimizer) {
  return tensora::AbiGuard("parameter_optimizer_step", [&] {
    return tensora::training::nn_v2_optimizer::Step(optimizer);
  });
}

ts_status_t ts_parameter_optimizer_release(ts_optimizer_t optimizer) {
  return tensora::AbiGuard("parameter_optimizer_release", [&] {
    return tensora::training::nn_v2_optimizer::Release(optimizer);
  });
}

ts_status_t ts_runtime_live_module_count(uint64_t* out_count) {
  return tensora::AbiGuard("runtime_live_module_count", [&] {
    if (out_count == nullptr) {
      return tensora::InvalidArgument(
          "runtime_live_module_count: output pointer is null");
    }
    *out_count = tensora::HandleRegistry::Instance().Count(
        tensora::HandleType::kModule);
    return tensora::Status::Ok();
  });
}

ts_status_t ts_runtime_live_optimizer_count(uint64_t* out_count) {
  return tensora::AbiGuard("runtime_live_optimizer_count", [&] {
    if (out_count == nullptr) {
      return tensora::InvalidArgument(
          "runtime_live_optimizer_count: output pointer is null");
    }
    *out_count =
        tensora::HandleRegistry::Instance().Count(
            tensora::HandleType::kOptimizer) +
        tensora::HandleRegistry::Instance().Count(
            tensora::HandleType::kParameterOptimizer);
    return tensora::Status::Ok();
  });
}

}  // extern "C"
