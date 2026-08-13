#include "training/training_bridge.h"

#include <cmath>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "runtime/handle_registry.h"
#include "training/torch_backend.h"
#include "training/torch_storage.h"

namespace tensora::training {
namespace {

struct LinearState {
  LinearState(int64_t in_features, int64_t out_features, bool use_bias)
      : module(torch::nn::LinearOptions(in_features, out_features)
                   .bias(use_bias)) {}

  torch::nn::Linear module;
};

struct OptimizerState {
  std::shared_ptr<LinearState> module;
  std::unique_ptr<torch::optim::Optimizer> optimizer;
};

Status TorchFailure(const char* operation, const c10::Error& error) {
  return InternalError(std::string(operation) + ": " + error.what());
}

Status LookupModule(uint64_t handle, std::shared_ptr<LinearState>* out) {
  return HandleRegistry::Instance().Lookup<LinearState>(
      handle, HandleType::kModule, out);
}

Status LookupOptimizer(uint64_t handle,
                       std::shared_ptr<OptimizerState>* out) {
  return HandleRegistry::Instance().Lookup<OptimizerState>(
      handle, HandleType::kOptimizer, out);
}

Status ValidateModuleDimensions(int64_t in_features, int64_t out_features) {
  if (in_features <= 0 || out_features <= 0) {
    return InvalidArgument(
        "linear_create: in_features and out_features must be positive");
  }
  return Status::Ok();
}

Status ValidateLearningRate(double learning_rate, const char* operation) {
  if (!std::isfinite(learning_rate) || learning_rate <= 0.0) {
    return InvalidArgument(std::string(operation) +
                           ": learning rate must be finite and positive");
  }
  return Status::Ok();
}

Status ValidateWeightDecay(double weight_decay, const char* operation) {
  if (!std::isfinite(weight_decay) || weight_decay < 0.0) {
    return InvalidArgument(std::string(operation) +
                           ": weight decay must be finite and non-negative");
  }
  return Status::Ok();
}

Status ValidateBetas(double beta1, double beta2, const char* operation) {
  if (!std::isfinite(beta1) || !std::isfinite(beta2) || beta1 < 0.0 ||
      beta1 >= 1.0 || beta2 < 0.0 || beta2 >= 1.0) {
    return InvalidArgument(std::string(operation) +
                           ": beta values must be finite in [0, 1)");
  }
  return Status::Ok();
}

Status ValidateEpsilon(double epsilon, const char* operation) {
  if (!std::isfinite(epsilon) || epsilon <= 0.0) {
    return InvalidArgument(std::string(operation) +
                           ": epsilon must be finite and positive");
  }
  return Status::Ok();
}

Status EnsureSameDevice(const Tensor& left,
                        const Tensor& right,
                        const char* operation) {
  if (left.device() != right.device() ||
      left.device_index() != right.device_index()) {
    return InvalidArgument(std::string(operation) +
                           ": tensors must use the same device");
  }
  return Status::Ok();
}

Status CreateOptimizerState(
    std::shared_ptr<LinearState> module,
    std::unique_ptr<torch::optim::Optimizer> optimizer,
    uint64_t* out_optimizer) {
  if (out_optimizer == nullptr) {
    return InvalidArgument("optimizer: output handle pointer is null");
  }
  *out_optimizer = 0;
  try {
    auto state = std::make_shared<OptimizerState>();
    state->module = std::move(module);
    state->optimizer = std::move(optimizer);
    return HandleRegistry::Instance().Insert(
        HandleType::kOptimizer, std::move(state), out_optimizer);
  } catch (const std::bad_alloc&) {
    return OutOfMemory("optimizer: allocation failed");
  }
}

}  // namespace

Status IsAvailable(uint8_t* out_available) {
  if (out_available == nullptr) {
    return InvalidArgument("training_available: output pointer is null");
  }
  *out_available = 1;
  return Status::Ok();
}

Status CudaDeviceCount(uint32_t* out_count) {
  if (out_count == nullptr) {
    return InvalidArgument("cuda_device_count: output pointer is null");
  }
  *out_count = 0;
  try {
    if (!torch::cuda::is_available()) return Status::Ok();
    const auto count = torch::cuda::device_count();
    if (count < 0) {
      return InternalError("cuda_device_count: negative count reported");
    }
    *out_count = static_cast<uint32_t>(count);
    return Status::Ok();
  } catch (const c10::Error& error) {
    return TorchFailure("cuda_device_count", error);
  }
}

Status ManualSeed(uint64_t seed) {
  try {
    torch::manual_seed(static_cast<int64_t>(seed));
    return Status::Ok();
  } catch (const c10::Error& error) {
    return TorchFailure("manual_seed", error);
  }
}

Status Transfer(const Tensor& tensor,
                Device device,
                int32_t device_index,
                std::shared_ptr<Tensor>* out) {
  if (out == nullptr) {
    return InvalidArgument("tensor_to_device: output tensor pointer is null");
  }
  *out = nullptr;

  torch::Device target(torch::kCPU);
  Status status = TorchDevice(device, device_index, &target);
  if (!status.ok()) return status;

  torch::Tensor value;
  status = TensorToTorch(tensor, &value);
  if (!status.ok()) return status;

  try {
    torch::Tensor moved = value.to(target);
    if (moved.device() == value.device()) {
      moved = moved.clone();
    }
    return WrapTorchTensor(std::move(moved), out);
  } catch (const c10::Error& error) {
    return TorchFailure("tensor_to_device", error);
  }
}

Status WithRequiresGrad(const Tensor& tensor,
                        bool requires_grad,
                        std::shared_ptr<Tensor>* out) {
  if (out == nullptr) {
    return InvalidArgument(
        "tensor_with_requires_grad: output tensor pointer is null");
  }
  *out = nullptr;

  torch::Tensor value;
  Status status = TensorToTorch(tensor, &value);
  if (!status.ok()) return status;
  try {
    torch::Tensor leaf = value.detach().clone();
    leaf.set_requires_grad(requires_grad);
    return WrapTorchTensor(std::move(leaf), out);
  } catch (const c10::Error& error) {
    return TorchFailure("tensor_with_requires_grad", error);
  }
}

Status RequiresGrad(const Tensor& tensor, uint8_t* out_requires_grad) {
  if (out_requires_grad == nullptr) {
    return InvalidArgument("tensor_requires_grad: output pointer is null");
  }
  *out_requires_grad = 0;
  if (tensor.storage()->kind() != StorageKind::kTorch) {
    return Status::Ok();
  }

  torch::Tensor value;
  Status status = TensorToTorch(tensor, &value);
  if (!status.ok()) return status;
  *out_requires_grad = value.requires_grad() ? 1 : 0;
  return Status::Ok();
}

Status Backward(const Tensor& tensor) {
  if (tensor.numel() != 1) {
    return InvalidShape("tensor_backward: loss tensor must contain one value");
  }
  torch::Tensor value;
  Status status = TensorToTorch(tensor, &value);
  if (!status.ok()) return status;
  if (!value.requires_grad()) {
    return InvalidArgument(
        "tensor_backward: tensor does not require gradients");
  }
  try {
    value.backward();
    return Status::Ok();
  } catch (const c10::Error& error) {
    return TorchFailure("tensor_backward", error);
  }
}

Status Gradient(const Tensor& tensor, std::shared_ptr<Tensor>* out) {
  if (out == nullptr) {
    return InvalidArgument("tensor_grad: output tensor pointer is null");
  }
  *out = nullptr;

  torch::Tensor value;
  Status status = TensorToTorch(tensor, &value);
  if (!status.ok()) return status;
  if (!value.grad().defined()) {
    return InvalidArgument("tensor_grad: gradient is not available");
  }
  try {
    return WrapTorchTensor(value.grad().detach().clone(), out);
  } catch (const c10::Error& error) {
    return TorchFailure("tensor_grad", error);
  }
}

Status Relu(const Tensor& tensor, std::shared_ptr<Tensor>* out) {
  torch::Tensor value;
  Status status = TensorToTorch(tensor, &value);
  if (!status.ok()) return status;
  try {
    return WrapTorchTensor(torch::relu(value), out);
  } catch (const c10::Error& error) {
    return TorchFailure("tensor_relu", error);
  }
}

Status Sigmoid(const Tensor& tensor, std::shared_ptr<Tensor>* out) {
  torch::Tensor value;
  Status status = TensorToTorch(tensor, &value);
  if (!status.ok()) return status;
  try {
    return WrapTorchTensor(torch::sigmoid(value), out);
  } catch (const c10::Error& error) {
    return TorchFailure("tensor_sigmoid", error);
  }
}

Status Tanh(const Tensor& tensor, std::shared_ptr<Tensor>* out) {
  torch::Tensor value;
  Status status = TensorToTorch(tensor, &value);
  if (!status.ok()) return status;
  try {
    return WrapTorchTensor(torch::tanh(value), out);
  } catch (const c10::Error& error) {
    return TorchFailure("tensor_tanh", error);
  }
}

Status MseLoss(const Tensor& prediction,
               const Tensor& target,
               std::shared_ptr<Tensor>* out) {
  Status status = EnsureSameDevice(prediction, target, "mse_loss");
  if (!status.ok()) return status;
  if (!SameShape(prediction.shape(), target.shape())) {
    return InvalidShape("mse_loss: prediction and target shapes must match");
  }

  torch::Tensor prediction_value;
  torch::Tensor target_value;
  status = TensorToTorch(prediction, &prediction_value);
  if (!status.ok()) return status;
  status = TensorToTorch(target, &target_value);
  if (!status.ok()) return status;
  try {
    return WrapTorchTensor(
        torch::mse_loss(prediction_value, target_value), out);
  } catch (const c10::Error& error) {
    return TorchFailure("mse_loss", error);
  }
}

Status CrossEntropyLoss(const Tensor& logits,
                        const Tensor& one_hot_target,
                        std::shared_ptr<Tensor>* out) {
  Status status = EnsureSameDevice(logits, one_hot_target,
                                   "cross_entropy_loss");
  if (!status.ok()) return status;
  if (!SameShape(logits.shape(), one_hot_target.shape())) {
    return InvalidShape(
        "cross_entropy_loss: logits and one-hot target shapes must match");
  }
  if (logits.shape().dimensions.size() != 2) {
    return InvalidShape(
        "cross_entropy_loss: initial contract requires rank-2 tensors");
  }

  torch::Tensor logits_value;
  torch::Tensor target_value;
  status = TensorToTorch(logits, &logits_value);
  if (!status.ok()) return status;
  status = TensorToTorch(one_hot_target, &target_value);
  if (!status.ok()) return status;
  try {
    const torch::Tensor log_probabilities =
        torch::log_softmax(logits_value, 1);
    const torch::Tensor loss =
        -(target_value * log_probabilities).sum(1).mean();
    return WrapTorchTensor(loss, out);
  } catch (const c10::Error& error) {
    return TorchFailure("cross_entropy_loss", error);
  }
}

Status LinearCreate(int64_t in_features,
                    int64_t out_features,
                    bool use_bias,
                    uint64_t* out_module) {
  if (out_module == nullptr) {
    return InvalidArgument("linear_create: output handle pointer is null");
  }
  *out_module = 0;
  Status status = ValidateModuleDimensions(in_features, out_features);
  if (!status.ok()) return status;

  try {
    auto state =
        std::make_shared<LinearState>(in_features, out_features, use_bias);
    return HandleRegistry::Instance().Insert(
        HandleType::kModule, std::move(state), out_module);
  } catch (const c10::Error& error) {
    return TorchFailure("linear_create", error);
  } catch (const std::bad_alloc&) {
    return OutOfMemory("linear_create: allocation failed");
  }
}

Status ModuleForward(uint64_t module,
                     const Tensor& input,
                     std::shared_ptr<Tensor>* out) {
  if (out == nullptr) {
    return InvalidArgument("module_forward: output tensor pointer is null");
  }
  *out = nullptr;

  std::shared_ptr<LinearState> state;
  Status status = LookupModule(module, &state);
  if (!status.ok()) return status;

  torch::Tensor input_value;
  status = TensorToTorch(input, &input_value);
  if (!status.ok()) return status;
  try {
    return WrapTorchTensor(state->module->forward(input_value), out);
  } catch (const c10::Error& error) {
    return TorchFailure("module_forward", error);
  }
}

Status ModuleSetTraining(uint64_t module, bool training) {
  std::shared_ptr<LinearState> state;
  Status status = LookupModule(module, &state);
  if (!status.ok()) return status;
  state->module->train(training);
  return Status::Ok();
}

Status ModuleToDevice(uint64_t module, Device device, int32_t device_index) {
  std::shared_ptr<LinearState> state;
  Status status = LookupModule(module, &state);
  if (!status.ok()) return status;

  torch::Device target(torch::kCPU);
  status = TorchDevice(device, device_index, &target);
  if (!status.ok()) return status;
  try {
    state->module->to(target, torch::kFloat32);
    return Status::Ok();
  } catch (const c10::Error& error) {
    return TorchFailure("module_to_device", error);
  }
}

Status ModuleParameterCount(uint64_t module, size_t* out_count) {
  if (out_count == nullptr) {
    return InvalidArgument("module_parameter_count: output pointer is null");
  }
  *out_count = 0;
  std::shared_ptr<LinearState> state;
  Status status = LookupModule(module, &state);
  if (!status.ok()) return status;
  *out_count = state->module->parameters().size();
  return Status::Ok();
}

Status ModuleParameterAt(uint64_t module,
                         size_t index,
                         std::shared_ptr<Tensor>* out) {
  if (out == nullptr) {
    return InvalidArgument("module_parameter_at: output tensor pointer is null");
  }
  *out = nullptr;
  std::shared_ptr<LinearState> state;
  Status status = LookupModule(module, &state);
  if (!status.ok()) return status;
  const auto parameters = state->module->parameters();
  if (index >= parameters.size()) {
    return InvalidArgument("module_parameter_at: index is out of range");
  }
  return WrapTorchTensor(parameters[index], out);
}

Status ModuleBufferCount(uint64_t module, size_t* out_count) {
  if (out_count == nullptr) {
    return InvalidArgument("module_buffer_count: output pointer is null");
  }
  *out_count = 0;
  std::shared_ptr<LinearState> state;
  Status status = LookupModule(module, &state);
  if (!status.ok()) return status;
  *out_count = state->module->buffers().size();
  return Status::Ok();
}

Status ModuleBufferAt(uint64_t module,
                      size_t index,
                      std::shared_ptr<Tensor>* out) {
  if (out == nullptr) {
    return InvalidArgument("module_buffer_at: output tensor pointer is null");
  }
  *out = nullptr;
  std::shared_ptr<LinearState> state;
  Status status = LookupModule(module, &state);
  if (!status.ok()) return status;
  const auto buffers = state->module->buffers();
  if (index >= buffers.size()) {
    return InvalidArgument("module_buffer_at: index is out of range");
  }
  return WrapTorchTensor(buffers[index], out);
}

Status ModuleSave(uint64_t module, const std::string& path) {
  if (path.empty()) {
    return InvalidArgument("module_save: checkpoint path is empty");
  }
  std::shared_ptr<LinearState> state;
  Status status = LookupModule(module, &state);
  if (!status.ok()) return status;
  try {
    torch::serialize::OutputArchive archive;
    state->module->save(archive);
    archive.save_to(path);
    return Status::Ok();
  } catch (const c10::Error& error) {
    return TorchFailure("module_save", error);
  }
}

Status ModuleLoad(uint64_t module, const std::string& path) {
  if (path.empty()) {
    return InvalidArgument("module_load: checkpoint path is empty");
  }
  std::shared_ptr<LinearState> state;
  Status status = LookupModule(module, &state);
  if (!status.ok()) return status;
  try {
    torch::serialize::InputArchive archive;
    archive.load_from(path);
    state->module->load(archive);
    return Status::Ok();
  } catch (const c10::Error& error) {
    return TorchFailure("module_load", error);
  }
}

Status ModuleRelease(uint64_t module) {
  return HandleRegistry::Instance().Release(module, HandleType::kModule);
}

Status SgdCreate(uint64_t module,
                 double learning_rate,
                 double momentum,
                 double weight_decay,
                 uint64_t* out_optimizer) {
  if (out_optimizer == nullptr) {
    return InvalidArgument("sgd_create: output handle pointer is null");
  }
  *out_optimizer = 0;
  Status status = ValidateLearningRate(learning_rate, "sgd_create");
  if (!status.ok()) return status;
  status = ValidateWeightDecay(weight_decay, "sgd_create");
  if (!status.ok()) return status;
  if (!std::isfinite(momentum) || momentum < 0.0) {
    return InvalidArgument(
        "sgd_create: momentum must be finite and non-negative");
  }

  std::shared_ptr<LinearState> state;
  status = LookupModule(module, &state);
  if (!status.ok()) return status;
  try {
    auto options = torch::optim::SGDOptions(learning_rate)
                       .momentum(momentum)
                       .weight_decay(weight_decay);
    auto optimizer = std::make_unique<torch::optim::SGD>(
        state->module->parameters(), options);
    return CreateOptimizerState(std::move(state), std::move(optimizer),
                                out_optimizer);
  } catch (const c10::Error& error) {
    return TorchFailure("sgd_create", error);
  } catch (const std::bad_alloc&) {
    return OutOfMemory("sgd_create: allocation failed");
  }
}

Status AdamCreate(uint64_t module,
                  double learning_rate,
                  double beta1,
                  double beta2,
                  double epsilon,
                  double weight_decay,
                  uint64_t* out_optimizer) {
  if (out_optimizer == nullptr) {
    return InvalidArgument("adam_create: output handle pointer is null");
  }
  *out_optimizer = 0;
  Status status = ValidateLearningRate(learning_rate, "adam_create");
  if (!status.ok()) return status;
  status = ValidateBetas(beta1, beta2, "adam_create");
  if (!status.ok()) return status;
  status = ValidateEpsilon(epsilon, "adam_create");
  if (!status.ok()) return status;
  status = ValidateWeightDecay(weight_decay, "adam_create");
  if (!status.ok()) return status;

  std::shared_ptr<LinearState> state;
  status = LookupModule(module, &state);
  if (!status.ok()) return status;
  try {
    auto options = torch::optim::AdamOptions(learning_rate)
                       .betas(std::make_tuple(beta1, beta2))
                       .eps(epsilon)
                       .weight_decay(weight_decay);
    auto optimizer = std::make_unique<torch::optim::Adam>(
        state->module->parameters(), options);
    return CreateOptimizerState(std::move(state), std::move(optimizer),
                                out_optimizer);
  } catch (const c10::Error& error) {
    return TorchFailure("adam_create", error);
  } catch (const std::bad_alloc&) {
    return OutOfMemory("adam_create: allocation failed");
  }
}

Status AdamWCreate(uint64_t module,
                   double learning_rate,
                   double beta1,
                   double beta2,
                   double epsilon,
                   double weight_decay,
                   uint64_t* out_optimizer) {
  if (out_optimizer == nullptr) {
    return InvalidArgument("adamw_create: output handle pointer is null");
  }
  *out_optimizer = 0;
  Status status = ValidateLearningRate(learning_rate, "adamw_create");
  if (!status.ok()) return status;
  status = ValidateBetas(beta1, beta2, "adamw_create");
  if (!status.ok()) return status;
  status = ValidateEpsilon(epsilon, "adamw_create");
  if (!status.ok()) return status;
  status = ValidateWeightDecay(weight_decay, "adamw_create");
  if (!status.ok()) return status;

  std::shared_ptr<LinearState> state;
  status = LookupModule(module, &state);
  if (!status.ok()) return status;
  try {
    auto options = torch::optim::AdamWOptions(learning_rate)
                       .betas(std::make_tuple(beta1, beta2))
                       .eps(epsilon)
                       .weight_decay(weight_decay);
    auto optimizer = std::make_unique<torch::optim::AdamW>(
        state->module->parameters(), options);
    return CreateOptimizerState(std::move(state), std::move(optimizer),
                                out_optimizer);
  } catch (const c10::Error& error) {
    return TorchFailure("adamw_create", error);
  } catch (const std::bad_alloc&) {
    return OutOfMemory("adamw_create: allocation failed");
  }
}

Status OptimizerZeroGrad(uint64_t optimizer) {
  std::shared_ptr<OptimizerState> state;
  Status status = LookupOptimizer(optimizer, &state);
  if (!status.ok()) return status;
  try {
    state->optimizer->zero_grad();
    return Status::Ok();
  } catch (const c10::Error& error) {
    return TorchFailure("optimizer_zero_grad", error);
  }
}

Status OptimizerStep(uint64_t optimizer) {
  std::shared_ptr<OptimizerState> state;
  Status status = LookupOptimizer(optimizer, &state);
  if (!status.ok()) return status;
  try {
    state->optimizer->step();
    return Status::Ok();
  } catch (const c10::Error& error) {
    return TorchFailure("optimizer_step", error);
  }
}

Status OptimizerRelease(uint64_t optimizer) {
  return HandleRegistry::Instance().Release(optimizer,
                                            HandleType::kOptimizer);
}

uint64_t LiveStorageBytes() { return TorchStorage::LiveBytes(); }

}  // namespace tensora::training
