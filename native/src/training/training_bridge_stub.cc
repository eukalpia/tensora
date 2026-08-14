#include "training/training_bridge.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "autograd/autograd.h"
#include "backends/backend.h"
#include "core/allocation_guard.h"
#include "memory/cpu_storage.h"
#include "runtime/dispatcher.h"
#include "runtime/handle_registry.h"
#include "tensor/shape.h"

namespace tensora::training {
namespace {

constexpr uint32_t kCheckpointVersion = 1;
constexpr char kCheckpointMagic[8] = {'T', 'S', 'T', 'R', 'N', 'V', '1', '\0'};

std::mutex& RngMutex() {
  static std::mutex mutex;
  return mutex;
}

std::mt19937_64& Rng() {
  static std::mt19937_64 engine(0);
  return engine;
}

struct LinearState {
  int64_t in_features = 0;
  int64_t out_features = 0;
  bool use_bias = false;
  bool training = true;
  std::shared_ptr<Tensor> weight;
  std::shared_ptr<Tensor> bias;
  mutable std::mutex mutex;
};

enum class OptimizerKind : uint8_t {
  kSgd,
  kAdam,
  kAdamW,
};

struct OptimizerState {
  OptimizerKind kind = OptimizerKind::kSgd;
  std::shared_ptr<LinearState> module;
  double learning_rate = 0.0;
  double momentum = 0.0;
  double beta1 = 0.0;
  double beta2 = 0.0;
  double epsilon = 0.0;
  double weight_decay = 0.0;
  uint64_t step = 0;
  std::vector<std::vector<float>> first_moment;
  std::vector<std::vector<float>> second_moment;
};

Status UnsupportedDevice(const char* operation) {
  return Unsupported(std::string(operation) +
                     ": requested accelerator is not available in the core-only build");
}

Status ValidatePositiveFinite(double value,
                              const char* name,
                              const char* operation) {
  if (!std::isfinite(value) || value <= 0.0) {
    return InvalidArgument(std::string(operation) + ": " + name +
                           " must be finite and positive");
  }
  return Status::Ok();
}

Status ValidateNonNegativeFinite(double value,
                                 const char* name,
                                 const char* operation) {
  if (!std::isfinite(value) || value < 0.0) {
    return InvalidArgument(std::string(operation) + ": " + name +
                           " must be finite and non-negative");
  }
  return Status::Ok();
}

Status ValidateBeta(double value, const char* name, const char* operation) {
  if (!std::isfinite(value) || value < 0.0 || value >= 1.0) {
    return InvalidArgument(std::string(operation) + ": " + name +
                           " must be finite in [0, 1)");
  }
  return Status::Ok();
}

Status ValidateModuleDimensions(int64_t in_features, int64_t out_features) {
  if (in_features <= 0 || out_features <= 0) {
    return InvalidArgument("linear_create: feature dimensions must be positive");
  }
  if (static_cast<uint64_t>(in_features) >
      std::numeric_limits<uint64_t>::max() /
          static_cast<uint64_t>(out_features)) {
    return InvalidArgument("linear_create: parameter element count overflows");
  }
  const uint64_t numel = static_cast<uint64_t>(in_features) *
                         static_cast<uint64_t>(out_features);
  if (numel > std::numeric_limits<size_t>::max() / sizeof(float)) {
    return InvalidArgument("linear_create: parameter allocation is too large");
  }
  return Status::Ok();
}

Status LookupModule(uint64_t handle, std::shared_ptr<LinearState>* out) {
  return HandleRegistry::Instance().Lookup<LinearState>(
      handle, HandleType::kModule, out);
}

Status LookupOptimizer(uint64_t handle, std::shared_ptr<OptimizerState>* out) {
  return HandleRegistry::Instance().Lookup<OptimizerState>(
      handle, HandleType::kOptimizer, out);
}

std::vector<std::shared_ptr<Tensor>> Parameters(const LinearState& state) {
  std::vector<std::shared_ptr<Tensor>> parameters;
  parameters.reserve(state.use_bias ? 2 : 1);
  parameters.push_back(state.weight);
  if (state.use_bias) parameters.push_back(state.bias);
  return parameters;
}

Status MakeParameter(const ShapeInfo& shape,
                     std::vector<float> values,
                     std::shared_ptr<Tensor>* out) {
  Status status = autograd::MakeCpuTensor(shape, values, out);
  if (!status.ok()) return status;
  auto meta = std::make_shared<AutogradMeta>();
  meta->requires_grad = true;
  meta->is_leaf = true;
  (*out)->set_autograd_meta(std::move(meta));
  return Status::Ok();
}

Status TensorValues(const Tensor& tensor,
                    const char* operation,
                    const std::vector<float>** out) {
  return autograd::CpuValues(tensor, operation, out);
}

Status MutableTensorValues(Tensor& tensor,
                           const char* operation,
                           std::vector<float>** out) {
  return autograd::MutableCpuValues(tensor, operation, out);
}

Status MakeActivation(const Tensor& tensor,
                      autograd::Operation operation,
                      std::shared_ptr<Tensor>* out) {
  if (out == nullptr) {
    return InvalidArgument("activation: output tensor pointer is null");
  }
  *out = nullptr;
  const std::vector<float>* input = nullptr;
  Status status = TensorValues(tensor, "activation", &input);
  if (!status.ok()) return status;

  std::vector<float> values(input->size(), 0.0f);
  for (size_t index = 0; index < values.size(); ++index) {
    if (operation == autograd::Operation::kRelu) {
      values[index] = std::max(0.0f, (*input)[index]);
    } else if (operation == autograd::Operation::kSigmoid) {
      const float value = (*input)[index];
      if (value >= 0.0f) {
        const float exp_value = std::exp(-value);
        values[index] = 1.0f / (1.0f + exp_value);
      } else {
        const float exp_value = std::exp(value);
        values[index] = exp_value / (1.0f + exp_value);
      }
    } else if (operation == autograd::Operation::kTanh) {
      values[index] = std::tanh((*input)[index]);
    } else {
      return InternalError("activation: unsupported operation");
    }
  }

  status = autograd::MakeCpuTensor(tensor.shape(), values, out);
  if (!status.ok()) return status;
  return autograd::RecordUnary(operation, tensor, *out);
}

Status MakeScalar(float value, std::shared_ptr<Tensor>* out) {
  ShapeInfo scalar;
  Status status = ValidateShape(nullptr, 0, &scalar);
  if (!status.ok()) return status;
  return autograd::MakeCpuTensor(scalar, std::vector<float>{value}, out);
}

Status AddBias(const Tensor& matrix,
               const Tensor& bias,
               std::shared_ptr<Tensor>* out) {
  if (out == nullptr) {
    return InvalidArgument("linear_forward: output tensor pointer is null");
  }
  *out = nullptr;
  if (matrix.shape().dimensions.size() != 2 ||
      bias.shape().dimensions.size() != 1) {
    return InternalError("linear_forward: invalid internal bias shapes");
  }
  const int64_t rows = matrix.shape().dimensions[0];
  const int64_t cols = matrix.shape().dimensions[1];
  if (bias.shape().dimensions[0] != cols) {
    return InternalError("linear_forward: bias width mismatch");
  }

  const std::vector<float>* matrix_values = nullptr;
  const std::vector<float>* bias_values = nullptr;
  Status status = TensorValues(matrix, "linear_forward", &matrix_values);
  if (!status.ok()) return status;
  status = TensorValues(bias, "linear_forward", &bias_values);
  if (!status.ok()) return status;

  std::vector<float> values = *matrix_values;
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < cols; ++col) {
      values[static_cast<size_t>(row * cols + col)] +=
          (*bias_values)[static_cast<size_t>(col)];
    }
  }

  status = autograd::MakeCpuTensor(matrix.shape(), values, out);
  if (!status.ok()) return status;
  return autograd::RecordBiasAdd(matrix, bias, *out);
}

Status ValidateModuleInput(const LinearState& state, const Tensor& input) {
  Status status = autograd::EnsureCpuFloat32(input, "module_forward");
  if (!status.ok()) return status;
  if (input.shape().dimensions.size() != 2) {
    return InvalidShape("module_forward: Linear currently requires rank-2 input");
  }
  if (input.shape().dimensions[1] != state.in_features) {
    return InvalidShape(
        "module_forward: input feature dimension does not match Linear");
  }
  return Status::Ok();
}

Status CreateOptimizer(uint64_t module,
                       OptimizerKind kind,
                       double learning_rate,
                       double momentum,
                       double beta1,
                       double beta2,
                       double epsilon,
                       double weight_decay,
                       uint64_t* out_optimizer) {
  if (out_optimizer == nullptr) {
    return InvalidArgument("optimizer_create: output handle pointer is null");
  }
  *out_optimizer = 0;

  const char* operation = kind == OptimizerKind::kSgd
                              ? "sgd_create"
                              : (kind == OptimizerKind::kAdam ? "adam_create"
                                                             : "adamw_create");
  Status status = ValidatePositiveFinite(learning_rate, "learning_rate", operation);
  if (!status.ok()) return status;
  status = ValidateNonNegativeFinite(weight_decay, "weight_decay", operation);
  if (!status.ok()) return status;

  if (kind == OptimizerKind::kSgd) {
    status = ValidateNonNegativeFinite(momentum, "momentum", operation);
    if (!status.ok()) return status;
  } else {
    status = ValidateBeta(beta1, "beta1", operation);
    if (!status.ok()) return status;
    status = ValidateBeta(beta2, "beta2", operation);
    if (!status.ok()) return status;
    status = ValidatePositiveFinite(epsilon, "epsilon", operation);
    if (!status.ok()) return status;
  }

  std::shared_ptr<LinearState> module_state;
  status = LookupModule(module, &module_state);
  if (!status.ok()) return status;

  return AllocationGuard(operation, [&]() -> Status {
    auto state = std::make_shared<OptimizerState>();
    state->kind = kind;
    state->module = module_state;
    state->learning_rate = learning_rate;
    state->momentum = momentum;
    state->beta1 = beta1;
    state->beta2 = beta2;
    state->epsilon = epsilon;
    state->weight_decay = weight_decay;

    const auto parameters = Parameters(*module_state);
    state->first_moment.reserve(parameters.size());
    state->second_moment.reserve(parameters.size());
    for (const auto& parameter : parameters) {
      state->first_moment.emplace_back(static_cast<size_t>(parameter->numel()),
                                       0.0f);
      if (kind != OptimizerKind::kSgd) {
        state->second_moment.emplace_back(
            static_cast<size_t>(parameter->numel()), 0.0f);
      }
    }

    return HandleRegistry::Instance().Insert(
        HandleType::kOptimizer, std::move(state), out_optimizer);
  });
}

template <typename T>
bool WriteScalar(std::ofstream& stream, const T& value) {
  stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
  return stream.good();
}

template <typename T>
bool ReadScalar(std::ifstream& stream, T* value) {
  if (value == nullptr) return false;
  stream.read(reinterpret_cast<char*>(value), sizeof(T));
  return stream.good();
}

Status SaveTensor(std::ofstream& stream, const Tensor& tensor) {
  const uint64_t rank = static_cast<uint64_t>(tensor.shape().dimensions.size());
  if (!WriteScalar(stream, rank)) {
    return InternalError("module_save: failed to write tensor rank");
  }
  for (int64_t dimension : tensor.shape().dimensions) {
    if (!WriteScalar(stream, dimension)) {
      return InternalError("module_save: failed to write tensor shape");
    }
  }
  const uint64_t numel = tensor.numel();
  if (!WriteScalar(stream, numel)) {
    return InternalError("module_save: failed to write tensor element count");
  }
  const std::vector<float>* values = nullptr;
  Status status = TensorValues(tensor, "module_save", &values);
  if (!status.ok()) return status;
  if (!values->empty()) {
    stream.write(reinterpret_cast<const char*>(values->data()),
                 static_cast<std::streamsize>(values->size() * sizeof(float)));
  }
  if (!stream.good()) {
    return InternalError("module_save: failed to write tensor payload");
  }
  return Status::Ok();
}

Status ReadTensorPayload(std::ifstream& stream,
                         const ShapeInfo& expected_shape,
                         std::vector<float>* out_values) {
  if (out_values == nullptr) {
    return InvalidArgument("module_load: output values pointer is null");
  }
  uint64_t rank = 0;
  if (!ReadScalar(stream, &rank) || rank > 32) {
    return InternalError("module_load: invalid tensor rank");
  }
  std::vector<int64_t> dimensions(static_cast<size_t>(rank));
  for (uint64_t index = 0; index < rank; ++index) {
    if (!ReadScalar(stream, &dimensions[static_cast<size_t>(index)])) {
      return InternalError("module_load: truncated tensor shape");
    }
  }
  ShapeInfo shape;
  Status status = ValidateShape(dimensions.empty() ? nullptr : dimensions.data(),
                                dimensions.size(), &shape);
  if (!status.ok()) return InternalError("module_load: invalid tensor shape");
  if (!SameShape(shape, expected_shape)) {
    return InternalError("module_load: checkpoint tensor shape mismatch");
  }

  uint64_t numel = 0;
  if (!ReadScalar(stream, &numel) || numel != shape.numel ||
      numel > std::numeric_limits<size_t>::max() / sizeof(float)) {
    return InternalError("module_load: invalid tensor element count");
  }
  out_values->assign(static_cast<size_t>(numel), 0.0f);
  if (numel > 0) {
    stream.read(reinterpret_cast<char*>(out_values->data()),
                static_cast<std::streamsize>(numel * sizeof(float)));
  }
  if (!stream.good()) {
    return InternalError("module_load: truncated tensor payload");
  }
  return Status::Ok();
}

}  // namespace

Status IsAvailable(uint8_t* out_available) {
  if (out_available == nullptr) {
    return InvalidArgument("training_available: output pointer is null");
  }
  *out_available = 1;
  return Status::Ok();
}

Status DeviceCount(Device device, uint32_t* out_count) {
  if (out_count == nullptr) {
    return InvalidArgument("device_count: output pointer is null");
  }
  *out_count = device == Device::kCpu ? 1u : 0u;
  return Status::Ok();
}

Status CudaDeviceCount(uint32_t* out_count) {
  return DeviceCount(Device::kCuda, out_count);
}

Status ManualSeed(uint64_t seed) {
  std::lock_guard<std::mutex> lock(RngMutex());
  Rng().seed(seed);
  return Status::Ok();
}

Status Transfer(const Tensor& tensor,
                Device device,
                int32_t device_index,
                std::shared_ptr<Tensor>* out) {
  if (out == nullptr) {
    return InvalidArgument("tensor_to_device: output tensor pointer is null");
  }
  *out = nullptr;
  if (device_index < 0) {
    return InvalidArgument("tensor_to_device: device index must be non-negative");
  }
  if (device != Device::kCpu) return UnsupportedDevice("tensor_to_device");
  if (device_index != 0) {
    return InvalidArgument("tensor_to_device: CPU device index must be zero");
  }

  Status status = autograd::CloneDetached(tensor, out);
  if (!status.ok()) return status;
  return autograd::RecordUnary(autograd::Operation::kIdentity, tensor, *out);
}

Status WithRequiresGrad(const Tensor& tensor,
                        bool requires_grad,
                        std::shared_ptr<Tensor>* out) {
  if (out == nullptr) {
    return InvalidArgument(
        "tensor_with_requires_grad: output tensor pointer is null");
  }
  return autograd::CloneAsLeaf(tensor, requires_grad, out);
}

Status RequiresGrad(const Tensor& tensor, uint8_t* out_requires_grad) {
  if (out_requires_grad == nullptr) {
    return InvalidArgument("tensor_requires_grad: output pointer is null");
  }
  *out_requires_grad = autograd::RequiresGrad(tensor) ? 1 : 0;
  return Status::Ok();
}

Status Backward(const Tensor& tensor) { return autograd::Backward(tensor); }

Status Gradient(const Tensor& tensor, std::shared_ptr<Tensor>* out) {
  return autograd::GradientSnapshot(tensor, out);
}

Status Relu(const Tensor& tensor, std::shared_ptr<Tensor>* out) {
  return MakeActivation(tensor, autograd::Operation::kRelu, out);
}

Status Sigmoid(const Tensor& tensor, std::shared_ptr<Tensor>* out) {
  return MakeActivation(tensor, autograd::Operation::kSigmoid, out);
}

Status Tanh(const Tensor& tensor, std::shared_ptr<Tensor>* out) {
  return MakeActivation(tensor, autograd::Operation::kTanh, out);
}

Status MseLoss(const Tensor& prediction,
               const Tensor& target,
               std::shared_ptr<Tensor>* out) {
  if (out == nullptr) {
    return InvalidArgument("mse_loss: output tensor pointer is null");
  }
  *out = nullptr;
  Status status = autograd::EnsureCpuFloat32(prediction, "mse_loss");
  if (!status.ok()) return status;
  status = autograd::EnsureCpuFloat32(target, "mse_loss");
  if (!status.ok()) return status;
  if (!SameShape(prediction.shape(), target.shape())) {
    return InvalidShape("mse_loss: prediction and target shapes must match");
  }

  const std::vector<float>* prediction_values = nullptr;
  const std::vector<float>* target_values = nullptr;
  status = TensorValues(prediction, "mse_loss", &prediction_values);
  if (!status.ok()) return status;
  status = TensorValues(target, "mse_loss", &target_values);
  if (!status.ok()) return status;

  double squared_error = 0.0;
  for (size_t index = 0; index < prediction_values->size(); ++index) {
    const double difference = static_cast<double>((*prediction_values)[index]) -
                              static_cast<double>((*target_values)[index]);
    squared_error += difference * difference;
  }
  const float value = static_cast<float>(
      squared_error / static_cast<double>(prediction_values->size()));
  status = MakeScalar(value, out);
  if (!status.ok()) return status;
  return autograd::RecordBinary(autograd::Operation::kMse, prediction, target,
                                *out);
}

Status CrossEntropyLoss(const Tensor& logits,
                        const Tensor& one_hot_target,
                        std::shared_ptr<Tensor>* out) {
  if (out == nullptr) {
    return InvalidArgument(
        "cross_entropy_loss: output tensor pointer is null");
  }
  *out = nullptr;
  Status status = autograd::EnsureCpuFloat32(logits, "cross_entropy_loss");
  if (!status.ok()) return status;
  status =
      autograd::EnsureCpuFloat32(one_hot_target, "cross_entropy_loss");
  if (!status.ok()) return status;
  if (!SameShape(logits.shape(), one_hot_target.shape())) {
    return InvalidShape(
        "cross_entropy_loss: logits and one-hot target shapes must match");
  }
  if (logits.shape().dimensions.size() != 2) {
    return InvalidShape(
        "cross_entropy_loss: initial contract requires rank-2 tensors");
  }

  const int64_t batch = logits.shape().dimensions[0];
  const int64_t classes = logits.shape().dimensions[1];
  const std::vector<float>* logits_values = nullptr;
  const std::vector<float>* target_values = nullptr;
  status = TensorValues(logits, "cross_entropy_loss", &logits_values);
  if (!status.ok()) return status;
  status = TensorValues(one_hot_target, "cross_entropy_loss", &target_values);
  if (!status.ok()) return status;

  double total = 0.0;
  for (int64_t row = 0; row < batch; ++row) {
    float max_logit = (*logits_values)[static_cast<size_t>(row * classes)];
    for (int64_t col = 1; col < classes; ++col) {
      max_logit = std::max(
          max_logit,
          (*logits_values)[static_cast<size_t>(row * classes + col)]);
    }
    double denominator = 0.0;
    for (int64_t col = 0; col < classes; ++col) {
      denominator += std::exp(static_cast<double>(
          (*logits_values)[static_cast<size_t>(row * classes + col)] -
          max_logit));
    }
    const double log_denominator = std::log(denominator) + max_logit;
    for (int64_t col = 0; col < classes; ++col) {
      const size_t index = static_cast<size_t>(row * classes + col);
      total -= static_cast<double>((*target_values)[index]) *
               (static_cast<double>((*logits_values)[index]) -
                log_denominator);
    }
  }

  status = MakeScalar(static_cast<float>(total / static_cast<double>(batch)),
                      out);
  if (!status.ok()) return status;
  return autograd::RecordBinary(autograd::Operation::kCrossEntropy, logits,
                                one_hot_target, *out);
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

  const int64_t weight_dims[2] = {in_features, out_features};
  ShapeInfo weight_shape;
  status = ValidateShape(weight_dims, 2, &weight_shape);
  if (!status.ok()) return status;

  return AllocationGuard("linear_create", [&]() -> Status {
    const float bound = 1.0f / std::sqrt(static_cast<float>(in_features));
    std::vector<float> weight_values(static_cast<size_t>(weight_shape.numel));
    std::vector<float> bias_values;
    {
      std::lock_guard<std::mutex> lock(RngMutex());
      std::uniform_real_distribution<float> distribution(-bound, bound);
      for (float& value : weight_values) value = distribution(Rng());
      if (use_bias) {
        bias_values.resize(static_cast<size_t>(out_features));
        for (float& value : bias_values) value = distribution(Rng());
      }
    }

    auto state = std::make_shared<LinearState>();
    state->in_features = in_features;
    state->out_features = out_features;
    state->use_bias = use_bias;
    status = MakeParameter(weight_shape, std::move(weight_values),
                           &state->weight);
    if (!status.ok()) return status;

    if (use_bias) {
      const int64_t bias_dims[1] = {out_features};
      ShapeInfo bias_shape;
      status = ValidateShape(bias_dims, 1, &bias_shape);
      if (!status.ok()) return status;
      status =
          MakeParameter(bias_shape, std::move(bias_values), &state->bias);
      if (!status.ok()) return status;
    }

    return HandleRegistry::Instance().Insert(
        HandleType::kModule, std::move(state), out_module);
  });
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

  std::lock_guard<std::mutex> lock(state->mutex);
  status = ValidateModuleInput(*state, input);
  if (!status.ok()) return status;

  const Backend* backend = nullptr;
  status = Dispatcher::For(Device::kCpu, &backend);
  if (!status.ok()) return status;

  std::shared_ptr<Tensor> linear;
  status = backend->Matmul(input, *state->weight, &linear);
  if (!status.ok()) return status;
  if (!state->use_bias) {
    *out = std::move(linear);
    return Status::Ok();
  }
  return AddBias(*linear, *state->bias, out);
}

Status ModuleSetTraining(uint64_t module, bool training) {
  std::shared_ptr<LinearState> state;
  Status status = LookupModule(module, &state);
  if (!status.ok()) return status;
  std::lock_guard<std::mutex> lock(state->mutex);
  state->training = training;
  return Status::Ok();
}

Status ModuleToDevice(uint64_t module, Device device, int32_t device_index) {
  std::shared_ptr<LinearState> state;
  Status status = LookupModule(module, &state);
  if (!status.ok()) return status;
  if (device_index < 0) {
    return InvalidArgument("module_to_device: device index must be non-negative");
  }
  if (device == Device::kCpu) {
    if (device_index != 0) {
      return InvalidArgument("module_to_device: CPU device index must be zero");
    }
    return Status::Ok();
  }
  return UnsupportedDevice("module_to_device");
}

Status ModuleParameterCount(uint64_t module, size_t* out_count) {
  if (out_count == nullptr) {
    return InvalidArgument("module_parameter_count: output pointer is null");
  }
  *out_count = 0;
  std::shared_ptr<LinearState> state;
  Status status = LookupModule(module, &state);
  if (!status.ok()) return status;
  std::lock_guard<std::mutex> lock(state->mutex);
  *out_count = state->use_bias ? 2u : 1u;
  return Status::Ok();
}

Status ModuleParameterAt(uint64_t module,
                         size_t index,
                         std::shared_ptr<Tensor>* out) {
  if (out == nullptr) {
    return InvalidArgument("module_parameter_at: output pointer is null");
  }
  *out = nullptr;
  std::shared_ptr<LinearState> state;
  Status status = LookupModule(module, &state);
  if (!status.ok()) return status;
  std::lock_guard<std::mutex> lock(state->mutex);
  if (index == 0) {
    *out = state->weight;
    return Status::Ok();
  }
  if (index == 1 && state->use_bias) {
    *out = state->bias;
    return Status::Ok();
  }
  return InvalidArgument("module_parameter_at: parameter index is out of range");
}

Status ModuleBufferCount(uint64_t module, size_t* out_count) {
  if (out_count == nullptr) {
    return InvalidArgument("module_buffer_count: output pointer is null");
  }
  *out_count = 0;
  std::shared_ptr<LinearState> state;
  Status status = LookupModule(module, &state);
  if (!status.ok()) return status;
  return Status::Ok();
}

Status ModuleBufferAt(uint64_t module,
                      size_t,
                      std::shared_ptr<Tensor>* out) {
  if (out == nullptr) {
    return InvalidArgument("module_buffer_at: output pointer is null");
  }
  *out = nullptr;
  std::shared_ptr<LinearState> state;
  Status status = LookupModule(module, &state);
  if (!status.ok()) return status;
  return InvalidArgument("module_buffer_at: module has no buffers");
}

Status ModuleSave(uint64_t module, const std::string& path) {
  if (path.empty()) return InvalidArgument("module_save: path must not be empty");
  std::shared_ptr<LinearState> state;
  Status status = LookupModule(module, &state);
  if (!status.ok()) return status;
  std::lock_guard<std::mutex> lock(state->mutex);

  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream.is_open()) {
    return InternalError("module_save: could not open checkpoint for writing");
  }
  stream.write(kCheckpointMagic, sizeof(kCheckpointMagic));
  const uint8_t use_bias = state->use_bias ? 1 : 0;
  if (!stream.good() || !WriteScalar(stream, kCheckpointVersion) ||
      !WriteScalar(stream, state->in_features) ||
      !WriteScalar(stream, state->out_features) ||
      !WriteScalar(stream, use_bias)) {
    return InternalError("module_save: failed to write checkpoint header");
  }
  status = SaveTensor(stream, *state->weight);
  if (!status.ok()) return status;
  if (state->use_bias) {
    status = SaveTensor(stream, *state->bias);
    if (!status.ok()) return status;
  }
  stream.flush();
  if (!stream.good()) {
    return InternalError("module_save: failed to finalize checkpoint");
  }
  return Status::Ok();
}

Status ModuleLoad(uint64_t module, const std::string& path) {
  if (path.empty()) return InvalidArgument("module_load: path must not be empty");
  std::shared_ptr<LinearState> state;
  Status status = LookupModule(module, &state);
  if (!status.ok()) return status;

  std::ifstream stream(path, std::ios::binary);
  if (!stream.is_open()) {
    return InternalError("module_load: could not open checkpoint");
  }
  char magic[sizeof(kCheckpointMagic)] = {};
  stream.read(magic, sizeof(magic));
  if (!stream.good() ||
      !std::equal(std::begin(magic), std::end(magic),
                  std::begin(kCheckpointMagic))) {
    return InternalError("module_load: invalid checkpoint magic");
  }
  uint32_t version = 0;
  int64_t in_features = 0;
  int64_t out_features = 0;
  uint8_t use_bias = 0;
  if (!ReadScalar(stream, &version) || !ReadScalar(stream, &in_features) ||
      !ReadScalar(stream, &out_features) || !ReadScalar(stream, &use_bias)) {
    return InternalError("module_load: truncated checkpoint header");
  }
  if (version != kCheckpointVersion || in_features != state->in_features ||
      out_features != state->out_features ||
      (use_bias != 0) != state->use_bias) {
    return InternalError("module_load: checkpoint is incompatible with module");
  }

  std::vector<float> weight_values;
  std::vector<float> bias_values;
  status = ReadTensorPayload(stream, state->weight->shape(), &weight_values);
  if (!status.ok()) return status;
  if (state->use_bias) {
    status = ReadTensorPayload(stream, state->bias->shape(), &bias_values);
    if (!status.ok()) return status;
  }
  if (stream.peek() != std::ifstream::traits_type::eof()) {
    return InternalError("module_load: checkpoint contains trailing data");
  }

  std::lock_guard<std::mutex> lock(state->mutex);
  std::vector<float>* weight = nullptr;
  status = MutableTensorValues(*state->weight, "module_load", &weight);
  if (!status.ok()) return status;
  *weight = std::move(weight_values);
  autograd::IncrementVersion(*state->weight);
  autograd::ClearGradient(*state->weight);

  if (state->use_bias) {
    std::vector<float>* bias = nullptr;
    status = MutableTensorValues(*state->bias, "module_load", &bias);
    if (!status.ok()) return status;
    *bias = std::move(bias_values);
    autograd::IncrementVersion(*state->bias);
    autograd::ClearGradient(*state->bias);
  }
  return Status::Ok();
}

Status ModuleRelease(uint64_t module) {
  return HandleRegistry::Instance().Release(module, HandleType::kModule);
}

Status SgdCreate(uint64_t module,
                 double learning_rate,
                 double momentum,
                 double weight_decay,
                 uint64_t* out_optimizer) {
  return CreateOptimizer(module, OptimizerKind::kSgd, learning_rate, momentum,
                         0.0, 0.0, 0.0, weight_decay, out_optimizer);
}

Status AdamCreate(uint64_t module,
                  double learning_rate,
                  double beta1,
                  double beta2,
                  double epsilon,
                  double weight_decay,
                  uint64_t* out_optimizer) {
  return CreateOptimizer(module, OptimizerKind::kAdam, learning_rate, 0.0,
                         beta1, beta2, epsilon, weight_decay, out_optimizer);
}

Status AdamWCreate(uint64_t module,
                   double learning_rate,
                   double beta1,
                   double beta2,
                   double epsilon,
                   double weight_decay,
                   uint64_t* out_optimizer) {
  return CreateOptimizer(module, OptimizerKind::kAdamW, learning_rate, 0.0,
                         beta1, beta2, epsilon, weight_decay, out_optimizer);
}

Status OptimizerZeroGrad(uint64_t optimizer) {
  std::shared_ptr<OptimizerState> state;
  Status status = LookupOptimizer(optimizer, &state);
  if (!status.ok()) return status;
  std::lock_guard<std::mutex> lock(state->module->mutex);
  for (const auto& parameter : Parameters(*state->module)) {
    autograd::ClearGradient(*parameter);
  }
  return Status::Ok();
}

Status OptimizerStep(uint64_t optimizer) {
  std::shared_ptr<OptimizerState> state;
  Status status = LookupOptimizer(optimizer, &state);
  if (!status.ok()) return status;

  std::lock_guard<std::mutex> lock(state->module->mutex);
  const auto parameters = Parameters(*state->module);
  ++state->step;

  for (size_t parameter_index = 0; parameter_index < parameters.size();
       ++parameter_index) {
    const auto& parameter = parameters[parameter_index];
    const auto meta = parameter->autograd_meta();
    if (!meta) continue;

    std::shared_ptr<Tensor> gradient;
    {
      std::lock_guard<std::mutex> gradient_lock(meta->mutex);
      gradient = meta->gradient;
    }
    if (!gradient) continue;

    std::vector<float>* values = nullptr;
    const std::vector<float>* gradient_values = nullptr;
    status = MutableTensorValues(*parameter, "optimizer_step", &values);
    if (!status.ok()) return status;
    status = TensorValues(*gradient, "optimizer_step", &gradient_values);
    if (!status.ok()) return status;
    if (values->size() != gradient_values->size()) {
      return InternalError("optimizer_step: gradient size mismatch");
    }

    auto& first = state->first_moment[parameter_index];
    if (state->kind == OptimizerKind::kSgd) {
      for (size_t index = 0; index < values->size(); ++index) {
        double grad_value = static_cast<double>((*gradient_values)[index]) +
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
        double grad_value = static_cast<double>((*gradient_values)[index]);
        if (state->kind == OptimizerKind::kAdam) {
          grad_value += state->weight_decay * parameter_value;
        } else if (state->weight_decay > 0.0) {
          parameter_value *= 1.0 - state->learning_rate * state->weight_decay;
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

Status OptimizerRelease(uint64_t optimizer) {
  return HandleRegistry::Instance().Release(optimizer, HandleType::kOptimizer);
}

uint64_t LiveStorageBytes() {
  // CPU parameters use CpuStorage and are already included in the core counter.
  return 0;
}

}  // namespace tensora::training
