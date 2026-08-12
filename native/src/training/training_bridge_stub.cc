#include "training/training_bridge.h"

namespace tensora::training {
namespace {

Status Disabled(const char* operation) {
  return Unsupported(std::string(operation) +
                     ": native training backend is not enabled");
}

}  // namespace

Status IsAvailable(uint8_t* out_available) {
  if (out_available == nullptr) {
    return InvalidArgument("training_available: output pointer is null");
  }
  *out_available = 0;
  return Status::Ok();
}

Status CudaDeviceCount(uint32_t* out_count) {
  if (out_count == nullptr) {
    return InvalidArgument("cuda_device_count: output pointer is null");
  }
  *out_count = 0;
  return Status::Ok();
}

Status ManualSeed(uint64_t) { return Disabled("manual_seed"); }

Status Transfer(const Tensor& tensor,
                Device device,
                int32_t device_index,
                std::shared_ptr<Tensor>* out) {
  if (out == nullptr) {
    return InvalidArgument("tensor_to_device: output tensor pointer is null");
  }
  *out = nullptr;
  if (device != Device::kCpu) return Disabled("tensor_to_device");
  if (device_index != 0) {
    return InvalidArgument("tensor_to_device: CPU device index must be zero");
  }
  if (tensor.dtype() != DType::kFloat32) {
    return Unsupported("tensor_to_device: only float32 is supported");
  }

  std::vector<float> values(static_cast<size_t>(tensor.numel()));
  size_t written = 0;
  Status status =
      tensor.storage()->CopyToHostF32(values.data(), values.size(), &written);
  if (!status.ok()) return status;
  if (written != values.size()) {
    return InternalError(
        "tensor_to_device: storage returned an inconsistent element count");
  }

  const Backend* backend = nullptr;
  status = Dispatcher::For(Device::kCpu, &backend);
  if (!status.ok()) return status;
  return backend->FromData(tensor.shape(), values.data(), out);
}

Status WithRequiresGrad(const Tensor&, bool, std::shared_ptr<Tensor>* out) {
  if (out != nullptr) *out = nullptr;
  return Disabled("tensor_with_requires_grad");
}

Status RequiresGrad(const Tensor&, uint8_t* out_requires_grad) {
  if (out_requires_grad == nullptr) {
    return InvalidArgument("tensor_requires_grad: output pointer is null");
  }
  *out_requires_grad = 0;
  return Disabled("tensor_requires_grad");
}

Status Backward(const Tensor&) { return Disabled("tensor_backward"); }

Status Gradient(const Tensor&, std::shared_ptr<Tensor>* out) {
  if (out != nullptr) *out = nullptr;
  return Disabled("tensor_grad");
}

Status Relu(const Tensor&, std::shared_ptr<Tensor>* out) {
  if (out != nullptr) *out = nullptr;
  return Disabled("tensor_relu");
}

Status Sigmoid(const Tensor&, std::shared_ptr<Tensor>* out) {
  if (out != nullptr) *out = nullptr;
  return Disabled("tensor_sigmoid");
}

Status Tanh(const Tensor&, std::shared_ptr<Tensor>* out) {
  if (out != nullptr) *out = nullptr;
  return Disabled("tensor_tanh");
}

Status MseLoss(const Tensor&,
               const Tensor&,
               std::shared_ptr<Tensor>* out) {
  if (out != nullptr) *out = nullptr;
  return Disabled("mse_loss");
}

Status CrossEntropyLoss(const Tensor&,
                        const Tensor&,
                        std::shared_ptr<Tensor>* out) {
  if (out != nullptr) *out = nullptr;
  return Disabled("cross_entropy_loss");
}

Status LinearCreate(int64_t,
                    int64_t,
                    bool,
                    uint64_t* out_module) {
  if (out_module != nullptr) *out_module = 0;
  return Disabled("linear_create");
}

Status ModuleForward(uint64_t,
                     const Tensor&,
                     std::shared_ptr<Tensor>* out) {
  if (out != nullptr) *out = nullptr;
  return Disabled("module_forward");
}

Status ModuleSetTraining(uint64_t, bool) {
  return Disabled("module_set_training");
}

Status ModuleToDevice(uint64_t, Device, int32_t) {
  return Disabled("module_to_device");
}

Status ModuleParameterCount(uint64_t, size_t* out_count) {
  if (out_count != nullptr) *out_count = 0;
  return Disabled("module_parameter_count");
}

Status ModuleParameterAt(uint64_t,
                         size_t,
                         std::shared_ptr<Tensor>* out) {
  if (out != nullptr) *out = nullptr;
  return Disabled("module_parameter_at");
}

Status ModuleBufferCount(uint64_t, size_t* out_count) {
  if (out_count != nullptr) *out_count = 0;
  return Disabled("module_buffer_count");
}

Status ModuleBufferAt(uint64_t,
                      size_t,
                      std::shared_ptr<Tensor>* out) {
  if (out != nullptr) *out = nullptr;
  return Disabled("module_buffer_at");
}

Status ModuleSave(uint64_t, const std::string&) {
  return Disabled("module_save");
}

Status ModuleLoad(uint64_t, const std::string&) {
  return Disabled("module_load");
}

Status ModuleRelease(uint64_t) { return Disabled("module_release"); }

Status SgdCreate(uint64_t,
                 double,
                 double,
                 double,
                 uint64_t* out_optimizer) {
  if (out_optimizer != nullptr) *out_optimizer = 0;
  return Disabled("sgd_create");
}

Status AdamCreate(uint64_t,
                  double,
                  double,
                  double,
                  double,
                  double,
                  uint64_t* out_optimizer) {
  if (out_optimizer != nullptr) *out_optimizer = 0;
  return Disabled("adam_create");
}

Status AdamWCreate(uint64_t,
                   double,
                   double,
                   double,
                   double,
                   double,
                   uint64_t* out_optimizer) {
  if (out_optimizer != nullptr) *out_optimizer = 0;
  return Disabled("adamw_create");
}

Status OptimizerZeroGrad(uint64_t) { return Disabled("optimizer_zero_grad"); }
Status OptimizerStep(uint64_t) { return Disabled("optimizer_step"); }
Status OptimizerRelease(uint64_t) { return Disabled("optimizer_release"); }

uint64_t LiveStorageBytes() { return 0; }

}  // namespace tensora::training
