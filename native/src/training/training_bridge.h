#ifndef TENSORA_TRAINING_TRAINING_BRIDGE_H_
#define TENSORA_TRAINING_TRAINING_BRIDGE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include "autograd/autograd.h"
#include "core/status.h"
#include "tensor/tensor.h"

namespace tensora::training {

Status IsAvailable(uint8_t* out_available);
Status DeviceCount(Device device, uint32_t* out_count);
Status CudaDeviceCount(uint32_t* out_count);
Status ManualSeed(uint64_t seed);

Status Transfer(const Tensor& tensor,
                Device device,
                int32_t device_index,
                std::shared_ptr<Tensor>* out);
Status WithRequiresGrad(const Tensor& tensor,
                        bool requires_grad,
                        std::shared_ptr<Tensor>* out);
Status RequiresGrad(const Tensor& tensor, uint8_t* out_requires_grad);
#if defined(TENSORA_WITH_TORCH)
// A Torch-enabled runtime can still own canonical Tensora CPU tensors. Route
// those tensors through their native AutogradMeta instead of treating every
// non-Torch storage object as frozen. Torch-backed tensors continue through the
// provider implementation above.
inline Status RequiresGrad(Tensor& tensor, uint8_t* out_requires_grad) {
  if (out_requires_grad == nullptr) {
    return InvalidArgument("tensor_requires_grad: output pointer is null");
  }
  *out_requires_grad = 0;
  if (tensor.storage()->kind() == StorageKind::kTorch) {
    return RequiresGrad(static_cast<const Tensor&>(tensor), out_requires_grad);
  }

  const auto& meta = tensor.autograd_meta();
  if (!meta) return Status::Ok();
  std::lock_guard<std::mutex> lock(meta->mutex);
  *out_requires_grad = meta->requires_grad ? 1 : 0;
  return Status::Ok();
}
#endif
Status Backward(const Tensor& tensor);
Status Gradient(const Tensor& tensor, std::shared_ptr<Tensor>* out);
Status Relu(const Tensor& tensor, std::shared_ptr<Tensor>* out);
Status Sigmoid(const Tensor& tensor, std::shared_ptr<Tensor>* out);
Status Tanh(const Tensor& tensor, std::shared_ptr<Tensor>* out);
Status MseLoss(const Tensor& prediction,
               const Tensor& target,
               std::shared_ptr<Tensor>* out);
Status CrossEntropyLoss(const Tensor& logits,
                        const Tensor& one_hot_target,
                        std::shared_ptr<Tensor>* out);

Status LinearCreate(int64_t in_features,
                    int64_t out_features,
                    bool use_bias,
                    uint64_t* out_module);
Status ModuleForward(uint64_t module,
                     const Tensor& input,
                     std::shared_ptr<Tensor>* out);
Status ModuleSetTraining(uint64_t module, bool training);
Status ModuleToDevice(uint64_t module, Device device, int32_t device_index);
Status ModuleParameterCount(uint64_t module, size_t* out_count);
Status ModuleParameterAt(uint64_t module,
                         size_t index,
                         std::shared_ptr<Tensor>* out);
Status ModuleBufferCount(uint64_t module, size_t* out_count);
Status ModuleBufferAt(uint64_t module,
                      size_t index,
                      std::shared_ptr<Tensor>* out);
Status ModuleSave(uint64_t module, const std::string& path);
Status ModuleLoad(uint64_t module, const std::string& path);
Status ModuleRelease(uint64_t module);

Status SgdCreate(uint64_t module,
                 double learning_rate,
                 double momentum,
                 double weight_decay,
                 uint64_t* out_optimizer);
Status AdamCreate(uint64_t module,
                  double learning_rate,
                  double beta1,
                  double beta2,
                  double epsilon,
                  double weight_decay,
                  uint64_t* out_optimizer);
Status AdamWCreate(uint64_t module,
                   double learning_rate,
                   double beta1,
                   double beta2,
                   double epsilon,
                   double weight_decay,
                   uint64_t* out_optimizer);
Status OptimizerZeroGrad(uint64_t optimizer);
Status OptimizerStep(uint64_t optimizer);
Status OptimizerRelease(uint64_t optimizer);

uint64_t LiveStorageBytes();

}  // namespace tensora::training

#endif  // TENSORA_TRAINING_TRAINING_BRIDGE_H_
