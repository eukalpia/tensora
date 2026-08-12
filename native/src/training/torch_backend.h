#ifndef TENSORA_TRAINING_TORCH_BACKEND_H_
#define TENSORA_TRAINING_TORCH_BACKEND_H_

#include <memory>

#include <torch/torch.h>

#include "backends/backend.h"

namespace tensora::training {

class TorchBackend final : public Backend {
 public:
  Status FromData(const ShapeInfo& shape,
                  const float* data,
                  std::shared_ptr<Tensor>* out) const override;
  Status Full(const ShapeInfo& shape,
              float value,
              std::shared_ptr<Tensor>* out) const override;
  Status Reshape(const Tensor& tensor,
                 const ShapeInfo& shape,
                 std::shared_ptr<Tensor>* out) const override;
  Status Transpose2D(const Tensor& tensor,
                     std::shared_ptr<Tensor>* out) const override;
  Status Add(const Tensor& left,
             const Tensor& right,
             std::shared_ptr<Tensor>* out) const override;
  Status Multiply(const Tensor& left,
                  const Tensor& right,
                  std::shared_ptr<Tensor>* out) const override;
  Status Sum(const Tensor& tensor,
             std::shared_ptr<Tensor>* out) const override;
  Status Matmul(const Tensor& left,
                const Tensor& right,
                std::shared_ptr<Tensor>* out) const override;
};

Status TensorToTorch(const Tensor& tensor, torch::Tensor* out);
Status WrapTorchTensor(torch::Tensor tensor, std::shared_ptr<Tensor>* out);
Status TorchDevice(Device device, int32_t device_index, torch::Device* out);

}  // namespace tensora::training

#endif  // TENSORA_TRAINING_TORCH_BACKEND_H_
