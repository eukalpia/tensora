#ifndef TENSORA_TRAINING_TORCH_BACKEND_H_
#define TENSORA_TRAINING_TORCH_BACKEND_H_

#include <cstdint>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <utility>

#include <torch/torch.h>

#include "backends/backend.h"

namespace tensora::training {

namespace internal {

Status TorchFailure(const char* operation, const c10::Error& error);

/// Contains LibTorch exceptions at one deterministic backend boundary.
template <typename Operation>
Status GuardTorch(const char* operation, Operation&& body) {
  try {
    return std::forward<Operation>(body)();
  } catch (const c10::Error& error) {
    return TorchFailure(operation, error);
  }
}

/// Converts allocation failures into the stable Tensora status model.
template <typename Operation>
Status GuardAllocation(const char* operation, Operation&& body) {
  try {
    return std::forward<Operation>(body)();
  } catch (const std::bad_alloc&) {
    return OutOfMemory(std::string(operation) + ": allocation failed");
  }
}

bool MatchesAccelerator(Device device, c10::DeviceType accelerator);

/// Applies device-count semantics to a captured accelerator snapshot.
///
/// This helper is intentionally hardware-independent so error, mismatch, and
/// accelerator-count behavior can be validated on hosted CPU runners without
/// pretending that physical accelerator execution has been qualified.
Status DeviceCountFromSnapshot(
    Device device,
    std::optional<c10::DeviceType> accelerator,
    c10::DeviceIndex accelerator_count,
    uint32_t* out_count);

/// Resolves a requested Tensora device from a captured accelerator snapshot.
Status TorchDeviceFromSnapshot(
    Device device,
    int32_t device_index,
    std::optional<c10::DeviceType> accelerator,
    c10::DeviceIndex accelerator_count,
    torch::Device* out);

/// Maps a LibTorch device identity back to Tensora's public device model.
Status MapTorchDevice(c10::DeviceType torch_type,
                      c10::DeviceIndex torch_index,
                      std::optional<c10::DeviceType> accelerator,
                      Device* out_device,
                      int32_t* out_index);

}  // namespace internal

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
Status WrapTorchTensor(
    torch::Tensor tensor,
    std::shared_ptr<Tensor>* out,
    std::shared_ptr<TensorIdentityAnchor> identity_anchor = nullptr);
Status TorchDevice(Device device, int32_t device_index, torch::Device* out);

}  // namespace tensora::training

#endif  // TENSORA_TRAINING_TORCH_BACKEND_H_
