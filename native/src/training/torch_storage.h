#ifndef TENSORA_TRAINING_TORCH_STORAGE_H_
#define TENSORA_TRAINING_TORCH_STORAGE_H_

#include <atomic>
#include <cstdint>
#include <memory>

#include <torch/torch.h>

#include "memory/tensor_storage.h"

namespace tensora::training {

namespace internal {

Status ValidateTorchStorageSize(int64_t numel, uint64_t* out_bytes);
Status ValidateHostCopyMetadata(int64_t expected_numel,
                                torch::ScalarType actual_type,
                                int64_t actual_numel);

}  // namespace internal

class TorchStorage final : public TensorStorage {
 public:
  static Status FromTensor(torch::Tensor tensor,
                           std::shared_ptr<TorchStorage>* out);

  ~TorchStorage() override;

  TorchStorage(const TorchStorage&) = delete;
  TorchStorage& operator=(const TorchStorage&) = delete;

  StorageKind kind() const override { return StorageKind::kTorch; }
  const torch::Tensor& tensor() const { return tensor_; }
  torch::Tensor& mutable_tensor() { return tensor_; }

  Status CopyToHostF32(float* out_values,
                       size_t capacity,
                       size_t* out_written) const override;
  uint64_t byte_size() const override { return byte_size_; }

  static uint64_t LiveBytes();

 private:
  explicit TorchStorage(torch::Tensor tensor);

  torch::Tensor tensor_;
  uint64_t byte_size_;
  static std::atomic<uint64_t> live_bytes_;
};

}  // namespace tensora::training

#endif  // TENSORA_TRAINING_TORCH_STORAGE_H_
