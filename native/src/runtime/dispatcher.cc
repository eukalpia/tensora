#include "runtime/dispatcher.h"

#include "backends/cpu/cpu_backend.h"

#if defined(TENSORA_WITH_TORCH)
#include "training/torch_backend.h"
#endif

namespace tensora {
namespace {

const Backend* CpuBackendInstance() {
  static const CpuBackend backend;
  return &backend;
}

#if defined(TENSORA_WITH_TORCH)
const Backend* TorchBackendInstance() {
  static const training::TorchBackend backend;
  return &backend;
}
#endif

}  // namespace

Status Dispatcher::For(Device device, const Backend** out) {
  if (out == nullptr) {
    return InvalidArgument("dispatcher: output backend pointer is null");
  }
  *out = nullptr;

  if (device == Device::kCpu) {
    *out = CpuBackendInstance();
    return Status::Ok();
  }

#if defined(TENSORA_WITH_TORCH)
  if (device == Device::kCuda) {
    *out = TorchBackendInstance();
    return Status::Ok();
  }
#endif

  return Unsupported("dispatcher: requested device is not supported");
}

Status Dispatcher::ForTensor(const Tensor& tensor, const Backend** out) {
  if (out == nullptr) {
    return InvalidArgument("dispatcher: output backend pointer is null");
  }
  *out = nullptr;

#if defined(TENSORA_WITH_TORCH)
  if (tensor.storage()->kind() == StorageKind::kTorch) {
    *out = TorchBackendInstance();
    return Status::Ok();
  }
#endif

  return For(tensor.device(), out);
}

Status Dispatcher::ForTensors(const Tensor& left,
                              const Tensor& right,
                              const Backend** out) {
  if (out == nullptr) {
    return InvalidArgument("dispatcher: output backend pointer is null");
  }
  *out = nullptr;

  if (left.device() != right.device() ||
      left.device_index() != right.device_index()) {
    return InvalidArgument(
        "dispatcher: input tensors must use the same device");
  }

#if defined(TENSORA_WITH_TORCH)
  if (left.storage()->kind() == StorageKind::kTorch ||
      right.storage()->kind() == StorageKind::kTorch) {
    *out = TorchBackendInstance();
    return Status::Ok();
  }
#endif

  return For(left.device(), out);
}

}  // namespace tensora
