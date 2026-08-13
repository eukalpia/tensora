#include "tensor/tensor.h"

#include <utility>

namespace tensora {

Tensor::Tensor(ShapeInfo shape,
               std::shared_ptr<TensorStorage> storage,
               DType dtype,
               Device device,
               int32_t device_index)
    : shape_(std::move(shape)),
      storage_(std::move(storage)),
      dtype_(dtype),
      device_(device),
      device_index_(device_index) {}

}  // namespace tensora
