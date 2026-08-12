#include "tensor/tensor.h"

namespace tensora {

Tensor::Tensor(ShapeInfo shape,
               std::shared_ptr<CpuStorage> storage,
               DType dtype,
               Device device)
    : shape_(std::move(shape)),
      storage_(std::move(storage)),
      dtype_(dtype),
      device_(device) {}

}  // namespace tensora
