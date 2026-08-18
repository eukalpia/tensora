#ifndef TENSORA_TENSOR_TYPED_TENSOR_ABI_H_
#define TENSORA_TENSOR_TYPED_TENSOR_ABI_H_

#include <memory>

#include "core/status.h"
#include "tensor/dtype.h"
#include "tensor/shape.h"
#include "tensora.h"

namespace tensora {

class CpuStorage;
class Tensor;

namespace typed_tensor_abi {

Status Lookup(ts_tensor_t handle, std::shared_ptr<Tensor>* out);
Status Insert(std::shared_ptr<Tensor> tensor, ts_tensor_t* out_handle);
Status MakeCpuTensor(ShapeInfo shape,
                     DType dtype,
                     std::shared_ptr<CpuStorage> storage,
                     std::shared_ptr<Tensor>* out);

// Materializes the logical order of arbitrary CPU views before casting, so the
// returned tensor is contiguous and independent from the source storage.
// Callers validate the public output pointer and device contract first.
Status CastCpuTensor(const Tensor& source,
                     DType target_dtype,
                     std::shared_ptr<Tensor>* out);

}  // namespace typed_tensor_abi
}  // namespace tensora

#endif  // TENSORA_TENSOR_TYPED_TENSOR_ABI_H_
