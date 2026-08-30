#ifndef TENSORA_BACKENDS_CPU_CPU_BACKEND_INTERNAL_H_
#define TENSORA_BACKENDS_CPU_CPU_BACKEND_INTERNAL_H_

#include <memory>
#include <vector>

#include "core/status.h"
#include "memory/cpu_storage.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

namespace tensora::cpu_backend_internal {

Status MakeTensor(ShapeInfo shape,
                  std::shared_ptr<CpuStorage> storage,
                  std::shared_ptr<Tensor>* out);
Status MakeView(ShapeInfo shape,
                const Tensor& source,
                std::shared_ptr<Tensor>* out);
Status EnsureCpuFloat32(const Tensor& tensor, const char* operation);
Status LogicalValues(const Tensor& tensor,
                     const char* operation,
                     std::vector<float>* out);

// Resolves a rank-2 CPU tensor into a raw base pointer plus row/column strides
// after proving the view lies inside its backing storage. Kernels consume the
// strides directly, so a transposed or otherwise strided operand needs no
// materialization before it is multiplied.
Status CpuMatrixOperand(const Tensor& tensor,
                        const char* operation,
                        const float** out_base,
                        int64_t* out_row_stride,
                        int64_t* out_col_stride);

}  // namespace tensora::cpu_backend_internal

#endif  // TENSORA_BACKENDS_CPU_CPU_BACKEND_INTERNAL_H_
