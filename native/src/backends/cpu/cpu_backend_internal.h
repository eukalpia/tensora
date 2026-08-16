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

}  // namespace tensora::cpu_backend_internal

#endif  // TENSORA_BACKENDS_CPU_CPU_BACKEND_INTERNAL_H_
