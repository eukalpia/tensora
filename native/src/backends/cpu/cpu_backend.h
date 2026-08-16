#ifndef TENSORA_BACKENDS_CPU_CPU_BACKEND_H_
#define TENSORA_BACKENDS_CPU_CPU_BACKEND_H_

#include "backends/backend.h"
#include "memory/cpu_storage.h"

namespace tensora {

class CpuBackend final : public Backend {
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

}  // namespace tensora

#endif  // TENSORA_BACKENDS_CPU_CPU_BACKEND_H_
