#ifndef TENSORA_BACKENDS_BACKEND_H_
#define TENSORA_BACKENDS_BACKEND_H_

#include <memory>

#include "tensor/tensor.h"

namespace tensora {

class Backend {
 public:
  virtual ~Backend() = default;

  virtual Status FromData(const ShapeInfo& shape,
                          const float* data,
                          std::shared_ptr<Tensor>* out) const = 0;
  virtual Status Full(const ShapeInfo& shape,
                      float value,
                      std::shared_ptr<Tensor>* out) const = 0;
  virtual Status Reshape(const Tensor& tensor,
                         const ShapeInfo& shape,
                         std::shared_ptr<Tensor>* out) const = 0;
  virtual Status Transpose2D(const Tensor& tensor,
                             std::shared_ptr<Tensor>* out) const = 0;
  virtual Status Add(const Tensor& left,
                     const Tensor& right,
                     std::shared_ptr<Tensor>* out) const = 0;
  virtual Status Multiply(const Tensor& left,
                          const Tensor& right,
                          std::shared_ptr<Tensor>* out) const = 0;
  virtual Status Sum(const Tensor& tensor,
                     std::shared_ptr<Tensor>* out) const = 0;
  virtual Status Matmul(const Tensor& left,
                        const Tensor& right,
                        std::shared_ptr<Tensor>* out) const = 0;
};

}  // namespace tensora

#endif  // TENSORA_BACKENDS_BACKEND_H_
