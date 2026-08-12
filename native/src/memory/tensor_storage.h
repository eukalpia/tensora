#ifndef TENSORA_MEMORY_TENSOR_STORAGE_H_
#define TENSORA_MEMORY_TENSOR_STORAGE_H_

#include <cstddef>
#include <cstdint>

#include "core/status.h"

namespace tensora {

class TensorStorage {
 public:
  virtual ~TensorStorage() = default;

  TensorStorage(const TensorStorage&) = delete;
  TensorStorage& operator=(const TensorStorage&) = delete;

  virtual Status CopyToHostF32(float* out_values,
                               size_t capacity,
                               size_t* out_written) const = 0;
  virtual uint64_t byte_size() const = 0;

 protected:
  TensorStorage() = default;
};

}  // namespace tensora

#endif  // TENSORA_MEMORY_TENSOR_STORAGE_H_
