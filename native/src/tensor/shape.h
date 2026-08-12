#ifndef TENSORA_TENSOR_SHAPE_H_
#define TENSORA_TENSOR_SHAPE_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/status.h"

namespace tensora {

inline constexpr size_t kMaxTensorRank = 32;

struct ShapeInfo {
  std::vector<int64_t> dimensions;
  std::vector<uint64_t> strides;
  uint64_t numel = 0;
  size_t byte_size = 0;
};

Status ValidateShape(const int64_t* dims, size_t rank, ShapeInfo* out);
bool SameShape(const ShapeInfo& left, const ShapeInfo& right);

}  // namespace tensora

#endif  // TENSORA_TENSOR_SHAPE_H_
