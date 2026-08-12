#include "tensor/shape.h"

#include <limits>
#include <sstream>

namespace tensora {

Status ValidateShape(const int64_t* dims, size_t rank, ShapeInfo* out) {
  if (out == nullptr) {
    return InvalidArgument("shape: output metadata pointer is null");
  }
  if (rank > kMaxTensorRank) {
    std::ostringstream stream;
    stream << "shape: rank " << rank << " exceeds Milestone 1 limit "
           << kMaxTensorRank;
    return InvalidShape(stream.str());
  }
  if (rank > 0 && dims == nullptr) {
    return InvalidArgument("shape: dimensions pointer is null for nonzero rank");
  }

  ShapeInfo result;
  result.dimensions.reserve(rank);
  result.strides.assign(rank, 1);

  uint64_t numel = 1;
  constexpr uint64_t kMaxNumel =
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max());

  for (size_t i = 0; i < rank; ++i) {
    const int64_t dim = dims[i];
    if (dim <= 0) {
      std::ostringstream stream;
      stream << "shape: dimension " << i << " must be positive, got " << dim;
      return InvalidShape(stream.str());
    }

    const uint64_t value = static_cast<uint64_t>(dim);
    if (numel > kMaxNumel / value) {
      return InvalidShape("shape: element count overflows supported int64 range");
    }
    numel *= value;
    result.dimensions.push_back(dim);
  }

  if (rank == 0) {
    numel = 1;
  }

  if (numel >
      static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(float))) {
    return InvalidShape("shape: float32 allocation byte size overflows size_t");
  }

  if (rank > 0) {
    uint64_t stride = 1;
    for (size_t i = rank; i-- > 0;) {
      result.strides[i] = stride;
      const uint64_t dim = static_cast<uint64_t>(result.dimensions[i]);
      if (i > 0) {
        if (stride > kMaxNumel / dim) {
          return InvalidShape("shape: stride calculation overflow");
        }
        stride *= dim;
      }
    }
  }

  result.numel = numel;
  result.byte_size = static_cast<size_t>(numel) * sizeof(float);
  *out = std::move(result);
  return Status::Ok();
}

bool SameShape(const ShapeInfo& left, const ShapeInfo& right) {
  return left.dimensions == right.dimensions;
}

}  // namespace tensora
