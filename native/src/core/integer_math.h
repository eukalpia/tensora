#ifndef TENSORA_CORE_INTEGER_MATH_H_
#define TENSORA_CORE_INTEGER_MATH_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include "core/status.h"

namespace tensora {

inline Status CheckedAddU64(uint64_t left,
                            uint64_t right,
                            const char* operation,
                            uint64_t* out) {
  if (out == nullptr) {
    return InvalidArgument(std::string(operation) + ": output pointer is null");
  }
  *out = 0;
  if (right > std::numeric_limits<uint64_t>::max() - left) {
    return InternalError(std::string(operation) + ": counter overflow");
  }
  *out = left + right;
  return Status::Ok();
}

inline Status CheckedMultiplyU64(uint64_t left,
                                 uint64_t right,
                                 const char* operation,
                                 uint64_t* out) {
  if (out == nullptr) {
    return InvalidArgument(std::string(operation) + ": output pointer is null");
  }
  *out = 0;
  if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
    return InvalidShape(std::string(operation) + ": multiplication overflow");
  }
  *out = left * right;
  return Status::Ok();
}

inline Status CheckedByteSize(uint64_t element_count,
                              size_t element_width,
                              const char* operation,
                              size_t* out) {
  if (out == nullptr) {
    return InvalidArgument(std::string(operation) + ": output pointer is null");
  }
  *out = 0;
  if (element_width == 0) {
    return InvalidArgument(std::string(operation) +
                           ": element width must be positive");
  }
  if (element_count >
      static_cast<uint64_t>(std::numeric_limits<size_t>::max() /
                            element_width)) {
    return InvalidShape(std::string(operation) + ": byte size overflows size_t");
  }
  *out = static_cast<size_t>(element_count) * element_width;
  return Status::Ok();
}

}  // namespace tensora

#endif  // TENSORA_CORE_INTEGER_MATH_H_
