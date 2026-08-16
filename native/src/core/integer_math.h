#ifndef TENSORA_CORE_INTEGER_MATH_H_
#define TENSORA_CORE_INTEGER_MATH_H_

#include <cstdint>
#include <limits>
#include <string>

#include "core/status.h"

namespace tensora {

inline Status CheckedAddU64(uint64_t left, uint64_t right, const char* operation, uint64_t* out) {
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

}  // namespace tensora

#endif  // TENSORA_CORE_INTEGER_MATH_H_
