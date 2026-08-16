#ifndef TENSORA_TENSOR_DTYPE_H_
#define TENSORA_TENSOR_DTYPE_H_

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include "core/status.h"
#include "tensora.h"

namespace tensora {

enum class DType : uint32_t {
  kFloat32 = TS_DTYPE_FLOAT32,
  kFloat16 = TS_DTYPE_FLOAT16,
  kBFloat16 = TS_DTYPE_BFLOAT16,
  kFloat64 = TS_DTYPE_FLOAT64,
  kInt8 = TS_DTYPE_INT8,
  kUInt8 = TS_DTYPE_UINT8,
  kInt16 = TS_DTYPE_INT16,
  kInt32 = TS_DTYPE_INT32,
  kInt64 = TS_DTYPE_INT64,
  kBool = TS_DTYPE_BOOL,
};

inline const char* DTypeName(DType dtype) {
  switch (dtype) {
    case DType::kFloat16:
      return "float16";
    case DType::kBFloat16:
      return "bfloat16";
    case DType::kFloat32:
      return "float32";
    case DType::kFloat64:
      return "float64";
    case DType::kInt8:
      return "int8";
    case DType::kUInt8:
      return "uint8";
    case DType::kInt16:
      return "int16";
    case DType::kInt32:
      return "int32";
    case DType::kInt64:
      return "int64";
    case DType::kBool:
      return "bool";
  }
  return "unknown";
}

inline Status DTypeFromCode(uint32_t code, DType* out) {
  if (out == nullptr) {
    return InvalidArgument("dtype: output pointer is null");
  }
  switch (code) {
    case TS_DTYPE_FLOAT16:
      *out = DType::kFloat16;
      return Status::Ok();
    case TS_DTYPE_BFLOAT16:
      *out = DType::kBFloat16;
      return Status::Ok();
    case TS_DTYPE_FLOAT32:
      *out = DType::kFloat32;
      return Status::Ok();
    case TS_DTYPE_FLOAT64:
      *out = DType::kFloat64;
      return Status::Ok();
    case TS_DTYPE_INT8:
      *out = DType::kInt8;
      return Status::Ok();
    case TS_DTYPE_UINT8:
      *out = DType::kUInt8;
      return Status::Ok();
    case TS_DTYPE_INT16:
      *out = DType::kInt16;
      return Status::Ok();
    case TS_DTYPE_INT32:
      *out = DType::kInt32;
      return Status::Ok();
    case TS_DTYPE_INT64:
      *out = DType::kInt64;
      return Status::Ok();
    case TS_DTYPE_BOOL:
      *out = DType::kBool;
      return Status::Ok();
    default:
      return InvalidArgument("dtype: unknown stable dtype code " +
                             std::to_string(code));
  }
}

inline size_t DTypeByteWidth(DType dtype) {
  switch (dtype) {
    case DType::kFloat16:
    case DType::kBFloat16:
    case DType::kInt16:
      return 2;
    case DType::kFloat32:
    case DType::kInt32:
      return 4;
    case DType::kFloat64:
    case DType::kInt64:
      return 8;
    case DType::kInt8:
    case DType::kUInt8:
    case DType::kBool:
      return 1;
  }
  return 0;
}

inline bool DTypeIsFloatingPoint(DType dtype) {
  return dtype == DType::kFloat16 || dtype == DType::kBFloat16 ||
         dtype == DType::kFloat32 || dtype == DType::kFloat64;
}

inline bool DTypeIsInteger(DType dtype) {
  return dtype == DType::kInt8 || dtype == DType::kUInt8 ||
         dtype == DType::kInt16 || dtype == DType::kInt32 ||
         dtype == DType::kInt64;
}

inline bool DTypeSupportsArithmetic(DType dtype) {
  return dtype != DType::kBool;
}

inline DType DTypeReductionAccumulator(DType dtype) {
  switch (dtype) {
    case DType::kFloat16:
    case DType::kBFloat16:
    case DType::kFloat32:
      return DType::kFloat32;
    case DType::kFloat64:
      return DType::kFloat64;
    case DType::kInt8:
    case DType::kUInt8:
    case DType::kInt16:
    case DType::kInt32:
    case DType::kInt64:
    case DType::kBool:
      return DType::kInt64;
  }
  return DType::kFloat32;
}

inline DType PromoteDTypes(DType left, DType right) {
  if (left == right) return left;
  if (left == DType::kBool) return right;
  if (right == DType::kBool) return left;

  if (DTypeIsFloatingPoint(left) || DTypeIsFloatingPoint(right)) {
    if (left == DType::kFloat64 || right == DType::kFloat64) {
      return DType::kFloat64;
    }
    if (left == DType::kFloat32 || right == DType::kFloat32) {
      return DType::kFloat32;
    }
    if ((left == DType::kFloat16 && right == DType::kBFloat16) ||
        (left == DType::kBFloat16 && right == DType::kFloat16)) {
      return DType::kFloat32;
    }
    return DTypeIsFloatingPoint(left) ? left : right;
  }

  if ((left == DType::kUInt8 && right == DType::kInt8) ||
      (left == DType::kInt8 && right == DType::kUInt8)) {
    return DType::kInt16;
  }
  if (left == DType::kUInt8) return right;
  if (right == DType::kUInt8) return left;
  return DTypeByteWidth(left) >= DTypeByteWidth(right) ? left : right;
}

inline float Float16BitsToFloat(uint16_t value) {
  const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16u;
  uint32_t exponent = static_cast<uint32_t>((value >> 10u) & 0x1fu);
  uint32_t mantissa = static_cast<uint32_t>(value & 0x03ffu);
  uint32_t bits = 0;

  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      int32_t adjusted_exponent = -14;
      while ((mantissa & 0x0400u) == 0) {
        mantissa <<= 1u;
        --adjusted_exponent;
      }
      mantissa &= 0x03ffu;
      bits = sign |
             (static_cast<uint32_t>(adjusted_exponent + 127) << 23u) |
             (mantissa << 13u);
    }
  } else if (exponent == 0x1fu) {
    bits = sign | 0x7f800000u | (mantissa << 13u);
    if (mantissa != 0) bits |= 0x00400000u;
  } else {
    bits = sign | ((exponent + 112u) << 23u) | (mantissa << 13u);
  }
  return std::bit_cast<float>(bits);
}

inline uint16_t FloatToFloat16Bits(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  const uint16_t sign = static_cast<uint16_t>((bits >> 16u) & 0x8000u);
  const uint32_t exponent = (bits >> 23u) & 0xffu;
  uint32_t mantissa = bits & 0x007fffffu;

  if (exponent == 0xffu) {
    if (mantissa == 0) return static_cast<uint16_t>(sign | 0x7c00u);
    uint16_t payload = static_cast<uint16_t>(mantissa >> 13u);
    if (payload == 0) payload = 1;
    return static_cast<uint16_t>(sign | 0x7c00u | payload | 0x0200u);
  }

  int32_t target_exponent = static_cast<int32_t>(exponent) - 127 + 15;
  if (target_exponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00u);
  }
  if (target_exponent <= 0) {
    if (target_exponent < -10) return sign;
    mantissa |= 0x00800000u;
    const uint32_t shift = static_cast<uint32_t>(14 - target_exponent);
    uint32_t rounded = mantissa >> shift;
    const uint32_t remainder_mask = (uint32_t{1} << shift) - 1u;
    const uint32_t remainder = mantissa & remainder_mask;
    const uint32_t halfway = uint32_t{1} << (shift - 1u);
    if (remainder > halfway ||
        (remainder == halfway && (rounded & 1u) != 0)) {
      ++rounded;
    }
    return static_cast<uint16_t>(sign | rounded);
  }

  uint32_t rounded_mantissa = mantissa >> 13u;
  const uint32_t remainder = mantissa & 0x1fffu;
  if (remainder > 0x1000u ||
      (remainder == 0x1000u && (rounded_mantissa & 1u) != 0)) {
    ++rounded_mantissa;
    if (rounded_mantissa == 0x0400u) {
      rounded_mantissa = 0;
      ++target_exponent;
      if (target_exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
      }
    }
  }
  return static_cast<uint16_t>(
      sign | (static_cast<uint16_t>(target_exponent) << 10u) |
      static_cast<uint16_t>(rounded_mantissa));
}

inline float BFloat16BitsToFloat(uint16_t value) {
  return std::bit_cast<float>(static_cast<uint32_t>(value) << 16u);
}

inline uint16_t FloatToBFloat16Bits(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  if ((bits & 0x7f800000u) == 0x7f800000u &&
      (bits & 0x007fffffu) != 0) {
    return static_cast<uint16_t>((bits >> 16u) | 0x0040u);
  }
  const uint32_t least_significant = (bits >> 16u) & 1u;
  bits += 0x00007fffu + least_significant;
  return static_cast<uint16_t>(bits >> 16u);
}

}  // namespace tensora

#endif  // TENSORA_TENSOR_DTYPE_H_
