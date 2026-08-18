#include "memory/cpu_storage.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/allocation_guard.h"
#include "core/integer_math.h"

namespace tensora {
namespace {

struct ScalarValue {
  enum class Kind { kFloating, kSigned, kUnsigned, kBoolean };

  Kind kind = Kind::kBoolean;
  double floating = 0.0;
  int64_t signed_value = 0;
  uint64_t unsigned_value = 0;
};

Status ExpectedBytes(uint64_t numel,
                     DType dtype,
                     const char* operation,
                     size_t* out) {
  return CheckedByteSize(numel, DTypeByteWidth(dtype), operation, out);
}

template <typename T>
T ReadRawScalar(const void* data) {
  T value{};
  std::memcpy(&value, data, sizeof(T));
  return value;
}

template <typename T>
CpuStorage::Data CopyRawVector(const void* data, uint64_t numel) {
  std::vector<T> values(static_cast<size_t>(numel));
  if (!values.empty()) {
    std::memcpy(values.data(), data, values.size() * sizeof(T));
  }
  return CpuStorage::Data(std::move(values));
}

template <typename T>
CpuStorage::Data FilledVector(uint64_t numel, T value) {
  return CpuStorage::Data(
      std::vector<T>(static_cast<size_t>(numel), value));
}

double ScalarAsDouble(const ScalarValue& value) {
  switch (value.kind) {
    case ScalarValue::Kind::kFloating:
      return value.floating;
    case ScalarValue::Kind::kSigned:
      return static_cast<double>(value.signed_value);
    case ScalarValue::Kind::kUnsigned:
      return static_cast<double>(value.unsigned_value);
    case ScalarValue::Kind::kBoolean:
      return value.unsigned_value == 0 ? 0.0 : 1.0;
  }
  return 0.0;
}

template <typename T>
Status ConvertToSigned(const ScalarValue& value,
                       const char* operation,
                       T* out) {
  static_assert(std::is_integral_v<T> && std::is_signed_v<T>);
  if (out == nullptr) {
    return InvalidArgument(std::string(operation) +
                           ": signed conversion output is null");
  }

  if (value.kind == ScalarValue::Kind::kSigned) {
    if (value.signed_value < static_cast<int64_t>(std::numeric_limits<T>::min()) ||
        value.signed_value > static_cast<int64_t>(std::numeric_limits<T>::max())) {
      return InvalidArgument(std::string(operation) +
                             ": integer value is outside destination range");
    }
    *out = static_cast<T>(value.signed_value);
    return Status::Ok();
  }
  if (value.kind == ScalarValue::Kind::kUnsigned ||
      value.kind == ScalarValue::Kind::kBoolean) {
    if (value.unsigned_value >
        static_cast<uint64_t>(std::numeric_limits<T>::max())) {
      return InvalidArgument(std::string(operation) +
                             ": unsigned value is outside destination range");
    }
    *out = static_cast<T>(value.unsigned_value);
    return Status::Ok();
  }

  const double floating = value.floating;
  if (!std::isfinite(floating)) {
    return InvalidArgument(std::string(operation) +
                           ": non-finite value cannot be cast to integer");
  }
  const double truncated = std::trunc(floating);
  if constexpr (std::is_same_v<T, int64_t>) {
    constexpr double kMinimum = -9223372036854775808.0;
    constexpr double kExclusiveMaximum = 9223372036854775808.0;
    if (truncated < kMinimum || truncated >= kExclusiveMaximum) {
      return InvalidArgument(std::string(operation) +
                             ": floating value is outside int64 range");
    }
  } else {
    if (truncated < static_cast<double>(std::numeric_limits<T>::min()) ||
        truncated > static_cast<double>(std::numeric_limits<T>::max())) {
      return InvalidArgument(std::string(operation) +
                             ": floating value is outside destination range");
    }
  }
  *out = static_cast<T>(truncated);
  return Status::Ok();
}

Status ConvertToUInt8(const ScalarValue& value,
                      const char* operation,
                      uint8_t* out) {
  if (out == nullptr) {
    return InvalidArgument(std::string(operation) +
                           ": uint8 conversion output is null");
  }
  if (value.kind == ScalarValue::Kind::kSigned) {
    if (value.signed_value < 0 || value.signed_value > 255) {
      return InvalidArgument(std::string(operation) +
                             ": signed value is outside uint8 range");
    }
    *out = static_cast<uint8_t>(value.signed_value);
    return Status::Ok();
  }
  if (value.kind == ScalarValue::Kind::kUnsigned ||
      value.kind == ScalarValue::Kind::kBoolean) {
    if (value.unsigned_value > 255) {
      return InvalidArgument(std::string(operation) +
                             ": unsigned value is outside uint8 range");
    }
    *out = static_cast<uint8_t>(value.unsigned_value);
    return Status::Ok();
  }
  if (!std::isfinite(value.floating)) {
    return InvalidArgument(std::string(operation) +
                           ": non-finite value cannot be cast to uint8");
  }
  const double truncated = std::trunc(value.floating);
  if (truncated < 0.0 || truncated > 255.0) {
    return InvalidArgument(std::string(operation) +
                           ": floating value is outside uint8 range");
  }
  *out = static_cast<uint8_t>(truncated);
  return Status::Ok();
}

bool ConvertToBool(const ScalarValue& value) {
  switch (value.kind) {
    case ScalarValue::Kind::kFloating:
      return value.floating != 0.0;
    case ScalarValue::Kind::kSigned:
      return value.signed_value != 0;
    case ScalarValue::Kind::kUnsigned:
    case ScalarValue::Kind::kBoolean:
      return value.unsigned_value != 0;
  }
  return false;
}

}  // namespace

std::atomic<uint64_t> CpuStorage::live_bytes_{0};

CpuStorage::CpuStorage(DType dtype,
                       uint64_t numel,
                       Data data,
                       uint64_t byte_size)
    : dtype_(dtype),
      numel_(numel),
      data_(std::move(data)),
      byte_size_(byte_size) {
  live_bytes_.fetch_add(byte_size_, std::memory_order_relaxed);
}

CpuStorage::~CpuStorage() {
  live_bytes_.fetch_sub(byte_size_, std::memory_order_relaxed);
}

Status CpuStorage::Filled(uint64_t numel,
                          float value,
                          std::shared_ptr<CpuStorage>* out) {
  size_t bytes = 0;
  Status status =
      ExpectedBytes(numel, DType::kFloat32, "cpu storage full", &bytes);
  if (!status.ok()) {
    if (status.code() == TS_INVALID_SHAPE) {
      return OutOfMemory(
          "cpu storage full: requested float32 allocation is too large");
    }
    return status;
  }
  return Full(&value, sizeof(value), numel, DType::kFloat32, out);
}

Status CpuStorage::FromData(const float* data,
                            uint64_t numel,
                            std::shared_ptr<CpuStorage>* out) {
  size_t bytes = 0;
  Status status =
      ExpectedBytes(numel, DType::kFloat32, "cpu storage", &bytes);
  if (!status.ok()) {
    if (status.code() == TS_INVALID_SHAPE) {
      return OutOfMemory(
          "cpu storage: requested float32 allocation is too large");
    }
    return status;
  }
  return FromRaw(data, bytes, numel, DType::kFloat32, out);
}

Status CpuStorage::FromRaw(const void* data,
                           size_t data_bytes,
                           uint64_t numel,
                           DType dtype,
                           std::shared_ptr<CpuStorage>* out) {
  if (out == nullptr) {
    return InvalidArgument("cpu storage: output pointer is null");
  }
  *out = nullptr;

  size_t expected_bytes = 0;
  Status status = ExpectedBytes(numel, dtype, "cpu storage", &expected_bytes);
  if (!status.ok()) return status;
  if (data_bytes != expected_bytes) {
    return InvalidArgument(
        "cpu storage: input byte count does not match dtype and element count");
  }
  if (expected_bytes > 0 && data == nullptr) {
    return InvalidArgument("cpu storage: input data pointer is null");
  }
  if (dtype == DType::kBool) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t index = 0; index < data_bytes; ++index) {
      if (bytes[index] > 1) {
        return InvalidArgument(
            "cpu storage: bool host representation must contain only 0 or 1");
      }
    }
  }

  return AllocationGuard("cpu storage", [&]() -> Status {
    Data values;
    switch (dtype) {
      case DType::kFloat16:
      case DType::kBFloat16:
        values = CopyRawVector<uint16_t>(data, numel);
        break;
      case DType::kFloat32:
        values = CopyRawVector<float>(data, numel);
        break;
      case DType::kFloat64:
        values = CopyRawVector<double>(data, numel);
        break;
      case DType::kInt8:
        values = CopyRawVector<int8_t>(data, numel);
        break;
      case DType::kUInt8:
      case DType::kBool:
        values = CopyRawVector<uint8_t>(data, numel);
        break;
      case DType::kInt16:
        values = CopyRawVector<int16_t>(data, numel);
        break;
      case DType::kInt32:
        values = CopyRawVector<int32_t>(data, numel);
        break;
      case DType::kInt64:
        values = CopyRawVector<int64_t>(data, numel);
        break;
    }
    *out = std::shared_ptr<CpuStorage>(new CpuStorage(
        dtype, numel, std::move(values), static_cast<uint64_t>(expected_bytes)));
    return Status::Ok();
  });
}

Status CpuStorage::Full(const void* scalar,
                        size_t scalar_bytes,
                        uint64_t numel,
                        DType dtype,
                        std::shared_ptr<CpuStorage>* out) {
  if (out == nullptr) {
    return InvalidArgument("cpu storage full: output pointer is null");
  }
  *out = nullptr;
  const size_t expected_scalar_bytes = DTypeByteWidth(dtype);
  if (scalar_bytes != expected_scalar_bytes) {
    return InvalidArgument(
        "cpu storage full: scalar byte count does not match dtype");
  }
  if (scalar == nullptr) {
    return InvalidArgument("cpu storage full: scalar pointer is null");
  }

  size_t byte_size = 0;
  Status status =
      ExpectedBytes(numel, dtype, "cpu storage full", &byte_size);
  if (!status.ok()) return status;

  return AllocationGuard("cpu storage full", [&]() -> Status {
    Data values;
    switch (dtype) {
      case DType::kFloat16:
      case DType::kBFloat16:
        values = FilledVector<uint16_t>(numel, ReadRawScalar<uint16_t>(scalar));
        break;
      case DType::kFloat32:
        values = FilledVector<float>(numel, ReadRawScalar<float>(scalar));
        break;
      case DType::kFloat64:
        values = FilledVector<double>(numel, ReadRawScalar<double>(scalar));
        break;
      case DType::kInt8:
        values = FilledVector<int8_t>(numel, ReadRawScalar<int8_t>(scalar));
        break;
      case DType::kUInt8:
        values = FilledVector<uint8_t>(numel, ReadRawScalar<uint8_t>(scalar));
        break;
      case DType::kBool: {
        const uint8_t value = ReadRawScalar<uint8_t>(scalar);
        if (value > 1) {
          return InvalidArgument(
              "cpu storage full: bool scalar must be represented by 0 or 1");
        }
        values = FilledVector<uint8_t>(numel, value);
        break;
      }
      case DType::kInt16:
        values = FilledVector<int16_t>(numel, ReadRawScalar<int16_t>(scalar));
        break;
      case DType::kInt32:
        values = FilledVector<int32_t>(numel, ReadRawScalar<int32_t>(scalar));
        break;
      case DType::kInt64:
        values = FilledVector<int64_t>(numel, ReadRawScalar<int64_t>(scalar));
        break;
    }
    *out = std::shared_ptr<CpuStorage>(new CpuStorage(
        dtype, numel, std::move(values), static_cast<uint64_t>(byte_size)));
    return Status::Ok();
  });
}

Status CpuStorage::Cast(const CpuStorage& source,
                        DType target_dtype,
                        std::shared_ptr<CpuStorage>* out) {
  if (out == nullptr) {
    return InvalidArgument("cpu storage cast: output pointer is null");
  }
  *out = nullptr;

  if (source.dtype_ == target_dtype) {
    return AllocationGuard("cpu storage cast", [&]() -> Status {
      *out = std::shared_ptr<CpuStorage>(new CpuStorage(
          source.dtype_, source.numel_, source.data_, source.byte_size_));
      return Status::Ok();
    });
  }

  const auto read_value = [&](uint64_t index, ScalarValue* value) -> Status {
    if (value == nullptr || index >= source.numel_) {
      return InternalError("cpu storage cast: invalid source element request");
    }
    const size_t position = static_cast<size_t>(index);
    switch (source.dtype_) {
      case DType::kFloat16:
        value->kind = ScalarValue::Kind::kFloating;
        value->floating = Float16BitsToFloat(
            std::get<std::vector<uint16_t>>(source.data_)[position]);
        return Status::Ok();
      case DType::kBFloat16:
        value->kind = ScalarValue::Kind::kFloating;
        value->floating = BFloat16BitsToFloat(
            std::get<std::vector<uint16_t>>(source.data_)[position]);
        return Status::Ok();
      case DType::kFloat32:
        value->kind = ScalarValue::Kind::kFloating;
        value->floating =
            std::get<std::vector<float>>(source.data_)[position];
        return Status::Ok();
      case DType::kFloat64:
        value->kind = ScalarValue::Kind::kFloating;
        value->floating =
            std::get<std::vector<double>>(source.data_)[position];
        return Status::Ok();
      case DType::kInt8:
        value->kind = ScalarValue::Kind::kSigned;
        value->signed_value =
            std::get<std::vector<int8_t>>(source.data_)[position];
        return Status::Ok();
      case DType::kUInt8:
        value->kind = ScalarValue::Kind::kUnsigned;
        value->unsigned_value =
            std::get<std::vector<uint8_t>>(source.data_)[position];
        return Status::Ok();
      case DType::kInt16:
        value->kind = ScalarValue::Kind::kSigned;
        value->signed_value =
            std::get<std::vector<int16_t>>(source.data_)[position];
        return Status::Ok();
      case DType::kInt32:
        value->kind = ScalarValue::Kind::kSigned;
        value->signed_value =
            std::get<std::vector<int32_t>>(source.data_)[position];
        return Status::Ok();
      case DType::kInt64:
        value->kind = ScalarValue::Kind::kSigned;
        value->signed_value =
            std::get<std::vector<int64_t>>(source.data_)[position];
        return Status::Ok();
      case DType::kBool:
        value->kind = ScalarValue::Kind::kBoolean;
        value->unsigned_value =
            std::get<std::vector<uint8_t>>(source.data_)[position];
        return Status::Ok();
    }
    return InternalError("cpu storage cast: unsupported source dtype");
  };

  size_t target_bytes = 0;
  Status status = ExpectedBytes(source.numel_, target_dtype,
                                "cpu storage cast", &target_bytes);
  if (!status.ok()) return status;

  return AllocationGuard("cpu storage cast", [&]() -> Status {
    Data target_data;
    switch (target_dtype) {
      case DType::kFloat16: {
        std::vector<uint16_t> values(static_cast<size_t>(source.numel_));
        for (uint64_t index = 0; index < source.numel_; ++index) {
          ScalarValue value;
          Status read_status = read_value(index, &value);
          if (!read_status.ok()) return read_status;
          values[static_cast<size_t>(index)] =
              FloatToFloat16Bits(static_cast<float>(ScalarAsDouble(value)));
        }
        target_data = std::move(values);
        break;
      }
      case DType::kBFloat16: {
        std::vector<uint16_t> values(static_cast<size_t>(source.numel_));
        for (uint64_t index = 0; index < source.numel_; ++index) {
          ScalarValue value;
          Status read_status = read_value(index, &value);
          if (!read_status.ok()) return read_status;
          values[static_cast<size_t>(index)] =
              FloatToBFloat16Bits(static_cast<float>(ScalarAsDouble(value)));
        }
        target_data = std::move(values);
        break;
      }
      case DType::kFloat32: {
        std::vector<float> values(static_cast<size_t>(source.numel_));
        for (uint64_t index = 0; index < source.numel_; ++index) {
          ScalarValue value;
          Status read_status = read_value(index, &value);
          if (!read_status.ok()) return read_status;
          values[static_cast<size_t>(index)] =
              static_cast<float>(ScalarAsDouble(value));
        }
        target_data = std::move(values);
        break;
      }
      case DType::kFloat64: {
        std::vector<double> values(static_cast<size_t>(source.numel_));
        for (uint64_t index = 0; index < source.numel_; ++index) {
          ScalarValue value;
          Status read_status = read_value(index, &value);
          if (!read_status.ok()) return read_status;
          values[static_cast<size_t>(index)] = ScalarAsDouble(value);
        }
        target_data = std::move(values);
        break;
      }
      case DType::kInt8: {
        std::vector<int8_t> values(static_cast<size_t>(source.numel_));
        for (uint64_t index = 0; index < source.numel_; ++index) {
          ScalarValue value;
          Status read_status = read_value(index, &value);
          if (!read_status.ok()) return read_status;
          Status conversion = ConvertToSigned(
              value, "cpu storage cast", &values[static_cast<size_t>(index)]);
          if (!conversion.ok()) return conversion;
        }
        target_data = std::move(values);
        break;
      }
      case DType::kUInt8: {
        std::vector<uint8_t> values(static_cast<size_t>(source.numel_));
        for (uint64_t index = 0; index < source.numel_; ++index) {
          ScalarValue value;
          Status read_status = read_value(index, &value);
          if (!read_status.ok()) return read_status;
          Status conversion = ConvertToUInt8(
              value, "cpu storage cast", &values[static_cast<size_t>(index)]);
          if (!conversion.ok()) return conversion;
        }
        target_data = std::move(values);
        break;
      }
      case DType::kInt16: {
        std::vector<int16_t> values(static_cast<size_t>(source.numel_));
        for (uint64_t index = 0; index < source.numel_; ++index) {
          ScalarValue value;
          Status read_status = read_value(index, &value);
          if (!read_status.ok()) return read_status;
          Status conversion = ConvertToSigned(
              value, "cpu storage cast", &values[static_cast<size_t>(index)]);
          if (!conversion.ok()) return conversion;
        }
        target_data = std::move(values);
        break;
      }
      case DType::kInt32: {
        std::vector<int32_t> values(static_cast<size_t>(source.numel_));
        for (uint64_t index = 0; index < source.numel_; ++index) {
          ScalarValue value;
          Status read_status = read_value(index, &value);
          if (!read_status.ok()) return read_status;
          Status conversion = ConvertToSigned(
              value, "cpu storage cast", &values[static_cast<size_t>(index)]);
          if (!conversion.ok()) return conversion;
        }
        target_data = std::move(values);
        break;
      }
      case DType::kInt64: {
        std::vector<int64_t> values(static_cast<size_t>(source.numel_));
        for (uint64_t index = 0; index < source.numel_; ++index) {
          ScalarValue value;
          Status read_status = read_value(index, &value);
          if (!read_status.ok()) return read_status;
          Status conversion = ConvertToSigned(
              value, "cpu storage cast", &values[static_cast<size_t>(index)]);
          if (!conversion.ok()) return conversion;
        }
        target_data = std::move(values);
        break;
      }
      case DType::kBool: {
        std::vector<uint8_t> values(static_cast<size_t>(source.numel_));
        for (uint64_t index = 0; index < source.numel_; ++index) {
          ScalarValue value;
          Status read_status = read_value(index, &value);
          if (!read_status.ok()) return read_status;
          values[static_cast<size_t>(index)] =
              ConvertToBool(value) ? uint8_t{1} : uint8_t{0};
        }
        target_data = std::move(values);
        break;
      }
    }

    *out = std::shared_ptr<CpuStorage>(new CpuStorage(
        target_dtype, source.numel_, std::move(target_data),
        static_cast<uint64_t>(target_bytes)));
    return Status::Ok();
  });
}

Status CpuStorage::CopyToHostRaw(void* out_data,
                                 size_t capacity_bytes,
                                 size_t* out_written_bytes) const {
  if (out_written_bytes == nullptr) {
    return InvalidArgument("cpu storage: output byte count pointer is null");
  }
  *out_written_bytes = 0;
  if (capacity_bytes < byte_size_) {
    return InvalidArgument("cpu storage: output byte capacity is too small");
  }
  if (byte_size_ > 0 && out_data == nullptr) {
    return InvalidArgument("cpu storage: output data pointer is null");
  }

  std::visit(
      [&](const auto& values) {
        if (!values.empty()) {
          std::memcpy(out_data, values.data(),
                      values.size() * sizeof(typename std::decay_t<decltype(values)>::value_type));
        }
      },
      data_);
  *out_written_bytes = static_cast<size_t>(byte_size_);
  return Status::Ok();
}

Status CpuStorage::CopyElementTo(uint64_t storage_index,
                                 void* out_data,
                                 size_t element_bytes) const {
  if (storage_index >= numel_) {
    return InternalError("cpu storage: element index exceeds storage");
  }
  const size_t expected = DTypeByteWidth(dtype_);
  if (element_bytes != expected) {
    return InternalError("cpu storage: element byte width is inconsistent");
  }
  if (out_data == nullptr) {
    return InvalidArgument("cpu storage: element output pointer is null");
  }

  const size_t position = static_cast<size_t>(storage_index);
  std::visit(
      [&](const auto& values) {
        std::memcpy(out_data, &values[position], expected);
      },
      data_);
  return Status::Ok();
}

Status CpuStorage::CopyToHostF32(float* out_values,
                                 size_t capacity,
                                 size_t* out_written) const {
  if (out_written == nullptr) {
    return InvalidArgument("cpu storage: output count pointer is null");
  }
  *out_written = 0;
  if (dtype_ != DType::kFloat32) {
    return Unsupported("cpu storage: exact float32 copy requires float32 storage");
  }
  const auto& values = std::get<std::vector<float>>(data_);
  if (capacity < values.size()) {
    return InvalidArgument("cpu storage: output capacity is too small");
  }
  if (!values.empty() && out_values == nullptr) {
    return InvalidArgument("cpu storage: output values pointer is null");
  }

  std::copy(values.begin(), values.end(), out_values);
  *out_written = values.size();
  return Status::Ok();
}

uint64_t CpuStorage::LiveBytes() {
  return live_bytes_.load(std::memory_order_relaxed);
}

}  // namespace tensora
