#include "tensora.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "tensor/dtype.h"

namespace {

int failures = 0;

#define CHECK_TRUE(expr)                                                        \
  do {                                                                          \
    if (!(expr)) {                                                              \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " #expr     \
                << "\n";                                                       \
      ++failures;                                                               \
    }                                                                           \
  } while (false)

#define CHECK_STATUS(actual, expected)                                          \
  do {                                                                          \
    const ts_status_t _actual = (actual);                                        \
    if (_actual != (expected)) {                                                \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__                    \
                << ": expected status " << static_cast<int>(expected)         \
                << ", got " << static_cast<int>(_actual)                      \
                << " error=" << ts_last_error_message() << "\n";            \
      ++failures;                                                               \
    }                                                                           \
  } while (false)

void Release(ts_tensor_t* tensor) {
  if (tensor == nullptr || *tensor == 0) return;
  CHECK_STATUS(ts_tensor_release(*tensor), TS_OK);
  *tensor = 0;
}

uint64_t LiveBytes() {
  uint64_t bytes = 0;
  CHECK_STATUS(ts_runtime_live_storage_bytes(&bytes), TS_OK);
  return bytes;
}

template <typename T>
std::vector<T> CopyRaw(ts_tensor_t tensor, size_t count) {
  std::vector<T> values(count);
  size_t written = std::numeric_limits<size_t>::max();
  CHECK_STATUS(ts_tensor_copy_to_host(tensor, values.data(),
                                      values.size() * sizeof(T), &written),
               TS_OK);
  CHECK_TRUE(written == values.size() * sizeof(T));
  return values;
}

template <typename T>
void CheckExact(const std::vector<T>& actual, const std::vector<T>& expected) {
  CHECK_TRUE(actual.size() == expected.size());
  if (actual.size() != expected.size()) return;
  if (!actual.empty()) {
    CHECK_TRUE(std::memcmp(actual.data(), expected.data(),
                           actual.size() * sizeof(T)) == 0);
  }
}

template <typename T>
ts_tensor_t MakeTyped(uint32_t dtype,
                      const std::vector<T>& values,
                      const std::vector<int64_t>& shape) {
  ts_tensor_t tensor = 0;
  CHECK_STATUS(ts_tensor_from_host(values.data(), values.size() * sizeof(T),
                                   dtype, shape.data(), shape.size(), &tensor),
               TS_OK);
  CHECK_TRUE(tensor != 0);
  return tensor;
}

template <typename T>
void CheckRoundTrip(uint32_t dtype,
                    const std::vector<T>& values,
                    size_t width) {
  const uint64_t baseline = LiveBytes();
  const std::vector<int64_t> shape{2, 2};
  ts_tensor_t tensor = MakeTyped(dtype, values, shape);
  CHECK_TRUE(LiveBytes() == baseline + values.size() * width);

  uint32_t actual_dtype = 0;
  uint64_t numel = 0;
  CHECK_STATUS(ts_tensor_dtype(tensor, &actual_dtype), TS_OK);
  CHECK_STATUS(ts_tensor_numel(tensor, &numel), TS_OK);
  CHECK_TRUE(actual_dtype == dtype);
  CHECK_TRUE(numel == values.size());
  CheckExact(CopyRaw<T>(tensor, values.size()), values);

  Release(&tensor);
  CHECK_TRUE(LiveBytes() == baseline);
}

void TestDTypeModel() {
  using tensora::DType;
  constexpr std::array<DType, 10> kTypes{
      DType::kFloat16,  DType::kBFloat16, DType::kFloat32,
      DType::kFloat64,  DType::kInt8,     DType::kUInt8,
      DType::kInt16,    DType::kInt32,    DType::kInt64,
      DType::kBool,
  };
  constexpr std::array<size_t, 10> kWidths{2, 2, 4, 8, 1, 1, 2, 4, 8, 1};

  for (size_t index = 0; index < kTypes.size(); ++index) {
    CHECK_TRUE(tensora::DTypeByteWidth(kTypes[index]) == kWidths[index]);
    DType decoded = DType::kFloat32;
    CHECK_STATUS(tensora::DTypeFromCode(
                     static_cast<uint32_t>(kTypes[index]), &decoded)
                     .code(),
                 TS_OK);
    CHECK_TRUE(decoded == kTypes[index]);
    CHECK_TRUE(std::string(tensora::DTypeName(kTypes[index])) != "unknown");
  }

  DType untouched = DType::kInt64;
  CHECK_STATUS(tensora::DTypeFromCode(0, &untouched).code(),
               TS_INVALID_ARGUMENT);
  CHECK_TRUE(untouched == DType::kInt64);
  CHECK_STATUS(tensora::DTypeFromCode(TS_DTYPE_FLOAT32, nullptr).code(),
               TS_INVALID_ARGUMENT);

  for (DType left : kTypes) {
    for (DType right : kTypes) {
      CHECK_TRUE(tensora::PromoteDTypes(left, right) ==
                 tensora::PromoteDTypes(right, left));
    }
  }

  CHECK_TRUE(tensora::PromoteDTypes(DType::kBool, DType::kBool) ==
             DType::kBool);
  CHECK_TRUE(tensora::PromoteDTypes(DType::kUInt8, DType::kInt8) ==
             DType::kInt16);
  CHECK_TRUE(tensora::PromoteDTypes(DType::kFloat16, DType::kBFloat16) ==
             DType::kFloat32);
  CHECK_TRUE(tensora::PromoteDTypes(DType::kFloat64, DType::kInt64) ==
             DType::kFloat64);

  CHECK_TRUE(tensora::DTypeReductionAccumulator(DType::kFloat16) ==
             DType::kFloat32);
  CHECK_TRUE(tensora::DTypeReductionAccumulator(DType::kFloat64) ==
             DType::kFloat64);
  CHECK_TRUE(tensora::DTypeReductionAccumulator(DType::kBool) ==
             DType::kInt64);
}

void TestHalfCodecs() {
  const std::array<uint16_t, 8> half_bits{
      0x0000u, 0x8000u, 0x0001u, 0x03ffu,
      0x3c00u, 0xbc00u, 0x7c00u, 0xfc00u,
  };
  for (uint16_t bits : half_bits) {
    const float decoded = tensora::Float16BitsToFloat(bits);
    CHECK_TRUE(tensora::FloatToFloat16Bits(decoded) == bits);
  }

  const float quiet_nan = std::numeric_limits<float>::quiet_NaN();
  const uint16_t half_nan = tensora::FloatToFloat16Bits(quiet_nan);
  CHECK_TRUE((half_nan & 0x7c00u) == 0x7c00u);
  CHECK_TRUE((half_nan & 0x03ffu) != 0);
  CHECK_TRUE(std::isnan(tensora::Float16BitsToFloat(half_nan)));

  CHECK_TRUE(tensora::FloatToFloat16Bits(1.0f) == 0x3c00u);
  CHECK_TRUE(tensora::FloatToFloat16Bits(-2.0f) == 0xc000u);
  CHECK_TRUE(tensora::FloatToFloat16Bits(65504.0f) == 0x7bffu);
  CHECK_TRUE(tensora::FloatToFloat16Bits(70000.0f) == 0x7c00u);

  const std::array<uint16_t, 6> bfloat_bits{
      0x0000u, 0x8000u, 0x3f80u, 0xc000u, 0x7f80u, 0xff80u,
  };
  for (uint16_t bits : bfloat_bits) {
    const float decoded = tensora::BFloat16BitsToFloat(bits);
    CHECK_TRUE(tensora::FloatToBFloat16Bits(decoded) == bits);
  }
  const uint16_t bfloat_nan = tensora::FloatToBFloat16Bits(quiet_nan);
  CHECK_TRUE((bfloat_nan & 0x7f80u) == 0x7f80u);
  CHECK_TRUE((bfloat_nan & 0x007fu) != 0);
  CHECK_TRUE(std::isnan(tensora::BFloat16BitsToFloat(bfloat_nan)));

  const float tie_even_low = std::bit_cast<float>(uint32_t{0x3f808000});
  const float tie_even_high = std::bit_cast<float>(uint32_t{0x3f818000});
  CHECK_TRUE(tensora::FloatToBFloat16Bits(tie_even_low) == 0x3f80u);
  CHECK_TRUE(tensora::FloatToBFloat16Bits(tie_even_high) == 0x3f82u);
}

void TestTypedRoundTrips() {
  CheckRoundTrip<uint16_t>(TS_DTYPE_FLOAT16,
                           {0x0000u, 0x3c00u, 0xc000u, 0x7bffu}, 2);
  CheckRoundTrip<uint16_t>(TS_DTYPE_BFLOAT16,
                           {0x0000u, 0x3f80u, 0xc000u, 0x7f80u}, 2);
  CheckRoundTrip<float>(TS_DTYPE_FLOAT32,
                        {0.0f, -0.0f, 1.25f, -3.5f}, 4);
  CheckRoundTrip<double>(TS_DTYPE_FLOAT64,
                         {0.0, -0.0, 1.25, -3.5}, 8);
  CheckRoundTrip<int8_t>(TS_DTYPE_INT8, {-128, -1, 0, 127}, 1);
  CheckRoundTrip<uint8_t>(TS_DTYPE_UINT8, {0, 1, 42, 255}, 1);
  CheckRoundTrip<int16_t>(TS_DTYPE_INT16, {-32768, -1, 0, 32767}, 2);
  CheckRoundTrip<int32_t>(TS_DTYPE_INT32,
                          {std::numeric_limits<int32_t>::min(), -1, 0,
                           std::numeric_limits<int32_t>::max()},
                          4);
  CheckRoundTrip<int64_t>(TS_DTYPE_INT64,
                          {std::numeric_limits<int64_t>::min(), -1, 0,
                           std::numeric_limits<int64_t>::max()},
                          8);
  CheckRoundTrip<uint8_t>(TS_DTYPE_BOOL, {0, 1, 1, 0}, 1);
}

template <typename T>
void CheckFull(uint32_t dtype, T scalar) {
  const int64_t shape[2] = {2, 3};
  ts_tensor_t tensor = 0;
  CHECK_STATUS(ts_tensor_full(&scalar, sizeof(T), dtype, shape, 2, &tensor),
               TS_OK);
  CheckExact(CopyRaw<T>(tensor, 6), std::vector<T>(6, scalar));
  Release(&tensor);
}

void TestTypedFull() {
  CheckFull<uint16_t>(TS_DTYPE_FLOAT16, 0x3e00u);
  CheckFull<uint16_t>(TS_DTYPE_BFLOAT16, 0x3fc0u);
  CheckFull<float>(TS_DTYPE_FLOAT32, 1.5f);
  CheckFull<double>(TS_DTYPE_FLOAT64, -2.25);
  CheckFull<int8_t>(TS_DTYPE_INT8, -7);
  CheckFull<uint8_t>(TS_DTYPE_UINT8, 250);
  CheckFull<int16_t>(TS_DTYPE_INT16, -1234);
  CheckFull<int32_t>(TS_DTYPE_INT32, 1234567);
  CheckFull<int64_t>(TS_DTYPE_INT64, int64_t{-1234567890123});
  CheckFull<uint8_t>(TS_DTYPE_BOOL, 1);
}

void TestTypedViews() {
  const std::vector<int64_t> shape{2, 3};
  const int64_t reshaped_dims[2] = {3, 2};

  const auto check = [&](uint32_t dtype, const auto& values) {
    using T = typename std::decay_t<decltype(values)>::value_type;
    ts_tensor_t source = MakeTyped<T>(dtype, values, shape);
    ts_tensor_t reshaped = 0;
    ts_tensor_t transposed = 0;
    CHECK_STATUS(ts_tensor_reshape(source, reshaped_dims, 2, &reshaped), TS_OK);
    CheckExact(CopyRaw<T>(reshaped, values.size()), values);
    CHECK_STATUS(ts_tensor_transpose2d(source, &transposed), TS_OK);
    const std::vector<T> expected{
        values[0], values[3], values[1], values[4], values[2], values[5]};
    CheckExact(CopyRaw<T>(transposed, values.size()), expected);
    Release(&transposed);
    Release(&reshaped);
    Release(&source);
  };

  check(TS_DTYPE_FLOAT16,
        std::vector<uint16_t>{0x3c00, 0x4000, 0x4200,
                              0x4400, 0x4500, 0x4600});
  check(TS_DTYPE_BFLOAT16,
        std::vector<uint16_t>{0x3f80, 0x4000, 0x4040,
                              0x4080, 0x40a0, 0x40c0});
  check(TS_DTYPE_FLOAT32, std::vector<float>{1, 2, 3, 4, 5, 6});
  check(TS_DTYPE_FLOAT64, std::vector<double>{1, 2, 3, 4, 5, 6});
  check(TS_DTYPE_INT8, std::vector<int8_t>{1, 2, 3, 4, 5, 6});
  check(TS_DTYPE_UINT8, std::vector<uint8_t>{1, 2, 3, 4, 5, 6});
  check(TS_DTYPE_INT16, std::vector<int16_t>{1, 2, 3, 4, 5, 6});
  check(TS_DTYPE_INT32, std::vector<int32_t>{1, 2, 3, 4, 5, 6});
  check(TS_DTYPE_INT64, std::vector<int64_t>{1, 2, 3, 4, 5, 6});
  check(TS_DTYPE_BOOL, std::vector<uint8_t>{0, 1, 0, 1, 1, 0});
}

void TestCasts() {
  const std::vector<int64_t> shape{3};
  ts_tensor_t source = MakeTyped<int32_t>(TS_DTYPE_INT32, {-2, 0, 7}, shape);
  ts_tensor_t result = 0;
  CHECK_STATUS(ts_tensor_cast(source, TS_DTYPE_FLOAT64, &result), TS_OK);
  CheckExact(CopyRaw<double>(result, 3), {-2.0, 0.0, 7.0});
  Release(&result);
  Release(&source);

  source = MakeTyped<float>(TS_DTYPE_FLOAT32, {1.9f, -2.2f, 0.0f}, shape);
  CHECK_STATUS(ts_tensor_cast(source, TS_DTYPE_INT16, &result), TS_OK);
  CheckExact(CopyRaw<int16_t>(result, 3), {1, -2, 0});
  Release(&result);
  Release(&source);

  source = MakeTyped<int32_t>(TS_DTYPE_INT32, {0, -3, 9}, shape);
  CHECK_STATUS(ts_tensor_cast(source, TS_DTYPE_BOOL, &result), TS_OK);
  CheckExact(CopyRaw<uint8_t>(result, 3), {0, 1, 1});
  Release(&result);
  Release(&source);

  source = MakeTyped<uint8_t>(TS_DTYPE_BOOL, {0, 1, 1}, shape);
  CHECK_STATUS(ts_tensor_cast(source, TS_DTYPE_INT8, &result), TS_OK);
  CheckExact(CopyRaw<int8_t>(result, 3), {0, 1, 1});
  Release(&result);
  Release(&source);

  source = MakeTyped<float>(TS_DTYPE_FLOAT32, {300.0f}, {1});
  result = 0xfeedu;
  CHECK_STATUS(ts_tensor_cast(source, TS_DTYPE_UINT8, &result),
               TS_INVALID_ARGUMENT);
  CHECK_TRUE(result == 0);
  Release(&source);

  source = MakeTyped<float>(TS_DTYPE_FLOAT32,
                            {std::numeric_limits<float>::quiet_NaN()}, {1});
  result = 0xfeedu;
  CHECK_STATUS(ts_tensor_cast(source, TS_DTYPE_INT32, &result),
               TS_INVALID_ARGUMENT);
  CHECK_TRUE(result == 0);
  Release(&source);
}

void TestMalformedTypedAbi() {
  const int64_t shape[1] = {2};
  const int32_t values[2] = {1, 2};
  ts_tensor_t tensor = 0xfeedu;

  CHECK_STATUS(ts_tensor_from_host(values, sizeof(values), 999, shape, 1,
                                   &tensor),
               TS_INVALID_ARGUMENT);
  CHECK_TRUE(tensor == 0);

  tensor = 0xfeedu;
  CHECK_STATUS(ts_tensor_from_host(values, sizeof(values) - 1, TS_DTYPE_INT32,
                                   shape, 1, &tensor),
               TS_INVALID_ARGUMENT);
  CHECK_TRUE(tensor == 0);

  tensor = 0xfeedu;
  CHECK_STATUS(ts_tensor_from_host(nullptr, sizeof(values), TS_DTYPE_INT32,
                                   shape, 1, &tensor),
               TS_INVALID_ARGUMENT);
  CHECK_TRUE(tensor == 0);

  CHECK_STATUS(ts_tensor_from_host(values, sizeof(values), TS_DTYPE_INT32,
                                   shape, 1, nullptr),
               TS_INVALID_ARGUMENT);

  const uint8_t bad_bool[2] = {0, 2};
  tensor = 0xfeedu;
  CHECK_STATUS(ts_tensor_from_host(bad_bool, sizeof(bad_bool), TS_DTYPE_BOOL,
                                   shape, 1, &tensor),
               TS_INVALID_ARGUMENT);
  CHECK_TRUE(tensor == 0);

  const uint8_t scalar = 1;
  tensor = 0xfeedu;
  CHECK_STATUS(ts_tensor_full(&scalar, 0, TS_DTYPE_BOOL, shape, 1, &tensor),
               TS_INVALID_ARGUMENT);
  CHECK_TRUE(tensor == 0);

  tensor = 0xfeedu;
  CHECK_STATUS(ts_tensor_full(nullptr, 1, TS_DTYPE_BOOL, shape, 1, &tensor),
               TS_INVALID_ARGUMENT);
  CHECK_TRUE(tensor == 0);

  const uint8_t bad_scalar = 2;
  tensor = 0xfeedu;
  CHECK_STATUS(ts_tensor_full(&bad_scalar, 1, TS_DTYPE_BOOL, shape, 1, &tensor),
               TS_INVALID_ARGUMENT);
  CHECK_TRUE(tensor == 0);

  CHECK_STATUS(ts_tensor_full(&scalar, 1, TS_DTYPE_BOOL, shape, 1, nullptr),
               TS_INVALID_ARGUMENT);

  ts_tensor_t valid = MakeTyped<int32_t>(TS_DTYPE_INT32, {1, 2}, {2});
  std::array<int32_t, 2> output{};
  size_t written = 123;
  CHECK_STATUS(ts_tensor_copy_to_host(valid, output.data(), sizeof(int32_t),
                                      &written),
               TS_INVALID_ARGUMENT);
  CHECK_TRUE(written == 0);

  written = 123;
  CHECK_STATUS(ts_tensor_copy_to_host(valid, nullptr, sizeof(output), &written),
               TS_INVALID_ARGUMENT);
  CHECK_TRUE(written == 0);

  CHECK_STATUS(ts_tensor_copy_to_host(valid, output.data(), sizeof(output),
                                      nullptr),
               TS_INVALID_ARGUMENT);

  written = 123;
  CHECK_STATUS(ts_tensor_copy_to_host(0x7fffffffffffffffu, output.data(),
                                      sizeof(output), &written),
               TS_INVALID_HANDLE);
  CHECK_TRUE(written == 0);

  ts_tensor_t cast = 0xfeedu;
  CHECK_STATUS(ts_tensor_cast(valid, 999, &cast), TS_INVALID_ARGUMENT);
  CHECK_TRUE(cast == 0);
  cast = 0xfeedu;
  CHECK_STATUS(ts_tensor_cast(0x7fffffffffffffffu, TS_DTYPE_FLOAT32, &cast),
               TS_INVALID_HANDLE);
  CHECK_TRUE(cast == 0);
  CHECK_STATUS(ts_tensor_cast(valid, TS_DTYPE_FLOAT32, nullptr),
               TS_INVALID_ARGUMENT);

  size_t float_written = 17;
  std::array<float, 2> float_output{};
  CHECK_STATUS(ts_tensor_copy_to_host_f32(valid, float_output.data(),
                                          float_output.size(), &float_written),
               TS_UNSUPPORTED);
  CHECK_TRUE(float_written == 0);

  ts_tensor_t unsupported = 0xfeedu;
  CHECK_STATUS(ts_tensor_add(valid, valid, &unsupported), TS_UNSUPPORTED);
  CHECK_TRUE(unsupported == 0);
  unsupported = 0xfeedu;
  CHECK_STATUS(ts_tensor_with_requires_grad(valid, 1, &unsupported),
               TS_UNSUPPORTED);
  CHECK_TRUE(unsupported == 0);

  Release(&valid);
}

void TestCpuCloneTransferForTypedStorage() {
  const std::vector<int64_t> shape{2, 3};
  ts_tensor_t source =
      MakeTyped<int64_t>(TS_DTYPE_INT64, {-3, 0, 9, 4, 7, 11}, shape);

  ts_tensor_t invalid = 0xfeedu;
  CHECK_STATUS(ts_tensor_to_device(source, TS_DEVICE_CPU, -1, &invalid),
               TS_INVALID_ARGUMENT);
  CHECK_TRUE(invalid == 0);
  invalid = 0xfeedu;
  CHECK_STATUS(ts_tensor_to_device(source, TS_DEVICE_CPU, 1, &invalid),
               TS_INVALID_ARGUMENT);
  CHECK_TRUE(invalid == 0);
  invalid = 0xfeedu;
  CHECK_STATUS(ts_tensor_to_device(source, TS_DEVICE_CUDA, 0, &invalid),
               TS_UNSUPPORTED);
  CHECK_TRUE(invalid == 0);

  ts_tensor_t copied = 0;
  CHECK_STATUS(ts_tensor_to_device(source, TS_DEVICE_CPU, 0, &copied), TS_OK);
  CHECK_TRUE(copied != 0 && copied != source);
  CheckExact(CopyRaw<int64_t>(copied, 6), {-3, 0, 9, 4, 7, 11});
  Release(&source);
  CheckExact(CopyRaw<int64_t>(copied, 6), {-3, 0, 9, 4, 7, 11});
  Release(&copied);

  source = MakeTyped<int64_t>(TS_DTYPE_INT64, {1, 2, 3, 4, 5, 6}, shape);
  ts_tensor_t transposed = 0;
  CHECK_STATUS(ts_tensor_transpose2d(source, &transposed), TS_OK);
  copied = 0;
  CHECK_STATUS(ts_tensor_to_device(transposed, TS_DEVICE_CPU, 0, &copied),
               TS_OK);
  CheckExact(CopyRaw<int64_t>(copied, 6), {1, 4, 2, 5, 3, 6});

  ts_tensor_t cast = 0;
  CHECK_STATUS(ts_tensor_cast(transposed, TS_DTYPE_FLOAT64, &cast), TS_OK);
  CheckExact(CopyRaw<double>(cast, 6), {1.0, 4.0, 2.0, 5.0, 3.0, 6.0});

  Release(&cast);
  Release(&copied);
  Release(&transposed);
  Release(&source);
}

}  // namespace

int main() {
  TestDTypeModel();
  TestHalfCodecs();
  TestTypedRoundTrips();
  TestTypedFull();
  TestTypedViews();
  TestCasts();
  TestMalformedTypedAbi();
  TestCpuCloneTransferForTypedStorage();
  if (failures != 0) {
    std::cerr << failures << " dtype storage test(s) failed\n";
    return 1;
  }
  std::cout << "dtype storage tests passed\n";
  return 0;
}
