#include "tensora.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

namespace {

int failures = 0;

#define CHECK_TRUE(expr) \
  do { \
    if (!(expr)) { \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " #expr "\n"; \
      ++failures; \
    } \
  } while (false)

#define CHECK_STATUS(actual, expected) \
  do { \
    const ts_status_t _actual = (actual); \
    if (_actual != (expected)) { \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ \
                << ": expected status " << static_cast<int>(expected) \
                << ", got " << static_cast<int>(_actual) \
                << " error=" << ts_last_error_message() << "\n"; \
      ++failures; \
    } \
  } while (false)

void check_near(float actual, float expected, float tolerance = 1e-5f) {
  if (std::fabs(actual - expected) > tolerance) {
    std::cerr << "FAIL numeric: expected " << expected << ", got " << actual
              << "\n";
    ++failures;
  }
}

ts_tensor_t make_tensor(const std::vector<float>& values,
                        const std::vector<int64_t>& shape) {
  ts_tensor_t handle = 0;
  CHECK_STATUS(ts_tensor_from_f32(values.data(), values.size(), shape.data(),
                                  shape.size(), &handle),
               TS_OK);
  CHECK_TRUE(handle != 0);
  return handle;
}

std::vector<float> read_tensor(ts_tensor_t handle) {
  uint64_t numel = 0;
  CHECK_STATUS(ts_tensor_numel(handle, &numel), TS_OK);
  std::vector<float> values(static_cast<size_t>(numel));
  size_t written = 0;
  CHECK_STATUS(ts_tensor_copy_to_host_f32(handle, values.data(), values.size(),
                                          &written),
               TS_OK);
  CHECK_TRUE(written == values.size());
  return values;
}

void test_abi_and_metadata() {
  CHECK_TRUE(ts_abi_version() == TS_ABI_VERSION);
  CHECK_STATUS(ts_noop(), TS_OK);
  CHECK_TRUE(std::string(ts_status_name(TS_OK)) == "OK");
  CHECK_TRUE(std::string(ts_status_name(TS_INVALID_ARGUMENT)) ==
             "INVALID_ARGUMENT");
  CHECK_TRUE(std::string(ts_status_name(TS_INVALID_SHAPE)) ==
             "INVALID_SHAPE");
  CHECK_TRUE(std::string(ts_status_name(TS_OUT_OF_MEMORY)) ==
             "OUT_OF_MEMORY");
  CHECK_TRUE(std::string(ts_status_name(TS_UNSUPPORTED)) == "UNSUPPORTED");
  CHECK_TRUE(std::string(ts_status_name(TS_INVALID_HANDLE)) ==
             "INVALID_HANDLE");
  CHECK_TRUE(std::string(ts_status_name(TS_INTERNAL_ERROR)) ==
             "INTERNAL_ERROR");
  CHECK_TRUE(std::string(ts_status_name(999)) == "UNKNOWN_STATUS");

  const std::vector<int64_t> shape{2, 2};
  const auto handle = make_tensor({1, 2, 3, 4}, shape);

  size_t rank = 0;
  CHECK_STATUS(ts_tensor_rank(handle, &rank), TS_OK);
  CHECK_TRUE(rank == 2);

  int64_t dims[2] = {0, 0};
  size_t written_rank = 0;
  CHECK_STATUS(ts_tensor_shape(handle, dims, 2, &written_rank), TS_OK);
  CHECK_TRUE(written_rank == 2);
  CHECK_TRUE(dims[0] == 2 && dims[1] == 2);

  uint32_t dtype = 0;
  uint32_t device = 0;
  uint64_t numel = 0;
  CHECK_STATUS(ts_tensor_dtype(handle, &dtype), TS_OK);
  CHECK_STATUS(ts_tensor_device(handle, &device), TS_OK);
  CHECK_STATUS(ts_tensor_numel(handle, &numel), TS_OK);
  CHECK_TRUE(dtype == TS_DTYPE_FLOAT32);
  CHECK_TRUE(device == TS_DEVICE_CPU);
  CHECK_TRUE(numel == 4);

  CHECK_STATUS(ts_tensor_release(handle), TS_OK);
}

void test_factories_and_transforms() {
  const int64_t shape[2] = {2, 3};

  ts_tensor_t zeros = 0;
  CHECK_STATUS(ts_tensor_full_f32(shape, 2, 0.0f, &zeros), TS_OK);
  for (float value : read_tensor(zeros)) check_near(value, 0.0f);

  ts_tensor_t ones = 0;
  CHECK_STATUS(ts_tensor_full_f32(shape, 2, 1.0f, &ones), TS_OK);
  for (float value : read_tensor(ones)) check_near(value, 1.0f);

  ts_tensor_t full = 0;
  CHECK_STATUS(ts_tensor_full_f32(shape, 2, 3.5f, &full), TS_OK);
  for (float value : read_tensor(full)) check_near(value, 3.5f);

  const int64_t reshaped_dims[2] = {3, 2};
  ts_tensor_t reshaped = 0;
  CHECK_STATUS(ts_tensor_reshape(full, reshaped_dims, 2, &reshaped), TS_OK);
  auto reshaped_values = read_tensor(reshaped);
  CHECK_TRUE(reshaped_values.size() == 6);

  ts_tensor_t transposed = 0;
  CHECK_STATUS(ts_tensor_transpose2d(reshaped, &transposed), TS_OK);
  int64_t transposed_shape[2] = {0, 0};
  size_t rank = 0;
  CHECK_STATUS(ts_tensor_shape(transposed, transposed_shape, 2, &rank), TS_OK);
  CHECK_TRUE(transposed_shape[0] == 2 && transposed_shape[1] == 3);

  CHECK_STATUS(ts_tensor_release(transposed), TS_OK);
  CHECK_STATUS(ts_tensor_release(reshaped), TS_OK);
  CHECK_STATUS(ts_tensor_release(full), TS_OK);
  CHECK_STATUS(ts_tensor_release(ones), TS_OK);
  CHECK_STATUS(ts_tensor_release(zeros), TS_OK);
}

void test_math_reference() {
  const auto a = make_tensor({1, 2, 3, 4}, {2, 2});
  const auto b = make_tensor({5, 6, 7, 8}, {2, 2});

  ts_tensor_t add = 0;
  CHECK_STATUS(ts_tensor_add(a, b, &add), TS_OK);
  const auto add_values = read_tensor(add);
  check_near(add_values[0], 6);
  check_near(add_values[1], 8);
  check_near(add_values[2], 10);
  check_near(add_values[3], 12);

  ts_tensor_t mul = 0;
  CHECK_STATUS(ts_tensor_multiply(a, b, &mul), TS_OK);
  const auto mul_values = read_tensor(mul);
  check_near(mul_values[0], 5);
  check_near(mul_values[1], 12);
  check_near(mul_values[2], 21);
  check_near(mul_values[3], 32);

  ts_tensor_t sum = 0;
  CHECK_STATUS(ts_tensor_sum(a, &sum), TS_OK);
  const auto sum_values = read_tensor(sum);
  CHECK_TRUE(sum_values.size() == 1);
  check_near(sum_values[0], 10);

  ts_tensor_t matmul = 0;
  CHECK_STATUS(ts_tensor_matmul(a, b, &matmul), TS_OK);
  const auto mm = read_tensor(matmul);
  check_near(mm[0], 19);
  check_near(mm[1], 22);
  check_near(mm[2], 43);
  check_near(mm[3], 50);

  CHECK_STATUS(ts_tensor_release(matmul), TS_OK);
  CHECK_STATUS(ts_tensor_release(sum), TS_OK);
  CHECK_STATUS(ts_tensor_release(mul), TS_OK);
  CHECK_STATUS(ts_tensor_release(add), TS_OK);
  CHECK_STATUS(ts_tensor_release(b), TS_OK);
  CHECK_STATUS(ts_tensor_release(a), TS_OK);
}

void test_invalid_inputs() {
  ts_tensor_t out = 0;

  const int64_t negative[2] = {2, -1};
  CHECK_STATUS(ts_tensor_full_f32(negative, 2, 0.0f, &out),
               TS_INVALID_SHAPE);

  const int64_t zero[2] = {2, 0};
  CHECK_STATUS(ts_tensor_full_f32(zero, 2, 0.0f, &out), TS_INVALID_SHAPE);

  const std::vector<int64_t> too_many_dims(33, 1);
  CHECK_STATUS(ts_tensor_full_f32(too_many_dims.data(), too_many_dims.size(),
                                  0.0f, &out),
               TS_INVALID_SHAPE);

  const int64_t overflow[2] = {std::numeric_limits<int64_t>::max(), 2};
  CHECK_STATUS(ts_tensor_full_f32(overflow, 2, 0.0f, &out),
               TS_INVALID_SHAPE);

  CHECK_STATUS(ts_tensor_full_f32(nullptr, 1, 0.0f, &out),
               TS_INVALID_ARGUMENT);
  CHECK_STATUS(ts_tensor_full_f32(nullptr, 0, 0.0f, nullptr),
               TS_INVALID_ARGUMENT);

  const int64_t shape[1] = {2};
  const float values[2] = {1.0f, 2.0f};
  CHECK_STATUS(ts_tensor_from_f32(nullptr, 2, shape, 1, &out),
               TS_INVALID_ARGUMENT);
  CHECK_STATUS(ts_tensor_from_f32(values, 1, shape, 1, &out),
               TS_INVALID_ARGUMENT);
  CHECK_STATUS(ts_tensor_from_f32(values, 3, shape, 1, &out),
               TS_INVALID_ARGUMENT);
  CHECK_STATUS(ts_tensor_from_f32(values, 2, shape, 1, nullptr),
               TS_INVALID_ARGUMENT);

  const float scalar = 7.0f;
  CHECK_STATUS(ts_tensor_from_f32(&scalar, 0, nullptr, 0, &out),
               TS_INVALID_ARGUMENT);

  CHECK_STATUS(ts_tensor_numel(999999999ULL, nullptr), TS_INVALID_ARGUMENT);

  uint64_t numel = 0;
  CHECK_STATUS(ts_tensor_numel(999999999ULL, &numel), TS_INVALID_HANDLE);
  CHECK_STATUS(ts_tensor_retain(999999999ULL), TS_INVALID_HANDLE);
  CHECK_STATUS(ts_tensor_release(0), TS_INVALID_HANDLE);
  CHECK_STATUS(ts_runtime_live_tensor_count(nullptr), TS_INVALID_ARGUMENT);
  CHECK_STATUS(ts_runtime_live_storage_bytes(nullptr), TS_INVALID_ARGUMENT);

  const auto a = make_tensor({1, 2, 3, 4}, {2, 2});
  const auto b = make_tensor({1, 2, 3, 4, 5, 6}, {3, 2});
  CHECK_STATUS(ts_tensor_matmul(a, b, &out), TS_INVALID_SHAPE);
  CHECK_STATUS(ts_tensor_add(a, b, &out), TS_INVALID_SHAPE);
  CHECK_STATUS(ts_tensor_multiply(a, b, &out), TS_INVALID_SHAPE);

  size_t rank = 0;
  int64_t one_dim = 0;
  CHECK_STATUS(ts_tensor_shape(a, &one_dim, 1, &rank), TS_INVALID_ARGUMENT);
  CHECK_STATUS(ts_tensor_shape(a, nullptr, 2, &rank), TS_INVALID_ARGUMENT);
  CHECK_STATUS(ts_tensor_shape(a, nullptr, 2, nullptr), TS_INVALID_ARGUMENT);
  CHECK_STATUS(ts_tensor_dtype(a, nullptr), TS_INVALID_ARGUMENT);
  CHECK_STATUS(ts_tensor_device(a, nullptr), TS_INVALID_ARGUMENT);
  CHECK_STATUS(ts_tensor_rank(a, nullptr), TS_INVALID_ARGUMENT);

  float one_value = 0.0f;
  size_t written = 0;
  CHECK_STATUS(ts_tensor_copy_to_host_f32(a, &one_value, 1, &written),
               TS_INVALID_ARGUMENT);
  CHECK_STATUS(ts_tensor_copy_to_host_f32(a, nullptr, 4, &written),
               TS_INVALID_ARGUMENT);
  CHECK_STATUS(ts_tensor_copy_to_host_f32(a, &one_value, 4, nullptr),
               TS_INVALID_ARGUMENT);

  const int64_t wrong_reshape[2] = {1, 3};
  CHECK_STATUS(ts_tensor_reshape(a, wrong_reshape, 2, &out),
               TS_INVALID_SHAPE);
  CHECK_STATUS(ts_tensor_reshape(a, wrong_reshape, 2, nullptr),
               TS_INVALID_ARGUMENT);
  CHECK_STATUS(ts_tensor_add(a, a, nullptr), TS_INVALID_ARGUMENT);
  CHECK_STATUS(ts_tensor_multiply(a, a, nullptr), TS_INVALID_ARGUMENT);
  CHECK_STATUS(ts_tensor_matmul(a, a, nullptr), TS_INVALID_ARGUMENT);
  CHECK_STATUS(ts_tensor_sum(a, nullptr), TS_INVALID_ARGUMENT);
  CHECK_STATUS(ts_tensor_transpose2d(a, nullptr), TS_INVALID_ARGUMENT);

  const auto vector = make_tensor({1, 2, 3}, {3});
  CHECK_STATUS(ts_tensor_transpose2d(vector, &out), TS_INVALID_SHAPE);
  CHECK_STATUS(ts_tensor_matmul(vector, vector, &out), TS_INVALID_SHAPE);
  CHECK_STATUS(ts_tensor_release(vector), TS_OK);

  CHECK_STATUS(ts_tensor_release(b), TS_OK);
  CHECK_STATUS(ts_tensor_release(a), TS_OK);
}

void test_retain_release_and_stale_handle() {
  const auto handle = make_tensor({1, 2, 3, 4}, {2, 2});

  CHECK_STATUS(ts_tensor_retain(handle), TS_OK);
  CHECK_STATUS(ts_tensor_release(handle), TS_OK);

  uint64_t numel = 0;
  CHECK_STATUS(ts_tensor_numel(handle, &numel), TS_OK);
  CHECK_TRUE(numel == 4);

  CHECK_STATUS(ts_tensor_release(handle), TS_OK);
  CHECK_STATUS(ts_tensor_numel(handle, &numel), TS_INVALID_HANDLE);
  CHECK_STATUS(ts_tensor_release(handle), TS_INVALID_HANDLE);
}

void test_concurrent_read_only_operations() {
  const auto a = make_tensor({1, 2, 3, 4}, {2, 2});
  const auto b = make_tensor({5, 6, 7, 8}, {2, 2});

  std::atomic<int> thread_failures{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < 8; ++t) {
    threads.emplace_back([&] {
      for (int i = 0; i < 100; ++i) {
        ts_tensor_t result = 0;
        if (ts_tensor_matmul(a, b, &result) != TS_OK) {
          ++thread_failures;
          continue;
        }
        auto values = read_tensor(result);
        if (values.size() != 4 || std::fabs(values[0] - 19.0f) > 1e-5f) {
          ++thread_failures;
        }
        if (ts_tensor_release(result) != TS_OK) {
          ++thread_failures;
        }
      }
    });
  }
  for (auto& thread : threads) thread.join();

  CHECK_TRUE(thread_failures.load() == 0);
  CHECK_STATUS(ts_tensor_release(b), TS_OK);
  CHECK_STATUS(ts_tensor_release(a), TS_OK);
}

void test_lifecycle_stress() {
  uint64_t start_count = 0;
  uint64_t start_bytes = 0;
  CHECK_STATUS(ts_runtime_live_tensor_count(&start_count), TS_OK);
  CHECK_STATUS(ts_runtime_live_storage_bytes(&start_bytes), TS_OK);

  for (int i = 0; i < 10000; ++i) {
    const auto a = make_tensor({1, 2, 3, 4}, {2, 2});
    const auto b = make_tensor({5, 6, 7, 8}, {2, 2});
    ts_tensor_t result = 0;
    CHECK_STATUS(ts_tensor_matmul(a, b, &result), TS_OK);
    CHECK_STATUS(ts_tensor_release(result), TS_OK);
    CHECK_STATUS(ts_tensor_release(b), TS_OK);
    CHECK_STATUS(ts_tensor_release(a), TS_OK);
  }

  uint64_t end_count = 0;
  uint64_t end_bytes = 0;
  CHECK_STATUS(ts_runtime_live_tensor_count(&end_count), TS_OK);
  CHECK_STATUS(ts_runtime_live_storage_bytes(&end_bytes), TS_OK);
  CHECK_TRUE(end_count == start_count);
  CHECK_TRUE(end_bytes == start_bytes);
}

}  // namespace

int main() {
  test_abi_and_metadata();
  test_factories_and_transforms();
  test_math_reference();
  test_invalid_inputs();
  test_retain_release_and_stale_handle();
  test_concurrent_read_only_operations();
  test_lifecycle_stress();

  if (failures != 0) {
    std::cerr << failures << " test failure(s)\n";
    return 1;
  }

  std::cout << "All Tensora native tests passed.\n";
  return 0;
}
