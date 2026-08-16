#include "tensora.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

#define CHECK_TRUE(expr)                                                       \
  do {                                                                         \
    if (!(expr)) {                                                             \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " #expr       \
                << "\n";                                                      \
      ++failures;                                                              \
    }                                                                          \
  } while (false)

#define CHECK_STATUS(actual, expected)                                         \
  do {                                                                         \
    const ts_status_t _actual = (actual);                                       \
    if (_actual != (expected)) {                                               \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__                     \
                << ": expected status " << static_cast<int>(expected)         \
                << ", got " << static_cast<int>(_actual)                       \
                << " error=" << ts_last_error_message() << "\n";             \
      ++failures;                                                              \
    }                                                                          \
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
  ts_tensor_t tensor = 0;
  CHECK_STATUS(ts_tensor_from_f32(values.data(), values.size(), shape.data(),
                                  shape.size(), &tensor),
               TS_OK);
  CHECK_TRUE(tensor != 0);
  return tensor;
}

void release_tensor(ts_tensor_t* tensor) {
  if (tensor == nullptr || *tensor == 0) return;
  CHECK_STATUS(ts_tensor_release(*tensor), TS_OK);
  *tensor = 0;
}

std::vector<float> read_tensor(ts_tensor_t tensor) {
  uint64_t numel = 0;
  CHECK_STATUS(ts_tensor_numel(tensor, &numel), TS_OK);
  std::vector<float> values(static_cast<size_t>(numel), 0.0f);
  size_t written = 0;
  CHECK_STATUS(ts_tensor_copy_to_host_f32(tensor, values.data(), values.size(),
                                          &written),
               TS_OK);
  CHECK_TRUE(written == values.size());
  return values;
}

uint64_t live_bytes() {
  uint64_t bytes = 0;
  CHECK_STATUS(ts_runtime_live_storage_bytes(&bytes), TS_OK);
  return bytes;
}

void check_values(const std::vector<float>& actual,
                  const std::vector<float>& expected) {
  CHECK_TRUE(actual.size() == expected.size());
  if (actual.size() != expected.size()) return;
  for (size_t index = 0; index < actual.size(); ++index) {
    check_near(actual[index], expected[index]);
  }
}

void test_zero_copy_reshape_and_transpose() {
  const uint64_t baseline = live_bytes();
  ts_tensor_t source = make_tensor({1, 2, 3, 4, 5, 6}, {2, 3});
  CHECK_TRUE(live_bytes() == baseline + 6 * sizeof(float));

  const int64_t reshape_dims[2] = {3, 2};
  ts_tensor_t reshaped = 0;
  CHECK_STATUS(ts_tensor_reshape(source, reshape_dims, 2, &reshaped), TS_OK);
  CHECK_TRUE(live_bytes() == baseline + 6 * sizeof(float));
  check_values(read_tensor(reshaped), {1, 2, 3, 4, 5, 6});

  ts_tensor_t transposed = 0;
  CHECK_STATUS(ts_tensor_transpose2d(source, &transposed), TS_OK);
  CHECK_TRUE(live_bytes() == baseline + 6 * sizeof(float));
  check_values(read_tensor(transposed), {1, 4, 2, 5, 3, 6});

  release_tensor(&source);
  CHECK_TRUE(live_bytes() == baseline + 6 * sizeof(float));
  check_values(read_tensor(transposed), {1, 4, 2, 5, 3, 6});

  release_tensor(&transposed);
  release_tensor(&reshaped);
  CHECK_TRUE(live_bytes() == baseline);
}

void test_non_contiguous_ops_use_logical_order() {
  const uint64_t baseline = live_bytes();
  ts_tensor_t source = make_tensor({1, 2, 3, 4, 5, 6}, {2, 3});
  ts_tensor_t transposed = 0;
  CHECK_STATUS(ts_tensor_transpose2d(source, &transposed), TS_OK);

  ts_tensor_t added = 0;
  CHECK_STATUS(ts_tensor_add(transposed, transposed, &added), TS_OK);
  check_values(read_tensor(added), {2, 8, 4, 10, 6, 12});

  ts_tensor_t multiplied = 0;
  CHECK_STATUS(ts_tensor_multiply(transposed, transposed, &multiplied), TS_OK);
  check_values(read_tensor(multiplied), {1, 16, 4, 25, 9, 36});

  ts_tensor_t sum = 0;
  CHECK_STATUS(ts_tensor_sum(transposed, &sum), TS_OK);
  check_values(read_tensor(sum), {21});

  ts_tensor_t right = make_tensor({2, -1}, {2, 1});
  ts_tensor_t product = 0;
  CHECK_STATUS(ts_tensor_matmul(transposed, right, &product), TS_OK);
  check_values(read_tensor(product), {-2, -1, 0});

  release_tensor(&product);
  release_tensor(&right);
  release_tensor(&sum);
  release_tensor(&multiplied);
  release_tensor(&added);
  release_tensor(&transposed);
  release_tensor(&source);
  CHECK_TRUE(live_bytes() == baseline);
}

void test_reshape_non_contiguous_materializes_logical_order() {
  const uint64_t baseline = live_bytes();
  ts_tensor_t source = make_tensor({1, 2, 3, 4, 5, 6}, {2, 3});
  ts_tensor_t transposed = 0;
  CHECK_STATUS(ts_tensor_transpose2d(source, &transposed), TS_OK);
  CHECK_TRUE(live_bytes() == baseline + 6 * sizeof(float));

  const int64_t flat_dims[1] = {6};
  ts_tensor_t flat = 0;
  CHECK_STATUS(ts_tensor_reshape(transposed, flat_dims, 1, &flat), TS_OK);
  CHECK_TRUE(live_bytes() == baseline + 12 * sizeof(float));
  check_values(read_tensor(flat), {1, 4, 2, 5, 3, 6});

  release_tensor(&flat);
  release_tensor(&transposed);
  release_tensor(&source);
  CHECK_TRUE(live_bytes() == baseline);
}

void test_autograd_through_non_contiguous_view() {
  ts_tensor_t raw = make_tensor({1, -2, 3, -4, 5, -6}, {2, 3});
  ts_tensor_t source = 0;
  CHECK_STATUS(ts_tensor_with_requires_grad(raw, 1, &source), TS_OK);
  release_tensor(&raw);

  ts_tensor_t transposed = 0;
  ts_tensor_t square = 0;
  ts_tensor_t loss = 0;
  CHECK_STATUS(ts_tensor_transpose2d(source, &transposed), TS_OK);
  CHECK_STATUS(ts_tensor_multiply(transposed, transposed, &square), TS_OK);
  CHECK_STATUS(ts_tensor_sum(square, &loss), TS_OK);
  CHECK_STATUS(ts_tensor_backward(loss), TS_OK);

  ts_tensor_t gradient = 0;
  CHECK_STATUS(ts_tensor_grad(source, &gradient), TS_OK);
  check_values(read_tensor(gradient), {2, -4, 6, -8, 10, -12});

  release_tensor(&gradient);
  release_tensor(&loss);
  release_tensor(&square);
  release_tensor(&transposed);
  release_tensor(&source);
}

#if !defined(TENSORA_WITH_TORCH)
void test_alias_mutation_is_detected_before_backward_publication() {
  CHECK_STATUS(ts_manual_seed(998877), TS_OK);
  ts_module_t module = 0;
  CHECK_STATUS(ts_linear_create(2, 2, 0, &module), TS_OK);

  ts_tensor_t weight = 0;
  CHECK_STATUS(ts_module_parameter_at(module, 0, &weight), TS_OK);
  ts_tensor_t view = 0;
  ts_tensor_t square = 0;
  ts_tensor_t pending_loss = 0;
  CHECK_STATUS(ts_tensor_transpose2d(weight, &view), TS_OK);
  CHECK_STATUS(ts_tensor_multiply(view, view, &square), TS_OK);
  CHECK_STATUS(ts_tensor_sum(square, &pending_loss), TS_OK);

  ts_optimizer_t optimizer = 0;
  CHECK_STATUS(ts_sgd_create(module, 0.01, 0.0, 0.0, &optimizer), TS_OK);
  ts_tensor_t input = make_tensor({1, 0}, {1, 2});
  ts_tensor_t target = make_tensor({0, 0}, {1, 2});
  CHECK_STATUS(ts_optimizer_zero_grad(optimizer), TS_OK);
  ts_tensor_t prediction = 0;
  ts_tensor_t training_loss = 0;
  CHECK_STATUS(ts_module_forward(module, input, &prediction), TS_OK);
  CHECK_STATUS(ts_mse_loss(prediction, target, &training_loss), TS_OK);
  CHECK_STATUS(ts_tensor_backward(training_loss), TS_OK);
  CHECK_STATUS(ts_optimizer_step(optimizer), TS_OK);
  CHECK_STATUS(ts_optimizer_zero_grad(optimizer), TS_OK);

  CHECK_STATUS(ts_tensor_backward(pending_loss), TS_INVALID_ARGUMENT);
  ts_tensor_t stale_gradient = 777;
  CHECK_STATUS(ts_tensor_grad(weight, &stale_gradient), TS_INVALID_ARGUMENT);
  CHECK_TRUE(stale_gradient == 0);

  release_tensor(&training_loss);
  release_tensor(&prediction);
  release_tensor(&target);
  release_tensor(&input);
  CHECK_STATUS(ts_optimizer_release(optimizer), TS_OK);
  release_tensor(&pending_loss);
  release_tensor(&square);
  release_tensor(&view);
  release_tensor(&weight);
  CHECK_STATUS(ts_module_release(module), TS_OK);
}
#endif

}  // namespace

int main() {
  test_zero_copy_reshape_and_transpose();
  test_non_contiguous_ops_use_logical_order();
  test_reshape_non_contiguous_materializes_logical_order();
  test_autograd_through_non_contiguous_view();
#if !defined(TENSORA_WITH_TORCH)
  test_alias_mutation_is_detected_before_backward_publication();
#endif

  if (failures != 0) {
    std::cerr << failures << " tensor view assertion(s) failed\n";
    return 1;
  }
  std::cout << "Tensora tensor view validation passed\n";
  return 0;
}
