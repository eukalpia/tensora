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

void check_near(float actual, float expected, float tolerance = 1e-4f) {
  if (std::fabs(actual - expected) > tolerance) {
    std::cerr << "FAIL numeric: expected " << expected << ", got " << actual
              << " tolerance=" << tolerance << "\n";
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

ts_tensor_t require_grad(ts_tensor_t source) {
  ts_tensor_t result = 0;
  CHECK_STATUS(ts_tensor_with_requires_grad(source, 1, &result), TS_OK);
  CHECK_TRUE(result != 0);
  return result;
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

float read_scalar(ts_tensor_t handle) {
  const auto values = read_tensor(handle);
  CHECK_TRUE(values.size() == 1);
  return values.empty() ? 0.0f : values[0];
}

void release_if_live(ts_tensor_t* handle) {
  if (*handle != 0) {
    CHECK_STATUS(ts_tensor_release(*handle), TS_OK);
    *handle = 0;
  }
}

void test_core_training_is_available_without_libtorch() {
  uint8_t available = 0;
  CHECK_STATUS(ts_training_available(&available), TS_OK);
  CHECK_TRUE(available == 1);

  CHECK_STATUS(ts_manual_seed(1234), TS_OK);
}

void test_leaf_state_and_add_accumulation() {
  ts_tensor_t raw = make_tensor({1.0f, -2.0f, 3.0f}, {3});
  ts_tensor_t x = require_grad(raw);
  CHECK_STATUS(ts_tensor_release(raw), TS_OK);

  uint8_t requires = 0;
  CHECK_STATUS(ts_tensor_requires_grad(x, &requires), TS_OK);
  CHECK_TRUE(requires == 1);

  ts_tensor_t doubled = 0;
  CHECK_STATUS(ts_tensor_add(x, x, &doubled), TS_OK);
  CHECK_STATUS(ts_tensor_requires_grad(doubled, &requires), TS_OK);
  CHECK_TRUE(requires == 1);

  ts_tensor_t loss = 0;
  CHECK_STATUS(ts_tensor_sum(doubled, &loss), TS_OK);
  CHECK_STATUS(ts_tensor_backward(loss), TS_OK);

  ts_tensor_t grad = 0;
  CHECK_STATUS(ts_tensor_grad(x, &grad), TS_OK);
  const auto values = read_tensor(grad);
  CHECK_TRUE(values.size() == 3);
  for (float value : values) check_near(value, 2.0f);

  release_if_live(&grad);
  release_if_live(&loss);
  release_if_live(&doubled);
  release_if_live(&x);
}

void test_multiply_and_sum_vjp() {
  ts_tensor_t raw = make_tensor({1.5f, -2.0f, 4.0f}, {3});
  ts_tensor_t x = require_grad(raw);
  CHECK_STATUS(ts_tensor_release(raw), TS_OK);

  ts_tensor_t square = 0;
  CHECK_STATUS(ts_tensor_multiply(x, x, &square), TS_OK);
  ts_tensor_t loss = 0;
  CHECK_STATUS(ts_tensor_sum(square, &loss), TS_OK);
  CHECK_STATUS(ts_tensor_backward(loss), TS_OK);

  ts_tensor_t grad = 0;
  CHECK_STATUS(ts_tensor_grad(x, &grad), TS_OK);
  const auto values = read_tensor(grad);
  CHECK_TRUE(values.size() == 3);
  check_near(values[0], 3.0f);
  check_near(values[1], -4.0f);
  check_near(values[2], 8.0f);

  release_if_live(&grad);
  release_if_live(&loss);
  release_if_live(&square);
  release_if_live(&x);
}

void test_matmul_vjp() {
  ts_tensor_t raw_a = make_tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  ts_tensor_t raw_b = make_tensor({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2});
  ts_tensor_t a = require_grad(raw_a);
  ts_tensor_t b = require_grad(raw_b);
  CHECK_STATUS(ts_tensor_release(raw_a), TS_OK);
  CHECK_STATUS(ts_tensor_release(raw_b), TS_OK);

  ts_tensor_t product = 0;
  CHECK_STATUS(ts_tensor_matmul(a, b, &product), TS_OK);
  ts_tensor_t loss = 0;
  CHECK_STATUS(ts_tensor_sum(product, &loss), TS_OK);
  CHECK_STATUS(ts_tensor_backward(loss), TS_OK);

  ts_tensor_t grad_a = 0;
  ts_tensor_t grad_b = 0;
  CHECK_STATUS(ts_tensor_grad(a, &grad_a), TS_OK);
  CHECK_STATUS(ts_tensor_grad(b, &grad_b), TS_OK);

  const auto da = read_tensor(grad_a);
  const auto db = read_tensor(grad_b);
  CHECK_TRUE(da.size() == 4);
  CHECK_TRUE(db.size() == 4);

  check_near(da[0], 11.0f);
  check_near(da[1], 15.0f);
  check_near(da[2], 11.0f);
  check_near(da[3], 15.0f);

  check_near(db[0], 4.0f);
  check_near(db[1], 4.0f);
  check_near(db[2], 6.0f);
  check_near(db[3], 6.0f);

  release_if_live(&grad_b);
  release_if_live(&grad_a);
  release_if_live(&loss);
  release_if_live(&product);
  release_if_live(&b);
  release_if_live(&a);
}

void test_relu_and_mse_backward() {
  ts_tensor_t raw = make_tensor({-2.0f, 1.0f, 3.0f}, {3});
  ts_tensor_t prediction = require_grad(raw);
  CHECK_STATUS(ts_tensor_release(raw), TS_OK);
  ts_tensor_t target = make_tensor({0.0f, 0.0f, 1.0f}, {3});

  ts_tensor_t activated = 0;
  CHECK_STATUS(ts_tensor_relu(prediction, &activated), TS_OK);
  const auto relu_values = read_tensor(activated);
  check_near(relu_values[0], 0.0f);
  check_near(relu_values[1], 1.0f);
  check_near(relu_values[2], 3.0f);

  ts_tensor_t loss = 0;
  CHECK_STATUS(ts_mse_loss(activated, target, &loss), TS_OK);
  check_near(read_scalar(loss), 5.0f / 3.0f);
  CHECK_STATUS(ts_tensor_backward(loss), TS_OK);

  ts_tensor_t grad = 0;
  CHECK_STATUS(ts_tensor_grad(prediction, &grad), TS_OK);
  const auto values = read_tensor(grad);
  check_near(values[0], 0.0f);
  check_near(values[1], 2.0f / 3.0f);
  check_near(values[2], 4.0f / 3.0f);

  release_if_live(&grad);
  release_if_live(&loss);
  release_if_live(&activated);
  release_if_live(&target);
  release_if_live(&prediction);
}

void test_linear_sgd_converges_without_libtorch() {
  CHECK_STATUS(ts_manual_seed(7), TS_OK);

  ts_module_t module = 0;
  CHECK_STATUS(ts_linear_create(1, 1, 0, &module), TS_OK);
  CHECK_TRUE(module != 0);

  size_t parameter_count = 0;
  CHECK_STATUS(ts_module_parameter_count(module, &parameter_count), TS_OK);
  CHECK_TRUE(parameter_count == 1);

  ts_optimizer_t optimizer = 0;
  CHECK_STATUS(ts_sgd_create(module, 0.05, 0.0, 0.0, &optimizer), TS_OK);
  CHECK_TRUE(optimizer != 0);

  ts_tensor_t input = make_tensor({-2.0f, -1.0f, 1.0f, 2.0f}, {4, 1});
  ts_tensor_t target = make_tensor({-4.0f, -2.0f, 2.0f, 4.0f}, {4, 1});

  float initial_loss = 0.0f;
  float final_loss = 0.0f;

  for (int step = 0; step < 120; ++step) {
    CHECK_STATUS(ts_optimizer_zero_grad(optimizer), TS_OK);

    ts_tensor_t prediction = 0;
    CHECK_STATUS(ts_module_forward(module, input, &prediction), TS_OK);
    ts_tensor_t loss = 0;
    CHECK_STATUS(ts_mse_loss(prediction, target, &loss), TS_OK);

    const float value = read_scalar(loss);
    if (step == 0) initial_loss = value;
    final_loss = value;

    CHECK_STATUS(ts_tensor_backward(loss), TS_OK);
    CHECK_STATUS(ts_optimizer_step(optimizer), TS_OK);

    release_if_live(&loss);
    release_if_live(&prediction);
  }

  CHECK_TRUE(std::isfinite(initial_loss));
  CHECK_TRUE(std::isfinite(final_loss));
  CHECK_TRUE(initial_loss > 0.01f);
  CHECK_TRUE(final_loss < initial_loss * 0.01f);
  CHECK_TRUE(final_loss < 1e-3f);

  ts_tensor_t parameter = 0;
  CHECK_STATUS(ts_module_parameter_at(module, 0, &parameter), TS_OK);
  const auto weights = read_tensor(parameter);
  CHECK_TRUE(weights.size() == 1);
  check_near(weights[0], 2.0f, 1e-2f);

  release_if_live(&parameter);
  release_if_live(&target);
  release_if_live(&input);
  CHECK_STATUS(ts_optimizer_release(optimizer), TS_OK);
  CHECK_STATUS(ts_module_release(module), TS_OK);
}

void test_training_lifetime_returns_to_baseline() {
  uint64_t tensors_before = 0;
  uint64_t storage_before = 0;
  uint64_t modules_before = 0;
  uint64_t optimizers_before = 0;
  CHECK_STATUS(ts_runtime_live_tensor_count(&tensors_before), TS_OK);
  CHECK_STATUS(ts_runtime_live_storage_bytes(&storage_before), TS_OK);
  CHECK_STATUS(ts_runtime_live_module_count(&modules_before), TS_OK);
  CHECK_STATUS(ts_runtime_live_optimizer_count(&optimizers_before), TS_OK);

  for (int iteration = 0; iteration < 50; ++iteration) {
    ts_module_t module = 0;
    CHECK_STATUS(ts_linear_create(2, 1, 0, &module), TS_OK);
    ts_optimizer_t optimizer = 0;
    CHECK_STATUS(ts_sgd_create(module, 0.01, 0.9, 0.001, &optimizer), TS_OK);

    ts_tensor_t input = make_tensor({1.0f, 2.0f}, {1, 2});
    ts_tensor_t target = make_tensor({1.0f}, {1, 1});
    ts_tensor_t prediction = 0;
    ts_tensor_t loss = 0;

    CHECK_STATUS(ts_optimizer_zero_grad(optimizer), TS_OK);
    CHECK_STATUS(ts_module_forward(module, input, &prediction), TS_OK);
    CHECK_STATUS(ts_mse_loss(prediction, target, &loss), TS_OK);
    CHECK_STATUS(ts_tensor_backward(loss), TS_OK);
    CHECK_STATUS(ts_optimizer_step(optimizer), TS_OK);

    release_if_live(&loss);
    release_if_live(&prediction);
    release_if_live(&target);
    release_if_live(&input);
    CHECK_STATUS(ts_optimizer_release(optimizer), TS_OK);
    CHECK_STATUS(ts_module_release(module), TS_OK);
  }

  uint64_t tensors_after = 0;
  uint64_t storage_after = 0;
  uint64_t modules_after = 0;
  uint64_t optimizers_after = 0;
  CHECK_STATUS(ts_runtime_live_tensor_count(&tensors_after), TS_OK);
  CHECK_STATUS(ts_runtime_live_storage_bytes(&storage_after), TS_OK);
  CHECK_STATUS(ts_runtime_live_module_count(&modules_after), TS_OK);
  CHECK_STATUS(ts_runtime_live_optimizer_count(&optimizers_after), TS_OK);

  CHECK_TRUE(tensors_after == tensors_before);
  CHECK_TRUE(storage_after == storage_before);
  CHECK_TRUE(modules_after == modules_before);
  CHECK_TRUE(optimizers_after == optimizers_before);
}

}  // namespace

int main() {
  test_core_training_is_available_without_libtorch();
  test_leaf_state_and_add_accumulation();
  test_multiply_and_sum_vjp();
  test_matmul_vjp();
  test_relu_and_mse_backward();
  test_linear_sgd_converges_without_libtorch();
  test_training_lifetime_returns_to_baseline();

  if (failures != 0) {
    std::cerr << failures << " CPU training engine assertion(s) failed\n";
    return 1;
  }

  std::cout << "Tensora CPU training engine tests passed\n";
  return 0;
}
