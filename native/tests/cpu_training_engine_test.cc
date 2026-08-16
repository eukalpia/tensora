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

void seed_unit_gradient(ts_tensor_t parameter) {
  ts_tensor_t loss = 0;
  CHECK_STATUS(ts_tensor_sum(parameter, &loss), TS_OK);
  CHECK_STATUS(ts_tensor_backward(loss), TS_OK);
  release_if_live(&loss);
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

  uint8_t requires_grad = 0;
  CHECK_STATUS(ts_tensor_requires_grad(x, &requires_grad), TS_OK);
  CHECK_TRUE(requires_grad == 1);

  ts_tensor_t doubled = 0;
  CHECK_STATUS(ts_tensor_add(x, x, &doubled), TS_OK);
  CHECK_STATUS(ts_tensor_requires_grad(doubled, &requires_grad), TS_OK);
  CHECK_TRUE(requires_grad == 1);

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

void test_nn_v2_activations_and_autograd() {
  ts_tensor_t input = make_tensor({-1.0f, 0.0f, 1.0f}, {3});
  ts_tensor_t gelu = 0;
  ts_tensor_t silu = 0;
  CHECK_STATUS(ts_tensor_gelu(input, &gelu), TS_OK);
  CHECK_STATUS(ts_tensor_silu(input, &silu), TS_OK);
  const auto gelu_values = read_tensor(gelu);
  const auto silu_values = read_tensor(silu);
  check_near(gelu_values[0], -0.15865526f, 2e-5f);
  check_near(gelu_values[1], 0.0f, 2e-5f);
  check_near(gelu_values[2], 0.8413447f, 2e-5f);
  check_near(silu_values[0], -0.26894143f, 2e-5f);
  check_near(silu_values[1], 0.0f, 2e-5f);
  check_near(silu_values[2], 0.7310586f, 2e-5f);

  ts_tensor_t gelu_leaf = require_grad(input);
  ts_tensor_t silu_leaf = require_grad(input);
  ts_tensor_t gelu_value = 0;
  ts_tensor_t silu_value = 0;
  ts_tensor_t gelu_loss = 0;
  ts_tensor_t silu_loss = 0;
  CHECK_STATUS(ts_tensor_gelu(gelu_leaf, &gelu_value), TS_OK);
  CHECK_STATUS(ts_tensor_silu(silu_leaf, &silu_value), TS_OK);
  CHECK_STATUS(ts_tensor_sum(gelu_value, &gelu_loss), TS_OK);
  CHECK_STATUS(ts_tensor_sum(silu_value, &silu_loss), TS_OK);
  CHECK_STATUS(ts_tensor_backward(gelu_loss), TS_OK);
  CHECK_STATUS(ts_tensor_backward(silu_loss), TS_OK);
  ts_tensor_t gelu_grad = 0;
  ts_tensor_t silu_grad = 0;
  CHECK_STATUS(ts_tensor_grad(gelu_leaf, &gelu_grad), TS_OK);
  CHECK_STATUS(ts_tensor_grad(silu_leaf, &silu_grad), TS_OK);
  const auto gelu_grad_values = read_tensor(gelu_grad);
  const auto silu_grad_values = read_tensor(silu_grad);
  check_near(gelu_grad_values[0], -0.08331547f, 3e-5f);
  check_near(gelu_grad_values[1], 0.5f, 3e-5f);
  check_near(gelu_grad_values[2], 1.0833155f, 3e-5f);
  check_near(silu_grad_values[0], 0.07232949f, 3e-5f);
  check_near(silu_grad_values[1], 0.5f, 3e-5f);
  check_near(silu_grad_values[2], 0.9276705f, 3e-5f);

  ts_tensor_t swiglu_input = make_tensor({1.0f, -1.0f, 2.0f, 3.0f}, {1, 4});
  ts_tensor_t swiglu_leaf = require_grad(swiglu_input);
  ts_tensor_t swiglu = 0;
  CHECK_STATUS(ts_tensor_swiglu(swiglu_leaf, &swiglu), TS_OK);
  size_t rank = 0;
  int64_t shape[2] = {0, 0};
  CHECK_STATUS(ts_tensor_shape(swiglu, shape, 2, &rank), TS_OK);
  CHECK_TRUE(rank == 2);
  CHECK_TRUE(shape[0] == 1);
  CHECK_TRUE(shape[1] == 2);
  const auto swiglu_values = read_tensor(swiglu);
  check_near(swiglu_values[0], 1.4621172f, 3e-5f);
  check_near(swiglu_values[1], -0.80682427f, 3e-5f);
  ts_tensor_t swiglu_loss = 0;
  CHECK_STATUS(ts_tensor_sum(swiglu, &swiglu_loss), TS_OK);
  CHECK_STATUS(ts_tensor_backward(swiglu_loss), TS_OK);
  ts_tensor_t swiglu_grad = 0;
  CHECK_STATUS(ts_tensor_grad(swiglu_leaf, &swiglu_grad), TS_OK);
  const auto swiglu_grad_values = read_tensor(swiglu_grad);
  check_near(swiglu_grad_values[0], 1.855341f, 4e-5f);
  check_near(swiglu_grad_values[1], 0.21698847f, 4e-5f);
  check_near(swiglu_grad_values[2], 0.7310586f, 4e-5f);
  check_near(swiglu_grad_values[3], -0.26894143f, 4e-5f);

  ts_tensor_t scalar = make_tensor({1.0f}, {});
  ts_tensor_t odd = make_tensor({1.0f, 2.0f, 3.0f}, {1, 3});
  ts_tensor_t invalid_output = 123;
  CHECK_STATUS(ts_tensor_swiglu(scalar, &invalid_output), TS_INVALID_SHAPE);
  CHECK_TRUE(invalid_output == 0);
  invalid_output = 123;
  CHECK_STATUS(ts_tensor_swiglu(odd, &invalid_output), TS_INVALID_SHAPE);
  CHECK_TRUE(invalid_output == 0);

  release_if_live(&odd);
  release_if_live(&scalar);
  release_if_live(&swiglu_grad);
  release_if_live(&swiglu_loss);
  release_if_live(&swiglu);
  release_if_live(&swiglu_leaf);
  release_if_live(&swiglu_input);
  release_if_live(&silu_grad);
  release_if_live(&gelu_grad);
  release_if_live(&silu_loss);
  release_if_live(&gelu_loss);
  release_if_live(&silu_value);
  release_if_live(&gelu_value);
  release_if_live(&silu_leaf);
  release_if_live(&gelu_leaf);
  release_if_live(&silu);
  release_if_live(&gelu);
  release_if_live(&input);
}

void test_nn_v2_parameter_optimizer() {
  ts_optimizer_t optimizer = 123;
  CHECK_STATUS(ts_sgd_create_for_tensors(nullptr, 0, 0.1, 0.0, 0.0,
                                         &optimizer),
               TS_INVALID_ARGUMENT);
  CHECK_TRUE(optimizer == 0);

  ts_tensor_t raw_first = make_tensor({1.0f}, {1});
  ts_tensor_t raw_second = make_tensor({3.0f}, {1});
  ts_tensor_t first = require_grad(raw_first);
  ts_tensor_t second = require_grad(raw_second);
  release_if_live(&raw_first);
  release_if_live(&raw_second);
  seed_unit_gradient(first);
  seed_unit_gradient(second);

  const ts_tensor_t parameters[2] = {first, second};
  CHECK_STATUS(ts_sgd_create_for_tensors(parameters, 2, 0.1, 0.0, 0.0,
                                         &optimizer),
               TS_OK);
  CHECK_TRUE(optimizer != 0);
  release_if_live(&first);
  CHECK_STATUS(ts_parameter_optimizer_step(optimizer), TS_OK);
  check_near(read_scalar(second), 2.9f, 1e-5f);
  CHECK_STATUS(ts_parameter_optimizer_zero_grad(optimizer), TS_OK);
  CHECK_STATUS(ts_parameter_optimizer_release(optimizer), TS_OK);

  const ts_tensor_t duplicate[2] = {second, second};
  optimizer = 999;
  CHECK_STATUS(ts_sgd_create_for_tensors(duplicate, 2, 0.1, 0.0, 0.0,
                                         &optimizer),
               TS_INVALID_ARGUMENT);
  CHECK_TRUE(optimizer == 0);

  ts_tensor_t frozen = make_tensor({7.0f}, {1});
  const ts_tensor_t all_frozen[1] = {frozen};
  CHECK_STATUS(ts_adam_create_for_tensors(all_frozen, 1, 0.001, 0.9, 0.999,
                                          1e-8, 0.0, &optimizer),
               TS_INVALID_ARGUMENT);
  const ts_tensor_t mixed[2] = {second, frozen};
  CHECK_STATUS(ts_adamw_create_for_tensors(mixed, 2, 0.001, 0.9, 0.999, 1e-8,
                                           0.01, &optimizer),
               TS_OK);
  CHECK_STATUS(ts_parameter_optimizer_release(optimizer), TS_OK);

  const ts_tensor_t invalid[1] = {UINT64_C(0xffffffffffffffff)};
  optimizer = 0;
  CHECK_STATUS(ts_sgd_create_for_tensors(invalid, 1, 0.1, 0.0, 0.0,
                                         &optimizer),
               TS_INVALID_HANDLE);

  release_if_live(&second);
  release_if_live(&frozen);
}

void test_nn_v2_state_identity_and_transaction() {
  ts_module_t linear = 0;
  CHECK_STATUS(ts_linear_create(1, 1, 1, &linear), TS_OK);
  ts_tensor_t weight_a = 0;
  ts_tensor_t weight_b = 0;
  CHECK_STATUS(ts_module_parameter_at(linear, 0, &weight_a), TS_OK);
  CHECK_STATUS(ts_module_parameter_at(linear, 0, &weight_b), TS_OK);

  uint64_t identity_a = 0;
  uint64_t identity_b = 0;
  CHECK_STATUS(ts_tensor_identity(weight_a, &identity_a), TS_OK);
  CHECK_STATUS(ts_tensor_identity(weight_b, &identity_b), TS_OK);
  CHECK_TRUE(identity_a != 0);
  CHECK_TRUE(identity_a == identity_b);

  ts_tensor_t original = make_tensor({3.0f}, {1});
  ts_tensor_t snapshot = 0;
  CHECK_STATUS(ts_tensor_clone_detached(original, &snapshot), TS_OK);
  uint8_t snapshot_requires_grad = 99;
  CHECK_STATUS(ts_tensor_requires_grad(snapshot, &snapshot_requires_grad),
               TS_OK);
  CHECK_TRUE(snapshot_requires_grad == 0);

  ts_tensor_t target_a = make_tensor({1.0f}, {1});
  ts_tensor_t target_b = make_tensor({2.0f}, {1});
  ts_tensor_t source_a = make_tensor({5.0f}, {1});
  ts_tensor_t wrong_source = make_tensor({7.0f, 8.0f}, {1, 2});
  const ts_tensor_t targets[2] = {target_a, target_b};
  const ts_tensor_t invalid_sources[2] = {source_a, wrong_source};
  CHECK_STATUS(ts_tensor_assign_many(targets, invalid_sources, 2),
               TS_INVALID_SHAPE);
  check_near(read_scalar(target_a), 1.0f, 1e-6f);
  check_near(read_scalar(target_b), 2.0f, 1e-6f);

  ts_tensor_t source_b = make_tensor({6.0f}, {1});
  const ts_tensor_t valid_sources[2] = {source_a, source_b};
  CHECK_STATUS(ts_tensor_assign_many(targets, valid_sources, 2), TS_OK);
  check_near(read_scalar(target_a), 5.0f, 1e-6f);
  check_near(read_scalar(target_b), 6.0f, 1e-6f);
  check_near(read_scalar(snapshot), 3.0f, 1e-6f);

  const ts_tensor_t duplicate_targets[2] = {target_a, target_a};
  CHECK_STATUS(ts_tensor_assign_many(duplicate_targets, valid_sources, 2),
               TS_INVALID_ARGUMENT);
  uint64_t invalid_identity = 0;
  CHECK_STATUS(ts_tensor_identity(UINT64_C(0xffffffffffffffff),
                                  &invalid_identity),
               TS_INVALID_HANDLE);

  release_if_live(&weight_a);
  release_if_live(&weight_b);
  CHECK_STATUS(ts_module_release(linear), TS_OK);
  release_if_live(&original);
  release_if_live(&snapshot);
  release_if_live(&target_a);
  release_if_live(&target_b);
  release_if_live(&source_a);
  release_if_live(&source_b);
  release_if_live(&wrong_source);
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
  test_nn_v2_activations_and_autograd();
  test_nn_v2_parameter_optimizer();
  test_nn_v2_state_identity_and_transaction();
  test_linear_sgd_converges_without_libtorch();
  test_training_lifetime_returns_to_baseline();

  if (failures != 0) {
    std::cerr << failures << " CPU training engine assertion(s) failed\n";
    return 1;
  }

  std::cout << "Tensora CPU training engine tests passed\n";
  return 0;
}
