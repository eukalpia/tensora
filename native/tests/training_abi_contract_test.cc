#include "tensora.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect_status(ts_status_t actual,
                   ts_status_t expected,
                   const char* operation) {
  if (actual == expected) return;
  std::cerr << "FAIL " << operation << ": expected " << expected << ", got "
            << actual << " error=" << ts_last_error_message() << "\n";
  ++failures;
}

void expect_true(bool value, const char* operation) {
  if (value) return;
  std::cerr << "FAIL " << operation << "\n";
  ++failures;
}

ts_tensor_t make_tensor(const std::vector<float>& values,
                        const std::vector<int64_t>& shape) {
  ts_tensor_t tensor = 0;
  expect_status(ts_tensor_from_f32(values.data(), values.size(), shape.data(),
                                   shape.size(), &tensor),
                TS_OK, "make tensor");
  return tensor;
}

void release_tensor(ts_tensor_t* tensor) {
  if (tensor == nullptr || *tensor == 0) return;
  expect_status(ts_tensor_release(*tensor), TS_OK, "tensor release");
  *tensor = 0;
}

void test_tensor_autograd_contracts() {
  ts_tensor_t x = make_tensor({1.0f, -2.0f, 3.0f, 4.0f}, {2, 2});
  ts_tensor_t out = 777;
  uint8_t requires_grad = 77;

  expect_status(ts_tensor_with_requires_grad(x, 1, nullptr),
                TS_INVALID_ARGUMENT, "requires grad null output");
  expect_status(ts_tensor_with_requires_grad(UINT64_C(999999999), 1, &out),
                TS_INVALID_HANDLE, "requires grad invalid handle");
  expect_true(out == 0, "requires grad zeroes output on invalid handle");

  expect_status(ts_tensor_requires_grad(x, nullptr), TS_INVALID_ARGUMENT,
                "requires grad query null output");
  expect_status(ts_tensor_requires_grad(UINT64_C(999999999), &requires_grad),
                TS_INVALID_HANDLE, "requires grad query invalid handle");

  expect_status(ts_tensor_backward(x), TS_INVALID_SHAPE,
                "backward rejects non-scalar root");

  out = 777;
  expect_status(ts_tensor_grad(x, nullptr), TS_INVALID_ARGUMENT,
                "grad null output");
  expect_status(ts_tensor_grad(x, &out), TS_INVALID_ARGUMENT,
                "grad unavailable before backward");
  expect_true(out == 0, "grad zeroes output when unavailable");

  out = 777;
  expect_status(ts_tensor_relu(x, nullptr), TS_INVALID_ARGUMENT,
                "relu null output");
  expect_status(ts_tensor_relu(UINT64_C(999999999), &out), TS_INVALID_HANDLE,
                "relu invalid handle");
  expect_true(out == 0, "relu zeroes output on invalid handle");

  out = 777;
  expect_status(ts_tensor_sigmoid(x, nullptr), TS_INVALID_ARGUMENT,
                "sigmoid null output");
  expect_status(ts_tensor_tanh(x, nullptr), TS_INVALID_ARGUMENT,
                "tanh null output");

  ts_tensor_t mismatch = make_tensor({1.0f, 2.0f}, {2});
  out = 777;
  expect_status(ts_mse_loss(x, mismatch, &out), TS_INVALID_SHAPE,
                "mse shape mismatch");
  expect_true(out == 0, "mse zeroes output on shape mismatch");
  expect_status(ts_mse_loss(x, mismatch, nullptr), TS_INVALID_ARGUMENT,
                "mse null output");

  out = 777;
  expect_status(ts_cross_entropy_loss(x, mismatch, &out), TS_INVALID_SHAPE,
                "cross entropy shape mismatch");
  expect_true(out == 0, "cross entropy zeroes output on shape mismatch");
  expect_status(ts_cross_entropy_loss(x, x, nullptr), TS_INVALID_ARGUMENT,
                "cross entropy null output");

  release_tensor(&mismatch);
  release_tensor(&x);
}

void test_module_contracts() {
  ts_module_t module = 777;
  ts_tensor_t out = 777;
  size_t count = 777;

  expect_status(ts_linear_create(1, 1, 1, nullptr), TS_INVALID_ARGUMENT,
                "linear null output");
  expect_status(ts_linear_create(0, 1, 1, &module), TS_INVALID_ARGUMENT,
                "linear zero input features");
  expect_true(module == 0, "linear zeroes output on invalid dimensions");
  module = 777;
  expect_status(ts_linear_create(-1, 1, 1, &module), TS_INVALID_ARGUMENT,
                "linear negative input features");
  expect_true(module == 0, "linear zeroes output on negative dimensions");

  expect_status(ts_module_forward(0, 0, nullptr), TS_INVALID_ARGUMENT,
                "module forward null output");
  out = 777;
  expect_status(ts_module_forward(UINT64_C(999999999), 0, &out),
                TS_INVALID_HANDLE, "module forward invalid module");
  expect_true(out == 0, "module forward zeroes output on invalid module");

  expect_status(ts_module_parameter_count(0, nullptr), TS_INVALID_ARGUMENT,
                "module parameter count null output");
  count = 777;
  expect_status(ts_module_parameter_count(UINT64_C(999999999), &count),
                TS_INVALID_HANDLE, "module parameter count invalid module");
  expect_true(count == 0, "parameter count zeroed on invalid module");

  expect_status(ts_module_buffer_count(0, nullptr), TS_INVALID_ARGUMENT,
                "module buffer count null output");

  expect_status(ts_linear_create(2, 3, 1, &module), TS_OK,
                "linear valid creation");
  expect_true(module != 0, "linear handle is non-zero");

  expect_status(ts_module_parameter_count(module, &count), TS_OK,
                "module parameter count");
  expect_true(count == 2, "linear with bias exposes two parameters");
  out = 0;
  expect_status(ts_module_parameter_at(module, 0, &out), TS_OK,
                "module weight parameter");
  release_tensor(&out);
  expect_status(ts_module_parameter_at(module, 2, &out), TS_INVALID_ARGUMENT,
                "module parameter index bounds");
  expect_true(out == 0, "parameter output zeroed on bounds failure");

  count = 777;
  expect_status(ts_module_buffer_count(module, &count), TS_OK,
                "module buffer count");
  expect_true(count == 0, "linear has no buffers");
  out = 777;
  expect_status(ts_module_buffer_at(module, 0, &out), TS_INVALID_ARGUMENT,
                "module buffer bounds");
  expect_true(out == 0, "buffer output zeroed on bounds failure");

  expect_status(ts_module_set_training(module, 1), TS_OK, "module train mode");
  expect_status(ts_module_set_training(module, 0), TS_OK, "module eval mode");
  expect_status(ts_module_to_device(module, TS_DEVICE_CPU, 0), TS_OK,
                "module CPU placement");
  expect_status(ts_module_to_device(module, TS_DEVICE_CPU, 1),
                TS_INVALID_ARGUMENT, "module invalid CPU index");
  expect_status(ts_module_to_device(module, TS_DEVICE_CUDA, 0), TS_UNSUPPORTED,
                "module unavailable accelerator");

  expect_status(ts_module_save(module, nullptr), TS_INVALID_ARGUMENT,
                "module save null path");
  expect_status(ts_module_save(module, ""), TS_INVALID_ARGUMENT,
                "module save empty path");
  expect_status(ts_module_load(module, nullptr), TS_INVALID_ARGUMENT,
                "module load null path");
  expect_status(ts_module_load(module, ""), TS_INVALID_ARGUMENT,
                "module load empty path");

  expect_status(ts_module_release(module), TS_OK, "module release");
  expect_status(ts_module_release(module), TS_INVALID_HANDLE,
                "module double release");
}

void test_optimizer_contracts() {
  ts_module_t module = 0;
  expect_status(ts_linear_create(2, 1, 0, &module), TS_OK,
                "optimizer module creation");

  ts_optimizer_t optimizer = 777;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();

  expect_status(ts_sgd_create(module, 0.01, 0.0, 0.0, nullptr),
                TS_INVALID_ARGUMENT, "sgd null output");
  expect_status(ts_sgd_create(module, 0.0, 0.0, 0.0, &optimizer),
                TS_INVALID_ARGUMENT, "sgd zero learning rate");
  expect_true(optimizer == 0, "sgd zeroes output on invalid learning rate");
  optimizer = 777;
  expect_status(ts_sgd_create(module, nan, 0.0, 0.0, &optimizer),
                TS_INVALID_ARGUMENT, "sgd NaN learning rate");
  expect_true(optimizer == 0, "sgd zeroes output on NaN");
  optimizer = 777;
  expect_status(ts_sgd_create(module, 0.01, -0.1, 0.0, &optimizer),
                TS_INVALID_ARGUMENT, "sgd negative momentum");
  optimizer = 777;
  expect_status(ts_sgd_create(module, 0.01, 0.0, -0.1, &optimizer),
                TS_INVALID_ARGUMENT, "sgd negative weight decay");
  optimizer = 777;
  expect_status(ts_sgd_create(UINT64_C(999999999), 0.01, 0.0, 0.0,
                              &optimizer),
                TS_INVALID_HANDLE, "sgd invalid module");
  expect_true(optimizer == 0, "sgd zeroes output on invalid module");

  optimizer = 777;
  expect_status(ts_adam_create(module, 0.001, 1.0, 0.999, 1e-8, 0.0,
                               &optimizer),
                TS_INVALID_ARGUMENT, "adam beta1 upper bound");
  optimizer = 777;
  expect_status(ts_adam_create(module, 0.001, 0.9, -0.1, 1e-8, 0.0,
                               &optimizer),
                TS_INVALID_ARGUMENT, "adam beta2 lower bound");
  optimizer = 777;
  expect_status(ts_adam_create(module, 0.001, 0.9, 0.999, 0.0, 0.0,
                               &optimizer),
                TS_INVALID_ARGUMENT, "adam zero epsilon");
  optimizer = 777;
  expect_status(ts_adamw_create(module, infinity, 0.9, 0.999, 1e-8, 0.01,
                                &optimizer),
                TS_INVALID_ARGUMENT, "adamw infinite learning rate");

  expect_status(ts_sgd_create(module, 0.01, 0.9, 0.001, &optimizer), TS_OK,
                "sgd valid creation");
  expect_true(optimizer != 0, "sgd handle is non-zero");
  expect_status(ts_optimizer_zero_grad(optimizer), TS_OK,
                "optimizer zero grad without gradients");
  expect_status(ts_optimizer_step(optimizer), TS_OK,
                "optimizer step without gradients");
  expect_status(ts_optimizer_release(optimizer), TS_OK, "optimizer release");
  expect_status(ts_optimizer_release(optimizer), TS_INVALID_HANDLE,
                "optimizer double release");

  expect_status(ts_module_release(module), TS_OK,
                "optimizer module release");
}

void test_live_counts_return_to_baseline() {
  uint64_t tensor_count = 777;
  uint64_t module_count = 777;
  uint64_t optimizer_count = 777;
  expect_status(ts_runtime_live_tensor_count(nullptr), TS_INVALID_ARGUMENT,
                "live tensor null output");
  expect_status(ts_runtime_live_module_count(nullptr), TS_INVALID_ARGUMENT,
                "live module null output");
  expect_status(ts_runtime_live_optimizer_count(nullptr), TS_INVALID_ARGUMENT,
                "live optimizer null output");
  expect_status(ts_runtime_live_tensor_count(&tensor_count), TS_OK,
                "live tensor count");
  expect_status(ts_runtime_live_module_count(&module_count), TS_OK,
                "live module count");
  expect_status(ts_runtime_live_optimizer_count(&optimizer_count), TS_OK,
                "live optimizer count");
  expect_true(tensor_count == 0, "no leaked public tensor handles");
  expect_true(module_count == 0, "no leaked module handles");
  expect_true(optimizer_count == 0, "no leaked optimizer handles");
}

}  // namespace

int main() {
  uint8_t available = 0;
  expect_status(ts_training_available(nullptr), TS_INVALID_ARGUMENT,
                "training available null output");
  expect_status(ts_training_available(&available), TS_OK,
                "training availability");
  expect_true(available == 1, "CPU training is a core capability");
  expect_status(ts_manual_seed(UINT64_C(123456789)), TS_OK,
                "manual seed");

  test_tensor_autograd_contracts();
  test_module_contracts();
  test_optimizer_contracts();
  test_live_counts_return_to_baseline();

  if (failures != 0) {
    std::cerr << failures << " training ABI contract assertion(s) failed\n";
    return 1;
  }
  std::cout << "Tensora autonomous training ABI contracts passed\n";
  return 0;
}
