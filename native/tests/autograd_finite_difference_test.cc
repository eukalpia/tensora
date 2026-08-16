#include "tensora.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr float kEpsilon = 1e-3f;
constexpr float kAbsoluteTolerance = 3e-3f;
constexpr float kRelativeTolerance = 3e-3f;

int failures = 0;

void fail(const std::string& message) {
  std::cerr << "FAIL: " << message << "\n";
  ++failures;
}

bool ok(ts_status_t status, const char* operation) {
  if (status == TS_OK) return true;
  fail(std::string(operation) + " status=" + std::to_string(status) +
       " error=" + ts_last_error_message());
  return false;
}

void release(ts_tensor_t* tensor) {
  if (tensor == nullptr || *tensor == 0) return;
  if (ts_tensor_release(*tensor) != TS_OK) {
    fail("tensor release failed");
  }
  *tensor = 0;
}

ts_tensor_t make_tensor(const std::vector<float>& values,
                        const std::vector<int64_t>& shape) {
  ts_tensor_t tensor = 0;
  if (!ok(ts_tensor_from_f32(values.data(), values.size(), shape.data(),
                             shape.size(), &tensor),
          "tensor_from_f32")) {
    return 0;
  }
  return tensor;
}

ts_tensor_t require_grad(ts_tensor_t source) {
  ts_tensor_t tensor = 0;
  if (!ok(ts_tensor_with_requires_grad(source, 1, &tensor),
          "tensor_with_requires_grad")) {
    return 0;
  }
  return tensor;
}

std::vector<float> read(ts_tensor_t tensor) {
  uint64_t numel = 0;
  if (!ok(ts_tensor_numel(tensor, &numel), "tensor_numel")) return {};
  std::vector<float> values(static_cast<size_t>(numel), 0.0f);
  size_t written = 0;
  if (!ok(ts_tensor_copy_to_host_f32(tensor, values.data(), values.size(),
                                     &written),
          "tensor_copy_to_host_f32")) {
    return {};
  }
  if (written != values.size()) {
    fail("tensor copy wrote unexpected number of values");
    return {};
  }
  return values;
}

float scalar(ts_tensor_t tensor) {
  const auto values = read(tensor);
  if (values.size() != 1) {
    fail("expected scalar tensor");
    return 0.0f;
  }
  return values[0];
}

bool near(float actual, float expected) {
  const float difference = std::fabs(actual - expected);
  const float scale = std::max(std::fabs(actual), std::fabs(expected));
  return difference <= kAbsoluteTolerance + kRelativeTolerance * scale;
}

void check_vector(const std::vector<float>& actual,
                  const std::vector<float>& expected,
                  const char* label) {
  if (actual.size() != expected.size()) {
    fail(std::string(label) + " size mismatch");
    return;
  }
  for (size_t index = 0; index < actual.size(); ++index) {
    if (!near(actual[index], expected[index])) {
      fail(std::string(label) + " gradient mismatch at index " +
           std::to_string(index) + ": analytic=" +
           std::to_string(actual[index]) + " numeric=" +
           std::to_string(expected[index]));
    }
  }
}

float evaluate_square_sum(const std::vector<float>& values) {
  ts_tensor_t x = make_tensor(values, {static_cast<int64_t>(values.size())});
  ts_tensor_t square = 0;
  ts_tensor_t loss = 0;
  if (x != 0) ok(ts_tensor_multiply(x, x, &square), "multiply");
  if (square != 0) ok(ts_tensor_sum(square, &loss), "sum");
  const float result = loss == 0 ? 0.0f : scalar(loss);
  release(&loss);
  release(&square);
  release(&x);
  return result;
}

void test_multiply_sum_finite_difference() {
  const std::vector<float> base = {-1.7f, -0.3f, 0.8f, 2.2f};
  ts_tensor_t raw = make_tensor(base, {4});
  ts_tensor_t x = raw == 0 ? 0 : require_grad(raw);
  release(&raw);
  ts_tensor_t square = 0;
  ts_tensor_t loss = 0;
  if (x != 0) ok(ts_tensor_multiply(x, x, &square), "multiply analytic");
  if (square != 0) ok(ts_tensor_sum(square, &loss), "sum analytic");
  if (loss != 0) ok(ts_tensor_backward(loss), "backward square sum");
  ts_tensor_t gradient = 0;
  if (x != 0) ok(ts_tensor_grad(x, &gradient), "grad square sum");
  const auto analytic = gradient == 0 ? std::vector<float>{} : read(gradient);

  std::vector<float> numeric(base.size(), 0.0f);
  for (size_t index = 0; index < base.size(); ++index) {
    auto plus = base;
    auto minus = base;
    plus[index] += kEpsilon;
    minus[index] -= kEpsilon;
    numeric[index] =
        (evaluate_square_sum(plus) - evaluate_square_sum(minus)) /
        (2.0f * kEpsilon);
  }
  check_vector(analytic, numeric, "multiply/sum");

  release(&gradient);
  release(&loss);
  release(&square);
  release(&x);
}

float evaluate_matmul_sum(const std::vector<float>& a,
                          const std::vector<float>& b) {
  ts_tensor_t left = make_tensor(a, {2, 3});
  ts_tensor_t right = make_tensor(b, {3, 2});
  ts_tensor_t product = 0;
  ts_tensor_t loss = 0;
  if (left != 0 && right != 0) {
    ok(ts_tensor_matmul(left, right, &product), "matmul numeric");
  }
  if (product != 0) ok(ts_tensor_sum(product, &loss), "matmul sum numeric");
  const float result = loss == 0 ? 0.0f : scalar(loss);
  release(&loss);
  release(&product);
  release(&right);
  release(&left);
  return result;
}

void test_matmul_finite_difference() {
  const std::vector<float> a = {0.3f, -1.2f, 2.0f, 0.7f, 1.1f, -0.4f};
  const std::vector<float> b = {1.5f, -0.5f, 0.2f, 0.9f, -1.0f, 1.7f};

  ts_tensor_t raw_a = make_tensor(a, {2, 3});
  ts_tensor_t raw_b = make_tensor(b, {3, 2});
  ts_tensor_t left = raw_a == 0 ? 0 : require_grad(raw_a);
  ts_tensor_t right = raw_b == 0 ? 0 : require_grad(raw_b);
  release(&raw_a);
  release(&raw_b);
  ts_tensor_t product = 0;
  ts_tensor_t loss = 0;
  if (left != 0 && right != 0) {
    ok(ts_tensor_matmul(left, right, &product), "matmul analytic");
  }
  if (product != 0) ok(ts_tensor_sum(product, &loss), "matmul sum analytic");
  if (loss != 0) ok(ts_tensor_backward(loss), "backward matmul");
  ts_tensor_t grad_a = 0;
  ts_tensor_t grad_b = 0;
  if (left != 0) ok(ts_tensor_grad(left, &grad_a), "grad matmul left");
  if (right != 0) ok(ts_tensor_grad(right, &grad_b), "grad matmul right");
  const auto analytic_a = grad_a == 0 ? std::vector<float>{} : read(grad_a);
  const auto analytic_b = grad_b == 0 ? std::vector<float>{} : read(grad_b);

  std::vector<float> numeric_a(a.size(), 0.0f);
  for (size_t index = 0; index < a.size(); ++index) {
    auto plus = a;
    auto minus = a;
    plus[index] += kEpsilon;
    minus[index] -= kEpsilon;
    numeric_a[index] =
        (evaluate_matmul_sum(plus, b) - evaluate_matmul_sum(minus, b)) /
        (2.0f * kEpsilon);
  }
  std::vector<float> numeric_b(b.size(), 0.0f);
  for (size_t index = 0; index < b.size(); ++index) {
    auto plus = b;
    auto minus = b;
    plus[index] += kEpsilon;
    minus[index] -= kEpsilon;
    numeric_b[index] =
        (evaluate_matmul_sum(a, plus) - evaluate_matmul_sum(a, minus)) /
        (2.0f * kEpsilon);
  }

  check_vector(analytic_a, numeric_a, "matmul left");
  check_vector(analytic_b, numeric_b, "matmul right");

  release(&grad_b);
  release(&grad_a);
  release(&loss);
  release(&product);
  release(&right);
  release(&left);
}

enum class Activation { kRelu, kSigmoid, kTanh };

bool activation(ts_tensor_t input, Activation kind, ts_tensor_t* out) {
  switch (kind) {
    case Activation::kRelu:
      return ok(ts_tensor_relu(input, out), "relu");
    case Activation::kSigmoid:
      return ok(ts_tensor_sigmoid(input, out), "sigmoid");
    case Activation::kTanh:
      return ok(ts_tensor_tanh(input, out), "tanh");
  }
  return false;
}

float evaluate_activation_sum(const std::vector<float>& values,
                              Activation kind) {
  ts_tensor_t input = make_tensor(values, {static_cast<int64_t>(values.size())});
  ts_tensor_t output = 0;
  ts_tensor_t loss = 0;
  if (input != 0) activation(input, kind, &output);
  if (output != 0) ok(ts_tensor_sum(output, &loss), "activation sum");
  const float result = loss == 0 ? 0.0f : scalar(loss);
  release(&loss);
  release(&output);
  release(&input);
  return result;
}

void test_activation_finite_difference(Activation kind, const char* label) {
  const std::vector<float> base = {-2.0f, -0.4f, 0.6f, 1.8f};
  ts_tensor_t raw = make_tensor(base, {4});
  ts_tensor_t input = raw == 0 ? 0 : require_grad(raw);
  release(&raw);
  ts_tensor_t output = 0;
  ts_tensor_t loss = 0;
  if (input != 0) activation(input, kind, &output);
  if (output != 0) ok(ts_tensor_sum(output, &loss), "activation sum analytic");
  if (loss != 0) ok(ts_tensor_backward(loss), "activation backward");
  ts_tensor_t gradient = 0;
  if (input != 0) ok(ts_tensor_grad(input, &gradient), "activation grad");
  const auto analytic = gradient == 0 ? std::vector<float>{} : read(gradient);

  std::vector<float> numeric(base.size(), 0.0f);
  for (size_t index = 0; index < base.size(); ++index) {
    auto plus = base;
    auto minus = base;
    plus[index] += kEpsilon;
    minus[index] -= kEpsilon;
    numeric[index] =
        (evaluate_activation_sum(plus, kind) -
         evaluate_activation_sum(minus, kind)) /
        (2.0f * kEpsilon);
  }
  check_vector(analytic, numeric, label);

  release(&gradient);
  release(&loss);
  release(&output);
  release(&input);
}

float evaluate_mse(const std::vector<float>& prediction,
                   const std::vector<float>& target) {
  ts_tensor_t p = make_tensor(prediction, {4});
  ts_tensor_t t = make_tensor(target, {4});
  ts_tensor_t loss = 0;
  if (p != 0 && t != 0) ok(ts_mse_loss(p, t, &loss), "mse numeric");
  const float result = loss == 0 ? 0.0f : scalar(loss);
  release(&loss);
  release(&t);
  release(&p);
  return result;
}

void test_mse_finite_difference() {
  const std::vector<float> prediction = {-1.3f, 0.2f, 1.7f, 2.1f};
  const std::vector<float> target = {-0.5f, -0.4f, 1.0f, 3.0f};
  ts_tensor_t raw = make_tensor(prediction, {4});
  ts_tensor_t p = raw == 0 ? 0 : require_grad(raw);
  release(&raw);
  ts_tensor_t t = make_tensor(target, {4});
  ts_tensor_t loss = 0;
  if (p != 0 && t != 0) ok(ts_mse_loss(p, t, &loss), "mse analytic");
  if (loss != 0) ok(ts_tensor_backward(loss), "mse backward");
  ts_tensor_t gradient = 0;
  if (p != 0) ok(ts_tensor_grad(p, &gradient), "mse grad");
  const auto analytic = gradient == 0 ? std::vector<float>{} : read(gradient);

  std::vector<float> numeric(prediction.size(), 0.0f);
  for (size_t index = 0; index < prediction.size(); ++index) {
    auto plus = prediction;
    auto minus = prediction;
    plus[index] += kEpsilon;
    minus[index] -= kEpsilon;
    numeric[index] =
        (evaluate_mse(plus, target) - evaluate_mse(minus, target)) /
        (2.0f * kEpsilon);
  }
  check_vector(analytic, numeric, "mse");

  release(&gradient);
  release(&loss);
  release(&t);
  release(&p);
}

float evaluate_cross_entropy(const std::vector<float>& logits,
                             const std::vector<float>& target) {
  ts_tensor_t l = make_tensor(logits, {2, 3});
  ts_tensor_t t = make_tensor(target, {2, 3});
  ts_tensor_t loss = 0;
  if (l != 0 && t != 0) {
    ok(ts_cross_entropy_loss(l, t, &loss), "cross entropy numeric");
  }
  const float result = loss == 0 ? 0.0f : scalar(loss);
  release(&loss);
  release(&t);
  release(&l);
  return result;
}

void test_cross_entropy_finite_difference() {
  const std::vector<float> logits = {0.2f, 1.1f, -0.7f, 1.3f, -0.2f, 0.4f};
  const std::vector<float> target = {0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f};
  ts_tensor_t raw = make_tensor(logits, {2, 3});
  ts_tensor_t l = raw == 0 ? 0 : require_grad(raw);
  release(&raw);
  ts_tensor_t t = make_tensor(target, {2, 3});
  ts_tensor_t loss = 0;
  if (l != 0 && t != 0) {
    ok(ts_cross_entropy_loss(l, t, &loss), "cross entropy analytic");
  }
  if (loss != 0) ok(ts_tensor_backward(loss), "cross entropy backward");
  ts_tensor_t gradient = 0;
  if (l != 0) ok(ts_tensor_grad(l, &gradient), "cross entropy grad");
  const auto analytic = gradient == 0 ? std::vector<float>{} : read(gradient);

  std::vector<float> numeric(logits.size(), 0.0f);
  for (size_t index = 0; index < logits.size(); ++index) {
    auto plus = logits;
    auto minus = logits;
    plus[index] += kEpsilon;
    minus[index] -= kEpsilon;
    numeric[index] =
        (evaluate_cross_entropy(plus, target) -
         evaluate_cross_entropy(minus, target)) /
        (2.0f * kEpsilon);
  }
  check_vector(analytic, numeric, "cross entropy logits");

  release(&gradient);
  release(&loss);
  release(&t);
  release(&l);
}

void test_view_like_transform_gradient_connectivity() {
  const std::vector<float> values = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
  ts_tensor_t raw = make_tensor(values, {2, 3});
  ts_tensor_t x = raw == 0 ? 0 : require_grad(raw);
  release(&raw);

  const int64_t reshaped_dims[2] = {3, 2};
  ts_tensor_t reshaped = 0;
  ts_tensor_t transposed = 0;
  ts_tensor_t loss = 0;
  if (x != 0) {
    ok(ts_tensor_reshape(x, reshaped_dims, 2, &reshaped), "reshape grad");
  }
  if (reshaped != 0) {
    ok(ts_tensor_transpose2d(reshaped, &transposed), "transpose grad");
  }
  if (transposed != 0) ok(ts_tensor_sum(transposed, &loss), "transform sum");
  if (loss != 0) ok(ts_tensor_backward(loss), "transform backward");

  ts_tensor_t gradient = 0;
  if (x != 0) ok(ts_tensor_grad(x, &gradient), "transform grad");
  const auto actual = gradient == 0 ? std::vector<float>{} : read(gradient);
  check_vector(actual, std::vector<float>(values.size(), 1.0f),
               "reshape/transpose connectivity");

  release(&gradient);
  release(&loss);
  release(&transposed);
  release(&reshaped);
  release(&x);
}

}  // namespace

int main() {
  test_multiply_sum_finite_difference();
  test_matmul_finite_difference();
  test_activation_finite_difference(Activation::kRelu, "relu");
  test_activation_finite_difference(Activation::kSigmoid, "sigmoid");
  test_activation_finite_difference(Activation::kTanh, "tanh");
  test_mse_finite_difference();
  test_cross_entropy_finite_difference();
  test_view_like_transform_gradient_connectivity();

  if (failures != 0) {
    std::cerr << failures << " autograd finite-difference assertion(s) failed\n";
    return 1;
  }
  std::cout << "Tensora autograd finite-difference validation passed\n";
  return 0;
}
