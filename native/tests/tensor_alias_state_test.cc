#include "tensora.h"

#include <cstdint>
#include <iostream>
#include <memory>

#include "runtime/handle_registry.h"
#include "tensor/tensor.h"

namespace {

using tensora::HandleRegistry;
using tensora::HandleType;
using tensora::Tensor;

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

std::shared_ptr<Tensor> lookup(ts_tensor_t handle) {
  std::shared_ptr<Tensor> tensor;
  const auto status =
      HandleRegistry::Instance().Lookup<Tensor>(handle, HandleType::kTensor,
                                                &tensor);
  CHECK_TRUE(status.ok());
  CHECK_TRUE(tensor != nullptr);
  return tensor;
}

void release_tensor(ts_tensor_t* tensor) {
  if (tensor == nullptr || *tensor == 0) return;
  CHECK_STATUS(ts_tensor_release(*tensor), TS_OK);
  *tensor = 0;
}

}  // namespace

int main() {
  CHECK_STATUS(ts_manual_seed(424242), TS_OK);

  ts_module_t module = 0;
  CHECK_STATUS(ts_linear_create(2, 2, 0, &module), TS_OK);
  ts_tensor_t weight = 0;
  CHECK_STATUS(ts_module_parameter_at(module, 0, &weight), TS_OK);

  auto weight_object = lookup(weight);
  const uint64_t initial_version = weight_object->version();

  ts_tensor_t view = 0;
  CHECK_STATUS(ts_tensor_transpose2d(weight, &view), TS_OK);
  auto view_object = lookup(view);
  CHECK_TRUE(view_object->storage().get() == weight_object->storage().get());
  CHECK_TRUE(view_object->version_counter().get() ==
             weight_object->version_counter().get());
  CHECK_TRUE(view_object->version() == initial_version);

  ts_tensor_t square = 0;
  ts_tensor_t pending_loss = 0;
  CHECK_STATUS(ts_tensor_multiply(view, view, &square), TS_OK);
  CHECK_STATUS(ts_tensor_sum(square, &pending_loss), TS_OK);

  ts_optimizer_t optimizer = 0;
  CHECK_STATUS(ts_sgd_create(module, 0.01, 0.0, 0.0, &optimizer), TS_OK);

  const float input_values[2] = {1.0f, 0.0f};
  const int64_t input_shape[2] = {1, 2};
  ts_tensor_t input = 0;
  CHECK_STATUS(ts_tensor_from_f32(input_values, 2, input_shape, 2, &input),
               TS_OK);
  const float target_values[2] = {0.0f, 0.0f};
  ts_tensor_t target = 0;
  CHECK_STATUS(ts_tensor_from_f32(target_values, 2, input_shape, 2, &target),
               TS_OK);

  CHECK_STATUS(ts_optimizer_zero_grad(optimizer), TS_OK);
  ts_tensor_t prediction = 0;
  ts_tensor_t training_loss = 0;
  CHECK_STATUS(ts_module_forward(module, input, &prediction), TS_OK);
  CHECK_STATUS(ts_mse_loss(prediction, target, &training_loss), TS_OK);
  CHECK_STATUS(ts_tensor_backward(training_loss), TS_OK);
  CHECK_STATUS(ts_optimizer_step(optimizer), TS_OK);

  CHECK_TRUE(weight_object->version() == initial_version + 1);
  CHECK_TRUE(view_object->version() == weight_object->version());

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
  view_object.reset();
  weight_object.reset();
  CHECK_STATUS(ts_module_release(module), TS_OK);

  if (failures != 0) {
    std::cerr << failures << " tensor alias-state assertion(s) failed\n";
    return 1;
  }
  std::cout << "Tensora tensor alias-state validation passed\n";
  return 0;
}
