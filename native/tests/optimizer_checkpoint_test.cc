#include "tensora.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
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

void check_near(float actual, float expected, float tolerance = 2e-5f) {
  if (std::fabs(actual - expected) > tolerance) {
    std::cerr << "FAIL numeric: expected " << expected << ", got " << actual
              << " tolerance=" << tolerance << "\n";
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

float parameter_value(ts_module_t module, size_t index = 0) {
  ts_tensor_t parameter = 0;
  CHECK_STATUS(ts_module_parameter_at(module, index, &parameter), TS_OK);
  const auto values = read_tensor(parameter);
  CHECK_TRUE(values.size() == 1);
  const float result = values.empty() ? 0.0f : values[0];
  release_tensor(&parameter);
  return result;
}

float training_step(ts_module_t module,
                    ts_optimizer_t optimizer,
                    ts_tensor_t input,
                    ts_tensor_t target) {
  CHECK_STATUS(ts_optimizer_zero_grad(optimizer), TS_OK);
  ts_tensor_t prediction = 0;
  ts_tensor_t loss = 0;
  CHECK_STATUS(ts_module_forward(module, input, &prediction), TS_OK);
  CHECK_STATUS(ts_mse_loss(prediction, target, &loss), TS_OK);
  const auto loss_values = read_tensor(loss);
  CHECK_TRUE(loss_values.size() == 1);
  const float value = loss_values.empty() ? 0.0f : loss_values[0];
  CHECK_STATUS(ts_tensor_backward(loss), TS_OK);
  CHECK_STATUS(ts_optimizer_step(optimizer), TS_OK);
  release_tensor(&loss);
  release_tensor(&prediction);
  return value;
}

void test_adam_first_step_matches_reference() {
  CHECK_STATUS(ts_manual_seed(918273), TS_OK);
  ts_module_t module = 0;
  CHECK_STATUS(ts_linear_create(1, 1, 0, &module), TS_OK);
  const float before = parameter_value(module);
  CHECK_TRUE(std::fabs(before) > 1e-5f);

  ts_optimizer_t optimizer = 0;
  constexpr double learning_rate = 0.01;
  CHECK_STATUS(ts_adam_create(module, learning_rate, 0.9, 0.999, 1e-8, 0.0,
                              &optimizer),
               TS_OK);

  ts_tensor_t input = make_tensor({1.0f}, {1, 1});
  ts_tensor_t target = make_tensor({0.0f}, {1, 1});
  training_step(module, optimizer, input, target);

  const float after = parameter_value(module);
  const double gradient = 2.0 * static_cast<double>(before);
  const double expected = static_cast<double>(before) -
                          learning_rate * gradient /
                              (std::fabs(gradient) + 1e-8);
  check_near(after, static_cast<float>(expected));

  release_tensor(&target);
  release_tensor(&input);
  CHECK_STATUS(ts_optimizer_release(optimizer), TS_OK);
  CHECK_STATUS(ts_module_release(module), TS_OK);
}

void test_adamw_first_step_uses_decoupled_weight_decay() {
  CHECK_STATUS(ts_manual_seed(182736), TS_OK);
  ts_module_t module = 0;
  CHECK_STATUS(ts_linear_create(1, 1, 0, &module), TS_OK);
  const float before = parameter_value(module);
  CHECK_TRUE(std::fabs(before) > 1e-5f);

  ts_optimizer_t optimizer = 0;
  constexpr double learning_rate = 0.01;
  constexpr double weight_decay = 0.1;
  CHECK_STATUS(ts_adamw_create(module, learning_rate, 0.9, 0.999, 1e-8,
                               weight_decay, &optimizer),
               TS_OK);

  ts_tensor_t input = make_tensor({1.0f}, {1, 1});
  ts_tensor_t target = make_tensor({0.0f}, {1, 1});
  training_step(module, optimizer, input, target);

  const float after = parameter_value(module);
  const double gradient = 2.0 * static_cast<double>(before);
  const double decayed = static_cast<double>(before) *
                         (1.0 - learning_rate * weight_decay);
  const double expected = decayed -
                          learning_rate * gradient /
                              (std::fabs(gradient) + 1e-8);
  check_near(after, static_cast<float>(expected));

  release_tensor(&target);
  release_tensor(&input);
  CHECK_STATUS(ts_optimizer_release(optimizer), TS_OK);
  CHECK_STATUS(ts_module_release(module), TS_OK);
}

void test_adam_and_adamw_converge() {
  for (int variant = 0; variant < 2; ++variant) {
    CHECK_STATUS(ts_manual_seed(static_cast<uint64_t>(400 + variant)), TS_OK);
    ts_module_t module = 0;
    CHECK_STATUS(ts_linear_create(1, 1, 0, &module), TS_OK);
    ts_optimizer_t optimizer = 0;
    if (variant == 0) {
      CHECK_STATUS(ts_adam_create(module, 0.05, 0.9, 0.999, 1e-8, 0.0,
                                  &optimizer),
                   TS_OK);
    } else {
      CHECK_STATUS(ts_adamw_create(module, 0.05, 0.9, 0.999, 1e-8, 0.0,
                                   &optimizer),
                   TS_OK);
    }

    ts_tensor_t input =
        make_tensor({-3.0f, -1.0f, 1.0f, 3.0f}, {4, 1});
    ts_tensor_t target =
        make_tensor({-7.5f, -2.5f, 2.5f, 7.5f}, {4, 1});

    float initial_loss = 0.0f;
    float final_loss = 0.0f;
    for (int step = 0; step < 240; ++step) {
      const float loss = training_step(module, optimizer, input, target);
      if (step == 0) initial_loss = loss;
      final_loss = loss;
    }
    CHECK_TRUE(std::isfinite(initial_loss));
    CHECK_TRUE(std::isfinite(final_loss));
    CHECK_TRUE(final_loss < initial_loss * 1e-4f);
    CHECK_TRUE(final_loss < 2e-4f);
    check_near(parameter_value(module), 2.5f, 5e-3f);

    release_tensor(&target);
    release_tensor(&input);
    CHECK_STATUS(ts_optimizer_release(optimizer), TS_OK);
    CHECK_STATUS(ts_module_release(module), TS_OK);
  }
}

std::vector<std::vector<float>> module_parameters(ts_module_t module) {
  size_t count = 0;
  CHECK_STATUS(ts_module_parameter_count(module, &count), TS_OK);
  std::vector<std::vector<float>> result;
  result.reserve(count);
  for (size_t index = 0; index < count; ++index) {
    ts_tensor_t parameter = 0;
    CHECK_STATUS(ts_module_parameter_at(module, index, &parameter), TS_OK);
    result.push_back(read_tensor(parameter));
    release_tensor(&parameter);
  }
  return result;
}

void check_parameters_equal(const std::vector<std::vector<float>>& actual,
                            const std::vector<std::vector<float>>& expected,
                            const char* label) {
  if (actual.size() != expected.size()) {
    std::cerr << "FAIL " << label << ": parameter count mismatch\n";
    ++failures;
    return;
  }
  for (size_t parameter = 0; parameter < actual.size(); ++parameter) {
    if (actual[parameter].size() != expected[parameter].size()) {
      std::cerr << "FAIL " << label << ": parameter size mismatch\n";
      ++failures;
      continue;
    }
    for (size_t index = 0; index < actual[parameter].size(); ++index) {
      check_near(actual[parameter][index], expected[parameter][index], 0.0f);
    }
  }
}

void test_checkpoint_restore_and_failure_rollback() {
  const std::filesystem::path checkpoint =
      std::filesystem::current_path() / "tensora_training_checkpoint.bin";
  const std::filesystem::path corrupt =
      std::filesystem::current_path() / "tensora_training_checkpoint_corrupt.bin";
  const std::filesystem::path trailing =
      std::filesystem::current_path() / "tensora_training_checkpoint_trailing.bin";
  std::error_code ignored;
  std::filesystem::remove(checkpoint, ignored);
  std::filesystem::remove(corrupt, ignored);
  std::filesystem::remove(trailing, ignored);

  CHECK_STATUS(ts_manual_seed(727272), TS_OK);
  ts_module_t module = 0;
  CHECK_STATUS(ts_linear_create(2, 1, 1, &module), TS_OK);
  ts_optimizer_t optimizer = 0;
  CHECK_STATUS(ts_sgd_create(module, 0.02, 0.8, 0.0, &optimizer), TS_OK);
  ts_tensor_t input = make_tensor({1.0f, -1.0f, 2.0f, 0.5f}, {2, 2});
  ts_tensor_t target = make_tensor({2.0f, 4.0f}, {2, 1});

  for (int step = 0; step < 12; ++step) {
    training_step(module, optimizer, input, target);
  }
  const auto saved_parameters = module_parameters(module);
  CHECK_STATUS(ts_module_save(module, checkpoint.string().c_str()), TS_OK);
  CHECK_TRUE(std::filesystem::exists(checkpoint));
  CHECK_TRUE(std::filesystem::file_size(checkpoint) > 32);

  for (int step = 0; step < 8; ++step) {
    training_step(module, optimizer, input, target);
  }
  const auto changed_parameters = module_parameters(module);
  bool changed = false;
  for (size_t parameter = 0; parameter < saved_parameters.size(); ++parameter) {
    for (size_t index = 0; index < saved_parameters[parameter].size(); ++index) {
      if (saved_parameters[parameter][index] != changed_parameters[parameter][index]) {
        changed = true;
      }
    }
  }
  CHECK_TRUE(changed);

  CHECK_STATUS(ts_module_load(module, checkpoint.string().c_str()), TS_OK);
  check_parameters_equal(module_parameters(module), saved_parameters,
                         "checkpoint restore");

  {
    std::ofstream stream(corrupt, std::ios::binary | std::ios::trunc);
    stream << "not-a-tensora-checkpoint";
  }
  const auto before_corrupt_load = module_parameters(module);
  CHECK_STATUS(ts_module_load(module, corrupt.string().c_str()), TS_INTERNAL_ERROR);
  check_parameters_equal(module_parameters(module), before_corrupt_load,
                         "corrupt checkpoint rollback");

  std::filesystem::copy_file(checkpoint, trailing,
                             std::filesystem::copy_options::overwrite_existing);
  {
    std::ofstream stream(trailing, std::ios::binary | std::ios::app);
    const char trailing_byte = '\x7f';
    stream.write(&trailing_byte, 1);
  }
  const auto before_trailing_load = module_parameters(module);
  CHECK_STATUS(ts_module_load(module, trailing.string().c_str()),
               TS_INTERNAL_ERROR);
  check_parameters_equal(module_parameters(module), before_trailing_load,
                         "trailing-data checkpoint rollback");

  ts_module_t incompatible = 0;
  CHECK_STATUS(ts_linear_create(3, 1, 1, &incompatible), TS_OK);
  const auto incompatible_before = module_parameters(incompatible);
  CHECK_STATUS(ts_module_load(incompatible, checkpoint.string().c_str()),
               TS_INTERNAL_ERROR);
  check_parameters_equal(module_parameters(incompatible), incompatible_before,
                         "incompatible checkpoint rollback");

  CHECK_STATUS(ts_module_release(incompatible), TS_OK);
  release_tensor(&target);
  release_tensor(&input);
  CHECK_STATUS(ts_optimizer_release(optimizer), TS_OK);
  CHECK_STATUS(ts_module_release(module), TS_OK);

  std::filesystem::remove(checkpoint, ignored);
  std::filesystem::remove(corrupt, ignored);
  std::filesystem::remove(trailing, ignored);
}

}  // namespace

int main() {
  test_adam_first_step_matches_reference();
  test_adamw_first_step_uses_decoupled_weight_decay();
  test_adam_and_adamw_converge();
  test_checkpoint_restore_and_failure_rollback();

  if (failures != 0) {
    std::cerr << failures << " optimizer/checkpoint assertion(s) failed\n";
    return 1;
  }
  std::cout << "Tensora optimizer and checkpoint validation passed\n";
  return 0;
}
