#include "tensora.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace {

bool Check(ts_status_t status, const char* operation) {
  if (status == TS_OK) return true;
  std::fprintf(stderr, "%s failed: %s\n", operation, ts_last_error_message());
  return false;
}

bool ExpectStatus(ts_status_t actual,
                  ts_status_t expected,
                  const char* operation) {
  if (actual == expected) return true;
  std::fprintf(stderr, "%s expected status %d, got %d: %s\n", operation,
               expected, actual, ts_last_error_message());
  return false;
}

bool ReadValues(ts_tensor_t tensor, std::vector<float>* out) {
  uint64_t numel = 0;
  if (!Check(ts_tensor_numel(tensor, &numel), "tensor_numel")) return false;
  out->assign(static_cast<size_t>(numel), 0.0f);
  size_t written = 0;
  if (!Check(ts_tensor_copy_to_host_f32(
                 tensor, out->data(), out->size(), &written),
             "tensor_copy_to_host_f32")) {
    return false;
  }
  return written == out->size();
}

bool Close(float actual, float expected, float tolerance = 1e-4f) {
  return std::fabs(actual - expected) <= tolerance;
}

int TestAutograd() {
  const int64_t dims[1] = {2};
  const float values[2] = {-1.0f, 2.0f};
  ts_tensor_t input = 0;
  ts_tensor_t leaf = 0;
  ts_tensor_t activated = 0;
  ts_tensor_t loss = 0;
  ts_tensor_t gradient = 0;

  if (!Check(ts_tensor_from_f32(values, 2, dims, 1, &input),
             "tensor_from_f32"))
    return 10;
  if (!Check(ts_tensor_with_requires_grad(input, 1, &leaf),
             "tensor_with_requires_grad"))
    return 11;
  uint8_t requires_grad = 0;
  if (!Check(ts_tensor_requires_grad(leaf, &requires_grad),
             "tensor_requires_grad") ||
      requires_grad != 1)
    return 12;
  if (!Check(ts_tensor_relu(leaf, &activated), "tensor_relu")) return 13;
  if (!Check(ts_tensor_sum(activated, &loss), "tensor_sum")) return 14;
  if (!Check(ts_tensor_backward(loss), "tensor_backward")) return 15;
  if (!Check(ts_tensor_grad(leaf, &gradient), "tensor_grad")) return 16;

  std::vector<float> gradient_values;
  if (!ReadValues(gradient, &gradient_values)) return 17;
  if (gradient_values.size() != 2 || !Close(gradient_values[0], 0.0f) ||
      !Close(gradient_values[1], 1.0f))
    return 18;

  if (!Check(ts_tensor_release(gradient), "release gradient")) return 19;
  if (!Check(ts_tensor_release(loss), "release loss")) return 20;
  if (!Check(ts_tensor_release(activated), "release activated")) return 21;
  if (!Check(ts_tensor_release(leaf), "release leaf")) return 22;
  if (!Check(ts_tensor_release(input), "release input")) return 23;
  return 0;
}

int TestCrossEntropy() {
  const int64_t dims[2] = {1, 2};
  const float logits_values[2] = {2.0f, 1.0f};
  const float target_values[2] = {1.0f, 0.0f};
  ts_tensor_t logits = 0;
  ts_tensor_t target = 0;
  ts_tensor_t loss = 0;

  if (!Check(ts_tensor_from_f32(logits_values, 2, dims, 2, &logits),
             "logits"))
    return 30;
  if (!Check(ts_tensor_from_f32(target_values, 2, dims, 2, &target),
             "target"))
    return 31;
  if (!Check(ts_cross_entropy_loss(logits, target, &loss),
             "cross_entropy_loss"))
    return 32;

  std::vector<float> values;
  if (!ReadValues(loss, &values)) return 33;
  if (values.size() != 1 || !Close(values[0], 0.31326166f, 1e-4f)) return 34;

  if (!Check(ts_tensor_release(loss), "release ce loss")) return 35;
  if (!Check(ts_tensor_release(target), "release ce target")) return 36;
  if (!Check(ts_tensor_release(logits), "release ce logits")) return 37;
  return 0;
}

int TestTrainingAndCheckpoint() {
  const int64_t dims[2] = {4, 1};
  const float x_values[4] = {-1.0f, 0.0f, 1.0f, 2.0f};
  const float y_values[4] = {-1.0f, 1.0f, 3.0f, 5.0f};
  ts_tensor_t x = 0;
  ts_tensor_t y = 0;
  ts_module_t module = 0;
  ts_optimizer_t optimizer = 0;

  if (!Check(ts_manual_seed(123456), "manual_seed")) return 40;
  if (!Check(ts_tensor_from_f32(x_values, 4, dims, 2, &x), "x")) return 41;
  if (!Check(ts_tensor_from_f32(y_values, 4, dims, 2, &y), "y")) return 42;
  if (!Check(ts_linear_create(1, 1, 1, &module), "linear_create")) return 43;

  size_t parameter_count = 0;
  if (!Check(ts_module_parameter_count(module, &parameter_count),
             "module_parameter_count") ||
      parameter_count != 2)
    return 44;
  size_t buffer_count = 1;
  if (!Check(ts_module_buffer_count(module, &buffer_count),
             "module_buffer_count") ||
      buffer_count != 0)
    return 45;

  ts_tensor_t parameter = 0;
  if (!Check(ts_module_parameter_at(module, 0, &parameter),
             "module_parameter_at"))
    return 46;
  std::vector<float> initial_parameter;
  if (!ReadValues(parameter, &initial_parameter) || initial_parameter.empty())
    return 47;
  if (!Check(ts_tensor_release(parameter), "release parameter")) return 48;

  if (!Check(ts_sgd_create(module, 0.1, 0.0, 0.0, &optimizer),
             "sgd_create"))
    return 49;
  if (ts_optimizer_step(module) != TS_INVALID_HANDLE) return 50;

  float initial_loss = 0.0f;
  float final_loss = 0.0f;
  for (int step = 0; step < 200; ++step) {
    ts_tensor_t prediction = 0;
    ts_tensor_t loss = 0;
    if (!Check(ts_optimizer_zero_grad(optimizer), "optimizer_zero_grad"))
      return 51;
    if (!Check(ts_module_forward(module, x, &prediction), "module_forward"))
      return 52;
    if (!Check(ts_mse_loss(prediction, y, &loss), "mse_loss")) return 53;

    std::vector<float> loss_values;
    if (!ReadValues(loss, &loss_values) || loss_values.size() != 1 ||
        !std::isfinite(loss_values[0]))
      return 54;
    if (step == 0) initial_loss = loss_values[0];
    final_loss = loss_values[0];

    if (!Check(ts_tensor_backward(loss), "loss backward")) return 55;
    if (!Check(ts_optimizer_step(optimizer), "optimizer_step")) return 56;
    if (!Check(ts_tensor_release(loss), "release train loss")) return 57;
    if (!Check(ts_tensor_release(prediction), "release prediction")) return 58;
  }

  if (!(final_loss < initial_loss && final_loss < 1e-3f)) return 59;

  if (!Check(ts_module_parameter_at(module, 0, &parameter),
             "module_parameter_at after train"))
    return 60;
  std::vector<float> trained_parameter;
  if (!ReadValues(parameter, &trained_parameter) || trained_parameter.empty())
    return 61;
  if (Close(trained_parameter[0], initial_parameter[0], 1e-6f)) return 62;
  if (!Check(ts_tensor_release(parameter), "release trained parameter"))
    return 63;

  const std::string path =
      (std::filesystem::temp_directory_path() /
       "tensora-training-checkpoint.pt")
          .string();
  if (!Check(ts_module_save(module, path.c_str()), "module_save")) return 64;

  ts_tensor_t saved_output = 0;
  if (!Check(ts_module_forward(module, x, &saved_output),
             "saved module forward"))
    return 65;
  std::vector<float> saved_values;
  if (!ReadValues(saved_output, &saved_values)) return 66;
  if (!Check(ts_tensor_release(saved_output), "release saved output")) return 67;

  for (int step = 0; step < 10; ++step) {
    ts_tensor_t prediction = 0;
    ts_tensor_t loss = 0;
    if (!Check(ts_optimizer_zero_grad(optimizer), "zero grad mutate"))
      return 68;
    if (!Check(ts_module_forward(module, x, &prediction), "forward mutate"))
      return 69;
    if (!Check(ts_mse_loss(prediction, y, &loss), "loss mutate")) return 70;
    if (!Check(ts_tensor_backward(loss), "backward mutate")) return 71;
    if (!Check(ts_optimizer_step(optimizer), "step mutate")) return 72;
    if (!Check(ts_tensor_release(loss), "release mutate loss")) return 73;
    if (!Check(ts_tensor_release(prediction), "release mutate output"))
      return 74;
  }

  if (!Check(ts_module_load(module, path.c_str()), "module_load")) return 75;
  ts_tensor_t restored_output = 0;
  if (!Check(ts_module_forward(module, x, &restored_output),
             "restored module forward"))
    return 76;
  std::vector<float> restored_values;
  if (!ReadValues(restored_output, &restored_values)) return 77;
  if (saved_values.size() != restored_values.size()) return 78;
  for (size_t i = 0; i < saved_values.size(); ++i) {
    if (!Close(saved_values[i], restored_values[i], 1e-5f)) return 79;
  }
  if (!Check(ts_tensor_release(restored_output), "release restored output"))
    return 80;

  ts_optimizer_t adam = 0;
  ts_optimizer_t adamw = 0;
  if (!Check(ts_adam_create(module, 0.001, 0.9, 0.999, 1e-8, 0.0, &adam),
             "adam_create"))
    return 81;
  if (!Check(ts_adamw_create(module, 0.001, 0.9, 0.999, 1e-8, 0.01,
                             &adamw),
             "adamw_create"))
    return 82;
  if (!Check(ts_optimizer_release(adamw), "release adamw")) return 83;
  if (!Check(ts_optimizer_release(adam), "release adam")) return 84;

  if (!Check(ts_optimizer_release(optimizer), "release optimizer")) return 85;
  if (!Check(ts_module_release(module), "release module")) return 86;
  if (!Check(ts_tensor_release(y), "release y")) return 87;
  if (!Check(ts_tensor_release(x), "release x")) return 88;
  std::remove(path.c_str());
  return 0;
}

int TestFailureContracts() {
  const int64_t scalar_dims[1] = {1};
  const int64_t vector_dims[1] = {2};
  const int64_t other_dims[1] = {3};
  const float scalar_values[1] = {1.0f};
  const float vector_values[2] = {1.0f, 2.0f};
  const float other_values[3] = {1.0f, 2.0f, 3.0f};
  ts_tensor_t scalar = 0;
  ts_tensor_t vector = 0;
  ts_tensor_t other = 0;
  ts_tensor_t output = 0;
  ts_module_t module = 0;
  ts_optimizer_t optimizer = 0;
  size_t count = 0;
  uint64_t live = 0;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();

  if (!Check(ts_tensor_from_f32(scalar_values, 1, scalar_dims, 1, &scalar),
             "failure scalar"))
    return 100;
  if (!Check(ts_tensor_from_f32(vector_values, 2, vector_dims, 1, &vector),
             "failure vector"))
    return 101;
  if (!Check(ts_tensor_from_f32(other_values, 3, other_dims, 1, &other),
             "failure other"))
    return 102;

  if (!ExpectStatus(ts_tensor_with_requires_grad(vector, 1, nullptr),
                    TS_INVALID_ARGUMENT, "with_requires_grad null output"))
    return 103;
  if (!ExpectStatus(ts_tensor_with_requires_grad(UINT64_C(999999999), 1,
                                                 &output),
                    TS_INVALID_HANDLE, "with_requires_grad invalid handle"))
    return 104;
  if (!ExpectStatus(ts_tensor_relu(vector, nullptr), TS_INVALID_ARGUMENT,
                    "relu null output"))
    return 105;
  if (!ExpectStatus(ts_tensor_sigmoid(UINT64_C(999999999), &output),
                    TS_INVALID_HANDLE, "sigmoid invalid handle"))
    return 106;
  if (!ExpectStatus(ts_tensor_tanh(vector, nullptr), TS_INVALID_ARGUMENT,
                    "tanh null output"))
    return 107;
  if (!ExpectStatus(ts_tensor_backward(vector), TS_INVALID_SHAPE,
                    "backward non-scalar"))
    return 108;
  if (!ExpectStatus(ts_tensor_backward(scalar), TS_INVALID_ARGUMENT,
                    "backward no grad"))
    return 109;
  if (!ExpectStatus(ts_tensor_grad(scalar, &output), TS_INVALID_ARGUMENT,
                    "gradient unavailable"))
    return 110;
  if (!ExpectStatus(ts_tensor_grad(scalar, nullptr), TS_INVALID_ARGUMENT,
                    "gradient null output"))
    return 111;

  if (!ExpectStatus(ts_mse_loss(vector, other, &output), TS_INVALID_SHAPE,
                    "mse mismatched shape"))
    return 112;
  if (!ExpectStatus(ts_mse_loss(vector, vector, nullptr), TS_INVALID_ARGUMENT,
                    "mse null output"))
    return 113;
  if (!ExpectStatus(ts_mse_loss(UINT64_C(999999999), vector, &output),
                    TS_INVALID_HANDLE, "mse invalid prediction"))
    return 114;
  if (!ExpectStatus(ts_cross_entropy_loss(vector, vector, &output),
                    TS_INVALID_SHAPE, "cross entropy rank one"))
    return 115;
  if (!ExpectStatus(ts_cross_entropy_loss(vector, other, &output),
                    TS_INVALID_SHAPE, "cross entropy mismatched shape"))
    return 116;
  if (!ExpectStatus(ts_cross_entropy_loss(vector, vector, nullptr),
                    TS_INVALID_ARGUMENT, "cross entropy null output"))
    return 117;

  if (!ExpectStatus(ts_linear_create(0, 1, 1, &module), TS_INVALID_ARGUMENT,
                    "linear zero input"))
    return 118;
  if (!ExpectStatus(ts_linear_create(1, 0, 1, &module), TS_INVALID_ARGUMENT,
                    "linear zero output"))
    return 119;
  if (!ExpectStatus(ts_linear_create(-1, 1, 1, &module), TS_INVALID_ARGUMENT,
                    "linear negative input"))
    return 120;
  if (!ExpectStatus(ts_linear_create(1, 1, 1, nullptr), TS_INVALID_ARGUMENT,
                    "linear null output"))
    return 121;
  if (!Check(ts_linear_create(1, 1, 1, &module), "failure linear")) return 122;

  if (!ExpectStatus(ts_module_forward(module, scalar, nullptr),
                    TS_INVALID_ARGUMENT, "module forward null output"))
    return 123;
  if (!ExpectStatus(ts_module_forward(UINT64_C(999999999), scalar, &output),
                    TS_INVALID_HANDLE, "module forward invalid module"))
    return 124;
  if (!ExpectStatus(ts_module_forward(module, UINT64_C(999999999), &output),
                    TS_INVALID_HANDLE, "module forward invalid tensor"))
    return 125;
  if (!ExpectStatus(ts_module_set_training(UINT64_C(999999999), 1),
                    TS_INVALID_HANDLE, "module train invalid handle"))
    return 126;
  if (!Check(ts_module_set_training(module, 1), "module train true")) return 127;
  if (!Check(ts_module_set_training(module, 0), "module train false")) return 128;

  if (!ExpectStatus(ts_module_to_device(module, 999u, 0), TS_UNSUPPORTED,
                    "module unknown device"))
    return 129;
  if (!ExpectStatus(ts_module_to_device(module, TS_DEVICE_CPU, 1),
                    TS_INVALID_ARGUMENT, "module bad cpu index"))
    return 130;
  if (!ExpectStatus(ts_module_to_device(UINT64_C(999999999), TS_DEVICE_CPU, 0),
                    TS_INVALID_HANDLE, "module to invalid handle"))
    return 131;
  if (!ExpectStatus(ts_module_to_device(module, TS_DEVICE_CUDA, -1),
                    TS_INVALID_ARGUMENT, "module negative cuda index"))
    return 132;

  if (!ExpectStatus(ts_module_parameter_count(module, nullptr),
                    TS_INVALID_ARGUMENT, "parameter count null"))
    return 133;
  if (!ExpectStatus(ts_module_parameter_count(UINT64_C(999999999), &count),
                    TS_INVALID_HANDLE, "parameter count invalid module"))
    return 134;
  if (!ExpectStatus(ts_module_parameter_at(module, 0, nullptr),
                    TS_INVALID_ARGUMENT, "parameter at null"))
    return 135;
  if (!ExpectStatus(ts_module_parameter_at(module, 999, &output),
                    TS_INVALID_ARGUMENT, "parameter at out of range"))
    return 136;
  if (!ExpectStatus(ts_module_parameter_at(UINT64_C(999999999), 0, &output),
                    TS_INVALID_HANDLE, "parameter at invalid module"))
    return 137;

  if (!ExpectStatus(ts_module_buffer_count(module, nullptr), TS_INVALID_ARGUMENT,
                    "buffer count null"))
    return 138;
  count = 999;
  if (!Check(ts_module_buffer_count(module, &count), "buffer count") ||
      count != 0)
    return 139;
  if (!ExpectStatus(ts_module_buffer_at(module, 0, nullptr),
                    TS_INVALID_ARGUMENT, "buffer at null"))
    return 140;
  if (!ExpectStatus(ts_module_buffer_at(module, 0, &output),
                    TS_INVALID_ARGUMENT, "buffer at out of range"))
    return 141;

  if (!ExpectStatus(ts_module_save(module, nullptr), TS_INVALID_ARGUMENT,
                    "module save null path"))
    return 142;
  if (!ExpectStatus(ts_module_save(module, ""), TS_INVALID_ARGUMENT,
                    "module save empty path"))
    return 143;
  if (!ExpectStatus(ts_module_load(module, nullptr), TS_INVALID_ARGUMENT,
                    "module load null path"))
    return 144;
  if (!ExpectStatus(ts_module_load(module, ""), TS_INVALID_ARGUMENT,
                    "module load empty path"))
    return 145;
  if (!ExpectStatus(ts_module_load(module, "/definitely/missing/tensora.pt"),
                    TS_INTERNAL_ERROR, "module load missing checkpoint"))
    return 146;

  if (!ExpectStatus(ts_sgd_create(module, 0.1, 0.0, 0.0, nullptr),
                    TS_INVALID_ARGUMENT, "sgd null output"))
    return 147;
  if (!ExpectStatus(ts_sgd_create(module, 0.0, 0.0, 0.0, &optimizer),
                    TS_INVALID_ARGUMENT, "sgd zero learning rate"))
    return 148;
  if (!ExpectStatus(ts_sgd_create(module, nan, 0.0, 0.0, &optimizer),
                    TS_INVALID_ARGUMENT, "sgd nan learning rate"))
    return 149;
  if (!ExpectStatus(ts_sgd_create(module, 0.1, -1.0, 0.0, &optimizer),
                    TS_INVALID_ARGUMENT, "sgd negative momentum"))
    return 150;
  if (!ExpectStatus(ts_sgd_create(module, 0.1, inf, 0.0, &optimizer),
                    TS_INVALID_ARGUMENT, "sgd infinite momentum"))
    return 151;
  if (!ExpectStatus(ts_sgd_create(module, 0.1, 0.0, -1.0, &optimizer),
                    TS_INVALID_ARGUMENT, "sgd negative weight decay"))
    return 152;
  if (!ExpectStatus(ts_sgd_create(UINT64_C(999999999), 0.1, 0.0, 0.0,
                                  &optimizer),
                    TS_INVALID_HANDLE, "sgd invalid module"))
    return 153;

  if (!ExpectStatus(ts_adam_create(module, 0.001, 0.9, 0.999, 1e-8, 0.0,
                                   nullptr),
                    TS_INVALID_ARGUMENT, "adam null output"))
    return 154;
  if (!ExpectStatus(ts_adam_create(module, 0.0, 0.9, 0.999, 1e-8, 0.0,
                                   &optimizer),
                    TS_INVALID_ARGUMENT, "adam zero learning rate"))
    return 155;
  if (!ExpectStatus(ts_adam_create(module, 0.001, -0.1, 0.999, 1e-8, 0.0,
                                   &optimizer),
                    TS_INVALID_ARGUMENT, "adam beta1 negative"))
    return 156;
  if (!ExpectStatus(ts_adam_create(module, 0.001, 0.9, 1.0, 1e-8, 0.0,
                                   &optimizer),
                    TS_INVALID_ARGUMENT, "adam beta2 one"))
    return 157;
  if (!ExpectStatus(ts_adam_create(module, 0.001, nan, 0.999, 1e-8, 0.0,
                                   &optimizer),
                    TS_INVALID_ARGUMENT, "adam beta nan"))
    return 158;
  if (!ExpectStatus(ts_adam_create(module, 0.001, 0.9, 0.999, 0.0, 0.0,
                                   &optimizer),
                    TS_INVALID_ARGUMENT, "adam zero epsilon"))
    return 159;
  if (!ExpectStatus(ts_adam_create(module, 0.001, 0.9, 0.999, inf, 0.0,
                                   &optimizer),
                    TS_INVALID_ARGUMENT, "adam infinite epsilon"))
    return 160;
  if (!ExpectStatus(ts_adam_create(module, 0.001, 0.9, 0.999, 1e-8, -1.0,
                                   &optimizer),
                    TS_INVALID_ARGUMENT, "adam negative weight decay"))
    return 161;
  if (!ExpectStatus(ts_adam_create(UINT64_C(999999999), 0.001, 0.9, 0.999,
                                   1e-8, 0.0, &optimizer),
                    TS_INVALID_HANDLE, "adam invalid module"))
    return 162;

  if (!ExpectStatus(ts_adamw_create(module, 0.001, 0.9, 0.999, 1e-8, 0.01,
                                    nullptr),
                    TS_INVALID_ARGUMENT, "adamw null output"))
    return 163;
  if (!ExpectStatus(ts_adamw_create(module, -0.1, 0.9, 0.999, 1e-8, 0.01,
                                    &optimizer),
                    TS_INVALID_ARGUMENT, "adamw negative learning rate"))
    return 164;
  if (!ExpectStatus(ts_adamw_create(module, 0.001, 1.0, 0.999, 1e-8, 0.01,
                                    &optimizer),
                    TS_INVALID_ARGUMENT, "adamw invalid beta"))
    return 165;
  if (!ExpectStatus(ts_adamw_create(module, 0.001, 0.9, 0.999, -1e-8, 0.01,
                                    &optimizer),
                    TS_INVALID_ARGUMENT, "adamw invalid epsilon"))
    return 166;
  if (!ExpectStatus(ts_adamw_create(module, 0.001, 0.9, 0.999, 1e-8, -0.01,
                                    &optimizer),
                    TS_INVALID_ARGUMENT, "adamw invalid weight decay"))
    return 167;

  if (!Check(ts_sgd_create(module, 0.1, 0.0, 0.0, &optimizer),
             "failure valid sgd"))
    return 168;
  if (!ExpectStatus(ts_optimizer_zero_grad(UINT64_C(999999999)),
                    TS_INVALID_HANDLE, "zero grad invalid optimizer"))
    return 169;
  if (!ExpectStatus(ts_optimizer_zero_grad(module), TS_INVALID_HANDLE,
                    "zero grad wrong type"))
    return 170;
  if (!ExpectStatus(ts_optimizer_step(UINT64_C(999999999)), TS_INVALID_HANDLE,
                    "step invalid optimizer"))
    return 171;
  if (!ExpectStatus(ts_optimizer_step(module), TS_INVALID_HANDLE,
                    "step wrong type"))
    return 172;
  if (!ExpectStatus(ts_optimizer_release(module), TS_INVALID_HANDLE,
                    "optimizer release wrong type"))
    return 173;
  if (!ExpectStatus(ts_module_release(optimizer), TS_INVALID_HANDLE,
                    "module release wrong type"))
    return 174;

  if (!ExpectStatus(ts_runtime_live_module_count(nullptr), TS_INVALID_ARGUMENT,
                    "live module null"))
    return 175;
  if (!ExpectStatus(ts_runtime_live_optimizer_count(nullptr),
                    TS_INVALID_ARGUMENT, "live optimizer null"))
    return 176;
  if (!Check(ts_runtime_live_module_count(&live), "live module count") ||
      live != 1)
    return 177;
  if (!Check(ts_runtime_live_optimizer_count(&live), "live optimizer count") ||
      live != 1)
    return 178;

  if (!Check(ts_optimizer_release(optimizer), "failure release optimizer"))
    return 179;
  if (!ExpectStatus(ts_optimizer_release(optimizer), TS_INVALID_HANDLE,
                    "optimizer double release"))
    return 180;
  if (!Check(ts_module_release(module), "failure release module")) return 181;
  if (!ExpectStatus(ts_module_release(module), TS_INVALID_HANDLE,
                    "module double release"))
    return 182;

  if (!Check(ts_tensor_release(other), "failure release other")) return 183;
  if (!Check(ts_tensor_release(vector), "failure release vector")) return 184;
  if (!Check(ts_tensor_release(scalar), "failure release scalar")) return 185;
  return 0;
}

}  // namespace

int main() {
  uint8_t available = 0;
  if (!Check(ts_training_available(&available), "training_available") ||
      available != 1)
    return 1;
  if (ts_training_available(nullptr) != TS_INVALID_ARGUMENT) return 2;

  uint32_t cuda_count = 0;
  if (!Check(ts_runtime_cuda_device_count(&cuda_count), "cuda_device_count"))
    return 3;

  const int autograd = TestAutograd();
  if (autograd != 0) return autograd;
  const int cross_entropy = TestCrossEntropy();
  if (cross_entropy != 0) return cross_entropy;
  const int training = TestTrainingAndCheckpoint();
  if (training != 0) return training;
  const int failures = TestFailureContracts();
  if (failures != 0) return failures;

  uint64_t live_tensors = 1;
  uint64_t live_modules = 1;
  uint64_t live_optimizers = 1;
  if (!Check(ts_runtime_live_tensor_count(&live_tensors), "live tensors") ||
      live_tensors != 0)
    return 190;
  if (!Check(ts_runtime_live_module_count(&live_modules), "live modules") ||
      live_modules != 0)
    return 191;
  if (!Check(ts_runtime_live_optimizer_count(&live_optimizers),
             "live optimizers") ||
      live_optimizers != 0)
    return 192;

  return 0;
}
