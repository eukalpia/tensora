#include "tensora.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define TS_STATIC_ASSERT(name, expression) \
  typedef char name[(expression) ? 1 : -1]

#define CHECK_STATUS(expr, expected, code) \
  do { \
    if ((expr) != (expected)) return (code); \
  } while (0)

#define CHECK_TRUE(expr, code) \
  do { \
    if (!(expr)) return (code); \
  } while (0)

TS_STATIC_ASSERT(ts_status_is_exactly_32_bits, sizeof(ts_status_t) == 4);
TS_STATIC_ASSERT(ts_tensor_handle_is_exactly_64_bits, sizeof(ts_tensor_t) == 8);
TS_STATIC_ASSERT(ts_module_handle_is_exactly_64_bits,
                 sizeof(ts_module_t) == 8);
TS_STATIC_ASSERT(ts_optimizer_handle_is_exactly_64_bits,
                 sizeof(ts_optimizer_t) == 8);
TS_STATIC_ASSERT(ts_onnx_session_handle_is_exactly_64_bits,
                 sizeof(ts_onnx_session_t) == 8);
TS_STATIC_ASSERT(ts_dtype_is_exactly_32_bits, sizeof(ts_dtype_t) == 4);
TS_STATIC_ASSERT(ts_device_is_exactly_32_bits, sizeof(ts_device_t) == 4);

static int test_disabled_training_contract(ts_tensor_t tensor) {
#if defined(TENSORA_WITH_TORCH)
  (void)tensor;
  return 0;
#else
  ts_tensor_t out_tensor = 777;
  ts_module_t module = 777;
  ts_optimizer_t optimizer = 777;
  size_t count = 777;
  uint8_t requires_grad = 1;
  uint64_t live = 777;

  CHECK_STATUS(ts_tensor_with_requires_grad(tensor, 1, NULL),
               TS_INVALID_ARGUMENT, 101);
  CHECK_STATUS(ts_tensor_with_requires_grad(tensor, 1, &out_tensor),
               TS_UNSUPPORTED, 102);
  CHECK_TRUE(out_tensor == 0, 103);

  CHECK_STATUS(ts_tensor_requires_grad(tensor, NULL), TS_INVALID_ARGUMENT,
               104);
  CHECK_STATUS(ts_tensor_requires_grad(tensor, &requires_grad), TS_OK, 105);
  CHECK_TRUE(requires_grad == 0, 106);
  CHECK_STATUS(ts_tensor_requires_grad(999999999ULL, &requires_grad),
               TS_INVALID_HANDLE, 107);

  CHECK_STATUS(ts_tensor_backward(tensor), TS_UNSUPPORTED, 108);
  CHECK_STATUS(ts_tensor_backward(999999999ULL), TS_INVALID_HANDLE, 109);

  out_tensor = 777;
  CHECK_STATUS(ts_tensor_grad(tensor, NULL), TS_INVALID_ARGUMENT, 110);
  CHECK_STATUS(ts_tensor_grad(tensor, &out_tensor), TS_UNSUPPORTED, 111);
  CHECK_TRUE(out_tensor == 0, 112);

  out_tensor = 777;
  CHECK_STATUS(ts_tensor_relu(tensor, NULL), TS_INVALID_ARGUMENT, 113);
  CHECK_STATUS(ts_tensor_relu(tensor, &out_tensor), TS_UNSUPPORTED, 114);
  CHECK_TRUE(out_tensor == 0, 115);

  out_tensor = 777;
  CHECK_STATUS(ts_tensor_sigmoid(tensor, NULL), TS_INVALID_ARGUMENT, 116);
  CHECK_STATUS(ts_tensor_sigmoid(tensor, &out_tensor), TS_UNSUPPORTED, 117);
  CHECK_TRUE(out_tensor == 0, 118);

  out_tensor = 777;
  CHECK_STATUS(ts_tensor_tanh(tensor, NULL), TS_INVALID_ARGUMENT, 119);
  CHECK_STATUS(ts_tensor_tanh(tensor, &out_tensor), TS_UNSUPPORTED, 120);
  CHECK_TRUE(out_tensor == 0, 121);

  out_tensor = 777;
  CHECK_STATUS(ts_mse_loss(tensor, tensor, NULL), TS_INVALID_ARGUMENT, 122);
  CHECK_STATUS(ts_mse_loss(tensor, tensor, &out_tensor), TS_UNSUPPORTED, 123);
  CHECK_TRUE(out_tensor == 0, 124);
  CHECK_STATUS(ts_mse_loss(tensor, 999999999ULL, &out_tensor),
               TS_INVALID_HANDLE, 125);

  out_tensor = 777;
  CHECK_STATUS(ts_cross_entropy_loss(tensor, tensor, NULL),
               TS_INVALID_ARGUMENT, 126);
  CHECK_STATUS(ts_cross_entropy_loss(tensor, tensor, &out_tensor),
               TS_UNSUPPORTED, 127);
  CHECK_TRUE(out_tensor == 0, 128);

  CHECK_STATUS(ts_linear_create(1, 1, 1, NULL), TS_INVALID_ARGUMENT, 129);
  CHECK_STATUS(ts_linear_create(1, 1, 1, &module), TS_UNSUPPORTED, 130);
  CHECK_TRUE(module == 0, 131);

  out_tensor = 777;
  CHECK_STATUS(ts_module_forward(0, tensor, NULL), TS_INVALID_ARGUMENT, 132);
  CHECK_STATUS(ts_module_forward(0, tensor, &out_tensor), TS_UNSUPPORTED, 133);
  CHECK_TRUE(out_tensor == 0, 134);
  CHECK_STATUS(ts_module_forward(0, 999999999ULL, &out_tensor),
               TS_INVALID_HANDLE, 135);

  CHECK_STATUS(ts_module_set_training(0, 1), TS_UNSUPPORTED, 136);
  CHECK_STATUS(ts_module_to_device(0, TS_DEVICE_CPU, 0), TS_UNSUPPORTED, 137);
  CHECK_STATUS(ts_module_to_device(0, 999u, 0), TS_UNSUPPORTED, 138);

  count = 777;
  CHECK_STATUS(ts_module_parameter_count(0, NULL), TS_INVALID_ARGUMENT, 139);
  CHECK_STATUS(ts_module_parameter_count(0, &count), TS_UNSUPPORTED, 140);
  CHECK_TRUE(count == 0, 141);

  out_tensor = 777;
  CHECK_STATUS(ts_module_parameter_at(0, 0, NULL), TS_INVALID_ARGUMENT, 142);
  CHECK_STATUS(ts_module_parameter_at(0, 0, &out_tensor), TS_UNSUPPORTED, 143);
  CHECK_TRUE(out_tensor == 0, 144);

  count = 777;
  CHECK_STATUS(ts_module_buffer_count(0, NULL), TS_INVALID_ARGUMENT, 145);
  CHECK_STATUS(ts_module_buffer_count(0, &count), TS_UNSUPPORTED, 146);
  CHECK_TRUE(count == 0, 147);

  out_tensor = 777;
  CHECK_STATUS(ts_module_buffer_at(0, 0, NULL), TS_INVALID_ARGUMENT, 148);
  CHECK_STATUS(ts_module_buffer_at(0, 0, &out_tensor), TS_UNSUPPORTED, 149);
  CHECK_TRUE(out_tensor == 0, 150);

  CHECK_STATUS(ts_module_save(0, NULL), TS_INVALID_ARGUMENT, 151);
  CHECK_STATUS(ts_module_save(0, "checkpoint.bin"), TS_UNSUPPORTED, 152);
  CHECK_STATUS(ts_module_load(0, NULL), TS_INVALID_ARGUMENT, 153);
  CHECK_STATUS(ts_module_load(0, "checkpoint.bin"), TS_UNSUPPORTED, 154);
  CHECK_STATUS(ts_module_release(0), TS_UNSUPPORTED, 155);

  optimizer = 777;
  CHECK_STATUS(ts_sgd_create(0, 0.1, 0.0, 0.0, NULL),
               TS_INVALID_ARGUMENT, 156);
  CHECK_STATUS(ts_sgd_create(0, 0.1, 0.0, 0.0, &optimizer), TS_UNSUPPORTED,
               157);
  CHECK_TRUE(optimizer == 0, 158);

  optimizer = 777;
  CHECK_STATUS(ts_adam_create(0, 0.001, 0.9, 0.999, 1e-8, 0.0, NULL),
               TS_INVALID_ARGUMENT, 159);
  CHECK_STATUS(
      ts_adam_create(0, 0.001, 0.9, 0.999, 1e-8, 0.0, &optimizer),
      TS_UNSUPPORTED, 160);
  CHECK_TRUE(optimizer == 0, 161);

  optimizer = 777;
  CHECK_STATUS(ts_adamw_create(0, 0.001, 0.9, 0.999, 1e-8, 0.01, NULL),
               TS_INVALID_ARGUMENT, 162);
  CHECK_STATUS(
      ts_adamw_create(0, 0.001, 0.9, 0.999, 1e-8, 0.01, &optimizer),
      TS_UNSUPPORTED, 163);
  CHECK_TRUE(optimizer == 0, 164);

  CHECK_STATUS(ts_optimizer_zero_grad(0), TS_UNSUPPORTED, 165);
  CHECK_STATUS(ts_optimizer_step(0), TS_UNSUPPORTED, 166);
  CHECK_STATUS(ts_optimizer_release(0), TS_UNSUPPORTED, 167);

  CHECK_STATUS(ts_runtime_live_module_count(NULL), TS_INVALID_ARGUMENT, 168);
  CHECK_STATUS(ts_runtime_live_module_count(&live), TS_OK, 169);
  CHECK_TRUE(live == 0, 170);
  live = 777;
  CHECK_STATUS(ts_runtime_live_optimizer_count(NULL), TS_INVALID_ARGUMENT,
               171);
  CHECK_STATUS(ts_runtime_live_optimizer_count(&live), TS_OK, 172);
  CHECK_TRUE(live == 0, 173);
  return 0;
#endif
}

static int test_disabled_inference_contract(ts_tensor_t tensor) {
#if defined(TENSORA_WITH_ONNXRUNTIME)
  (void)tensor;
  return 0;
#else
  uint8_t available = 1;
  size_t count = 777;
  size_t required = 777;
  size_t written = 777;
  uint64_t live = 777;
  ts_onnx_session_t session = 777;
  char buffer[32] = {0};
  const char* input_names[1] = {"X"};
  const ts_tensor_t input_tensors[1] = {tensor};
  const char* output_names[1] = {"Y"};
  ts_tensor_t output_tensors[1] = {777};
  const char* null_name[1] = {NULL};

  CHECK_STATUS(ts_onnx_available(NULL), TS_INVALID_ARGUMENT, 201);
  CHECK_STATUS(ts_onnx_available(&available), TS_OK, 202);
  CHECK_TRUE(available == 0, 203);

  CHECK_STATUS(ts_onnx_provider_count(NULL), TS_INVALID_ARGUMENT, 204);
  CHECK_STATUS(ts_onnx_provider_count(&count), TS_OK, 205);
  CHECK_TRUE(count == 0, 206);

  CHECK_STATUS(ts_onnx_provider_name(0, buffer, sizeof(buffer), &required),
               TS_UNSUPPORTED, 207);

  CHECK_STATUS(ts_onnx_session_create("model.onnx", 0, NULL, NULL),
               TS_INVALID_ARGUMENT, 208);
  CHECK_STATUS(ts_onnx_session_create(NULL, 0, NULL, &session),
               TS_INVALID_ARGUMENT, 209);
  CHECK_STATUS(ts_onnx_session_create("model.onnx", 0, NULL, &session),
               TS_UNSUPPORTED, 210);
  CHECK_TRUE(session == 0, 211);

  session = 777;
  CHECK_STATUS(ts_onnx_session_create_with_provider(
                   "model.onnx", NULL, 0, NULL, &session),
               TS_INVALID_ARGUMENT, 212);
  CHECK_STATUS(ts_onnx_session_create_with_provider(
                   "model.onnx", "CPUExecutionProvider", 0, "profile", &session),
               TS_UNSUPPORTED, 213);
  CHECK_TRUE(session == 0, 214);

  CHECK_STATUS(ts_onnx_session_provider(0, buffer, sizeof(buffer), &required),
               TS_UNSUPPORTED, 215);
  CHECK_STATUS(ts_onnx_session_input_count(0, NULL), TS_INVALID_ARGUMENT, 216);
  CHECK_STATUS(ts_onnx_session_input_count(0, &count), TS_UNSUPPORTED, 217);
  CHECK_TRUE(count == 0, 218);
  count = 777;
  CHECK_STATUS(ts_onnx_session_output_count(0, NULL), TS_INVALID_ARGUMENT,
               219);
  CHECK_STATUS(ts_onnx_session_output_count(0, &count), TS_UNSUPPORTED, 220);
  CHECK_TRUE(count == 0, 221);

  CHECK_STATUS(
      ts_onnx_session_input_name(0, 0, buffer, sizeof(buffer), &required),
      TS_UNSUPPORTED, 222);
  CHECK_STATUS(
      ts_onnx_session_output_name(0, 0, buffer, sizeof(buffer), &required),
      TS_UNSUPPORTED, 223);

  CHECK_STATUS(ts_onnx_session_run(0, NULL, NULL, 0, output_names, 1,
                                   output_tensors, 1, NULL),
               TS_INVALID_ARGUMENT, 224);
  CHECK_STATUS(ts_onnx_session_run(0, NULL, NULL, 1, output_names, 1,
                                   output_tensors, 1, &written),
               TS_INVALID_ARGUMENT, 225);
  CHECK_STATUS(ts_onnx_session_run(0, input_names, input_tensors, 1, NULL, 0,
                                   output_tensors, 1, &written),
               TS_INVALID_ARGUMENT, 226);
  CHECK_STATUS(ts_onnx_session_run(0, input_names, input_tensors, 1,
                                   output_names, 1, output_tensors, 0, &written),
               TS_INVALID_ARGUMENT, 227);
  CHECK_STATUS(ts_onnx_session_run(0, input_names, input_tensors, 1,
                                   output_names, 1, NULL, 1, &written),
               TS_INVALID_ARGUMENT, 228);
  CHECK_STATUS(ts_onnx_session_run(0, null_name, input_tensors, 1,
                                   output_names, 1, output_tensors, 1, &written),
               TS_INVALID_ARGUMENT, 229);
  {
    const ts_tensor_t invalid_tensor[1] = {999999999ULL};
    CHECK_STATUS(ts_onnx_session_run(0, input_names, invalid_tensor, 1,
                                     output_names, 1, output_tensors, 1,
                                     &written),
                 TS_INVALID_HANDLE, 230);
  }
  CHECK_STATUS(ts_onnx_session_run(0, input_names, input_tensors, 1,
                                   null_name, 1, output_tensors, 1, &written),
               TS_INVALID_ARGUMENT, 231);
  output_tensors[0] = 777;
  CHECK_STATUS(ts_onnx_session_run(0, input_names, input_tensors, 1,
                                   output_names, 1, output_tensors, 1, &written),
               TS_UNSUPPORTED, 232);
  CHECK_TRUE(output_tensors[0] == 0, 233);
  CHECK_TRUE(written == 0, 234);

  CHECK_STATUS(ts_onnx_session_end_profiling(0, buffer, sizeof(buffer),
                                             &required),
               TS_UNSUPPORTED, 235);
  CHECK_STATUS(ts_onnx_session_release(0), TS_UNSUPPORTED, 236);

  CHECK_STATUS(ts_runtime_live_onnx_session_count(NULL), TS_INVALID_ARGUMENT,
               237);
  CHECK_STATUS(ts_runtime_live_onnx_session_count(&live), TS_OK, 238);
  CHECK_TRUE(live == 0, 239);
#endif
  return 0;
}

int main(void) {
  const int64_t dims[2] = {2, 2};
  const float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  ts_tensor_t tensor = 0;
  ts_tensor_t copied = 0;
  ts_module_t module = 0;
  ts_optimizer_t optimizer = 0;
  uint64_t numel = 0;
  uint32_t device = 0;
  int32_t device_index = -1;
  uint32_t cuda_count = 0;
  uint8_t training_available = 0;
  uint8_t requires_grad = 1;
  float values[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  size_t written = 0;

  if (ts_abi_version() != TS_ABI_VERSION) return 1;
  if (ts_noop() != TS_OK) return 2;
  if (TS_DEVICE_CPU == TS_DEVICE_CUDA) return 3;
  if (ts_training_available(&training_available) != TS_OK) return 4;
#if defined(TENSORA_WITH_TORCH)
  if (training_available != 1) return 5;
#else
  if (training_available != 0) return 5;
#endif
  if (ts_training_available(NULL) != TS_INVALID_ARGUMENT) return 6;
  if (ts_runtime_cuda_device_count(&cuda_count) != TS_OK) return 7;
  if (ts_runtime_cuda_device_count(NULL) != TS_INVALID_ARGUMENT) return 8;
#if defined(TENSORA_WITH_TORCH)
  if (ts_manual_seed(42) != TS_OK) return 9;
#else
  if (cuda_count != 0) return 9;
  if (ts_manual_seed(42) != TS_UNSUPPORTED) return 10;
#endif

  if (ts_tensor_from_f32(input, 3, dims, 2, &tensor) != TS_INVALID_ARGUMENT)
    return 11;
  if (tensor != 0) return 12;
  if (ts_tensor_from_f32(input, 5, dims, 2, &tensor) != TS_INVALID_ARGUMENT)
    return 13;
  if (tensor != 0) return 14;
  if (ts_tensor_from_f32(input, 4, dims, 2, &tensor) != TS_OK) return 15;
  if (tensor == 0) return 16;
  if (ts_tensor_numel(tensor, &numel) != TS_OK || numel != 4) return 17;
  if (ts_tensor_device(tensor, &device) != TS_OK || device != TS_DEVICE_CPU)
    return 18;
  if (ts_tensor_device_index(tensor, &device_index) != TS_OK ||
      device_index != 0)
    return 19;
  if (ts_tensor_copy_to_host_f32(tensor, values, 4, &written) != TS_OK)
    return 20;
  if (written != 4) return 21;
  if (values[0] != 1.0f || values[3] != 4.0f) return 22;

  if (ts_tensor_requires_grad(tensor, &requires_grad) != TS_OK ||
      requires_grad != 0)
    return 23;
  if (ts_tensor_requires_grad(tensor, NULL) != TS_INVALID_ARGUMENT) return 24;
#if defined(TENSORA_WITH_TORCH)
  if (ts_tensor_with_requires_grad(tensor, 1, &copied) != TS_OK) return 25;
  if (copied == 0) return 26;
  if (ts_tensor_release(copied) != TS_OK) return 27;
  copied = 0;
  if (ts_tensor_relu(tensor, &copied) != TS_OK) return 28;
  if (copied == 0) return 29;
  if (ts_tensor_release(copied) != TS_OK) return 30;
  copied = 0;
#else
  if (ts_tensor_with_requires_grad(tensor, 1, &copied) != TS_UNSUPPORTED)
    return 25;
  if (copied != 0) return 26;
  if (ts_tensor_relu(tensor, &copied) != TS_UNSUPPORTED) return 27;
  if (copied != 0) return 28;
#endif

  if (test_disabled_training_contract(tensor) != 0)
    return test_disabled_training_contract(tensor);
  if (test_disabled_inference_contract(tensor) != 0)
    return test_disabled_inference_contract(tensor);

  if (ts_tensor_to_device(tensor, TS_DEVICE_CPU, 0, &copied) != TS_OK)
    return 31;
  if (copied == 0) return 32;
  if (ts_tensor_device(copied, &device) != TS_OK || device != TS_DEVICE_CPU)
    return 33;
  if (ts_tensor_device_index(copied, &device_index) != TS_OK ||
      device_index != 0)
    return 34;
  if (ts_tensor_release(copied) != TS_OK) return 35;
  copied = 0;

  if (ts_tensor_to_device(tensor, 999u, 0, &copied) != TS_UNSUPPORTED)
    return 36;
  if (copied != 0) return 37;
  if (ts_tensor_to_device(tensor, TS_DEVICE_CPU, 1, &copied) !=
      TS_INVALID_ARGUMENT)
    return 38;
  if (copied != 0) return 39;
  if (cuda_count == 0) {
    if (ts_tensor_to_device(tensor, TS_DEVICE_CUDA, 0, &copied) !=
        TS_UNSUPPORTED)
      return 40;
    if (copied != 0) return 41;
  }

#if defined(TENSORA_WITH_TORCH)
  if (ts_linear_create(1, 1, 1, &module) != TS_OK) return 42;
  if (module == 0) return 43;
  if (ts_sgd_create(module, 0.1, 0.0, 0.0, &optimizer) != TS_OK) return 44;
  if (optimizer == 0) return 45;
  if (ts_optimizer_release(optimizer) != TS_OK) return 46;
  if (ts_module_release(module) != TS_OK) return 47;
#else
  if (ts_linear_create(1, 1, 1, &module) != TS_UNSUPPORTED) return 42;
  if (module != 0) return 43;
  if (ts_sgd_create(module, 0.1, 0.0, 0.0, &optimizer) != TS_UNSUPPORTED)
    return 44;
  if (optimizer != 0) return 45;
#endif

  if (ts_tensor_release(tensor) != TS_OK) return 48;

  tensor = 0;
  if (ts_tensor_full_f32(dims, 2, 3.0f, &tensor) != TS_OK) return 49;
  if (ts_tensor_release(tensor) != TS_OK) return 50;

  if (strcmp(ts_status_name(TS_MODEL_ERROR), "MODEL_ERROR") != 0) return 51;
  if (strcmp(ts_status_name(999), "UNKNOWN_STATUS") != 0) return 52;

  if (TS_DEVICE_CPU == TS_DEVICE_MPS || TS_DEVICE_CPU == TS_DEVICE_XPU ||
      TS_DEVICE_CPU == TS_DEVICE_HIP || TS_DEVICE_CUDA == TS_DEVICE_MPS ||
      TS_DEVICE_CUDA == TS_DEVICE_XPU || TS_DEVICE_CUDA == TS_DEVICE_HIP ||
      TS_DEVICE_MPS == TS_DEVICE_XPU || TS_DEVICE_MPS == TS_DEVICE_HIP ||
      TS_DEVICE_XPU == TS_DEVICE_HIP)
    return 53;

  {
    uint32_t count = 99;
    if (ts_runtime_device_count(TS_DEVICE_CPU, &count) != TS_OK || count != 1)
      return 54;
    if (ts_runtime_device_count(999u, &count) != TS_UNSUPPORTED) return 55;
    if (ts_runtime_device_count(TS_DEVICE_CPU, NULL) != TS_INVALID_ARGUMENT)
      return 56;
#if !defined(TENSORA_WITH_TORCH)
    if (ts_runtime_device_count(TS_DEVICE_CUDA, &count) != TS_OK || count != 0)
      return 57;
    if (ts_runtime_device_count(TS_DEVICE_MPS, &count) != TS_OK || count != 0)
      return 58;
    if (ts_runtime_device_count(TS_DEVICE_XPU, &count) != TS_OK || count != 0)
      return 59;
    if (ts_runtime_device_count(TS_DEVICE_HIP, &count) != TS_OK || count != 0)
      return 60;
#endif
    if (ts_runtime_device_count(TS_DEVICE_CUDA, &count) != TS_OK) return 61;
    if (count != cuda_count) return 62;
  }

  return 0;
}
