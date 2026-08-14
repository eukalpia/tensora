#include "tensora.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint32_t g_abi_version = TS_ABI_VERSION;
static int32_t g_forced_status = TS_OK;
static uint8_t g_null_diagnostic = 0;
static uint8_t g_null_handle = 0;
static size_t g_rank = 1;
static size_t g_shape_returned_rank = SIZE_MAX;
static int64_t g_shape_dimension = 1;
static uint32_t g_dtype = TS_DTYPE_FLOAT32;
static uint32_t g_device = TS_DEVICE_CPU;
static int32_t g_device_index = 0;
static uint64_t g_numel = 1;
static size_t g_copy_written = SIZE_MAX;
static int32_t g_training_mode = 0;
static int32_t g_inference_mode = 0;
static uint64_t g_next_handle = 100;
static uint64_t g_malformed_tensor_handle = 0;
static uint64_t g_tensor_release_count = 0;
static uint64_t g_module_release_count = 0;
static uint64_t g_optimizer_release_count = 0;
static uint64_t g_session_release_count = 0;
static char g_requested_provider[96] = "auto";

TS_API void ts_test_reset(void) {
  g_abi_version = TS_ABI_VERSION;
  g_forced_status = TS_OK;
  g_null_diagnostic = 0;
  g_null_handle = 0;
  g_rank = 1;
  g_shape_returned_rank = SIZE_MAX;
  g_shape_dimension = 1;
  g_dtype = TS_DTYPE_FLOAT32;
  g_device = TS_DEVICE_CPU;
  g_device_index = 0;
  g_numel = 1;
  g_copy_written = SIZE_MAX;
  g_training_mode = 0;
  g_inference_mode = 0;
  g_malformed_tensor_handle = 0;
  g_tensor_release_count = 0;
  g_module_release_count = 0;
  g_optimizer_release_count = 0;
  g_session_release_count = 0;
  memcpy(g_requested_provider, "auto", 5);
}

TS_API void ts_test_set_abi_version(uint32_t version) { g_abi_version = version; }
TS_API void ts_test_set_forced_status(int32_t status) { g_forced_status = status; }
TS_API void ts_test_set_null_diagnostic(uint8_t enabled) { g_null_diagnostic = enabled; }
TS_API void ts_test_set_null_handle(uint8_t enabled) { g_null_handle = enabled; }
TS_API void ts_test_set_rank(size_t rank) { g_rank = rank; }
TS_API void ts_test_set_shape_returned_rank(size_t rank) { g_shape_returned_rank = rank; }
TS_API void ts_test_set_shape_dimension(int64_t dimension) { g_shape_dimension = dimension; }
TS_API void ts_test_set_dtype(uint32_t dtype) { g_dtype = dtype; }
TS_API void ts_test_set_device(uint32_t device, int32_t index) {
  g_device = device;
  g_device_index = index;
}
TS_API void ts_test_set_numel(uint64_t numel) { g_numel = numel; }
TS_API void ts_test_set_copy_written(size_t written) { g_copy_written = written; }
TS_API void ts_test_set_training_mode(int32_t mode) { g_training_mode = mode; }
TS_API void ts_test_set_inference_mode(int32_t mode) { g_inference_mode = mode; }
TS_API uint64_t ts_test_tensor_release_count(void) { return g_tensor_release_count; }
TS_API uint64_t ts_test_module_release_count(void) { return g_module_release_count; }
TS_API uint64_t ts_test_optimizer_release_count(void) { return g_optimizer_release_count; }
TS_API uint64_t ts_test_session_release_count(void) { return g_session_release_count; }

static ts_status_t forced_status(void) { return g_forced_status; }

static uint64_t next_handle(void) {
  if (g_null_handle != 0) {
    return 0;
  }
  return g_next_handle++;
}

static ts_status_t write_utf8(const char* value,
                              char* out,
                              size_t capacity,
                              size_t* out_required) {
  if (out_required == NULL) {
    return TS_INVALID_ARGUMENT;
  }

  const size_t required = strlen(value) + 1;
  if (g_inference_mode == 2) {
    *out_required = 0;
    return TS_OK;
  }

  if (out == NULL || capacity == 0) {
    *out_required = required;
    return TS_OK;
  }
  if (capacity < required) {
    *out_required = required;
    return TS_INVALID_ARGUMENT;
  }

  memcpy(out, value, required);
  *out_required = g_inference_mode == 3 ? required + 1 : required;
  return TS_OK;
}

uint32_t ts_abi_version(void) { return g_abi_version; }

const char* ts_last_error_message(void) {
  return g_null_diagnostic != 0 ? NULL : "contract fixture failure";
}

const char* ts_status_name(int32_t status) {
  switch (status) {
    case TS_OK:
      return "OK";
    case TS_INVALID_ARGUMENT:
      return "INVALID_ARGUMENT";
    case TS_INVALID_SHAPE:
      return "INVALID_SHAPE";
    case TS_OUT_OF_MEMORY:
      return "OUT_OF_MEMORY";
    case TS_UNSUPPORTED:
      return "UNSUPPORTED";
    case TS_INVALID_HANDLE:
      return "INVALID_HANDLE";
    case TS_INTERNAL_ERROR:
      return "INTERNAL_ERROR";
    case TS_MODEL_ERROR:
      return "MODEL_ERROR";
    default:
      return "UNKNOWN";
  }
}

ts_status_t ts_noop(void) { return forced_status(); }

ts_status_t ts_runtime_device_count(uint32_t device, uint32_t* out_count) {
  if (out_count == NULL) {
    return TS_INVALID_ARGUMENT;
  }
  switch (device) {
    case TS_DEVICE_CPU:
      *out_count = 1;
      break;
    case TS_DEVICE_CUDA:
      *out_count = 2;
      break;
    case TS_DEVICE_MPS:
      *out_count = 1;
      break;
    case TS_DEVICE_XPU:
      *out_count = 1;
      break;
    case TS_DEVICE_HIP:
      *out_count = 1;
      break;
    default:
      *out_count = 0;
      break;
  }
  return TS_OK;
}

ts_status_t ts_runtime_cuda_device_count(uint32_t* out_count) {
  return ts_runtime_device_count(TS_DEVICE_CUDA, out_count);
}

ts_status_t ts_tensor_from_f32(const float* data,
                               size_t data_length,
                               const int64_t* dims,
                               size_t rank,
                               ts_tensor_t* out_tensor) {
  (void)data;
  (void)data_length;
  (void)dims;
  (void)rank;
  if (out_tensor == NULL) {
    return TS_INVALID_ARGUMENT;
  }
  *out_tensor = next_handle();
  return TS_OK;
}

ts_status_t ts_tensor_full_f32(const int64_t* dims,
                               size_t rank,
                               float value,
                               ts_tensor_t* out_tensor) {
  (void)dims;
  (void)rank;
  (void)value;
  if (out_tensor == NULL) {
    return TS_INVALID_ARGUMENT;
  }
  *out_tensor = next_handle();
  return TS_OK;
}

ts_status_t ts_tensor_rank(ts_tensor_t tensor, size_t* out_rank) {
  (void)tensor;
  if (out_rank == NULL) {
    return TS_INVALID_ARGUMENT;
  }
  *out_rank = g_rank;
  return TS_OK;
}

ts_status_t ts_tensor_shape(ts_tensor_t tensor,
                            int64_t* out_dims,
                            size_t capacity,
                            size_t* out_rank) {
  (void)tensor;
  if (out_rank == NULL) {
    return TS_INVALID_ARGUMENT;
  }
  *out_rank = g_shape_returned_rank == SIZE_MAX ? g_rank : g_shape_returned_rank;
  if (g_rank > 0 && out_dims != NULL && capacity > 0) {
    out_dims[0] = g_shape_dimension;
    for (size_t index = 1; index < capacity; ++index) {
      out_dims[index] = 1;
    }
  }
  return TS_OK;
}

ts_status_t ts_tensor_dtype(ts_tensor_t tensor, uint32_t* out_dtype) {
  if (out_dtype == NULL) {
    return TS_INVALID_ARGUMENT;
  }
  *out_dtype = tensor == g_malformed_tensor_handle ? UINT32_MAX : g_dtype;
  return TS_OK;
}

ts_status_t ts_tensor_device(ts_tensor_t tensor, uint32_t* out_device) {
  (void)tensor;
  if (out_device == NULL) {
    return TS_INVALID_ARGUMENT;
  }
  *out_device = g_device;
  return TS_OK;
}

ts_status_t ts_tensor_device_index(ts_tensor_t tensor, int32_t* out_device_index) {
  (void)tensor;
  if (out_device_index == NULL) {
    return TS_INVALID_ARGUMENT;
  }
  *out_device_index = g_device_index;
  return TS_OK;
}

ts_status_t ts_tensor_numel(ts_tensor_t tensor, uint64_t* out_numel) {
  (void)tensor;
  if (out_numel == NULL) {
    return TS_INVALID_ARGUMENT;
  }
  *out_numel = g_numel;
  return TS_OK;
}

static ts_status_t unary_tensor(ts_tensor_t tensor, ts_tensor_t* out_tensor) {
  (void)tensor;
  if (out_tensor == NULL) {
    return TS_INVALID_ARGUMENT;
  }
  *out_tensor = next_handle();
  return TS_OK;
}

static ts_status_t binary_tensor(ts_tensor_t left,
                                 ts_tensor_t right,
                                 ts_tensor_t* out_tensor) {
  (void)left;
  (void)right;
  return unary_tensor(0, out_tensor);
}

ts_status_t ts_tensor_to_device(ts_tensor_t tensor,
                                uint32_t device,
                                int32_t device_index,
                                ts_tensor_t* out_tensor) {
  (void)device;
  (void)device_index;
  return unary_tensor(tensor, out_tensor);
}

ts_status_t ts_tensor_reshape(ts_tensor_t tensor,
                              const int64_t* dims,
                              size_t rank,
                              ts_tensor_t* out_tensor) {
  (void)dims;
  (void)rank;
  return unary_tensor(tensor, out_tensor);
}

ts_status_t ts_tensor_transpose2d(ts_tensor_t tensor, ts_tensor_t* out_tensor) {
  return unary_tensor(tensor, out_tensor);
}

ts_status_t ts_tensor_add(ts_tensor_t left,
                          ts_tensor_t right,
                          ts_tensor_t* out_tensor) {
  return binary_tensor(left, right, out_tensor);
}

ts_status_t ts_tensor_multiply(ts_tensor_t left,
                               ts_tensor_t right,
                               ts_tensor_t* out_tensor) {
  return binary_tensor(left, right, out_tensor);
}

ts_status_t ts_tensor_sum(ts_tensor_t tensor, ts_tensor_t* out_tensor) {
  return unary_tensor(tensor, out_tensor);
}

ts_status_t ts_tensor_matmul(ts_tensor_t left,
                             ts_tensor_t right,
                             ts_tensor_t* out_tensor) {
  return binary_tensor(left, right, out_tensor);
}

ts_status_t ts_tensor_copy_to_host_f32(ts_tensor_t tensor,
                                       float* out_values,
                                       size_t capacity,
                                       size_t* out_written) {
  (void)tensor;
  if (out_written == NULL) {
    return TS_INVALID_ARGUMENT;
  }
  if (out_values != NULL && capacity > 0) {
    for (size_t index = 0; index < capacity; ++index) {
      out_values[index] = (float)(index + 1);
    }
  }
  *out_written = g_copy_written == SIZE_MAX ? capacity : g_copy_written;
  return TS_OK;
}

ts_status_t ts_tensor_retain(ts_tensor_t tensor) {
  (void)tensor;
  return TS_OK;
}

ts_status_t ts_tensor_release(ts_tensor_t tensor) {
  (void)tensor;
  ++g_tensor_release_count;
  return TS_OK;
}

ts_status_t ts_runtime_live_tensor_count(uint64_t* out_count) {
  if (out_count == NULL) {
    return TS_INVALID_ARGUMENT;
  }
  *out_count = 0;
  return TS_OK;
}

ts_status_t ts_runtime_live_storage_bytes(uint64_t* out_bytes) {
  if (out_bytes == NULL) {
    return TS_INVALID_ARGUMENT;
  }
  *out_bytes = 0;
  return TS_OK;
}

ts_status_t ts_training_available(uint8_t* out_available) {
  if (out_available == NULL) {
    return TS_INVALID_ARGUMENT;
  }
  *out_available = 1;
  return TS_OK;
}

ts_status_t ts_manual_seed(uint64_t seed) {
  (void)seed;
  return forced_status();
}

ts_status_t ts_tensor_with_requires_grad(ts_tensor_t tensor,
                                         uint8_t requires_grad,
                                         ts_tensor_t* out_tensor) {
  (void)requires_grad;
  return unary_tensor(tensor, out_tensor);
}

ts_status_t ts_tensor_requires_grad(ts_tensor_t tensor,
                                    uint8_t* out_requires_grad) {
  (void)tensor;
  if (out_requires_grad == NULL) {
    return TS_INVALID_ARGUMENT;
  }
  *out_requires_grad = 1;
  return TS_OK;
}

ts_status_t ts_tensor_backward(ts_tensor_t tensor) {
  (void)tensor;
  return TS_OK;
}

ts_status_t ts_tensor_grad(ts_tensor_t tensor, ts_tensor_t* out_tensor) {
  return unary_tensor(tensor, out_tensor);
}

ts_status_t ts_tensor_relu(ts_tensor_t tensor, ts_tensor_t* out_tensor) {
  return unary_tensor(tensor, out_tensor);
}

ts_status_t ts_tensor_sigmoid(ts_tensor_t tensor, ts_tensor_t* out_tensor) {
  return unary_tensor(tensor, out_tensor);
}

ts_status_t ts_tensor_tanh(ts_tensor_t tensor, ts_tensor_t* out_tensor) {
  return unary_tensor(tensor, out_tensor);
}

ts_status_t ts_mse_loss(ts_tensor_t prediction,
                        ts_tensor_t target,
                        ts_tensor_t* out_tensor) {
  return binary_tensor(prediction, target, out_tensor);
}

ts_status_t ts_cross_entropy_loss(ts_tensor_t logits,
                                  ts_tensor_t one_hot_target,
                                  ts_tensor_t* out_tensor) {
  return binary_tensor(logits, one_hot_target, out_tensor);
}

ts_status_t ts_linear_create(int64_t in_features,
                             int64_t out_features,
                             uint8_t use_bias,
                             ts_module_t* out_module) {
  (void)in_features;
  (void)out_features;
  (void)use_bias;
  if (out_module == NULL) {
    return TS_INVALID_ARGUMENT;
  }
  *out_module = next_handle();
  return TS_OK;
}

ts_status_t ts_module_forward(ts_module_t module,
                              ts_tensor_t input,
                              ts_tensor_t* out_tensor) {
  (void)module;
  return unary_tensor(input, out_tensor);
}

ts_status_t ts_module_set_training(ts_module_t module, uint8_t training) {
  (void)module;
  (void)training;
  return TS_OK;
}

ts_status_t ts_module_to_device(ts_module_t module,
                                uint32_t device,
                                int32_t device_index) {
  (void)module;
  (void)device;
  (void)device_index;
  return TS_OK;
}

ts_status_t ts_module_parameter_count(ts_module_t module, size_t* out_count) {
  (void)module;
  if (out_count == NULL) {
    return TS_INVALID_ARGUMENT;
  }
  *out_count = g_training_mode == 1 ? 2 : 1;
  return TS_OK;
}

ts_status_t ts_module_parameter_at(ts_module_t module,
                                   size_t index,
                                   ts_tensor_t* out_tensor) {
  (void)module;
  if (out_tensor == NULL) {
    return TS_INVALID_ARGUMENT;
  }
  *out_tensor = next_handle();
  if (g_training_mode == 1 && index == 1) {
    g_malformed_tensor_handle = *out_tensor;
  }
  return TS_OK;
}

ts_status_t ts_module_buffer_count(ts_module_t module, size_t* out_count) {
  (void)module;
  if (out_count == NULL) {
    return TS_INVALID_ARGUMENT;
  }
  *out_count = 0;
  return TS_OK;
}

ts_status_t ts_module_buffer_at(ts_module_t module,
                                size_t index,
                                ts_tensor_t* out_tensor) {
  (void)module;
  (void)index;
  return unary_tensor(0, out_tensor);
}

ts_status_t ts_module_save(ts_module_t module, const char* path) {
  (void)module;
  (void)path;
  return TS_OK;
}

ts_status_t ts_module_load(ts_module_t module, const char* path) {
  (void)module;
  (void)path;
  return TS_OK;
}

ts_status_t ts_module_release(ts_module_t module) {
  (void)module;
  ++g_module_release_count;
  return TS_OK;
}

static ts_status_t create_optimizer(ts_optimizer_t* out_optimizer) {
  if (out_optimizer == NULL) {
    return TS_INVALID_ARGUMENT;
  }
  *out_optimizer = next_handle();
  return TS_OK;
}

ts_status_t ts_sgd_create(ts_module_t module,
                          double learning_rate,
                          double momentum,
                          double weight_decay,
                          ts_optimizer_t* out_optimizer) {
  (void)module;
  (void)learning_rate;
  (void)momentum;
  (void)weight_decay;
  return create_optimizer(out_optimizer);
}

ts_status_t ts_adam_create(ts_module_t module,
                           double learning_rate,
                           double beta1,
                           double beta2,
                           double epsilon,
                           double weight_decay,
                           ts_optimizer_t* out_optimizer) {
  (void)module;
  (void)learning_rate;
  (void)beta1;
  (void)beta2;
  (void)epsilon;
  (void)weight_decay;
  return create_optimizer(out_optimizer);
}

ts_status_t ts_adamw_create(ts_module_t module,
                            double learning_rate,
                            double beta1,
                            double beta2,
                            double epsilon,
                            double weight_decay,
                            ts_optimizer_t* out_optimizer) {
  return ts_adam_create(module,
                        learning_rate,
                        beta1,
                        beta2,
                        epsilon,
                        weight_decay,
                        out_optimizer);
}

ts_status_t ts_optimizer_zero_grad(ts_optimizer_t optimizer) {
  (void)optimizer;
  return TS_OK;
}

ts_status_t ts_optimizer_step(ts_optimizer_t optimizer) {
  (void)optimizer;
  return TS_OK;
}

ts_status_t ts_optimizer_release(ts_optimizer_t optimizer) {
  (void)optimizer;
  ++g_optimizer_release_count;
  return TS_OK;
}

ts_status_t ts_runtime_live_module_count(uint64_t* out_count) {
  if (out_count == NULL) {
    return TS_INVALID_ARGUMENT;
  }
  *out_count = 0;
  return TS_OK;
}

ts_status_t ts_runtime_live_optimizer_count(uint64_t* out_count) {
  if (out_count == NULL) {
    return TS_INVALID_ARGUMENT;
  }
  *out_count = 0;
  return TS_OK;
}

ts_status_t ts_onnx_available(uint8_t* out_available) {
  if (g_forced_status != TS_OK) {
    return g_forced_status;
  }
  if (out_available == NULL) {
    return TS_INVALID_ARGUMENT;
  }
  *out_available = 1;
  return TS_OK;
}

ts_status_t ts_onnx_provider_count(size_t* out_count) {
  if (out_count == NULL) {
    return TS_INVALID_ARGUMENT;
  }
  *out_count = 2;
  return TS_OK;
}

ts_status_t ts_onnx_provider_name(size_t index,
                                  char* out_name,
                                  size_t capacity,
                                  size_t* out_required) {
  const char* provider = index == 0 ? "CPUExecutionProvider" : "CUDAExecutionProvider";
  return write_utf8(provider, out_name, capacity, out_required);
}

ts_status_t ts_onnx_session_create(const char* model_path,
                                   uint8_t enable_profiling,
                                   const char* profiling_prefix,
                                   ts_onnx_session_t* out_session) {
  return ts_onnx_session_create_with_provider(model_path,
                                              "auto",
                                              enable_profiling,
                                              profiling_prefix,
                                              out_session);
}

ts_status_t ts_onnx_session_create_with_provider(
    const char* model_path,
    const char* requested_provider,
    uint8_t enable_profiling,
    const char* profiling_prefix,
    ts_onnx_session_t* out_session) {
  (void)model_path;
  (void)enable_profiling;
  (void)profiling_prefix;
  if (out_session == NULL || requested_provider == NULL) {
    return TS_INVALID_ARGUMENT;
  }
  snprintf(g_requested_provider, sizeof(g_requested_provider), "%s", requested_provider);
  *out_session = next_handle();
  return TS_OK;
}

ts_status_t ts_onnx_session_provider(ts_onnx_session_t session,
                                     char* out_provider,
                                     size_t capacity,
                                     size_t* out_required) {
  (void)session;
  const char* provider;
  if (g_inference_mode == 4) {
    provider = "UnknownExecutionProvider";
  } else if (g_inference_mode == 5) {
    provider = "CPUExecutionProvider";
  } else if (strcmp(g_requested_provider, "auto") == 0) {
    provider = "CPUExecutionProvider";
  } else {
    provider = g_requested_provider;
  }
  return write_utf8(provider, out_provider, capacity, out_required);
}

ts_status_t ts_onnx_session_input_count(ts_onnx_session_t session,
                                        size_t* out_count) {
  (void)session;
  if (out_count == NULL) {
    return TS_INVALID_ARGUMENT;
  }
  *out_count = g_inference_mode == 6 ? 0 : 1;
  return TS_OK;
}

ts_status_t ts_onnx_session_output_count(ts_onnx_session_t session,
                                         size_t* out_count) {
  (void)session;
  if (out_count == NULL) {
    return TS_INVALID_ARGUMENT;
  }
  if (g_inference_mode == 7) {
    *out_count = 0;
  } else if (g_inference_mode == 9) {
    *out_count = 2;
  } else {
    *out_count = 1;
  }
  return TS_OK;
}

ts_status_t ts_onnx_session_input_name(ts_onnx_session_t session,
                                       size_t index,
                                       char* out_name,
                                       size_t capacity,
                                       size_t* out_required) {
  (void)session;
  (void)index;
  return write_utf8("X", out_name, capacity, out_required);
}

ts_status_t ts_onnx_session_output_name(ts_onnx_session_t session,
                                        size_t index,
                                        char* out_name,
                                        size_t capacity,
                                        size_t* out_required) {
  (void)session;
  if (g_inference_mode == 9) {
    return write_utf8(index == 0 ? "Y0" : "Y1", out_name, capacity, out_required);
  }
  return write_utf8("Y", out_name, capacity, out_required);
}

ts_status_t ts_onnx_session_run(ts_onnx_session_t session,
                                const char* const* input_names,
                                const ts_tensor_t* input_tensors,
                                size_t input_count,
                                const char* const* output_names,
                                size_t output_count,
                                ts_tensor_t* out_tensors,
                                size_t out_capacity,
                                size_t* out_written) {
  (void)session;
  (void)input_names;
  (void)input_tensors;
  (void)input_count;
  (void)output_names;
  if (out_tensors == NULL || out_written == NULL || out_capacity < output_count) {
    return TS_INVALID_ARGUMENT;
  }
  for (size_t index = 0; index < output_count; ++index) {
    out_tensors[index] = next_handle();
  }
  if (g_inference_mode == 9 && output_count > 1) {
    g_malformed_tensor_handle = out_tensors[1];
  }
  if (g_inference_mode == 8 && output_count > 1) {
    *out_written = 1;
  } else {
    *out_written = output_count;
  }
  return TS_OK;
}

ts_status_t ts_onnx_session_end_profiling(ts_onnx_session_t session,
                                          char* out_path,
                                          size_t capacity,
                                          size_t* out_required) {
  (void)session;
  return write_utf8("profile.json", out_path, capacity, out_required);
}

ts_status_t ts_onnx_session_release(ts_onnx_session_t session) {
  (void)session;
  ++g_session_release_count;
  return TS_OK;
}

ts_status_t ts_runtime_live_onnx_session_count(uint64_t* out_count) {
  if (out_count == NULL) {
    return TS_INVALID_ARGUMENT;
  }
  *out_count = 0;
  return TS_OK;
}
