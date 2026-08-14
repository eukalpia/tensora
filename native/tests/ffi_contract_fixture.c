#include "tensora.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if defined(_WIN32)
#define TEST_API __declspec(dllexport)
#else
#define TEST_API __attribute__((visibility("default")))
#endif

enum {
  MODE_NORMAL = 0,
  MODE_ABI_MISMATCH = 1,
  MODE_INVALID_ARGUMENT = 10,
  MODE_INVALID_SHAPE_STATUS = 11,
  MODE_OUT_OF_MEMORY = 12,
  MODE_UNSUPPORTED = 13,
  MODE_INVALID_HANDLE = 14,
  MODE_INTERNAL_ERROR = 15,
  MODE_MODEL_ERROR = 16,
  MODE_UNKNOWN_STATUS = 17,
  MODE_NULL_DIAGNOSTIC = 18,
  MODE_NULL_TENSOR_HANDLE = 20,
  MODE_RANK_ABOVE_LIMIT = 21,
  MODE_SCALAR_RANK_CHANGED = 22,
  MODE_RANK_CHANGED = 23,
  MODE_INVALID_SHAPE_METADATA = 24,
  MODE_UNKNOWN_DTYPE = 25,
  MODE_INVALID_CPU_INDEX = 26,
  MODE_INVALID_CUDA_INDEX = 27,
  MODE_INVALID_MPS_INDEX = 28,
  MODE_INVALID_XPU_INDEX = 29,
  MODE_INVALID_HIP_INDEX = 30,
  MODE_UNKNOWN_DEVICE = 31,
  MODE_INCONSISTENT_NUMEL = 32,
  MODE_VALID_CUDA_TENSOR = 33,
  MODE_NULL_TRAINING_HANDLE = 40,
  MODE_ACCELERATOR_COUNTS = 41,
  MODE_MODULE_BUFFER = 42,
  MODE_MODULE_ADOPTION_FAILURE = 43,
  MODE_NULL_SESSION_HANDLE = 50,
  MODE_PROVIDER_MISMATCH = 51,
  MODE_NO_OUTPUTS = 52,
  MODE_INVALID_UTF8_SIZE = 53,
  MODE_CHANGED_UTF8_SIZE = 54,
  MODE_OUTPUT_COUNT_MISMATCH = 55,
  MODE_OUTPUT_ADOPTION_FAILURE = 56,
  MODE_DTYPE_FLOAT16 = 60,
  MODE_DTYPE_BFLOAT16 = 61,
  MODE_DTYPE_FLOAT64 = 62,
  MODE_DTYPE_INT8 = 63,
  MODE_DTYPE_UINT8 = 64,
  MODE_DTYPE_INT16 = 65,
  MODE_DTYPE_INT32 = 66,
  MODE_DTYPE_INT64 = 67,
  MODE_DTYPE_BOOL = 68
};

static int32_t g_mode = MODE_NORMAL;
static uint64_t g_next_handle = 1000;
static uint64_t g_live_tensors = 0;
static uint64_t g_live_modules = 0;
static uint64_t g_live_optimizers = 0;
static uint64_t g_live_sessions = 0;
static uint64_t g_release_count = 0;
static const char g_diagnostic[] = "fault runtime diagnostic";

TEST_API void ts_test_set_mode(int32_t mode) { g_mode = mode; }
TEST_API uint64_t ts_test_release_count(void) { return g_release_count; }
TEST_API void ts_test_reset_counters(void) { g_release_count = 0; }

static ts_status_t generic_status(void) {
  switch (g_mode) {
    case MODE_INVALID_ARGUMENT:
      return TS_INVALID_ARGUMENT;
    case MODE_INVALID_SHAPE_STATUS:
      return TS_INVALID_SHAPE;
    case MODE_OUT_OF_MEMORY:
      return TS_OUT_OF_MEMORY;
    case MODE_UNSUPPORTED:
      return TS_UNSUPPORTED;
    case MODE_INVALID_HANDLE:
      return TS_INVALID_HANDLE;
    case MODE_INTERNAL_ERROR:
    case MODE_NULL_DIAGNOSTIC:
      return TS_INTERNAL_ERROR;
    case MODE_MODEL_ERROR:
      return TS_MODEL_ERROR;
    case MODE_UNKNOWN_STATUS:
      return 99;
    default:
      return TS_OK;
  }
}

static uint64_t new_tensor(void) {
  ++g_live_tensors;
  return ++g_next_handle;
}

static uint64_t new_module(void) {
  ++g_live_modules;
  return ++g_next_handle;
}

static uint64_t new_optimizer(void) {
  ++g_live_optimizers;
  return ++g_next_handle;
}

static uint64_t new_session(void) {
  ++g_live_sessions;
  return ++g_next_handle;
}

static ts_status_t write_string(const char* value,
                                char* output,
                                size_t capacity,
                                size_t* required) {
  const size_t size = strlen(value) + 1;
  if (required == NULL) return TS_INVALID_ARGUMENT;
  *required = size;
  if (output == NULL || capacity == 0) return TS_OK;
  if (capacity < size) return TS_INVALID_ARGUMENT;
  memcpy(output, value, size);
  return TS_OK;
}

uint32_t ts_abi_version(void) {
  return g_mode == MODE_ABI_MISMATCH ? 999u : TS_ABI_VERSION;
}

const char* ts_last_error_message(void) {
  return g_mode == MODE_NULL_DIAGNOSTIC ? NULL : g_diagnostic;
}

const char* ts_status_name(int32_t status) {
  (void)status;
  return "fixture";
}

ts_status_t ts_noop(void) { return generic_status(); }

ts_status_t ts_training_available(uint8_t* out_available) {
  const ts_status_t status = generic_status();
  if (status != TS_OK) return status;
  if (out_available == NULL) return TS_INVALID_ARGUMENT;
  *out_available = 1;
  return TS_OK;
}

ts_status_t ts_onnx_available(uint8_t* out_available) {
  const ts_status_t status = generic_status();
  if (status != TS_OK) return status;
  if (out_available == NULL) return TS_INVALID_ARGUMENT;
  *out_available = 1;
  return TS_OK;
}

ts_status_t ts_runtime_device_count(uint32_t device, uint32_t* out_count) {
  if (out_count == NULL) return TS_INVALID_ARGUMENT;
  if (g_mode == MODE_ACCELERATOR_COUNTS) {
    switch (device) {
      case TS_DEVICE_CPU:
        *out_count = 1;
        break;
      case TS_DEVICE_CUDA:
      case TS_DEVICE_XPU:
      case TS_DEVICE_HIP:
        *out_count = 2;
        break;
      case TS_DEVICE_MPS:
        *out_count = 1;
        break;
      default:
        *out_count = 0;
        break;
    }
    return TS_OK;
  }
  *out_count = device == TS_DEVICE_CPU ? 1u : 0u;
  return TS_OK;
}

ts_status_t ts_runtime_cuda_device_count(uint32_t* out_count) {
  return ts_runtime_device_count(TS_DEVICE_CUDA, out_count);
}

ts_status_t ts_manual_seed(uint64_t seed) {
  (void)seed;
  return generic_status();
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
  if (out_tensor == NULL) return TS_INVALID_ARGUMENT;
  if (g_mode == MODE_NULL_TENSOR_HANDLE) {
    *out_tensor = 0;
    return TS_OK;
  }
  *out_tensor = new_tensor();
  return TS_OK;
}

ts_status_t ts_tensor_full_f32(const int64_t* dims,
                               size_t rank,
                               float value,
                               ts_tensor_t* out_tensor) {
  (void)dims;
  (void)rank;
  (void)value;
  return ts_tensor_from_f32(NULL, 0, NULL, 0, out_tensor);
}

ts_status_t ts_tensor_rank(ts_tensor_t tensor, size_t* out_rank) {
  (void)tensor;
  if (out_rank == NULL) return TS_INVALID_ARGUMENT;
  switch (g_mode) {
    case MODE_RANK_ABOVE_LIMIT:
      *out_rank = 65;
      break;
    case MODE_SCALAR_RANK_CHANGED:
      *out_rank = 0;
      break;
    case MODE_RANK_CHANGED:
      *out_rank = 2;
      break;
    default:
      *out_rank = 1;
      break;
  }
  return TS_OK;
}

ts_status_t ts_tensor_shape(ts_tensor_t tensor,
                            int64_t* out_dims,
                            size_t capacity,
                            size_t* out_rank) {
  (void)tensor;
  if (out_rank == NULL) return TS_INVALID_ARGUMENT;
  if (g_mode == MODE_SCALAR_RANK_CHANGED) {
    *out_rank = 1;
    return TS_OK;
  }
  if (g_mode == MODE_RANK_CHANGED) {
    *out_rank = 1;
    if (out_dims != NULL && capacity > 0) out_dims[0] = 1;
    return TS_OK;
  }
  *out_rank = 1;
  if (out_dims != NULL && capacity > 0) {
    out_dims[0] = g_mode == MODE_INVALID_SHAPE_METADATA ? 0 : 1;
  }
  return TS_OK;
}

ts_status_t ts_tensor_dtype(ts_tensor_t tensor, uint32_t* out_dtype) {
  if (out_dtype == NULL) return TS_INVALID_ARGUMENT;
  if (g_mode == MODE_UNKNOWN_DTYPE ||
      (g_mode == MODE_MODULE_ADOPTION_FAILURE && tensor == 4302) ||
      (g_mode == MODE_OUTPUT_ADOPTION_FAILURE && tensor == 5602)) {
    *out_dtype = 99;
  } else {
    switch (g_mode) {
      case MODE_DTYPE_FLOAT16:
        *out_dtype = TS_DTYPE_FLOAT16;
        break;
      case MODE_DTYPE_BFLOAT16:
        *out_dtype = TS_DTYPE_BFLOAT16;
        break;
      case MODE_DTYPE_FLOAT64:
        *out_dtype = TS_DTYPE_FLOAT64;
        break;
      case MODE_DTYPE_INT8:
        *out_dtype = TS_DTYPE_INT8;
        break;
      case MODE_DTYPE_UINT8:
        *out_dtype = TS_DTYPE_UINT8;
        break;
      case MODE_DTYPE_INT16:
        *out_dtype = TS_DTYPE_INT16;
        break;
      case MODE_DTYPE_INT32:
        *out_dtype = TS_DTYPE_INT32;
        break;
      case MODE_DTYPE_INT64:
        *out_dtype = TS_DTYPE_INT64;
        break;
      case MODE_DTYPE_BOOL:
        *out_dtype = TS_DTYPE_BOOL;
        break;
      default:
        *out_dtype = TS_DTYPE_FLOAT32;
        break;
    }
  }
  return TS_OK;
}

ts_status_t ts_tensor_device(ts_tensor_t tensor, uint32_t* out_device) {
  (void)tensor;
  if (out_device == NULL) return TS_INVALID_ARGUMENT;
  switch (g_mode) {
    case MODE_INVALID_CUDA_INDEX:
    case MODE_VALID_CUDA_TENSOR:
      *out_device = TS_DEVICE_CUDA;
      break;
    case MODE_INVALID_MPS_INDEX:
      *out_device = TS_DEVICE_MPS;
      break;
    case MODE_INVALID_XPU_INDEX:
      *out_device = TS_DEVICE_XPU;
      break;
    case MODE_INVALID_HIP_INDEX:
      *out_device = TS_DEVICE_HIP;
      break;
    case MODE_UNKNOWN_DEVICE:
      *out_device = 99;
      break;
    default:
      *out_device = TS_DEVICE_CPU;
      break;
  }
  return TS_OK;
}

ts_status_t ts_tensor_device_index(ts_tensor_t tensor,
                                   int32_t* out_device_index) {
  (void)tensor;
  if (out_device_index == NULL) return TS_INVALID_ARGUMENT;
  switch (g_mode) {
    case MODE_INVALID_CPU_INDEX:
    case MODE_INVALID_MPS_INDEX:
    case MODE_VALID_CUDA_TENSOR:
      *out_device_index = 1;
      break;
    case MODE_INVALID_CUDA_INDEX:
    case MODE_INVALID_XPU_INDEX:
    case MODE_INVALID_HIP_INDEX:
      *out_device_index = -1;
      break;
    default:
      *out_device_index = 0;
      break;
  }
  return TS_OK;
}

ts_status_t ts_tensor_numel(ts_tensor_t tensor, uint64_t* out_numel) {
  (void)tensor;
  if (out_numel == NULL) return TS_INVALID_ARGUMENT;
  *out_numel = g_mode == MODE_INCONSISTENT_NUMEL ? 2u : 1u;
  return TS_OK;
}

ts_status_t ts_tensor_to_device(ts_tensor_t tensor,
                                uint32_t device,
                                int32_t device_index,
                                ts_tensor_t* out_tensor) {
  (void)tensor;
  (void)device;
  (void)device_index;
  if (out_tensor == NULL) return TS_INVALID_ARGUMENT;
  *out_tensor = new_tensor();
  return TS_OK;
}

#define DEFINE_UNARY_TENSOR(name)                                      \
  ts_status_t name(ts_tensor_t tensor, ts_tensor_t* out_tensor) {      \
    (void)tensor;                                                       \
    if (out_tensor == NULL) return TS_INVALID_ARGUMENT;                 \
    *out_tensor = new_tensor();                                         \
    return TS_OK;                                                       \
  }

#define DEFINE_BINARY_TENSOR(name)                                     \
  ts_status_t name(ts_tensor_t left,                                   \
                   ts_tensor_t right,                                  \
                   ts_tensor_t* out_tensor) {                           \
    (void)left;                                                         \
    (void)right;                                                        \
    if (out_tensor == NULL) return TS_INVALID_ARGUMENT;                 \
    *out_tensor = new_tensor();                                         \
    return TS_OK;                                                       \
  }

ts_status_t ts_tensor_reshape(ts_tensor_t tensor,
                              const int64_t* dims,
                              size_t rank,
                              ts_tensor_t* out_tensor) {
  (void)dims;
  (void)rank;
  return ts_tensor_to_device(tensor, TS_DEVICE_CPU, 0, out_tensor);
}
DEFINE_UNARY_TENSOR(ts_tensor_transpose2d)
DEFINE_BINARY_TENSOR(ts_tensor_add)
DEFINE_BINARY_TENSOR(ts_tensor_multiply)
DEFINE_UNARY_TENSOR(ts_tensor_sum)
DEFINE_BINARY_TENSOR(ts_tensor_matmul)

ts_status_t ts_tensor_with_requires_grad(ts_tensor_t tensor,
                                         uint8_t requires_grad,
                                         ts_tensor_t* out_tensor) {
  (void)requires_grad;
  return ts_tensor_to_device(tensor, TS_DEVICE_CPU, 0, out_tensor);
}

ts_status_t ts_tensor_requires_grad(ts_tensor_t tensor,
                                    uint8_t* out_requires_grad) {
  (void)tensor;
  if (out_requires_grad == NULL) return TS_INVALID_ARGUMENT;
  *out_requires_grad = 1;
  return TS_OK;
}

ts_status_t ts_tensor_backward(ts_tensor_t tensor) {
  (void)tensor;
  return TS_OK;
}
DEFINE_UNARY_TENSOR(ts_tensor_grad)
DEFINE_UNARY_TENSOR(ts_tensor_relu)
DEFINE_UNARY_TENSOR(ts_tensor_sigmoid)
DEFINE_UNARY_TENSOR(ts_tensor_tanh)
DEFINE_BINARY_TENSOR(ts_mse_loss)
DEFINE_BINARY_TENSOR(ts_cross_entropy_loss)

ts_status_t ts_linear_create(int64_t in_features,
                             int64_t out_features,
                             uint8_t use_bias,
                             ts_module_t* out_module) {
  (void)in_features;
  (void)out_features;
  (void)use_bias;
  if (out_module == NULL) return TS_INVALID_ARGUMENT;
  if (g_mode == MODE_NULL_TRAINING_HANDLE) {
    *out_module = 0;
    return TS_OK;
  }
  *out_module = new_module();
  return TS_OK;
}

ts_status_t ts_module_forward(ts_module_t module,
                              ts_tensor_t input,
                              ts_tensor_t* out_tensor) {
  (void)module;
  (void)input;
  if (out_tensor == NULL) return TS_INVALID_ARGUMENT;
  *out_tensor = new_tensor();
  return TS_OK;
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
  if (out_count == NULL) return TS_INVALID_ARGUMENT;
  *out_count = g_mode == MODE_MODULE_ADOPTION_FAILURE ? 2u : 0u;
  return TS_OK;
}

ts_status_t ts_module_parameter_at(ts_module_t module,
                                   size_t index,
                                   ts_tensor_t* out_tensor) {
  (void)module;
  if (out_tensor == NULL) return TS_INVALID_ARGUMENT;
  if (g_mode == MODE_MODULE_ADOPTION_FAILURE) {
    *out_tensor = index == 0 ? 4301u : 4302u;
    ++g_live_tensors;
    return TS_OK;
  }
  *out_tensor = new_tensor();
  return TS_OK;
}

ts_status_t ts_module_buffer_count(ts_module_t module, size_t* out_count) {
  (void)module;
  if (out_count == NULL) return TS_INVALID_ARGUMENT;
  *out_count = g_mode == MODE_MODULE_BUFFER ? 1u : 0u;
  return TS_OK;
}

ts_status_t ts_module_buffer_at(ts_module_t module,
                                size_t index,
                                ts_tensor_t* out_tensor) {
  (void)module;
  (void)index;
  if (out_tensor == NULL) return TS_INVALID_ARGUMENT;
  *out_tensor = new_tensor();
  return TS_OK;
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
  ++g_release_count;
  if (g_live_modules > 0) --g_live_modules;
  return TS_OK;
}

static ts_status_t create_optimizer(ts_optimizer_t* out_optimizer) {
  if (out_optimizer == NULL) return TS_INVALID_ARGUMENT;
  if (g_mode == MODE_NULL_TRAINING_HANDLE) {
    *out_optimizer = 0;
    return TS_OK;
  }
  *out_optimizer = new_optimizer();
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
  return ts_adam_create(module, learning_rate, beta1, beta2, epsilon,
                        weight_decay, out_optimizer);
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
  ++g_release_count;
  if (g_live_optimizers > 0) --g_live_optimizers;
  return TS_OK;
}

ts_status_t ts_onnx_provider_count(size_t* out_count) {
  if (out_count == NULL) return TS_INVALID_ARGUMENT;
  *out_count = 1;
  return TS_OK;
}

ts_status_t ts_onnx_provider_name(size_t index,
                                  char* out_name,
                                  size_t capacity,
                                  size_t* out_required) {
  (void)index;
  if (out_required == NULL) return TS_INVALID_ARGUMENT;
  if (g_mode == MODE_INVALID_UTF8_SIZE) {
    *out_required = 0;
    return TS_OK;
  }
  if (g_mode == MODE_CHANGED_UTF8_SIZE) {
    if (out_name == NULL || capacity == 0) {
      *out_required = 4;
      return TS_OK;
    }
    *out_required = 5;
    if (capacity > 0) out_name[0] = '\0';
    return TS_OK;
  }
  return write_string("CPUExecutionProvider", out_name, capacity, out_required);
}

ts_status_t ts_onnx_session_create(const char* model_path,
                                   uint8_t enable_profiling,
                                   const char* profiling_prefix,
                                   ts_onnx_session_t* out_session) {
  return ts_onnx_session_create_with_provider(
      model_path, "auto", enable_profiling, profiling_prefix, out_session);
}

ts_status_t ts_onnx_session_create_with_provider(
    const char* model_path,
    const char* requested_provider,
    uint8_t enable_profiling,
    const char* profiling_prefix,
    ts_onnx_session_t* out_session) {
  (void)model_path;
  (void)requested_provider;
  (void)enable_profiling;
  (void)profiling_prefix;
  if (out_session == NULL) return TS_INVALID_ARGUMENT;
  if (g_mode == MODE_NULL_SESSION_HANDLE) {
    *out_session = 0;
    return TS_OK;
  }
  *out_session = new_session();
  return TS_OK;
}

ts_status_t ts_onnx_session_provider(ts_onnx_session_t session,
                                     char* out_provider,
                                     size_t capacity,
                                     size_t* out_required) {
  (void)session;
  return write_string("CPUExecutionProvider", out_provider, capacity,
                      out_required);
}

ts_status_t ts_onnx_session_input_count(ts_onnx_session_t session,
                                        size_t* out_count) {
  (void)session;
  if (out_count == NULL) return TS_INVALID_ARGUMENT;
  *out_count = 1;
  return TS_OK;
}

ts_status_t ts_onnx_session_output_count(ts_onnx_session_t session,
                                         size_t* out_count) {
  (void)session;
  if (out_count == NULL) return TS_INVALID_ARGUMENT;
  if (g_mode == MODE_NO_OUTPUTS) {
    *out_count = 0;
  } else if (g_mode == MODE_OUTPUT_ADOPTION_FAILURE) {
    *out_count = 3;
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
  return write_string("X", out_name, capacity, out_required);
}

ts_status_t ts_onnx_session_output_name(ts_onnx_session_t session,
                                        size_t index,
                                        char* out_name,
                                        size_t capacity,
                                        size_t* out_required) {
  (void)session;
  if (g_mode == MODE_OUTPUT_ADOPTION_FAILURE) {
    const char* name = index == 0 ? "Y0" : (index == 1 ? "Y1" : "Y2");
    return write_string(name, out_name, capacity, out_required);
  }
  return write_string("Y", out_name, capacity, out_required);
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
  if (out_tensors == NULL || out_written == NULL) return TS_INVALID_ARGUMENT;
  if (g_mode == MODE_OUTPUT_COUNT_MISMATCH) {
    if (out_capacity > 0) out_tensors[0] = new_tensor();
    *out_written = 1;
    return TS_OK;
  }
  if (g_mode == MODE_OUTPUT_ADOPTION_FAILURE) {
    if (out_capacity < 3) return TS_INVALID_ARGUMENT;
    out_tensors[0] = 5601;
    out_tensors[1] = 5602;
    out_tensors[2] = 5603;
    g_live_tensors += 3;
    *out_written = 3;
    return TS_OK;
  }
  if (out_capacity < output_count) return TS_INVALID_ARGUMENT;
  for (size_t index = 0; index < output_count; ++index) {
    out_tensors[index] = new_tensor();
  }
  *out_written = output_count;
  return TS_OK;
}

ts_status_t ts_onnx_session_end_profiling(ts_onnx_session_t session,
                                          char* out_path,
                                          size_t capacity,
                                          size_t* out_required) {
  (void)session;
  return write_string("profile.json", out_path, capacity, out_required);
}

ts_status_t ts_onnx_session_release(ts_onnx_session_t session) {
  (void)session;
  ++g_release_count;
  if (g_live_sessions > 0) --g_live_sessions;
  return TS_OK;
}

ts_status_t ts_tensor_copy_to_host_f32(ts_tensor_t tensor,
                                       float* out_values,
                                       size_t capacity,
                                       size_t* out_written) {
  (void)tensor;
  if (out_written == NULL) return TS_INVALID_ARGUMENT;
  if (capacity > 0 && out_values != NULL) out_values[0] = 1.0f;
  *out_written = capacity;
  return TS_OK;
}

ts_status_t ts_tensor_retain(ts_tensor_t tensor) {
  (void)tensor;
  ++g_live_tensors;
  return TS_OK;
}

ts_status_t ts_tensor_release(ts_tensor_t tensor) {
  (void)tensor;
  ++g_release_count;
  if (g_live_tensors > 0) --g_live_tensors;
  return TS_OK;
}

ts_status_t ts_runtime_live_tensor_count(uint64_t* out_count) {
  if (out_count == NULL) return TS_INVALID_ARGUMENT;
  *out_count = g_live_tensors;
  return TS_OK;
}

ts_status_t ts_runtime_live_storage_bytes(uint64_t* out_bytes) {
  if (out_bytes == NULL) return TS_INVALID_ARGUMENT;
  *out_bytes = g_live_tensors * sizeof(float);
  return TS_OK;
}

ts_status_t ts_runtime_live_module_count(uint64_t* out_count) {
  if (out_count == NULL) return TS_INVALID_ARGUMENT;
  *out_count = g_live_modules;
  return TS_OK;
}

ts_status_t ts_runtime_live_optimizer_count(uint64_t* out_count) {
  if (out_count == NULL) return TS_INVALID_ARGUMENT;
  *out_count = g_live_optimizers;
  return TS_OK;
}

ts_status_t ts_runtime_live_onnx_session_count(uint64_t* out_count) {
  if (out_count == NULL) return TS_INVALID_ARGUMENT;
  *out_count = g_live_sessions;
  return TS_OK;
}
