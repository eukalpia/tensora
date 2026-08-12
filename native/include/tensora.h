#ifndef TENSORA_H_
#define TENSORA_H_

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(TENSORA_NATIVE_BUILD)
#define TS_API __declspec(dllexport)
#else
#define TS_API __declspec(dllimport)
#endif
#else
#define TS_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define TS_ABI_VERSION 2u

typedef uint64_t ts_tensor_t;
typedef uint64_t ts_module_t;
typedef uint64_t ts_optimizer_t;
typedef uint64_t ts_onnx_session_t;
typedef int32_t ts_status_t;
typedef uint32_t ts_dtype_t;
typedef uint32_t ts_device_t;

enum {
  TS_OK = 0,
  TS_INVALID_ARGUMENT = 1,
  TS_INVALID_SHAPE = 2,
  TS_OUT_OF_MEMORY = 3,
  TS_UNSUPPORTED = 4,
  TS_INVALID_HANDLE = 5,
  TS_INTERNAL_ERROR = 6
};

enum {
  TS_DTYPE_FLOAT32 = 1u
};

enum {
  TS_DEVICE_CPU = 1u,
  TS_DEVICE_CUDA = 2u
};

/*
 * ABI and error diagnostics.
 */
TS_API uint32_t ts_abi_version(void);
TS_API const char* ts_last_error_message(void);
TS_API const char* ts_status_name(int32_t status);
TS_API ts_status_t ts_noop(void);

/* Runtime/device discovery. */
TS_API ts_status_t ts_training_available(uint8_t* out_available);
TS_API ts_status_t ts_runtime_cuda_device_count(uint32_t* out_count);
TS_API ts_status_t ts_manual_seed(uint64_t seed);

/* Tensor creation. */
TS_API ts_status_t ts_tensor_from_f32(const float* data,
                                      size_t data_length,
                                      const int64_t* dims,
                                      size_t rank,
                                      ts_tensor_t* out_tensor);
TS_API ts_status_t ts_tensor_full_f32(const int64_t* dims,
                                      size_t rank,
                                      float value,
                                      ts_tensor_t* out_tensor);

/* Tensor metadata. */
TS_API ts_status_t ts_tensor_rank(ts_tensor_t tensor, size_t* out_rank);
TS_API ts_status_t ts_tensor_shape(ts_tensor_t tensor,
                                   int64_t* out_dims,
                                   size_t capacity,
                                   size_t* out_rank);
TS_API ts_status_t ts_tensor_dtype(ts_tensor_t tensor, uint32_t* out_dtype);
TS_API ts_status_t ts_tensor_device(ts_tensor_t tensor, uint32_t* out_device);
TS_API ts_status_t ts_tensor_device_index(ts_tensor_t tensor,
                                          int32_t* out_device_index);
TS_API ts_status_t ts_tensor_numel(ts_tensor_t tensor, uint64_t* out_numel);

/* Device transfer. */
TS_API ts_status_t ts_tensor_to_device(ts_tensor_t tensor,
                                       uint32_t device,
                                       int32_t device_index,
                                       ts_tensor_t* out_tensor);

/* Core tensor operations. */
TS_API ts_status_t ts_tensor_reshape(ts_tensor_t tensor,
                                     const int64_t* dims,
                                     size_t rank,
                                     ts_tensor_t* out_tensor);
TS_API ts_status_t ts_tensor_transpose2d(ts_tensor_t tensor,
                                         ts_tensor_t* out_tensor);
TS_API ts_status_t ts_tensor_add(ts_tensor_t left,
                                 ts_tensor_t right,
                                 ts_tensor_t* out_tensor);
TS_API ts_status_t ts_tensor_multiply(ts_tensor_t left,
                                      ts_tensor_t right,
                                      ts_tensor_t* out_tensor);
TS_API ts_status_t ts_tensor_sum(ts_tensor_t tensor, ts_tensor_t* out_tensor);
TS_API ts_status_t ts_tensor_matmul(ts_tensor_t left,
                                    ts_tensor_t right,
                                    ts_tensor_t* out_tensor);

/* Autograd and selected training operations. */
TS_API ts_status_t ts_tensor_with_requires_grad(ts_tensor_t tensor,
                                                uint8_t requires_grad,
                                                ts_tensor_t* out_tensor);
TS_API ts_status_t ts_tensor_requires_grad(ts_tensor_t tensor,
                                           uint8_t* out_requires_grad);
TS_API ts_status_t ts_tensor_backward(ts_tensor_t tensor);
TS_API ts_status_t ts_tensor_grad(ts_tensor_t tensor,
                                  ts_tensor_t* out_tensor);
TS_API ts_status_t ts_tensor_relu(ts_tensor_t tensor,
                                  ts_tensor_t* out_tensor);
TS_API ts_status_t ts_tensor_sigmoid(ts_tensor_t tensor,
                                     ts_tensor_t* out_tensor);
TS_API ts_status_t ts_tensor_tanh(ts_tensor_t tensor,
                                  ts_tensor_t* out_tensor);
TS_API ts_status_t ts_mse_loss(ts_tensor_t prediction,
                               ts_tensor_t target,
                               ts_tensor_t* out_tensor);
TS_API ts_status_t ts_cross_entropy_loss(ts_tensor_t logits,
                                         ts_tensor_t one_hot_target,
                                         ts_tensor_t* out_tensor);

/* Linear module and generic module operations. */
TS_API ts_status_t ts_linear_create(int64_t in_features,
                                    int64_t out_features,
                                    uint8_t use_bias,
                                    ts_module_t* out_module);
TS_API ts_status_t ts_module_forward(ts_module_t module,
                                     ts_tensor_t input,
                                     ts_tensor_t* out_tensor);
TS_API ts_status_t ts_module_set_training(ts_module_t module,
                                          uint8_t training);
TS_API ts_status_t ts_module_to_device(ts_module_t module,
                                       uint32_t device,
                                       int32_t device_index);
TS_API ts_status_t ts_module_parameter_count(ts_module_t module,
                                             size_t* out_count);
TS_API ts_status_t ts_module_parameter_at(ts_module_t module,
                                          size_t index,
                                          ts_tensor_t* out_tensor);
TS_API ts_status_t ts_module_buffer_count(ts_module_t module,
                                          size_t* out_count);
TS_API ts_status_t ts_module_buffer_at(ts_module_t module,
                                       size_t index,
                                       ts_tensor_t* out_tensor);
TS_API ts_status_t ts_module_save(ts_module_t module, const char* path);
TS_API ts_status_t ts_module_load(ts_module_t module, const char* path);
TS_API ts_status_t ts_module_release(ts_module_t module);

/* Optimizers. */
TS_API ts_status_t ts_sgd_create(ts_module_t module,
                                 double learning_rate,
                                 double momentum,
                                 double weight_decay,
                                 ts_optimizer_t* out_optimizer);
TS_API ts_status_t ts_adam_create(ts_module_t module,
                                  double learning_rate,
                                  double beta1,
                                  double beta2,
                                  double epsilon,
                                  double weight_decay,
                                  ts_optimizer_t* out_optimizer);
TS_API ts_status_t ts_adamw_create(ts_module_t module,
                                   double learning_rate,
                                   double beta1,
                                   double beta2,
                                   double epsilon,
                                   double weight_decay,
                                   ts_optimizer_t* out_optimizer);
TS_API ts_status_t ts_optimizer_zero_grad(ts_optimizer_t optimizer);
TS_API ts_status_t ts_optimizer_step(ts_optimizer_t optimizer);
TS_API ts_status_t ts_optimizer_release(ts_optimizer_t optimizer);

/* Explicit native -> host copy. */
TS_API ts_status_t ts_tensor_copy_to_host_f32(ts_tensor_t tensor,
                                              float* out_values,
                                              size_t capacity,
                                              size_t* out_written);

/* Tensor lifetime. */
TS_API ts_status_t ts_tensor_retain(ts_tensor_t tensor);
TS_API ts_status_t ts_tensor_release(ts_tensor_t tensor);

/* Runtime diagnostics. */
TS_API ts_status_t ts_runtime_live_tensor_count(uint64_t* out_count);
TS_API ts_status_t ts_runtime_live_storage_bytes(uint64_t* out_bytes);
TS_API ts_status_t ts_runtime_live_module_count(uint64_t* out_count);
TS_API ts_status_t ts_runtime_live_optimizer_count(uint64_t* out_count);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* TENSORA_H_ */
