#include "tensora.h"

#include <stddef.h>
#include <stdint.h>

// The NN V2 coverage fixture is linked with dart_contract_fixture.c after the
// two symbols below are preprocessor-renamed. Keep all of the base fixture's
// behavior for modes 0/1, while reserving mode 2 for a stable two-parameter
// module whose second Tensor has valid metadata. This lets the Dart layer test
// a post-move parameter-count contract violation without conflating it with the
// separate malformed-metadata contract represented by mode 1.
void ts_fixture_base_set_training_mode(int32_t mode);
ts_status_t ts_fixture_base_module_parameter_count(ts_module_t module,
                                                    size_t* out_count);

#undef ts_test_set_training_mode
#undef ts_module_parameter_count

static int32_t g_nn_v2_training_mode = 0;

TS_API void ts_test_set_training_mode(int32_t mode) {
  g_nn_v2_training_mode = mode;
  ts_fixture_base_set_training_mode(mode);
}

ts_status_t ts_module_parameter_count(ts_module_t module, size_t* out_count) {
  if (g_nn_v2_training_mode == 2) {
    (void)module;
    if (out_count == NULL) {
      return TS_INVALID_ARGUMENT;
    }
    *out_count = 2;
    return TS_OK;
  }
  return ts_fixture_base_module_parameter_count(module, out_count);
}

static uint64_t g_nn_v2_next_handle = UINT64_C(900000);

static ts_status_t nn_v2_unary_tensor(ts_tensor_t tensor,
                                      ts_tensor_t* out_tensor) {
  (void)tensor;
  if (out_tensor == NULL) {
    return TS_INVALID_ARGUMENT;
  }
  *out_tensor = g_nn_v2_next_handle++;
  return TS_OK;
}

static ts_status_t nn_v2_optimizer_create(const ts_tensor_t* parameters,
                                          size_t count,
                                          ts_optimizer_t* out_optimizer) {
  if (parameters == NULL || count == 0 || out_optimizer == NULL) {
    if (out_optimizer != NULL) {
      *out_optimizer = 0;
    }
    return TS_INVALID_ARGUMENT;
  }
  *out_optimizer = g_nn_v2_next_handle++;
  return TS_OK;
}

ts_status_t ts_tensor_identity(ts_tensor_t tensor, uint64_t* out_identity) {
  if (tensor == 0 || out_identity == NULL) {
    return TS_INVALID_ARGUMENT;
  }
  *out_identity = tensor + UINT64_C(1000000);
  return TS_OK;
}

ts_status_t ts_tensor_clone_detached(ts_tensor_t tensor,
                                     ts_tensor_t* out_tensor) {
  return nn_v2_unary_tensor(tensor, out_tensor);
}

ts_status_t ts_tensor_assign_many(const ts_tensor_t* targets,
                                  const ts_tensor_t* sources,
                                  size_t count) {
  if (count == 0) {
    return TS_OK;
  }
  if (targets == NULL || sources == NULL) {
    return TS_INVALID_ARGUMENT;
  }
  return TS_OK;
}

ts_status_t ts_tensor_set_requires_grad(ts_tensor_t tensor,
                                        uint8_t requires_grad) {
  (void)requires_grad;
  return tensor == 0 ? TS_INVALID_HANDLE : TS_OK;
}

ts_status_t ts_tensor_gelu(ts_tensor_t tensor, ts_tensor_t* out_tensor) {
  return nn_v2_unary_tensor(tensor, out_tensor);
}

ts_status_t ts_tensor_silu(ts_tensor_t tensor, ts_tensor_t* out_tensor) {
  return nn_v2_unary_tensor(tensor, out_tensor);
}

ts_status_t ts_tensor_swiglu(ts_tensor_t tensor, ts_tensor_t* out_tensor) {
  return nn_v2_unary_tensor(tensor, out_tensor);
}

ts_status_t ts_sgd_create_for_tensors(const ts_tensor_t* parameters,
                                      size_t count,
                                      double learning_rate,
                                      double momentum,
                                      double weight_decay,
                                      ts_optimizer_t* out_optimizer) {
  (void)learning_rate;
  (void)momentum;
  (void)weight_decay;
  return nn_v2_optimizer_create(parameters, count, out_optimizer);
}

ts_status_t ts_adam_create_for_tensors(const ts_tensor_t* parameters,
                                       size_t count,
                                       double learning_rate,
                                       double beta1,
                                       double beta2,
                                       double epsilon,
                                       double weight_decay,
                                       ts_optimizer_t* out_optimizer) {
  (void)learning_rate;
  (void)beta1;
  (void)beta2;
  (void)epsilon;
  (void)weight_decay;
  return nn_v2_optimizer_create(parameters, count, out_optimizer);
}

ts_status_t ts_adamw_create_for_tensors(const ts_tensor_t* parameters,
                                        size_t count,
                                        double learning_rate,
                                        double beta1,
                                        double beta2,
                                        double epsilon,
                                        double weight_decay,
                                        ts_optimizer_t* out_optimizer) {
  return ts_adam_create_for_tensors(parameters, count, learning_rate, beta1,
                                    beta2, epsilon, weight_decay,
                                    out_optimizer);
}

ts_status_t ts_parameter_optimizer_zero_grad(ts_optimizer_t optimizer) {
  (void)optimizer;
  return TS_OK;
}

ts_status_t ts_parameter_optimizer_step(ts_optimizer_t optimizer) {
  (void)optimizer;
  return TS_OK;
}

ts_status_t ts_parameter_optimizer_release(ts_optimizer_t optimizer) {
  (void)optimizer;
  return TS_OK;
}
