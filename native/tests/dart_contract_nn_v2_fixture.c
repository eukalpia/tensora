#include "tensora.h"

#include <stddef.h>
#include <stdint.h>

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

ts_status_t ts_tensor_gelu(ts_tensor_t tensor, ts_tensor_t* out_tensor) {
  return nn_v2_unary_tensor(tensor, out_tensor);
}

ts_status_t ts_tensor_silu(ts_tensor_t tensor, ts_tensor_t* out_tensor) {
  return nn_v2_unary_tensor(tensor, out_tensor);
}

ts_status_t ts_tensor_swiglu(ts_tensor_t tensor, ts_tensor_t* out_tensor) {
  return nn_v2_unary_tensor(tensor, out_tensor);
}
