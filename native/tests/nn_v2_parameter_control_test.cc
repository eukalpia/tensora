#include "tensora.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

void Require(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\nlast_error=%s\n", message,
                 ts_last_error_message());
    std::exit(1);
  }
}

void RequireStatus(ts_status_t actual,
                   ts_status_t expected,
                   const char* message) {
  if (actual != expected) {
    std::fprintf(stderr,
                 "FAIL: %s expected=%d actual=%d last_error=%s\n",
                 message, static_cast<int>(expected), static_cast<int>(actual),
                 ts_last_error_message());
    std::exit(1);
  }
}

void Release(ts_tensor_t tensor) {
  if (tensor != 0) {
    RequireStatus(ts_tensor_release(tensor), TS_OK, "release tensor");
  }
}

}  // namespace

int main() {
  const float value = 2.0f;
  const int64_t dims[1] = {1};
  ts_tensor_t parameter = 0;
  RequireStatus(ts_tensor_from_f32(&value, 1, dims, 1, &parameter), TS_OK,
                "create parameter");

  uint8_t requires_grad = 99;
  RequireStatus(ts_tensor_requires_grad(parameter, &requires_grad), TS_OK,
                "initial requiresGrad query");
  Require(requires_grad == 0, "fresh tensor is frozen by default");

  RequireStatus(ts_tensor_set_requires_grad(parameter, 0), TS_OK,
                "freezing tensor without autograd metadata is idempotent");
  RequireStatus(ts_tensor_set_requires_grad(parameter, 1), TS_OK,
                "unfreeze creates trainable leaf metadata");
  RequireStatus(ts_tensor_requires_grad(parameter, &requires_grad), TS_OK,
                "unfrozen requiresGrad query");
  Require(requires_grad == 1, "unfreeze enables requiresGrad");

  ts_tensor_t loss = 0;
  RequireStatus(ts_tensor_sum(parameter, &loss), TS_OK,
                "create scalar parameter loss");
  RequireStatus(ts_tensor_backward(loss), TS_OK,
                "seed parameter gradient");
  ts_tensor_t gradient = 0;
  RequireStatus(ts_tensor_grad(parameter, &gradient), TS_OK,
                "gradient exists before freeze");
  Release(gradient);
  gradient = 123;

  RequireStatus(ts_tensor_set_requires_grad(parameter, 0), TS_OK,
                "freeze trainable leaf");
  RequireStatus(ts_tensor_requires_grad(parameter, &requires_grad), TS_OK,
                "frozen requiresGrad query");
  Require(requires_grad == 0, "freeze disables requiresGrad");
  RequireStatus(ts_tensor_grad(parameter, &gradient), TS_INVALID_ARGUMENT,
                "freeze clears existing gradient");
  Require(gradient == 0, "failed gradient query clears output handle");

  RequireStatus(ts_tensor_set_requires_grad(parameter, 1), TS_OK,
                "unfreeze existing leaf");
  RequireStatus(ts_tensor_requires_grad(parameter, &requires_grad), TS_OK,
                "re-unfrozen requiresGrad query");
  Require(requires_grad == 1, "unfreeze restores requiresGrad");

  ts_tensor_t doubled = 0;
  RequireStatus(ts_tensor_add(parameter, parameter, &doubled), TS_OK,
                "create non-leaf tensor");
  RequireStatus(ts_tensor_set_requires_grad(doubled, 0), TS_INVALID_ARGUMENT,
                "non-leaf freeze is rejected");
  RequireStatus(ts_tensor_set_requires_grad(doubled, 1), TS_INVALID_ARGUMENT,
                "non-leaf unfreeze is rejected");

  RequireStatus(ts_tensor_set_requires_grad(UINT64_C(0xffffffffffffffff), 1),
                TS_INVALID_HANDLE, "invalid parameter handle is rejected");

  Release(doubled);
  Release(loss);
  Release(parameter);
  std::puts("NN V2 parameter freeze/unfreeze contract passed");
  return 0;
}
