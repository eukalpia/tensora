#include "tensora.h"

#include <cmath>
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

ts_tensor_t MakeLeaf(float value) {
  const int64_t dims[1] = {1};
  ts_tensor_t raw = 0;
  RequireStatus(ts_tensor_from_f32(&value, 1, dims, 1, &raw), TS_OK,
                "create raw parameter");
  ts_tensor_t leaf = 0;
  RequireStatus(ts_tensor_with_requires_grad(raw, 1, &leaf), TS_OK,
                "create trainable leaf");
  RequireStatus(ts_tensor_release(raw), TS_OK, "release raw parameter");
  return leaf;
}

float ReadScalar(ts_tensor_t tensor) {
  float value = 0.0f;
  size_t written = 0;
  RequireStatus(ts_tensor_copy_to_host_f32(tensor, &value, 1, &written), TS_OK,
                "copy scalar");
  Require(written == 1, "scalar copy writes exactly one value");
  return value;
}

void SeedUnitGradient(ts_tensor_t parameter) {
  ts_tensor_t loss = 0;
  RequireStatus(ts_tensor_sum(parameter, &loss), TS_OK, "sum parameter");
  RequireStatus(ts_tensor_backward(loss), TS_OK, "backward parameter sum");
  RequireStatus(ts_tensor_release(loss), TS_OK, "release scalar loss");
}

}  // namespace

int main() {
  ts_optimizer_t optimizer = 123;
  RequireStatus(ts_sgd_create_for_tensors(nullptr, 0, 0.1, 0.0, 0.0,
                                          &optimizer),
                TS_INVALID_ARGUMENT, "empty SGD collection rejected");
  Require(optimizer == 0, "failed optimizer creation clears output handle");

  ts_tensor_t first = MakeLeaf(1.0f);
  ts_tensor_t second = MakeLeaf(3.0f);
  SeedUnitGradient(first);
  SeedUnitGradient(second);

  const ts_tensor_t parameters[2] = {first, second};
  RequireStatus(ts_sgd_create_for_tensors(parameters, 2, 0.1, 0.0, 0.0,
                                          &optimizer),
                TS_OK, "create SGD over arbitrary tensors");
  Require(optimizer != 0, "SGD returns a handle");

  RequireStatus(ts_tensor_release(first), TS_OK,
                "optimizer retains first parameter after handle release");
  RequireStatus(ts_parameter_optimizer_step(optimizer), TS_OK,
                "step retained parameter collection");

  Require(std::fabs(ReadScalar(second) - 2.9f) < 1e-5f,
          "second parameter updated by SGD");
  RequireStatus(ts_parameter_optimizer_zero_grad(optimizer), TS_OK,
                "zero gradients for parameter collection");
  RequireStatus(ts_parameter_optimizer_release(optimizer), TS_OK,
                "release parameter optimizer");

  const ts_tensor_t duplicate[2] = {second, second};
  optimizer = 999;
  RequireStatus(ts_sgd_create_for_tensors(duplicate, 2, 0.1, 0.0, 0.0,
                                          &optimizer),
                TS_INVALID_ARGUMENT, "duplicate tensor rejected");
  Require(optimizer == 0, "duplicate failure clears output handle");

  ts_tensor_t freeze_probe = MakeLeaf(7.0f);
  uint64_t identity_before = 0;
  uint64_t identity_after = 0;
  RequireStatus(ts_tensor_identity(freeze_probe, &identity_before), TS_OK,
                "identity before freeze");
  RequireStatus(ts_tensor_set_requires_grad(freeze_probe, 0), TS_OK,
                "freeze leaf parameter");
  uint8_t requires_grad = 1;
  RequireStatus(ts_tensor_requires_grad(freeze_probe, &requires_grad), TS_OK,
                "query frozen parameter");
  Require(requires_grad == 0, "freeze clears requiresGrad");
  RequireStatus(ts_tensor_identity(freeze_probe, &identity_after), TS_OK,
                "identity after freeze");
  Require(identity_before == identity_after,
          "freeze preserves opaque parameter identity");

  const ts_tensor_t all_frozen[1] = {freeze_probe};
  RequireStatus(ts_adam_create_for_tensors(all_frozen, 1, 0.001, 0.9, 0.999,
                                           1e-8, 0.0, &optimizer),
                TS_INVALID_ARGUMENT, "all-frozen collection rejected");

  const ts_tensor_t mixed[2] = {second, freeze_probe};
  RequireStatus(ts_adamw_create_for_tensors(mixed, 2, 0.001, 0.9, 0.999, 1e-8,
                                            0.01, &optimizer),
                TS_OK, "frozen tensor is skipped when trainable tensors exist");
  RequireStatus(ts_parameter_optimizer_release(optimizer), TS_OK,
                "release AdamW parameter optimizer");

  RequireStatus(ts_tensor_set_requires_grad(freeze_probe, 1), TS_OK,
                "unfreeze leaf parameter");
  RequireStatus(ts_tensor_requires_grad(freeze_probe, &requires_grad), TS_OK,
                "query unfrozen parameter");
  Require(requires_grad == 1, "unfreeze restores requiresGrad");
  RequireStatus(ts_adam_create_for_tensors(all_frozen, 1, 0.001, 0.9, 0.999,
                                           1e-8, 0.0, &optimizer),
                TS_OK, "unfrozen parameter is optimizer eligible");
  RequireStatus(ts_parameter_optimizer_release(optimizer), TS_OK,
                "release Adam parameter optimizer");

  SeedUnitGradient(freeze_probe);
  ts_tensor_t non_leaf = 0;
  RequireStatus(ts_tensor_sum(freeze_probe, &non_leaf), TS_OK,
                "create non-leaf tensor");
  RequireStatus(ts_tensor_set_requires_grad(non_leaf, 0), TS_INVALID_ARGUMENT,
                "non-leaf requiresGrad mutation rejected");
  RequireStatus(ts_tensor_release(non_leaf), TS_OK, "release non-leaf tensor");

  const ts_tensor_t invalid[1] = {UINT64_C(0xffffffffffffffff)};
  optimizer = 0;
  RequireStatus(ts_sgd_create_for_tensors(invalid, 1, 0.1, 0.0, 0.0,
                                          &optimizer),
                TS_INVALID_HANDLE, "invalid tensor handle rejected");
  RequireStatus(ts_tensor_set_requires_grad(UINT64_C(0xffffffffffffffff), 0),
                TS_INVALID_HANDLE, "invalid freeze handle rejected");

  RequireStatus(ts_tensor_release(second), TS_OK, "release second parameter");
  RequireStatus(ts_tensor_release(freeze_probe), TS_OK,
                "release freeze probe");

  std::puts("NN V2 optimizer parameter-collection contract passed");
  return 0;
}
