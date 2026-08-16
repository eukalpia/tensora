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

ts_tensor_t Scalar(float value) {
  const int64_t dims[1] = {1};
  ts_tensor_t tensor = 0;
  RequireStatus(ts_tensor_from_f32(&value, 1, dims, 1, &tensor), TS_OK,
                "create scalar");
  return tensor;
}

float ReadScalar(ts_tensor_t tensor) {
  float value = 0.0f;
  size_t written = 0;
  RequireStatus(ts_tensor_copy_to_host_f32(tensor, &value, 1, &written), TS_OK,
                "copy scalar");
  Require(written == 1, "scalar copy writes exactly one value");
  return value;
}

}  // namespace

int main() {
  ts_module_t linear = 0;
  RequireStatus(ts_linear_create(1, 1, 1, &linear), TS_OK,
                "create linear module");
  ts_tensor_t weight_a = 0;
  ts_tensor_t weight_b = 0;
  RequireStatus(ts_module_parameter_at(linear, 0, &weight_a), TS_OK,
                "first weight view");
  RequireStatus(ts_module_parameter_at(linear, 0, &weight_b), TS_OK,
                "second weight view");

  uint64_t identity_a = 0;
  uint64_t identity_b = 0;
  RequireStatus(ts_tensor_identity(weight_a, &identity_a), TS_OK,
                "first identity");
  RequireStatus(ts_tensor_identity(weight_b, &identity_b), TS_OK,
                "second identity");
  Require(identity_a != 0, "opaque identity is non-zero");
  Require(identity_a == identity_b,
          "retained wrappers for one parameter share native identity");

  ts_tensor_t weight_snapshot = 0;
  RequireStatus(ts_tensor_clone_detached(weight_a, &weight_snapshot), TS_OK,
                "clone module parameter state");
  uint64_t weight_snapshot_identity = 0;
  RequireStatus(ts_tensor_identity(weight_snapshot, &weight_snapshot_identity),
                TS_OK, "module parameter snapshot identity");
  Require(weight_snapshot_identity != identity_a,
          "module parameter snapshot owns distinct identity");

  ts_module_t source_linear = 0;
  RequireStatus(ts_linear_create(1, 1, 0, &source_linear), TS_OK,
                "create source linear module");
  ts_tensor_t source_weight = 0;
  RequireStatus(ts_module_parameter_at(source_linear, 0, &source_weight), TS_OK,
                "source weight view");
  const float source_weight_value = ReadScalar(source_weight);
  const ts_tensor_t module_targets[1] = {weight_a};
  const ts_tensor_t module_sources[1] = {source_weight};
  RequireStatus(ts_tensor_assign_many(module_targets, module_sources, 1), TS_OK,
                "assign module parameter state");
  Require(std::fabs(ReadScalar(weight_a) - source_weight_value) < 1e-6f,
          "module parameter state assigned transactionally");

  const float alias_values[2] = {1.0f, 2.0f};
  const int64_t alias_dims[2] = {1, 2};
  ts_tensor_t alias_source = 0;
  RequireStatus(ts_tensor_from_f32(alias_values, 2, alias_dims, 2,
                                   &alias_source),
                TS_OK, "create alias source");
  const int64_t alias_view_dims[1] = {2};
  ts_tensor_t alias_view = 0;
  RequireStatus(ts_tensor_reshape(alias_source, alias_view_dims, 1,
                                  &alias_view),
                TS_OK, "create zero-copy alias view");
  uint64_t source_identity = 0;
  uint64_t view_identity = 0;
  RequireStatus(ts_tensor_identity(alias_source, &source_identity), TS_OK,
                "source alias identity");
  RequireStatus(ts_tensor_identity(alias_view, &view_identity), TS_OK,
                "view alias identity");
  Require(source_identity == view_identity,
          "zero-copy aliases share one opaque identity lineage");

  ts_tensor_t alias_clone = 0;
  RequireStatus(ts_tensor_clone_detached(alias_source, &alias_clone), TS_OK,
                "clone alias source");
  uint64_t clone_identity = 0;
  RequireStatus(ts_tensor_identity(alias_clone, &clone_identity), TS_OK,
                "detached clone identity");
  Require(clone_identity != source_identity,
          "detached clone receives a distinct opaque identity");

  ts_tensor_t original = Scalar(3.0f);
  ts_tensor_t snapshot = 0;
  RequireStatus(ts_tensor_clone_detached(original, &snapshot), TS_OK,
                "clone detached state");
  uint8_t snapshot_requires_grad = 99;
  RequireStatus(ts_tensor_requires_grad(snapshot, &snapshot_requires_grad),
                TS_OK, "snapshot requires-grad query");
  Require(snapshot_requires_grad == 0, "state snapshot is detached");

  ts_tensor_t target_a = Scalar(1.0f);
  ts_tensor_t target_b = Scalar(2.0f);
  ts_tensor_t source_a = Scalar(5.0f);
  const int64_t wrong_dims[2] = {1, 2};
  const float wrong_values[2] = {7.0f, 8.0f};
  ts_tensor_t wrong_source = 0;
  RequireStatus(
      ts_tensor_from_f32(wrong_values, 2, wrong_dims, 2, &wrong_source), TS_OK,
      "create wrong-shape source");

  const ts_tensor_t failing_targets[2] = {target_a, target_b};
  const ts_tensor_t failing_sources[2] = {source_a, wrong_source};
  RequireStatus(ts_tensor_assign_many(failing_targets, failing_sources, 2),
                TS_INVALID_SHAPE, "batch assignment validates before mutation");
  Require(std::fabs(ReadScalar(target_a) - 1.0f) < 1e-6f,
          "failed batch leaves first target unchanged");
  Require(std::fabs(ReadScalar(target_b) - 2.0f) < 1e-6f,
          "failed batch leaves second target unchanged");

  ts_tensor_t source_b = Scalar(6.0f);
  const ts_tensor_t valid_sources[2] = {source_a, source_b};
  RequireStatus(ts_tensor_assign_many(failing_targets, valid_sources, 2), TS_OK,
                "valid batch assignment");
  Require(std::fabs(ReadScalar(target_a) - 5.0f) < 1e-6f,
          "first target assigned");
  Require(std::fabs(ReadScalar(target_b) - 6.0f) < 1e-6f,
          "second target assigned");
  Require(std::fabs(ReadScalar(snapshot) - 3.0f) < 1e-6f,
          "detached snapshot remains independent");

  const ts_tensor_t duplicate_targets[2] = {target_a, target_a};
  RequireStatus(ts_tensor_assign_many(duplicate_targets, valid_sources, 2),
                TS_INVALID_ARGUMENT, "duplicate assignment target rejected");

  // In a core-only runtime both pairs are CPU-backed and the operation is
  // valid. In a LibTorch-enabled runtime module parameters are Torch-backed,
  // so a single transaction spanning the CPU scalar pair and Torch parameter
  // pair must be rejected before mutation.
  const ts_tensor_t mixed_targets[2] = {target_a, weight_a};
  const ts_tensor_t mixed_sources[2] = {source_a, source_weight};
  const ts_status_t mixed_status =
      ts_tensor_assign_many(mixed_targets, mixed_sources, 2);
  Require(mixed_status == TS_OK || mixed_status == TS_INVALID_ARGUMENT,
          "mixed-backend transaction has deterministic provider policy");

  uint64_t invalid_identity = 0;
  RequireStatus(ts_tensor_identity(UINT64_C(0xffffffffffffffff),
                                   &invalid_identity),
                TS_INVALID_HANDLE, "invalid identity handle rejected");

  RequireStatus(ts_tensor_release(weight_snapshot), TS_OK,
                "release weight snapshot");
  RequireStatus(ts_tensor_release(source_weight), TS_OK,
                "release source weight");
  RequireStatus(ts_module_release(source_linear), TS_OK,
                "release source linear module");
  RequireStatus(ts_tensor_release(weight_a), TS_OK, "release weight view a");
  RequireStatus(ts_tensor_release(weight_b), TS_OK, "release weight view b");
  RequireStatus(ts_module_release(linear), TS_OK, "release linear module");
  RequireStatus(ts_tensor_release(alias_clone), TS_OK, "release alias clone");
  RequireStatus(ts_tensor_release(alias_view), TS_OK, "release alias view");
  RequireStatus(ts_tensor_release(alias_source), TS_OK, "release alias source");
  RequireStatus(ts_tensor_release(original), TS_OK, "release original");
  RequireStatus(ts_tensor_release(snapshot), TS_OK, "release snapshot");
  RequireStatus(ts_tensor_release(target_a), TS_OK, "release target a");
  RequireStatus(ts_tensor_release(target_b), TS_OK, "release target b");
  RequireStatus(ts_tensor_release(source_a), TS_OK, "release source a");
  RequireStatus(ts_tensor_release(source_b), TS_OK, "release source b");
  RequireStatus(ts_tensor_release(wrong_source), TS_OK,
                "release wrong-shape source");

  std::puts("NN V2 state identity/transaction contract passed");
  return 0;
}
