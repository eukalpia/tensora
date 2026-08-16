#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <vector>

#include "../src/memory/tensor_storage.h"
#include "../src/training/training_c_api.cc"

namespace {

void Require(bool condition) {
  if (!condition) std::abort();
}

void RequireCode(const tensora::Status& status, int32_t code) {
  Require(status.code() == code);
}

void RequireStatus(ts_status_t status, int32_t code) {
  Require(status == code);
}

ts_tensor_t MakeTensor(const std::vector<float>& values,
                       const std::vector<int64_t>& shape) {
  ts_tensor_t handle = 0;
  const int64_t* dimensions = shape.empty() ? nullptr : shape.data();
  RequireStatus(ts_tensor_from_f32(values.data(), values.size(), dimensions,
                                   shape.size(), &handle),
                TS_OK);
  Require(handle != 0);
  return handle;
}

std::shared_ptr<tensora::Tensor> LookupTensor(ts_tensor_t handle) {
  std::shared_ptr<tensora::Tensor> tensor;
  RequireCode(tensora::LookupTrainingTensor(handle, &tensor), TS_OK);
  Require(tensor != nullptr);
  return tensor;
}

ts_tensor_t RequireGrad(ts_tensor_t source) {
  ts_tensor_t result = 0;
  RequireStatus(ts_tensor_with_requires_grad(source, 1, &result), TS_OK);
  return result;
}

void SeedGradient(ts_tensor_t parameter) {
  ts_tensor_t loss = 0;
  RequireStatus(ts_tensor_sum(parameter, &loss), TS_OK);
  RequireStatus(ts_tensor_backward(loss), TS_OK);
  RequireStatus(ts_tensor_release(loss), TS_OK);
}

class UnsupportedStorage final : public tensora::TensorStorage {
 public:
  tensora::StorageKind kind() const override {
    return static_cast<tensora::StorageKind>(99);
  }

  tensora::Status CopyToHostF32(float*, size_t, size_t* out_written) const override {
    if (out_written != nullptr) *out_written = 0;
    return tensora::Unsupported("test storage has no host copy");
  }

  uint64_t byte_size() const override { return sizeof(float); }
};

ts_tensor_t InsertTensor(std::shared_ptr<tensora::Tensor> tensor) {
  ts_tensor_t handle = 0;
  RequireCode(tensora::HandleRegistry::Instance().Insert(
                  tensora::HandleType::kTensor, std::move(tensor), &handle),
              TS_OK);
  return handle;
}

std::shared_ptr<tensora::Tensor> MakeUnsupportedTensor(
    const tensora::ShapeInfo& shape, bool requires_grad) {
  auto storage = std::make_shared<UnsupportedStorage>();
  auto tensor = std::make_shared<tensora::Tensor>(shape, std::move(storage));
  if (requires_grad) {
    auto meta = std::make_shared<tensora::AutogradMeta>();
    meta->requires_grad = true;
    meta->is_leaf = true;
    tensor->set_autograd_meta(std::move(meta));
  }
  return tensor;
}

void TestRuntimeNullOutputs(const tensora::Tensor& tensor) {
  RequireCode(tensora::training::nn_v2::Gelu(tensor, nullptr),
              TS_INVALID_ARGUMENT);
  RequireCode(tensora::training::nn_v2::Silu(tensor, nullptr),
              TS_INVALID_ARGUMENT);
  RequireCode(tensora::training::nn_v2::SwiGlu(tensor, nullptr),
              TS_INVALID_ARGUMENT);
}

void TestParameterControl(ts_tensor_t raw) {
  auto tensor = LookupTensor(raw);
  RequireCode(tensora::training::nn_v2_parameter_control::SetRequiresGrad(
                  *tensor, false),
              TS_OK);
  RequireCode(tensora::training::nn_v2_parameter_control::SetRequiresGrad(
                  *tensor, true),
              TS_OK);
  Require(tensor->autograd_meta() != nullptr);

  SeedGradient(raw);
  RequireCode(tensora::training::nn_v2_parameter_control::SetRequiresGrad(
                  *tensor, false),
              TS_OK);
  Require(tensor->autograd_meta()->gradient == nullptr);
  RequireCode(tensora::training::nn_v2_parameter_control::SetRequiresGrad(
                  *tensor, true),
              TS_OK);

  ts_tensor_t doubled = 0;
  RequireStatus(ts_tensor_add(raw, raw, &doubled), TS_OK);
  auto non_leaf = LookupTensor(doubled);
  RequireCode(tensora::training::nn_v2_parameter_control::SetRequiresGrad(
                  *non_leaf, false),
              TS_INVALID_ARGUMENT);
  RequireStatus(ts_tensor_release(doubled), TS_OK);
}

void TestOptimizerValidation() {
  using namespace tensora::training::nn_v2_optimizer;
  RequireCode(ValidateLearningRate(0.0, "coverage"), TS_INVALID_ARGUMENT);
  RequireCode(ValidateLearningRate(std::numeric_limits<double>::infinity(),
                                   "coverage"),
              TS_INVALID_ARGUMENT);
  RequireCode(ValidateWeightDecay(-1.0, "coverage"), TS_INVALID_ARGUMENT);
  RequireCode(ValidateMomentum(-1.0, "coverage"), TS_INVALID_ARGUMENT);
  RequireCode(ValidateBetas(1.0, 0.9, "coverage"), TS_INVALID_ARGUMENT);
  RequireCode(ValidateBetas(0.9, -1.0, "coverage"), TS_INVALID_ARGUMENT);
  RequireCode(ValidateEpsilon(0.0, "coverage"), TS_INVALID_ARGUMENT);

  tensora::StorageKind kind = tensora::StorageKind::kCpu;
  std::vector<std::shared_ptr<tensora::Tensor>> output;
  RequireCode(CollectParameters(nullptr, 0, "coverage", nullptr, &kind),
              TS_INVALID_ARGUMENT);
  RequireCode(CollectParameters(nullptr, 0, "coverage", &output, nullptr),
              TS_INVALID_ARGUMENT);
  RequireCode(CollectParameters(nullptr, 0, "coverage", &output, &kind),
              TS_INVALID_ARGUMENT);
  RequireCode(InitializeCpuMoments(nullptr, "coverage"), TS_INVALID_ARGUMENT);
  RequireCode(Create(nullptr, 0, Kind::kSgd, 0.1, 0.0, 0.0, 0.0, 0.0,
                     0.0, nullptr, "coverage"),
              TS_INVALID_ARGUMENT);
}

void TestOptimizerAlgorithms() {
  auto run = [](tensora::training::nn_v2_optimizer::Kind kind,
                double momentum, double weight_decay) {
    ts_tensor_t raw = MakeTensor({2.0f}, {1});
    ts_tensor_t parameter = RequireGrad(raw);
    RequireStatus(ts_tensor_release(raw), TS_OK);
    SeedGradient(parameter);

    ts_optimizer_t optimizer = 0;
    if (kind == tensora::training::nn_v2_optimizer::Kind::kSgd) {
      RequireStatus(ts_sgd_create_for_tensors(&parameter, 1, 0.1, momentum,
                                              weight_decay, &optimizer),
                    TS_OK);
    } else if (kind == tensora::training::nn_v2_optimizer::Kind::kAdam) {
      RequireStatus(ts_adam_create_for_tensors(&parameter, 1, 0.01, 0.9,
                                               0.999, 1e-8, weight_decay,
                                               &optimizer),
                    TS_OK);
    } else {
      RequireStatus(ts_adamw_create_for_tensors(&parameter, 1, 0.01, 0.9,
                                                0.999, 1e-8, weight_decay,
                                                &optimizer),
                    TS_OK);
    }
    RequireStatus(ts_parameter_optimizer_step(optimizer), TS_OK);
    RequireStatus(ts_parameter_optimizer_zero_grad(optimizer), TS_OK);
    RequireStatus(ts_parameter_optimizer_release(optimizer), TS_OK);
    RequireStatus(ts_tensor_release(parameter), TS_OK);
  };

  run(tensora::training::nn_v2_optimizer::Kind::kSgd, 0.9, 0.1);
  run(tensora::training::nn_v2_optimizer::Kind::kAdam, 0.0, 0.1);
  run(tensora::training::nn_v2_optimizer::Kind::kAdamW, 0.0, 0.1);
}

void TestOptimizerDefensiveStates(const tensora::ShapeInfo& shape) {
  using namespace tensora::training::nn_v2_optimizer;

  auto no_meta = MakeUnsupportedTensor(shape, false);
  State no_meta_state;
  no_meta_state.kind = Kind::kSgd;
  no_meta_state.parameters = {no_meta};
  no_meta_state.first_moment = {{0.0f}};
  no_meta_state.second_moment = {{0.0f}};
  RequireCode(CpuStep(&no_meta_state), TS_OK);

  ts_tensor_t raw = MakeTensor({1.0f}, {1});
  ts_tensor_t parameter_handle = RequireGrad(raw);
  RequireStatus(ts_tensor_release(raw), TS_OK);
  auto parameter = LookupTensor(parameter_handle);

  State no_gradient_state;
  no_gradient_state.kind = Kind::kSgd;
  no_gradient_state.learning_rate = 0.1;
  no_gradient_state.parameters = {parameter};
  no_gradient_state.first_moment = {{0.0f}};
  no_gradient_state.second_moment = {{0.0f}};
  RequireCode(CpuStep(&no_gradient_state), TS_OK);

  ts_tensor_t wrong_gradient_handle = MakeTensor({1.0f, 2.0f}, {2});
  auto wrong_gradient = LookupTensor(wrong_gradient_handle);
  {
    auto meta = parameter->autograd_meta();
    std::lock_guard<std::mutex> lock(meta->mutex);
    meta->gradient = wrong_gradient;
  }
  State malformed_state;
  malformed_state.kind = Kind::kSgd;
  malformed_state.learning_rate = 0.1;
  malformed_state.parameters = {parameter};
  malformed_state.first_moment = {{0.0f}};
  malformed_state.second_moment = {{0.0f}};
  RequireCode(CpuStep(&malformed_state), TS_INTERNAL_ERROR);

  auto unsupported = MakeUnsupportedTensor(shape, true);
  ts_tensor_t unsupported_handle = InsertTensor(unsupported);
  ts_optimizer_t optimizer = 0;
  RequireCode(Create(&unsupported_handle, 1, Kind::kSgd, 0.1, 0.0, 0.0,
                     0.0, 0.0, 0.0, &optimizer, "coverage"),
              TS_UNSUPPORTED);

  std::vector<std::shared_ptr<tensora::Tensor>> collected;
  tensora::StorageKind collected_kind = tensora::StorageKind::kCpu;
  const ts_tensor_t mixed[2] = {parameter_handle, unsupported_handle};
  RequireCode(CollectParameters(mixed, 2, "coverage", &collected,
                                &collected_kind),
              TS_INVALID_ARGUMENT);

  RequireStatus(ts_tensor_release(unsupported_handle), TS_OK);
  RequireStatus(ts_tensor_release(wrong_gradient_handle), TS_OK);
  {
    auto meta = parameter->autograd_meta();
    std::lock_guard<std::mutex> lock(meta->mutex);
    meta->gradient.reset();
  }
  RequireStatus(ts_tensor_release(parameter_handle), TS_OK);
}

void TestStateDefensivePaths(const tensora::ShapeInfo& shape,
                             const std::shared_ptr<tensora::Tensor>& cpu) {
  using namespace tensora::training::nn_v2_state;

  RequireCode(TensorIdentity(*cpu, nullptr), TS_INVALID_ARGUMENT);
  RequireCode(CloneDetached(*cpu, nullptr), TS_INVALID_ARGUMENT);
  RequireCode(AssignMany(nullptr, nullptr, 0), TS_OK);
  RequireCode(AssignMany(nullptr, nullptr, 1), TS_INVALID_ARGUMENT);

  auto wrong_dtype = std::make_shared<tensora::Tensor>(
      shape, cpu->storage(), static_cast<tensora::DType>(99),
      tensora::Device::kCpu, 0);
  RequireCode(internal::ValidatePair(*cpu, *wrong_dtype, "coverage"),
              TS_INVALID_ARGUMENT);

  auto wrong_device = std::make_shared<tensora::Tensor>(
      shape, cpu->storage(), tensora::DType::kFloat32,
      tensora::Device::kCuda, 0);
  RequireCode(internal::ValidatePair(*cpu, *wrong_device, "coverage"),
              TS_INVALID_ARGUMENT);

  auto unsupported_a = MakeUnsupportedTensor(shape, false);
  auto unsupported_b = MakeUnsupportedTensor(shape, false);
  RequireCode(internal::ValidatePair(*cpu, *unsupported_a, "coverage"),
              TS_INVALID_ARGUMENT);

  auto zero_counter = std::make_shared<tensora::TensorVersionCounter>(
      std::make_shared<tensora::TensorIdentityAnchor>(0));
  auto zero_identity = std::make_shared<tensora::Tensor>(
      shape, cpu->storage(), tensora::DType::kFloat32,
      tensora::Device::kCpu, 0, 0, zero_counter);
  uint64_t identity = 99;
  RequireCode(TensorIdentity(*zero_identity, &identity), TS_INTERNAL_ERROR);
  Require(identity == 0);

  ts_tensor_t unsupported_a_handle = InsertTensor(unsupported_a);
  ts_tensor_t unsupported_b_handle = InsertTensor(unsupported_b);
  const ts_tensor_t unsupported_targets[1] = {unsupported_a_handle};
  const ts_tensor_t unsupported_sources[1] = {unsupported_b_handle};
  RequireCode(AssignMany(unsupported_targets, unsupported_sources, 1),
              TS_UNSUPPORTED);

  ts_tensor_t cpu_target = MakeTensor({1.0f}, {1});
  ts_tensor_t cpu_source = MakeTensor({2.0f}, {1});
  const ts_tensor_t mixed_targets[2] = {cpu_target, unsupported_a_handle};
  const ts_tensor_t mixed_sources[2] = {cpu_source, unsupported_b_handle};
  RequireCode(AssignMany(mixed_targets, mixed_sources, 2),
              TS_INVALID_ARGUMENT);

  RequireStatus(ts_tensor_release(cpu_target), TS_OK);
  RequireStatus(ts_tensor_release(cpu_source), TS_OK);
  RequireStatus(ts_tensor_release(unsupported_a_handle), TS_OK);
  RequireStatus(ts_tensor_release(unsupported_b_handle), TS_OK);
}

void TestSwiGluMalformedBackward(const std::shared_ptr<tensora::Tensor>& scalar,
                                 const std::shared_ptr<tensora::Tensor>& odd,
                                 const std::shared_ptr<tensora::Tensor>& even) {
  using tensora::autograd::GradNode;
  using tensora::autograd::Operation;

  GradNode no_parent;
  no_parent.operation = Operation::kSwiGlu;
  std::vector<tensora::autograd::GradientContribution> contributions;
  RequireCode(tensora::autograd::ApplyNode(no_parent, *scalar, &contributions),
              TS_INTERNAL_ERROR);

  auto run = [&](const std::shared_ptr<tensora::Tensor>& parent,
                 const tensora::Tensor& upstream, int32_t expected) {
    GradNode node;
    node.operation = Operation::kSwiGlu;
    node.parents = {parent};
    node.parent_versions = {tensora::autograd::Version(*parent)};
    RequireCode(tensora::autograd::ApplyNode(node, upstream, &contributions),
                expected);
  };

  run(scalar, *scalar, TS_INTERNAL_ERROR);
  run(odd, *scalar, TS_INTERNAL_ERROR);
  run(even, *scalar, TS_INTERNAL_ERROR);
}

}  // namespace

int main() {
  const tensora::Status original =
      tensora::InsertTrainingTensor(nullptr, nullptr);
  RequireCode(original, TS_INVALID_ARGUMENT);

  ts_tensor_t scalar_handle = MakeTensor({1.0f}, {1});
  auto scalar = LookupTensor(scalar_handle);
  TestRuntimeNullOutputs(*scalar);
  TestParameterControl(scalar_handle);
  TestOptimizerValidation();
  TestOptimizerAlgorithms();
  TestOptimizerDefensiveStates(scalar->shape());
  TestStateDefensivePaths(scalar->shape(), scalar);

  ts_tensor_t rank_zero_handle = MakeTensor({1.0f}, {});
  ts_tensor_t odd_handle = MakeTensor({1.0f, 2.0f, 3.0f}, {1, 3});
  ts_tensor_t even_handle =
      MakeTensor({1.0f, 2.0f, 3.0f, 4.0f}, {1, 4});
  auto rank_zero = LookupTensor(rank_zero_handle);
  auto odd = LookupTensor(odd_handle);
  auto even = LookupTensor(even_handle);
  TestSwiGluMalformedBackward(rank_zero, odd, even);

  RequireStatus(ts_tensor_release(even_handle), TS_OK);
  RequireStatus(ts_tensor_release(odd_handle), TS_OK);
  RequireStatus(ts_tensor_release(rank_zero_handle), TS_OK);
  RequireStatus(ts_tensor_release(scalar_handle), TS_OK);
  return 0;
}
