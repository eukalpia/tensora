#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "autograd/autograd.h"
#include "backends/cpu/cpu_backend.h"
#include "backends/cpu/cpu_backend_internal.h"
#include "core/allocation_guard.h"
#include "core/integer_math.h"
#include "core/status.h"
#include "runtime/dispatcher.h"
#include "memory/tensor_storage.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"
#include "training/training_bridge.h"

namespace {

int failures = 0;

#define CHECK_STATUS(actual, expected)                                           \
  do {                                                                           \
    const tensora::Status _status = (actual);                                     \
    if (_status.code() != (expected)) {                                           \
      std::cerr << "FAIL " << __FILE__ << ':' << __LINE__                       \
                << ": expected status " << static_cast<int>(expected)            \
                << ", got " << static_cast<int>(_status.code())                  \
                << " message=" << _status.message() << '\n';                    \
      ++failures;                                                                 \
    }                                                                             \
  } while (false)

#define CHECK_TRUE(expression)                                                    \
  do {                                                                           \
    if (!(expression)) {                                                          \
      std::cerr << "FAIL " << __FILE__ << ':' << __LINE__ << ": "              \
                << #expression << '\n';                                           \
      ++failures;                                                                 \
    }                                                                             \
  } while (false)

std::shared_ptr<tensora::Tensor> MakeTensor(const std::vector<float>& values,
                                            const std::vector<int64_t>& dims) {
  tensora::ShapeInfo shape;
  CHECK_STATUS(tensora::ValidateShape(dims.empty() ? nullptr : dims.data(),
                                      dims.size(), &shape),
               TS_OK);
  tensora::CpuBackend backend;
  std::shared_ptr<tensora::Tensor> tensor;
  CHECK_STATUS(backend.FromData(shape, values.data(), &tensor), TS_OK);
  CHECK_TRUE(tensor != nullptr);
  return tensor;
}

class FakeCpuStorage final : public tensora::TensorStorage {
 public:
  explicit FakeCpuStorage(uint64_t byte_size) : byte_size_(byte_size) {}

  tensora::StorageKind kind() const override {
    return tensora::StorageKind::kCpu;
  }

  tensora::Status CopyToHostF32(float*, size_t, size_t* out_written) const override {
    if (out_written != nullptr) *out_written = 0;
    return tensora::Status::Ok();
  }

  uint64_t byte_size() const override { return byte_size_; }

 private:
  uint64_t byte_size_;
};

void TestAllocationGuardContracts() {
  using tensora::AllocationGuard;
  using tensora::Status;

  CHECK_STATUS(AllocationGuard("alloc_ok", [] { return Status::Ok(); }), TS_OK);
  CHECK_STATUS(AllocationGuard("alloc_bad", []() -> Status { throw std::bad_alloc(); }),
               TS_OUT_OF_MEMORY);
  CHECK_STATUS(AllocationGuard("alloc_length", []() -> Status {
                 throw std::length_error("too large");
               }),
               TS_OUT_OF_MEMORY);
}

void TestIntegerAndStatusContracts() {
  using namespace tensora;
  uint64_t result = 777;
  CHECK_STATUS(CheckedAddU64(2, 3, "checked_add", &result), TS_OK);
  CHECK_TRUE(result == 5);
  CHECK_STATUS(CheckedAddU64(1, 2, "checked_add", nullptr), TS_INVALID_ARGUMENT);
  CHECK_STATUS(CheckedAddU64(std::numeric_limits<uint64_t>::max(), 1,
                             "checked_add", &result), TS_INTERNAL_ERROR);
  CHECK_TRUE(result == 0);
  CHECK_STATUS(ModelError("model failure"), TS_MODEL_ERROR);
}

void TestCpuBackendInternalContracts() {
  using namespace tensora;

  ShapeInfo shape;
  const int64_t dims[1] = {2};
  CHECK_STATUS(ValidateShape(dims, 1, &shape), TS_OK);
  std::shared_ptr<CpuStorage> storage;
  CHECK_STATUS(CpuStorage::Filled(2, 1.0f, &storage), TS_OK);
  CHECK_TRUE(storage != nullptr);

  std::shared_ptr<Tensor> tensor;
  CHECK_STATUS(tensora::cpu_backend_internal::MakeTensor(shape, storage, nullptr), TS_INVALID_ARGUMENT);
  CHECK_STATUS(tensora::cpu_backend_internal::MakeTensor(shape, nullptr, &tensor), TS_INVALID_ARGUMENT);

  ShapeInfo corrupt = shape;
  corrupt.numel = 3;
  corrupt.byte_size = 3 * sizeof(float);
  CHECK_STATUS(tensora::cpu_backend_internal::MakeTensor(corrupt, storage, &tensor), TS_INTERNAL_ERROR);

  Tensor source(shape, storage);
  CHECK_STATUS(tensora::cpu_backend_internal::MakeView(shape, source, nullptr), TS_INVALID_ARGUMENT);
  CHECK_STATUS(tensora::cpu_backend_internal::EnsureCpuFloat32(source, "cpu_internal"), TS_OK);
  Tensor cuda_like(shape, storage, DType::kFloat32, Device::kCuda, 0);
  CHECK_STATUS(tensora::cpu_backend_internal::EnsureCpuFloat32(cuda_like, "cpu_internal"), TS_UNSUPPORTED);
  Tensor bad_dtype(shape, storage, static_cast<DType>(999), Device::kCpu, 0);
  CHECK_STATUS(tensora::cpu_backend_internal::EnsureCpuFloat32(bad_dtype, "cpu_internal"), TS_UNSUPPORTED);
  CHECK_STATUS(tensora::cpu_backend_internal::LogicalValues(source, "cpu_internal", nullptr), TS_INVALID_ARGUMENT);
  std::vector<float> values;
  CHECK_STATUS(tensora::cpu_backend_internal::LogicalValues(source, "cpu_internal", &values), TS_OK);
  CHECK_TRUE(values == std::vector<float>({1.0f, 1.0f}));
}

void TestDirectArgumentContracts() {
  using namespace tensora;
  using namespace tensora::training;

  const auto vector = MakeTensor({1.0f, 2.0f}, {2});
  const auto matrix = MakeTensor({1.0f, 2.0f}, {1, 2});
  const auto wrong_width = MakeTensor({1.0f, 2.0f, 3.0f}, {1, 3});
  const auto rank_one_logits = MakeTensor({1.0f, -1.0f}, {2});

  uint8_t available = 0;
  uint32_t count = 0;
  CHECK_STATUS(IsAvailable(nullptr), TS_INVALID_ARGUMENT);
  CHECK_STATUS(IsAvailable(&available), TS_OK);
  CHECK_TRUE(available == 1);
  CHECK_STATUS(DeviceCount(Device::kCpu, nullptr), TS_INVALID_ARGUMENT);
  CHECK_STATUS(DeviceCount(Device::kCpu, &count), TS_OK);
  CHECK_TRUE(count == 1);
  CHECK_STATUS(CudaDeviceCount(&count), TS_OK);
  CHECK_TRUE(count == 0);

  std::shared_ptr<Tensor> output;
  CHECK_STATUS(Transfer(*matrix, Device::kCpu, 0, nullptr), TS_INVALID_ARGUMENT);
  CHECK_STATUS(Transfer(*matrix, Device::kCpu, -1, &output), TS_INVALID_ARGUMENT);
  CHECK_STATUS(Transfer(*matrix, Device::kCuda, 0, &output), TS_UNSUPPORTED);
  CHECK_STATUS(Transfer(*matrix, Device::kCpu, 1, &output), TS_INVALID_ARGUMENT);
  CHECK_STATUS(Transfer(*matrix, Device::kCpu, 0, &output), TS_OK);
  CHECK_TRUE(output != nullptr);

  CHECK_STATUS(WithRequiresGrad(*matrix, true, nullptr), TS_INVALID_ARGUMENT);
  uint8_t requires_grad = 0;
  CHECK_STATUS(RequiresGrad(*matrix, nullptr), TS_INVALID_ARGUMENT);
  CHECK_STATUS(RequiresGrad(*matrix, &requires_grad), TS_OK);
  CHECK_TRUE(requires_grad == 0);

  CHECK_STATUS(Relu(*matrix, nullptr), TS_INVALID_ARGUMENT);
  CHECK_STATUS(Sigmoid(*matrix, nullptr), TS_INVALID_ARGUMENT);
  CHECK_STATUS(Tanh(*matrix, nullptr), TS_INVALID_ARGUMENT);
  CHECK_STATUS(MseLoss(*matrix, *matrix, nullptr), TS_INVALID_ARGUMENT);
  CHECK_STATUS(CrossEntropyLoss(*matrix, *matrix, nullptr), TS_INVALID_ARGUMENT);
  CHECK_STATUS(CrossEntropyLoss(*rank_one_logits, *rank_one_logits, &output),
               TS_INVALID_SHAPE);

  uint64_t module = 0;
  CHECK_STATUS(LinearCreate(1, 1, false, nullptr), TS_INVALID_ARGUMENT);
  CHECK_STATUS(LinearCreate(std::numeric_limits<int64_t>::max(),
                            std::numeric_limits<int64_t>::max(), false, &module),
               TS_INVALID_ARGUMENT);
  CHECK_TRUE(module == 0);
  CHECK_STATUS(LinearCreate(std::numeric_limits<int64_t>::max(), 1, false,
                            &module),
               TS_INVALID_ARGUMENT);
  CHECK_TRUE(module == 0);

  CHECK_STATUS(LinearCreate(2, 2, true, &module), TS_OK);
  CHECK_TRUE(module != 0);
  CHECK_STATUS(ModuleForward(module, *matrix, nullptr), TS_INVALID_ARGUMENT);
  CHECK_STATUS(ModuleForward(module, *vector, &output), TS_INVALID_SHAPE);
  CHECK_STATUS(ModuleForward(module, *wrong_width, &output), TS_INVALID_SHAPE);
  CHECK_STATUS(ModuleToDevice(module, Device::kCpu, -1), TS_INVALID_ARGUMENT);

  size_t parameter_count = 0;
  CHECK_STATUS(ModuleParameterCount(module, &parameter_count), TS_OK);
  CHECK_TRUE(parameter_count == 2);
  CHECK_STATUS(ModuleParameterAt(module, 0, nullptr), TS_INVALID_ARGUMENT);
  CHECK_STATUS(ModuleBufferAt(module, 0, nullptr), TS_INVALID_ARGUMENT);

  CHECK_STATUS(ModuleSave(module, "/definitely/missing/tensora/checkpoint.bin"),
               TS_INTERNAL_ERROR);
  CHECK_STATUS(ModuleLoad(module, "/definitely/missing/tensora/checkpoint.bin"),
               TS_INTERNAL_ERROR);

  CHECK_STATUS(ModuleRelease(module), TS_OK);
}

void TestAutogradInternalContracts() {
  using namespace tensora;
  using namespace tensora::autograd;

  const auto vector = MakeTensor({1.0f, -2.0f}, {2});
  const auto other = MakeTensor({3.0f, 4.0f}, {2});
  const auto scalar = MakeTensor({1.0f}, {});
  const auto matrix = MakeTensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});

  const std::vector<float>* const_values = nullptr;
  std::vector<float>* mutable_values = nullptr;
  CHECK_STATUS(CpuValues(*vector, "test", nullptr), TS_INVALID_ARGUMENT);
  CHECK_STATUS(MutableCpuValues(*vector, "test", nullptr), TS_INVALID_ARGUMENT);
  CHECK_STATUS(CpuValues(*vector, "test", &const_values), TS_OK);
  CHECK_TRUE(const_values != nullptr && const_values->size() == 2);
  CHECK_STATUS(MutableCpuValues(*vector, "test", &mutable_values), TS_OK);
  CHECK_TRUE(mutable_values != nullptr && mutable_values->size() == 2);

  std::vector<float> logical;
  CHECK_STATUS(ReadLogicalCpuValues(*vector, "test", nullptr),
               TS_INVALID_ARGUMENT);
  CHECK_STATUS(ReadLogicalCpuValues(*vector, "test", &logical), TS_OK);
  CHECK_TRUE(logical == std::vector<float>({1.0f, -2.0f}));

  std::shared_ptr<Tensor> output;
  CHECK_STATUS(MakeCpuTensor(vector->shape(), {1.0f}, &output),
               TS_INTERNAL_ERROR);
  CHECK_STATUS(MakeCpuTensor(vector->shape(), {1.0f, 2.0f}, nullptr),
               TS_INVALID_ARGUMENT);
  CHECK_STATUS(MakeCpuFull(vector->shape(), 1.0f, nullptr), TS_INVALID_ARGUMENT);

  std::shared_ptr<Tensor> leaf;
  CHECK_STATUS(CloneAsLeaf(*vector, true, &leaf), TS_OK);
  CHECK_TRUE(leaf != nullptr && RequiresGrad(*leaf));

  Tensor unmanaged(vector->shape(), vector->storage());
  CHECK_STATUS(Share(unmanaged, &output), TS_INTERNAL_ERROR);
  CHECK_STATUS(Share(*leaf, nullptr), TS_INVALID_ARGUMENT);

  CHECK_STATUS(AttachNode(nullptr, Operation::kIdentity, {}),
               TS_INVALID_ARGUMENT);
  const auto detached_result = MakeTensor({5.0f, 6.0f}, {2});
  CHECK_STATUS(AttachNode(detached_result, Operation::kAdd, {vector, other}),
               TS_OK);
  CHECK_TRUE(!RequiresGrad(*detached_result));

  CHECK_STATUS(Accumulate(nullptr, *vector), TS_INVALID_ARGUMENT);
  std::shared_ptr<Tensor> accumulation;
  CHECK_STATUS(Accumulate(&accumulation, *vector), TS_OK);
  CHECK_STATUS(Accumulate(&accumulation, *other), TS_OK);
  std::vector<float> accumulated_values;
  CHECK_STATUS(ReadLogicalCpuValues(*accumulation, "test", &accumulated_values),
               TS_OK);
  CHECK_TRUE(accumulated_values == std::vector<float>({4.0f, 2.0f}));
  CHECK_STATUS(AddInPlace(*accumulation, *scalar), TS_INTERNAL_ERROR);

  GradNode corrupt_versions;
  corrupt_versions.operation = Operation::kIdentity;
  corrupt_versions.parents = {leaf};
  CHECK_STATUS(ValidateSavedVersions(corrupt_versions), TS_INTERNAL_ERROR);
  corrupt_versions.parent_versions = {Version(*leaf)};
  CHECK_STATUS(ValidateSavedVersions(corrupt_versions), TS_OK);
  IncrementVersion(*leaf);
  CHECK_STATUS(ValidateSavedVersions(corrupt_versions), TS_INVALID_ARGUMENT);

  ShapeInfo wrong_shape;
  const int64_t wrong_dims[1] = {3};
  CHECK_STATUS(ValidateShape(wrong_dims, 1, &wrong_shape), TS_OK);
  CHECK_STATUS(CloneWithShape(*vector, wrong_shape, &output), TS_INTERNAL_ERROR);
  CHECK_STATUS(TransposeGradient(*vector, &output), TS_INTERNAL_ERROR);

  std::vector<GradientContribution> contributions;
  CHECK_STATUS(ApplyNode(corrupt_versions, *vector, nullptr), TS_INVALID_ARGUMENT);

  std::shared_ptr<Tensor> left_leaf;
  std::shared_ptr<Tensor> right_leaf;
  CHECK_STATUS(CloneAsLeaf(*vector, true, &left_leaf), TS_OK);
  CHECK_STATUS(CloneAsLeaf(*other, true, &right_leaf), TS_OK);

  GradNode multiply_bad;
  multiply_bad.operation = Operation::kMultiply;
  multiply_bad.parents = {left_leaf};
  multiply_bad.parent_versions = {Version(*left_leaf)};
  CHECK_STATUS(ApplyNode(multiply_bad, *vector, &contributions),
               TS_INTERNAL_ERROR);

  GradNode sum_bad;
  sum_bad.operation = Operation::kSum;
  sum_bad.parents = {left_leaf};
  sum_bad.parent_versions = {Version(*left_leaf)};
  CHECK_STATUS(ApplyNode(sum_bad, *vector, &contributions), TS_INTERNAL_ERROR);

  GradNode matmul_bad_count;
  matmul_bad_count.operation = Operation::kMatmul;
  matmul_bad_count.parents = {left_leaf};
  matmul_bad_count.parent_versions = {Version(*left_leaf)};
  CHECK_STATUS(ApplyNode(matmul_bad_count, *vector, &contributions),
               TS_INTERNAL_ERROR);

  GradNode matmul_bad_rank;
  matmul_bad_rank.operation = Operation::kMatmul;
  matmul_bad_rank.parents = {left_leaf, right_leaf};
  matmul_bad_rank.parent_versions = {Version(*left_leaf), Version(*right_leaf)};
  CHECK_STATUS(ApplyNode(matmul_bad_rank, *scalar, &contributions),
               TS_INTERNAL_ERROR);

  GradNode activation_bad;
  activation_bad.operation = Operation::kRelu;
  activation_bad.parents = {left_leaf};
  activation_bad.parent_versions = {Version(*left_leaf)};
  CHECK_STATUS(ApplyNode(activation_bad, *scalar, &contributions),
               TS_INTERNAL_ERROR);

  GradNode mse_bad;
  mse_bad.operation = Operation::kMse;
  mse_bad.parents = {left_leaf};
  mse_bad.parent_versions = {Version(*left_leaf)};
  CHECK_STATUS(ApplyNode(mse_bad, *scalar, &contributions), TS_INTERNAL_ERROR);

  GradNode cross_entropy_bad;
  cross_entropy_bad.operation = Operation::kCrossEntropy;
  cross_entropy_bad.parents = {left_leaf};
  cross_entropy_bad.parent_versions = {Version(*left_leaf)};
  CHECK_STATUS(ApplyNode(cross_entropy_bad, *scalar, &contributions),
               TS_INTERNAL_ERROR);

  GradNode bias_bad;
  bias_bad.operation = Operation::kBiasAdd;
  bias_bad.parents = {left_leaf};
  bias_bad.parent_versions = {Version(*left_leaf)};
  CHECK_STATUS(ApplyNode(bias_bad, *vector, &contributions), TS_INTERNAL_ERROR);

  GradNode unknown;
  unknown.operation = static_cast<Operation>(255);
  unknown.parents = {};
  unknown.parent_versions = {};
  CHECK_STATUS(ApplyNode(unknown, *scalar, &contributions), TS_INTERNAL_ERROR);

  std::unordered_set<Tensor*> visited;
  std::vector<std::shared_ptr<Tensor>> topology;
  BuildTopology(nullptr, &visited, &topology);
  BuildTopology(left_leaf, nullptr, &topology);
  BuildTopology(left_leaf, &visited, nullptr);
  BuildTopology(left_leaf, &visited, &topology);
  BuildTopology(left_leaf, &visited, &topology);
  CHECK_TRUE(topology.size() == 1);

  CHECK_STATUS(Backward(*vector), TS_INVALID_SHAPE);
  CHECK_STATUS(Backward(*scalar), TS_INVALID_ARGUMENT);
  CHECK_STATUS(GradientSnapshot(*vector, nullptr), TS_INVALID_ARGUMENT);
  CHECK_STATUS(GradientSnapshot(*vector, &output), TS_INVALID_ARGUMENT);

  auto empty_meta = std::make_shared<AutogradMeta>();
  empty_meta->requires_grad = true;
  vector->set_autograd_meta(empty_meta);
  CHECK_STATUS(GradientSnapshot(*vector, &output), TS_INVALID_ARGUMENT);
  ClearGradient(*other);
  ClearGradient(*vector);
}

void TestAutogradReachableEdgeContracts() {
  using namespace tensora;
  using namespace tensora::autograd;

  const auto vector = MakeTensor({1.0f, 2.0f}, {2});
  Tensor cuda_like(vector->shape(), vector->storage(), DType::kFloat32,
                   Device::kCuda, 0);
  CHECK_STATUS(EnsureCpuFloat32(cuda_like, "autograd_edge"), TS_UNSUPPORTED);
  Tensor bad_dtype(vector->shape(), vector->storage(), static_cast<DType>(999),
                   Device::kCpu, 0);
  CHECK_STATUS(EnsureCpuFloat32(bad_dtype, "autograd_edge"), TS_UNSUPPORTED);

  const auto matrix = MakeTensor({1, 2, 3, 4, 5, 6}, {2, 3});
  CpuBackend backend;
  std::shared_ptr<Tensor> transposed;
  CHECK_STATUS(backend.Transpose2D(*matrix, &transposed), TS_OK);
  const std::vector<float>* const_values = nullptr;
  std::vector<float>* mutable_values = nullptr;
  CHECK_STATUS(CpuValues(*transposed, "autograd_edge", &const_values),
               TS_INVALID_ARGUMENT);
  CHECK_STATUS(MutableCpuValues(*transposed, "autograd_edge", &mutable_values),
               TS_INVALID_ARGUMENT);

  ShapeInfo one_shape;
  const int64_t one_dim[1] = {1};
  CHECK_STATUS(ValidateShape(one_dim, 1, &one_shape), TS_OK);
  Tensor partial(one_shape, vector->storage());
  CHECK_STATUS(CpuValues(partial, "autograd_edge", &const_values),
               TS_INVALID_ARGUMENT);
  CHECK_STATUS(MutableCpuValues(partial, "autograd_edge", &mutable_values),
               TS_INVALID_ARGUMENT);

  const auto fake_storage = std::make_shared<FakeCpuStorage>(one_shape.byte_size);
  Tensor fake_tensor(one_shape, fake_storage);
  CHECK_STATUS(CpuValues(fake_tensor, "autograd_edge", &const_values),
               TS_UNSUPPORTED);
  CHECK_STATUS(MutableCpuValues(fake_tensor, "autograd_edge", &mutable_values),
               TS_UNSUPPORTED);

  ShapeInfo corrupt_shape;
  corrupt_shape.dimensions = {2};
  corrupt_shape.strides = {1};
  corrupt_shape.numel = 3;
  corrupt_shape.byte_size = 3 * sizeof(float);
  std::shared_ptr<Tensor> output;
  CHECK_STATUS(MakeCpuTensor(corrupt_shape, {1, 2, 3}, &output),
               TS_INTERNAL_ERROR);

  std::shared_ptr<Tensor> left;
  std::shared_ptr<Tensor> right;
  CHECK_STATUS(CloneAsLeaf(*vector, true, &left), TS_OK);
  const auto other = MakeTensor({3.0f, 4.0f}, {2});
  CHECK_STATUS(CloneAsLeaf(*other, true, &right), TS_OK);
  const auto upstream = MakeTensor({5.0f, 6.0f}, {2});
  const auto scalar = MakeTensor({1.0f}, {});
  std::vector<GradientContribution> contributions;

  GradNode identity{Operation::kIdentity, {left}, {Version(*left)}};
  CHECK_STATUS(ApplyNode(identity, *upstream, &contributions), TS_OK);
  CHECK_TRUE(contributions.size() == 1);

  GradNode multiply{Operation::kMultiply,
                    {left, right},
                    {Version(*left), Version(*right)}};
  CHECK_STATUS(ApplyNode(multiply, *scalar, &contributions), TS_INTERNAL_ERROR);

  GradNode mse{Operation::kMse,
               {left, right},
               {Version(*left), Version(*right)}};
  CHECK_STATUS(ApplyNode(mse, *scalar, &contributions), TS_OK);
  CHECK_TRUE(contributions.size() == 2);

  const auto logits_base = MakeTensor({1, 2, 3, 4}, {2, 2});
  const auto target_base = MakeTensor({1, 0, 0, 1}, {2, 2});
  std::shared_ptr<Tensor> logits;
  std::shared_ptr<Tensor> target;
  CHECK_STATUS(CloneAsLeaf(*logits_base, true, &logits), TS_OK);
  CHECK_STATUS(CloneAsLeaf(*target_base, true, &target), TS_OK);
  GradNode cross_entropy{Operation::kCrossEntropy,
                         {logits, target},
                         {Version(*logits), Version(*target)}};
  CHECK_STATUS(ApplyNode(cross_entropy, *scalar, &contributions), TS_OK);
  CHECK_TRUE(contributions.size() == 2);

  const auto bad_target_base = MakeTensor({1, 0}, {2});
  std::shared_ptr<Tensor> bad_target;
  CHECK_STATUS(CloneAsLeaf(*bad_target_base, true, &bad_target), TS_OK);
  GradNode cross_entropy_shape{Operation::kCrossEntropy,
                               {logits, bad_target},
                               {Version(*logits), Version(*bad_target)}};
  CHECK_STATUS(ApplyNode(cross_entropy_shape, *scalar, &contributions),
               TS_INTERNAL_ERROR);

  GradNode bias_rank{Operation::kBiasAdd,
                     {left, right},
                     {Version(*left), Version(*right)}};
  CHECK_STATUS(ApplyNode(bias_rank, *upstream, &contributions),
               TS_INTERNAL_ERROR);

  const auto matrix_base = MakeTensor({1, 2, 3, 4}, {2, 2});
  std::shared_ptr<Tensor> matrix_leaf;
  CHECK_STATUS(CloneAsLeaf(*matrix_base, true, &matrix_leaf), TS_OK);
  const auto wrong_bias_base = MakeTensor({1, 2, 3}, {3});
  std::shared_ptr<Tensor> wrong_bias;
  CHECK_STATUS(CloneAsLeaf(*wrong_bias_base, true, &wrong_bias), TS_OK);
  const auto matrix_upstream = MakeTensor({1, 1, 1, 1}, {2, 2});
  GradNode bias_width{Operation::kBiasAdd,
                      {matrix_leaf, wrong_bias},
                      {Version(*matrix_leaf), Version(*wrong_bias)}};
  CHECK_STATUS(ApplyNode(bias_width, *matrix_upstream, &contributions),
               TS_INTERNAL_ERROR);

  const auto bias_base = MakeTensor({1, 2}, {2});
  std::shared_ptr<Tensor> bias;
  CHECK_STATUS(CloneAsLeaf(*bias_base, true, &bias), TS_OK);
  GradNode bias_gradient_shape{Operation::kBiasAdd,
                               {matrix_leaf, bias},
                               {Version(*matrix_leaf), Version(*bias)}};
  CHECK_STATUS(ApplyNode(bias_gradient_shape, *upstream, &contributions),
               TS_INTERNAL_ERROR);
}

void WriteBytes(const std::filesystem::path& path,
                const std::vector<unsigned char>& bytes) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
}

template <typename T>
void StoreScalar(std::vector<unsigned char>* bytes, size_t offset, T value) {
  CHECK_TRUE(bytes != nullptr);
  CHECK_TRUE(offset + sizeof(T) <= bytes->size());
  if (bytes == nullptr || offset + sizeof(T) > bytes->size()) return;
  const auto* source = reinterpret_cast<const unsigned char*>(&value);
  std::copy(source, source + sizeof(T), bytes->begin() +
                                           static_cast<std::ptrdiff_t>(offset));
}

std::vector<unsigned char> ReadBytes(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  return std::vector<unsigned char>(std::istreambuf_iterator<char>(stream),
                                    std::istreambuf_iterator<char>());
}

void TestTensorAndStorageSafetyContracts() {
  using namespace tensora;

  ShapeInfo vector_shape;
  const int64_t vector_dims[1] = {2};
  CHECK_STATUS(ValidateShape(vector_dims, 1, &vector_shape), TS_OK);
  CHECK_STATUS(ValidateShape(vector_dims, 1, nullptr), TS_INVALID_ARGUMENT);

  std::shared_ptr<CpuStorage> storage;
  CHECK_STATUS(CpuStorage::Filled(2, 1.0f, nullptr), TS_INVALID_ARGUMENT);
  CHECK_STATUS(CpuStorage::FromData(nullptr, 1, &storage), TS_INVALID_ARGUMENT);
  CHECK_STATUS(CpuStorage::FromData(nullptr, 0, nullptr), TS_INVALID_ARGUMENT);
  CHECK_STATUS(CpuStorage::Filled(std::numeric_limits<uint64_t>::max(), 0.0f,
                                  &storage),
               TS_OUT_OF_MEMORY);
  const float one = 1.0f;
  CHECK_STATUS(CpuStorage::FromData(&one, std::numeric_limits<uint64_t>::max(),
                                    &storage),
               TS_OUT_OF_MEMORY);

  CHECK_STATUS(CpuStorage::Filled(2, 1.0f, &storage), TS_OK);
  CHECK_TRUE(storage != nullptr);
  size_t written = 777;
  float values[2] = {0.0f, 0.0f};
  CHECK_STATUS(storage->CopyToHostF32(values, 2, nullptr), TS_INVALID_ARGUMENT);
  CHECK_STATUS(storage->CopyToHostF32(values, 1, &written), TS_INVALID_ARGUMENT);
  CHECK_TRUE(written == 0);
  CHECK_STATUS(storage->CopyToHostF32(nullptr, 2, &written), TS_INVALID_ARGUMENT);
  CHECK_TRUE(written == 0);

  Tensor base(vector_shape, storage);
  CHECK_STATUS(base.CopyToHostF32(values, 2, nullptr), TS_INVALID_ARGUMENT);
  Tensor bad_dtype(vector_shape, storage, static_cast<DType>(999), Device::kCpu,
                   0);
  CHECK_STATUS(bad_dtype.CopyToHostF32(values, 2, &written), TS_UNSUPPORTED);
  CHECK_STATUS(base.CopyToHostF32(values, 1, &written), TS_INVALID_ARGUMENT);
  CHECK_STATUS(base.CopyToHostF32(nullptr, 2, &written), TS_INVALID_ARGUMENT);

  const auto inconsistent =
      std::make_shared<FakeCpuStorage>(vector_shape.byte_size);
  Tensor inconsistent_tensor(vector_shape, inconsistent);
  CHECK_STATUS(inconsistent_tensor.CopyToHostF32(values, 2, &written),
               TS_INTERNAL_ERROR);
  CHECK_TRUE(written == 0);

  ShapeInfo non_contiguous = vector_shape;
  non_contiguous.strides[0] = 2;
  Tensor accelerator_view(non_contiguous, storage, DType::kFloat32,
                          Device::kCuda, 0);
  CHECK_STATUS(accelerator_view.CopyToHostF32(values, 2, &written),
               TS_UNSUPPORTED);

  Tensor fake_cpu_view(non_contiguous, inconsistent);
  CHECK_STATUS(fake_cpu_view.CopyToHostF32(values, 2, &written), TS_UNSUPPORTED);

  ShapeInfo overflowing_view = vector_shape;
  overflowing_view.strides[0] = 3;
  Tensor out_of_storage_view(overflowing_view, storage);
  CHECK_STATUS(out_of_storage_view.CopyToHostF32(values, 2, &written),
               TS_INTERNAL_ERROR);
  CHECK_TRUE(written == 0);

  const Backend* backend = nullptr;
  CHECK_STATUS(Dispatcher::For(Device::kCpu, nullptr), TS_INVALID_ARGUMENT);
  CHECK_STATUS(Dispatcher::For(Device::kCpu, &backend), TS_OK);
  CHECK_TRUE(backend != nullptr);
  CHECK_STATUS(Dispatcher::For(static_cast<Device>(999), &backend), TS_UNSUPPORTED);
  CHECK_TRUE(backend == nullptr);
  CHECK_STATUS(Dispatcher::ForTensor(base, nullptr), TS_INVALID_ARGUMENT);
  CHECK_STATUS(Dispatcher::ForTensor(base, &backend), TS_OK);
  CHECK_TRUE(backend != nullptr);

  Tensor other_device(vector_shape, storage, DType::kFloat32, Device::kCpu, 1);
  CHECK_STATUS(Dispatcher::ForTensors(base, other_device, &backend),
               TS_INVALID_ARGUMENT);
  CHECK_STATUS(Dispatcher::ForTensors(base, base, nullptr), TS_INVALID_ARGUMENT);
}

void TestCheckpointCorruptionContracts() {
  using namespace tensora::training;
  namespace fs = std::filesystem;

  const fs::path root = fs::current_path();
  const fs::path valid = root / "tensora_internal_valid.bin";
  const fs::path mutated = root / "tensora_internal_mutated.bin";
  std::error_code ignored;
  fs::remove(valid, ignored);
  fs::remove(mutated, ignored);

  uint64_t module = 0;
  CHECK_STATUS(LinearCreate(2, 1, true, &module), TS_OK);
  CHECK_STATUS(ModuleSave(module, valid.string()), TS_OK);
  auto bytes = ReadBytes(valid);
  CHECK_TRUE(bytes.size() >= 97);

  auto candidate = bytes;
  candidate.resize(10);
  WriteBytes(mutated, candidate);
  CHECK_STATUS(ModuleLoad(module, mutated.string()), TS_INTERNAL_ERROR);

  candidate = bytes;
  StoreScalar<uint64_t>(&candidate, 29, 33);
  WriteBytes(mutated, candidate);
  CHECK_STATUS(ModuleLoad(module, mutated.string()), TS_INTERNAL_ERROR);

  candidate = bytes;
  candidate.resize(40);
  WriteBytes(mutated, candidate);
  CHECK_STATUS(ModuleLoad(module, mutated.string()), TS_INTERNAL_ERROR);

  candidate = bytes;
  StoreScalar<int64_t>(&candidate, 37, 1);
  WriteBytes(mutated, candidate);
  CHECK_STATUS(ModuleLoad(module, mutated.string()), TS_INTERNAL_ERROR);

  candidate = bytes;
  StoreScalar<uint64_t>(&candidate, 53, 3);
  WriteBytes(mutated, candidate);
  CHECK_STATUS(ModuleLoad(module, mutated.string()), TS_INTERNAL_ERROR);

  candidate = bytes;
  candidate.resize(64);
  WriteBytes(mutated, candidate);
  CHECK_STATUS(ModuleLoad(module, mutated.string()), TS_INTERNAL_ERROR);

  CHECK_STATUS(ModuleRelease(module), TS_OK);
  fs::remove(valid, ignored);
  fs::remove(mutated, ignored);
}

}  // namespace

int main() {
  TestAllocationGuardContracts();
  TestIntegerAndStatusContracts();
  TestCpuBackendInternalContracts();
  TestDirectArgumentContracts();
  TestAutogradInternalContracts();
  TestAutogradReachableEdgeContracts();
  TestTensorAndStorageSafetyContracts();
  TestCheckpointCorruptionContracts();

  if (failures != 0) {
    std::cerr << failures << " core training internal assertion(s) failed\n";
    return 1;
  }
  std::cout << "Tensora core training internal contracts passed\n";
  return 0;
}
