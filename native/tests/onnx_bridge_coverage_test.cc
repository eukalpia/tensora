#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "core/status.h"
#include "inference/inference_bridge.h"
#include "memory/cpu_storage.h"
#include "memory/tensor_storage.h"
#include "runtime/handle_registry.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

namespace coverage {
bool fail_make_shared = false;
bool provider_throw = false;
bool use_provider_snapshot = false;
bool force_non_tensor = false;
bool count_mismatch = false;
bool info_throw = false;
bool session_ctor_bad_alloc = false;
bool wrong_output_count = false;
bool end_profiling_throw = false;
std::vector<std::string> provider_snapshot;
}

namespace std {

template <typename T, typename... Args>
std::shared_ptr<T> CoverageMakeShared(Args&&... args) {
  if (coverage::fail_make_shared) {
    coverage::fail_make_shared = false;
    throw std::bad_alloc();
  }
  return std::make_shared<T>(std::forward<Args>(args)...);
}

}  // namespace std

namespace Ort {

std::vector<std::string> GetAvailableProvidersForCoverage() {
  if (coverage::provider_throw) {
    throw Ort::Exception("coverage provider enumeration failure", ORT_FAIL);
  }
  if (coverage::use_provider_snapshot) return coverage::provider_snapshot;
  return Ort::GetAvailableProviders();
}

class CoverageSessionOptions : public Ort::SessionOptions {
 public:
  CoverageSessionOptions() = default;
  CoverageSessionOptions(CoverageSessionOptions&&) noexcept = default;
  CoverageSessionOptions& operator=(CoverageSessionOptions&&) noexcept = default;

  CoverageSessionOptions& AppendExecutionProvider_CUDA(
      const OrtCUDAProviderOptions&) {
    return *this;
  }

  CoverageSessionOptions& AppendExecutionProvider_OpenVINO_V2(
      std::initializer_list<int>) {
    return *this;
  }

  CoverageSessionOptions& AppendExecutionProvider_MIGraphX(
      const OrtMIGraphXProviderOptions&) {
    return *this;
  }
};

class CoverageTensorInfo {
 public:
  explicit CoverageTensorInfo(Ort::TensorTypeAndShapeInfo info)
      : info_(std::move(info)) {}

  ONNXTensorElementDataType GetElementType() const {
    return info_.GetElementType();
  }

  std::vector<int64_t> GetShape() const { return info_.GetShape(); }

  size_t GetElementCount() const {
    const size_t count = info_.GetElementCount();
    return coverage::count_mismatch ? count + 1 : count;
  }

 private:
  Ort::TensorTypeAndShapeInfo info_;
};

class CoverageValue : public Ort::Value {
 public:
  explicit CoverageValue(Ort::Value value) : Ort::Value(std::move(value)) {}
  CoverageValue(CoverageValue&&) noexcept = default;
  CoverageValue& operator=(CoverageValue&&) noexcept = default;

  template <typename T, typename... Args>
  static CoverageValue CreateTensor(Args&&... args) {
    return CoverageValue(
        Ort::Value::CreateTensor<T>(std::forward<Args>(args)...));
  }

  bool IsTensor() const {
    if (coverage::force_non_tensor) return false;
    return Ort::Value::IsTensor();
  }

  CoverageTensorInfo GetTensorTypeAndShapeInfo() const {
    if (coverage::info_throw) {
      throw Ort::Exception("coverage tensor metadata failure", ORT_FAIL);
    }
    return CoverageTensorInfo(Ort::Value::GetTensorTypeAndShapeInfo());
  }
};

class CoverageSession {
 public:
  CoverageSession(Ort::Env& environment,
                  const ORTCHAR_T* model_path,
                  const CoverageSessionOptions& options)
      : session_(MakeSession(environment, model_path, options)) {}

  size_t GetInputCount() const { return session_.GetInputCount(); }
  size_t GetOutputCount() const { return session_.GetOutputCount(); }

  template <typename Allocator>
  auto GetInputNameAllocated(size_t index, Allocator& allocator) const {
    return session_.GetInputNameAllocated(index, allocator);
  }

  template <typename Allocator>
  auto GetOutputNameAllocated(size_t index, Allocator& allocator) const {
    return session_.GetOutputNameAllocated(index, allocator);
  }

  auto GetInputTypeInfo(size_t index) const {
    return session_.GetInputTypeInfo(index);
  }

  auto GetOutputTypeInfo(size_t index) const {
    return session_.GetOutputTypeInfo(index);
  }

  template <typename... Args>
  std::vector<CoverageValue> Run(Args&&...) {
    if (coverage::wrong_output_count) return {};
    throw Ort::Exception("coverage session run was not scripted", ORT_FAIL);
  }

  auto EndProfilingAllocated(Ort::AllocatorWithDefaultOptions& allocator) {
    if (coverage::end_profiling_throw) {
      throw Ort::Exception("coverage end profiling failure", ORT_FAIL);
    }
    return session_.EndProfilingAllocated(allocator);
  }

 private:
  static Ort::Session MakeSession(Ort::Env& environment,
                                  const ORTCHAR_T* model_path,
                                  const CoverageSessionOptions& options) {
    if (coverage::session_ctor_bad_alloc) {
      coverage::session_ctor_bad_alloc = false;
      throw std::bad_alloc();
    }
    return Ort::Session(
        environment, model_path,
        static_cast<const Ort::SessionOptions&>(options));
  }

  Ort::Session session_;
};

}  // namespace Ort

#define SessionOptions CoverageSessionOptions
#define Session CoverageSession
#define Value CoverageValue
#define GetAvailableProviders GetAvailableProvidersForCoverage
#define make_shared CoverageMakeShared
#include "../src/inference/inference_bridge_onnx.cc"
#undef make_shared
#undef GetAvailableProviders
#undef Value
#undef Session
#undef SessionOptions

namespace tensora::inference {
namespace {

bool ExpectStatus(const Status& status,
                  ts_status_t expected,
                  const char* operation) {
  if (status.code() == expected) return true;
  std::cerr << operation << " expected status " << expected << ", got "
            << status.code() << ": " << status.message() << '\n';
  return false;
}

std::filesystem::path FixturePath(const std::string& suffix) {
  const char* raw = std::getenv("TENSORA_ONNX_TEST_MODEL");
  if (raw == nullptr) return {};
  const std::filesystem::path base(raw);
  return base.parent_path() /
         (base.stem().string() + suffix + base.extension().string());
}

std::shared_ptr<Tensor> MakeTensor() {
  const int64_t dimensions[2] = {2, 2};
  ShapeInfo shape;
  if (!ValidateShape(dimensions, 2, &shape).ok()) return nullptr;
  const float values[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  std::shared_ptr<CpuStorage> storage;
  if (!CpuStorage::FromData(values, 4, &storage).ok()) return nullptr;
  return std::make_shared<Tensor>(std::move(shape), std::move(storage));
}

class ThrowingStorage final : public TensorStorage {
 public:
  explicit ThrowingStorage(uint64_t byte_size) : byte_size_(byte_size) {}

  StorageKind kind() const override { return StorageKind::kCpu; }

  Status CopyToHostF32(float*, size_t, size_t*) const override {
    throw std::bad_alloc();
  }

  uint64_t byte_size() const override { return byte_size_; }

 private:
  uint64_t byte_size_;
};

int CheckProviderCoverage() {
  coverage::use_provider_snapshot = true;
  coverage::provider_snapshot = {kDmlProvider};
  try {
    (void)ResolveProvider(kDmlProvider);
    return 1;
  } catch (const Ort::Exception& error) {
    if (error.GetOrtErrorCode() != ORT_NOT_IMPLEMENTED) return 2;
  }

  coverage::provider_snapshot = {kCpuProvider};
  if (ResolveProvider(kCpuProvider) != kCpuProvider) return 3;

  Ort::CoverageSessionOptions options;
  AppendProvider(&options, kCudaProvider);
  AppendProvider(&options, kOpenVinoProvider);
  AppendProvider(&options, kMiGraphXProvider);

  coverage::provider_throw = true;
  size_t count = 99;
  if (!ExpectStatus(ProviderCount(&count), TS_MODEL_ERROR,
                    "provider count failure")) {
    coverage::provider_throw = false;
    return 4;
  }
  std::string name;
  if (!ExpectStatus(ProviderName(0, &name), TS_MODEL_ERROR,
                    "provider name failure")) {
    coverage::provider_throw = false;
    return 5;
  }
  coverage::provider_throw = false;
  coverage::use_provider_snapshot = false;
  coverage::provider_snapshot.clear();
  return 0;
}

int CheckValueConversionCoverage() {
  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  float value = 1.0f;
  int64_t dimensions[1] = {1};
  auto make_value = [&] {
    return Ort::CoverageValue::CreateTensor<float>(
        memory_info, &value, 1, dimensions, 1);
  };
  std::shared_ptr<Tensor> output;

  auto non_tensor = make_value();
  coverage::force_non_tensor = true;
  if (!ExpectStatus(OrtValueToTensor(&non_tensor, &output), TS_UNSUPPORTED,
                    "non tensor output")) {
    coverage::force_non_tensor = false;
    return 10;
  }
  coverage::force_non_tensor = false;

  auto mismatched = make_value();
  coverage::count_mismatch = true;
  if (!ExpectStatus(OrtValueToTensor(&mismatched, &output), TS_MODEL_ERROR,
                    "output element mismatch")) {
    coverage::count_mismatch = false;
    return 11;
  }
  coverage::count_mismatch = false;

  auto allocation = make_value();
  coverage::fail_make_shared = true;
  if (!ExpectStatus(OrtValueToTensor(&allocation, &output), TS_OUT_OF_MEMORY,
                    "output tensor allocation")) {
    coverage::fail_make_shared = false;
    return 12;
  }
  coverage::fail_make_shared = false;

  auto metadata = make_value();
  coverage::info_throw = true;
  if (!ExpectStatus(OrtValueToTensor(&metadata, &output), TS_MODEL_ERROR,
                    "output metadata failure")) {
    coverage::info_throw = false;
    return 13;
  }
  coverage::info_throw = false;
  return 0;
}

int CheckSessionConstructionCoverage() {
  const std::filesystem::path reference = FixturePath("");
  const std::filesystem::path double_input = FixturePath("-double-input");
  const std::filesystem::path double_output = FixturePath("-double-output");
  if (reference.empty() || double_input.empty() || double_output.empty()) {
    return 20;
  }

  uint64_t session = 0;
  coverage::fail_make_shared = true;
  if (!ExpectStatus(SessionCreate(reference.string(), kCpuProvider, false, "",
                                  &session),
                    TS_OUT_OF_MEMORY, "session allocation")) {
    coverage::fail_make_shared = false;
    return 21;
  }
  coverage::fail_make_shared = false;

  if (!ExpectStatus(SessionCreate(double_input.string(), kCpuProvider, false,
                                  "", &session),
                    TS_UNSUPPORTED, "double input model")) {
    return 22;
  }
  if (!ExpectStatus(SessionCreate(double_output.string(), kCpuProvider, false,
                                  "", &session),
                    TS_UNSUPPORTED, "double output model")) {
    return 23;
  }

  const std::filesystem::path profile_prefix =
      reference.parent_path() / "coverage-profile";
  if (!ExpectStatus(SessionCreate(reference.string(), kCpuProvider, true,
                                  profile_prefix.string(), &session),
                    TS_OK, "profiled session create") ||
      session == 0) {
    return 24;
  }
  std::string profile_path;
  if (!ExpectStatus(SessionEndProfiling(session, &profile_path), TS_OK,
                    "profile end") ||
      profile_path.empty()) {
    SessionRelease(session);
    return 25;
  }
  if (!ExpectStatus(SessionRelease(session), TS_OK, "profile session release")) {
    return 26;
  }

  session = 0;
  if (!ExpectStatus(SessionCreate(reference.string(), kCpuProvider, true,
                                  profile_prefix.string(), &session),
                    TS_OK, "failing profile session create") ||
      session == 0) {
    return 27;
  }
  coverage::end_profiling_throw = true;
  if (!ExpectStatus(SessionEndProfiling(session, &profile_path), TS_MODEL_ERROR,
                    "profile end failure")) {
    coverage::end_profiling_throw = false;
    SessionRelease(session);
    return 28;
  }
  coverage::end_profiling_throw = false;
  if (!ExpectStatus(SessionRelease(session), TS_OK,
                    "failing profile session release")) {
    return 29;
  }
  return 0;
}

int CheckSessionRunCoverage() {
  const std::filesystem::path reference = FixturePath("");
  if (reference.empty()) return 30;

  uint64_t session = 0;
  if (!ExpectStatus(SessionCreate(reference.string(), kCpuProvider, false, "",
                                  &session),
                    TS_OK, "run session create") ||
      session == 0) {
    return 31;
  }

  std::string input_name;
  std::string output_name;
  if (!ExpectStatus(SessionInputName(session, 0, &input_name), TS_OK,
                    "input name") ||
      !ExpectStatus(SessionOutputName(session, 0, &output_name), TS_OK,
                    "output name")) {
    SessionRelease(session);
    return 32;
  }

  auto input = MakeTensor();
  if (!input) {
    SessionRelease(session);
    return 33;
  }
  std::vector<std::shared_ptr<Tensor>> outputs;
  coverage::wrong_output_count = true;
  if (!ExpectStatus(SessionRun(session, {input_name}, {input}, {output_name},
                               &outputs),
                    TS_MODEL_ERROR, "runtime output count")) {
    coverage::wrong_output_count = false;
    SessionRelease(session);
    return 34;
  }
  coverage::wrong_output_count = false;

  ShapeInfo shape;
  const int64_t dimensions[2] = {2, 2};
  if (!ValidateShape(dimensions, 2, &shape).ok()) {
    SessionRelease(session);
    return 35;
  }
  auto throwing_storage = std::make_shared<ThrowingStorage>(shape.byte_size);
  auto throwing_input = std::make_shared<Tensor>(
      shape, throwing_storage, DType::kFloat32, Device::kCpu, 0);
  if (!ExpectStatus(SessionRun(session, {input_name}, {throwing_input},
                               {output_name}, &outputs),
                    TS_OUT_OF_MEMORY, "runtime allocation failure")) {
    SessionRelease(session);
    return 36;
  }

  if (!ExpectStatus(SessionRelease(session), TS_OK, "run session release")) {
    return 37;
  }
  return 0;
}

}  // namespace

int RunOnnxCoverageContracts() {
  if (const int code = CheckProviderCoverage(); code != 0) return code;
  if (const int code = CheckValueConversionCoverage(); code != 0) return code;
  if (const int code = CheckSessionConstructionCoverage(); code != 0) return code;
  if (const int code = CheckSessionRunCoverage(); code != 0) return code;
  return 0;
}

}  // namespace tensora::inference

int main() { return tensora::inference::RunOnnxCoverageContracts(); }
