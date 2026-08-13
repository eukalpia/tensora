#include "inference/inference_bridge.h"

#include <algorithm>
#include <filesystem>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "memory/cpu_storage.h"
#include "runtime/handle_registry.h"
#include "tensor/shape.h"

#if defined(TENSORA_WITH_ORT_DML)
extern "C" OrtStatus* ORT_API_CALL OrtSessionOptionsAppendExecutionProvider_DML(
    OrtSessionOptions* options, int device_id);
#endif

#if defined(TENSORA_WITH_ORT_COREML)
extern "C" OrtStatus* ORT_API_CALL OrtSessionOptionsAppendExecutionProvider_CoreML(
    OrtSessionOptions* options, uint32_t coreml_flags);
#endif

namespace tensora::inference {
namespace {

constexpr const char* kAutoProvider = "auto";
constexpr const char* kCpuProvider = "CPUExecutionProvider";
constexpr const char* kCudaProvider = "CUDAExecutionProvider";
constexpr const char* kDmlProvider = "DmlExecutionProvider";
constexpr const char* kCoreMlProvider = "CoreMLExecutionProvider";
constexpr const char* kOpenVinoProvider = "OpenVINOExecutionProvider";
constexpr const char* kMiGraphXProvider = "MIGraphXExecutionProvider";

bool IsKnownProvider(const std::string& provider) {
  return provider == kAutoProvider || provider == kCpuProvider ||
         provider == kCudaProvider || provider == kDmlProvider ||
         provider == kCoreMlProvider || provider == kOpenVinoProvider ||
         provider == kMiGraphXProvider;
}

bool IsProviderAvailable(const std::vector<std::string>& available,
                         const char* provider) {
  return std::find(available.begin(), available.end(), provider) !=
         available.end();
}

bool CanConfigureProvider(const std::string& provider) {
  if (provider == kCpuProvider || provider == kCudaProvider ||
      provider == kOpenVinoProvider || provider == kMiGraphXProvider) {
    return true;
  }
#if defined(TENSORA_WITH_ORT_DML)
  if (provider == kDmlProvider) return true;
#endif
#if defined(TENSORA_WITH_ORT_COREML)
  if (provider == kCoreMlProvider) return true;
#endif
  return false;
}

std::string ResolveAutoProvider(const std::vector<std::string>& available) {
#if defined(_WIN32)
  const char* candidates[] = {kCudaProvider, kOpenVinoProvider, kDmlProvider,
                              kMiGraphXProvider, kCpuProvider};
#elif defined(__APPLE__)
  const char* candidates[] = {kCoreMlProvider, kCpuProvider};
#else
  const char* candidates[] = {kCudaProvider, kMiGraphXProvider,
                              kOpenVinoProvider, kCpuProvider};
#endif
  for (const char* candidate : candidates) {
    if (IsProviderAvailable(available, candidate) &&
        CanConfigureProvider(candidate)) {
      return candidate;
    }
  }
  throw Ort::Exception("No supported ONNX execution provider is available",
                       ORT_NOT_IMPLEMENTED);
}

std::string ResolveProvider(const std::string& requested_provider) {
  if (!IsKnownProvider(requested_provider)) {
    throw Ort::Exception("Unknown ONNX execution provider", ORT_INVALID_ARGUMENT);
  }
  const auto available = Ort::GetAvailableProviders();
  if (requested_provider == kAutoProvider) {
    return ResolveAutoProvider(available);
  }
  if (!IsProviderAvailable(available, requested_provider.c_str())) {
    throw Ort::Exception("Requested ONNX execution provider is not available",
                         ORT_NOT_IMPLEMENTED);
  }
  if (!CanConfigureProvider(requested_provider)) {
    throw Ort::Exception(
        "Requested ONNX execution provider is not enabled in this Tensora build",
        ORT_NOT_IMPLEMENTED);
  }
  return requested_provider;
}

void AppendProvider(Ort::SessionOptions* options, const std::string& provider) {
  if (options == nullptr) {
    throw Ort::Exception("Session options pointer is null", ORT_INVALID_ARGUMENT);
  }
  if (provider == kCpuProvider) return;
  if (provider == kCudaProvider) {
    OrtCUDAProviderOptions provider_options{};
    options->AppendExecutionProvider_CUDA(provider_options);
    return;
  }
  if (provider == kOpenVinoProvider) {
    options->AppendExecutionProvider_OpenVINO_V2({});
    return;
  }
  if (provider == kMiGraphXProvider) {
    OrtMIGraphXProviderOptions provider_options{};
    options->AppendExecutionProvider_MIGraphX(provider_options);
    return;
  }
#if defined(TENSORA_WITH_ORT_DML)
  if (provider == kDmlProvider) {
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(*options, 0));
    return;
  }
#endif
#if defined(TENSORA_WITH_ORT_COREML)
  if (provider == kCoreMlProvider) {
    Ort::ThrowOnError(
        OrtSessionOptionsAppendExecutionProvider_CoreML(*options, 0u));
    return;
  }
#endif
  throw Ort::Exception("ONNX execution provider cannot be configured",
                       ORT_NOT_IMPLEMENTED);
}

class SessionState {
 public:
  SessionState(const std::string& model_path,
               const std::string& requested_provider,
               bool enable_profiling,
               const std::string& profiling_prefix)
      : selected_provider(ResolveProvider(requested_provider)),
        session(Environment(),
                std::filesystem::path(model_path).c_str(),
                MakeOptions(selected_provider, enable_profiling,
                            profiling_prefix)),
        profiling_enabled(enable_profiling) {
    Ort::AllocatorWithDefaultOptions allocator;
    const size_t input_count = session.GetInputCount();
    input_names.reserve(input_count);
    for (size_t index = 0; index < input_count; ++index) {
      auto name = session.GetInputNameAllocated(index, allocator);
      input_names.emplace_back(name.get());
      const auto type_info = session.GetInputTypeInfo(index);
      const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      if (tensor_info.GetElementType() !=
          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw Ort::Exception("Tensora currently supports float32 ONNX inputs",
                             ORT_NOT_IMPLEMENTED);
      }
    }

    const size_t output_count = session.GetOutputCount();
    output_names.reserve(output_count);
    for (size_t index = 0; index < output_count; ++index) {
      auto name = session.GetOutputNameAllocated(index, allocator);
      output_names.emplace_back(name.get());
      const auto type_info = session.GetOutputTypeInfo(index);
      const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      if (tensor_info.GetElementType() !=
          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw Ort::Exception("Tensora currently supports float32 ONNX outputs",
                             ORT_NOT_IMPLEMENTED);
      }
    }
  }

  static Ort::Env& Environment() {
    static Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "tensora");
    static const bool telemetry_disabled = [] {
      environment.DisableTelemetryEvents();
      return true;
    }();
    (void)telemetry_disabled;
    return environment;
  }

  static Ort::SessionOptions MakeOptions(const std::string& provider,
                                         bool enable_profiling,
                                         const std::string& prefix) {
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    AppendProvider(&options, provider);
    if (enable_profiling) {
      const std::filesystem::path native_prefix(
          prefix.empty() ? "tensora-profile" : prefix);
      options.EnableProfiling(native_prefix.c_str());
    }
    return options;
  }

  std::string selected_provider;
  Ort::Session session;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  bool profiling_enabled = false;
  bool profiling_ended = false;
  std::string profiling_path;
  std::mutex mutex;
};

Status OrtFailure(const char* operation, const Ort::Exception& error) {
  if (error.GetOrtErrorCode() == ORT_NO_SUCHFILE ||
      error.GetOrtErrorCode() == ORT_INVALID_GRAPH ||
      error.GetOrtErrorCode() == ORT_INVALID_PROTOBUF ||
      error.GetOrtErrorCode() == ORT_MODEL_LOADED) {
    return ModelError(std::string(operation) + ": " + error.what());
  }
  if (error.GetOrtErrorCode() == ORT_INVALID_ARGUMENT) {
    return InvalidArgument(std::string(operation) + ": " + error.what());
  }
  if (error.GetOrtErrorCode() == ORT_NOT_IMPLEMENTED) {
    return Unsupported(std::string(operation) + ": " + error.what());
  }
  return ModelError(std::string(operation) + ": " + error.what());
}

Status LookupSession(uint64_t handle, std::shared_ptr<SessionState>* out) {
  return HandleRegistry::Instance().Lookup<SessionState>(
      handle, HandleType::kInferenceSession, out);
}

Status TensorToHost(const Tensor& tensor,
                    std::vector<float>* out_values,
                    std::vector<int64_t>* out_shape) {
  if (out_values == nullptr || out_shape == nullptr) {
    return InvalidArgument("onnx input: output buffer pointer is null");
  }
  if (tensor.dtype() != DType::kFloat32) {
    return Unsupported("onnx input: only float32 tensors are supported");
  }

  out_values->assign(static_cast<size_t>(tensor.numel()), 0.0f);
  size_t written = 0;
  Status status = tensor.storage()->CopyToHostF32(
      out_values->data(), out_values->size(), &written);
  if (!status.ok()) return status;
  if (written != out_values->size()) {
    return InternalError("onnx input: storage returned an inconsistent size");
  }
  *out_shape = tensor.shape().dimensions;
  return Status::Ok();
}

Status OrtValueToTensor(Ort::Value* value, std::shared_ptr<Tensor>* out) {
  if (value == nullptr || out == nullptr) {
    return InvalidArgument("onnx output: output pointer is null");
  }
  *out = nullptr;
  if (!value->IsTensor()) {
    return Unsupported("onnx output: only tensor outputs are supported");
  }

  try {
    const auto info = value->GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return Unsupported("onnx output: only float32 outputs are supported");
    }
    const std::vector<int64_t> dimensions = info.GetShape();
    ShapeInfo shape;
    const int64_t* dims = dimensions.empty() ? nullptr : dimensions.data();
    Status status = ValidateShape(dims, dimensions.size(), &shape);
    if (!status.ok()) return status;

    const size_t numel = info.GetElementCount();
    if (shape.numel != static_cast<uint64_t>(numel)) {
      return ModelError("onnx output: shape and element count are inconsistent");
    }
    const float* data = value->GetTensorData<float>();
    std::shared_ptr<CpuStorage> storage;
    status = CpuStorage::FromData(data, shape.numel, &storage);
    if (!status.ok()) return status;
    try {
      *out = std::make_shared<Tensor>(std::move(shape), std::move(storage));
      return Status::Ok();
    } catch (const std::bad_alloc&) {
      return OutOfMemory("onnx output: Tensor allocation failed");
    }
  } catch (const Ort::Exception& error) {
    return OrtFailure("onnx output", error);
  }
}

}  // namespace

Status IsAvailable(uint8_t* out_available) {
  if (out_available == nullptr) {
    return InvalidArgument("onnx_available: output pointer is null");
  }
  *out_available = 1;
  return Status::Ok();
}

Status ProviderCount(size_t* out_count) {
  if (out_count == nullptr) {
    return InvalidArgument("onnx_provider_count: output pointer is null");
  }
  *out_count = 0;
  try {
    *out_count = Ort::GetAvailableProviders().size();
    return Status::Ok();
  } catch (const Ort::Exception& error) {
    return OrtFailure("onnx_provider_count", error);
  }
}

Status ProviderName(size_t index, std::string* out_name) {
  if (out_name == nullptr) {
    return InvalidArgument("onnx_provider_name: output string pointer is null");
  }
  out_name->clear();
  try {
    const auto providers = Ort::GetAvailableProviders();
    if (index >= providers.size()) {
      return InvalidArgument("onnx_provider_name: index is out of range");
    }
    *out_name = providers[index];
    return Status::Ok();
  } catch (const Ort::Exception& error) {
    return OrtFailure("onnx_provider_name", error);
  }
}

Status SessionCreate(const std::string& model_path,
                     const std::string& requested_provider,
                     bool enable_profiling,
                     const std::string& profiling_prefix,
                     uint64_t* out_session) {
  if (out_session == nullptr) {
    return InvalidArgument("onnx_session_create: output handle pointer is null");
  }
  *out_session = 0;
  if (model_path.empty()) {
    return InvalidArgument("onnx_session_create: model path is empty");
  }
  try {
    auto state = std::make_shared<SessionState>(
        model_path, requested_provider, enable_profiling, profiling_prefix);
    return HandleRegistry::Instance().Insert(
        HandleType::kInferenceSession, std::move(state), out_session);
  } catch (const Ort::Exception& error) {
    return OrtFailure("onnx_session_create", error);
  } catch (const std::bad_alloc&) {
    return OutOfMemory("onnx_session_create: allocation failed");
  }
}

Status SessionProvider(uint64_t session, std::string* out_provider) {
  if (out_provider == nullptr) {
    return InvalidArgument("onnx_session_provider: output string pointer is null");
  }
  out_provider->clear();
  std::shared_ptr<SessionState> state;
  Status status = LookupSession(session, &state);
  if (!status.ok()) return status;
  *out_provider = state->selected_provider;
  return Status::Ok();
}

Status SessionInputCount(uint64_t session, size_t* out_count) {
  if (out_count == nullptr) {
    return InvalidArgument("onnx_session_input_count: output pointer is null");
  }
  *out_count = 0;
  std::shared_ptr<SessionState> state;
  Status status = LookupSession(session, &state);
  if (!status.ok()) return status;
  *out_count = state->input_names.size();
  return Status::Ok();
}

Status SessionOutputCount(uint64_t session, size_t* out_count) {
  if (out_count == nullptr) {
    return InvalidArgument("onnx_session_output_count: output pointer is null");
  }
  *out_count = 0;
  std::shared_ptr<SessionState> state;
  Status status = LookupSession(session, &state);
  if (!status.ok()) return status;
  *out_count = state->output_names.size();
  return Status::Ok();
}

Status SessionInputName(uint64_t session, size_t index, std::string* out_name) {
  if (out_name == nullptr) {
    return InvalidArgument("onnx_session_input_name: output string pointer is null");
  }
  out_name->clear();
  std::shared_ptr<SessionState> state;
  Status status = LookupSession(session, &state);
  if (!status.ok()) return status;
  if (index >= state->input_names.size()) {
    return InvalidArgument("onnx_session_input_name: index is out of range");
  }
  *out_name = state->input_names[index];
  return Status::Ok();
}

Status SessionOutputName(uint64_t session, size_t index, std::string* out_name) {
  if (out_name == nullptr) {
    return InvalidArgument("onnx_session_output_name: output string pointer is null");
  }
  out_name->clear();
  std::shared_ptr<SessionState> state;
  Status status = LookupSession(session, &state);
  if (!status.ok()) return status;
  if (index >= state->output_names.size()) {
    return InvalidArgument("onnx_session_output_name: index is out of range");
  }
  *out_name = state->output_names[index];
  return Status::Ok();
}

Status SessionRun(uint64_t session,
                  const std::vector<std::string>& input_names,
                  const std::vector<std::shared_ptr<Tensor>>& inputs,
                  const std::vector<std::string>& output_names,
                  std::vector<std::shared_ptr<Tensor>>* out_tensors) {
  if (out_tensors == nullptr) {
    return InvalidArgument("onnx_session_run: output vector pointer is null");
  }
  out_tensors->clear();
  if (input_names.size() != inputs.size()) {
    return InvalidArgument("onnx_session_run: input name/tensor counts differ");
  }

  std::shared_ptr<SessionState> state;
  Status status = LookupSession(session, &state);
  if (!status.ok()) return status;
  if (input_names.size() != state->input_names.size()) {
    return InvalidArgument("onnx_session_run: input count does not match model");
  }
  if (output_names.empty() || output_names.size() > state->output_names.size()) {
    return InvalidArgument("onnx_session_run: invalid output count");
  }

  try {
    std::lock_guard<std::mutex> lock(state->mutex);

    std::vector<std::vector<float>> host_buffers(inputs.size());
    std::vector<std::vector<int64_t>> shapes(inputs.size());
    std::vector<Ort::Value> ort_inputs;
    ort_inputs.reserve(inputs.size());
    const Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    for (size_t index = 0; index < inputs.size(); ++index) {
      if (!inputs[index]) {
        return InvalidArgument("onnx_session_run: null input tensor object");
      }
      status = TensorToHost(*inputs[index], &host_buffers[index], &shapes[index]);
      if (!status.ok()) return status;
      ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(
          memory_info, host_buffers[index].data(), host_buffers[index].size(),
          shapes[index].data(), shapes[index].size()));
    }

    std::vector<const char*> input_name_ptrs;
    input_name_ptrs.reserve(input_names.size());
    for (const auto& name : input_names) input_name_ptrs.push_back(name.c_str());
    std::vector<const char*> output_name_ptrs;
    output_name_ptrs.reserve(output_names.size());
    for (const auto& name : output_names) output_name_ptrs.push_back(name.c_str());

    auto outputs = state->session.Run(
        Ort::RunOptions{nullptr}, input_name_ptrs.data(), ort_inputs.data(),
        ort_inputs.size(), output_name_ptrs.data(), output_name_ptrs.size());
    if (outputs.size() != output_names.size()) {
      return ModelError("onnx_session_run: runtime returned unexpected output count");
    }

    std::vector<std::shared_ptr<Tensor>> converted;
    converted.reserve(outputs.size());
    for (auto& output : outputs) {
      std::shared_ptr<Tensor> tensor;
      status = OrtValueToTensor(&output, &tensor);
      if (!status.ok()) return status;
      converted.push_back(std::move(tensor));
    }
    *out_tensors = std::move(converted);
    return Status::Ok();
  } catch (const Ort::Exception& error) {
    return OrtFailure("onnx_session_run", error);
  } catch (const std::bad_alloc&) {
    return OutOfMemory("onnx_session_run: allocation failed");
  }
}

Status SessionEndProfiling(uint64_t session, std::string* out_path) {
  if (out_path == nullptr) {
    return InvalidArgument("onnx_session_end_profiling: output string pointer is null");
  }
  out_path->clear();
  std::shared_ptr<SessionState> state;
  Status status = LookupSession(session, &state);
  if (!status.ok()) return status;

  std::lock_guard<std::mutex> lock(state->mutex);
  if (!state->profiling_enabled) {
    return InvalidArgument("onnx_session_end_profiling: profiling is not enabled");
  }
  if (state->profiling_ended) {
    *out_path = state->profiling_path;
    return Status::Ok();
  }

  try {
    Ort::AllocatorWithDefaultOptions allocator;
    auto path = state->session.EndProfilingAllocated(allocator);
    state->profiling_path = path.get();
    state->profiling_ended = true;
    *out_path = state->profiling_path;
    return Status::Ok();
  } catch (const Ort::Exception& error) {
    return OrtFailure("onnx_session_end_profiling", error);
  }
}

Status SessionRelease(uint64_t session) {
  return HandleRegistry::Instance().Release(
      session, HandleType::kInferenceSession);
}

}  // namespace tensora::inference
