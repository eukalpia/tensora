#ifndef TENSORA_INFERENCE_ONNX_POLICY_H_
#define TENSORA_INFERENCE_ONNX_POLICY_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "core/status.h"

namespace tensora::inference::internal {

inline bool ContainsProvider(const std::vector<std::string>& providers,
                             const std::string& provider) {
  return std::find(providers.begin(), providers.end(), provider) !=
         providers.end();
}

inline bool IsKnownProvider(const std::string& provider) {
  return provider == "auto" || provider == "CPUExecutionProvider" ||
         provider == "CUDAExecutionProvider" ||
         provider == "DmlExecutionProvider" ||
         provider == "CoreMLExecutionProvider" ||
         provider == "OpenVINOExecutionProvider" ||
         provider == "MIGraphXExecutionProvider";
}

inline Status ResolveProviderFromSnapshot(
    const std::string& requested_provider,
    const std::vector<std::string>& available,
    const std::vector<std::string>& configurable,
    const std::vector<std::string>& auto_candidates,
    std::string* out_selected) {
  if (out_selected == nullptr) {
    return InvalidArgument("onnx provider: output string pointer is null");
  }
  out_selected->clear();
  if (!IsKnownProvider(requested_provider)) {
    return InvalidArgument("onnx provider: unknown execution provider");
  }

  if (requested_provider == "auto") {
    for (const std::string& candidate : auto_candidates) {
      if (ContainsProvider(available, candidate) &&
          ContainsProvider(configurable, candidate)) {
        *out_selected = candidate;
        return Status::Ok();
      }
    }
    return Unsupported("onnx provider: no supported execution provider is available");
  }

  if (!ContainsProvider(available, requested_provider)) {
    return Unsupported("onnx provider: requested execution provider is not available");
  }
  if (!ContainsProvider(configurable, requested_provider)) {
    return Unsupported(
        "onnx provider: requested execution provider is not enabled in this Tensora build");
  }
  *out_selected = requested_provider;
  return Status::Ok();
}

inline Status ValidateFloatTensorType(ONNXTensorElementDataType element_type,
                                      const char* operation) {
  if (element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return Unsupported(std::string(operation) +
                       ": only float32 tensors are supported");
  }
  return Status::Ok();
}

inline Status ValidateTensorElementCount(uint64_t expected,
                                         size_t actual,
                                         const char* operation) {
  if (expected != static_cast<uint64_t>(actual)) {
    return ModelError(std::string(operation) +
                      ": shape and element count are inconsistent");
  }
  return Status::Ok();
}

inline Status OrtFailure(const char* operation, const Ort::Exception& error) {
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

template <typename Operation>
Status GuardOrt(const char* operation, Operation&& body) {
  try {
    return std::forward<Operation>(body)();
  } catch (const Ort::Exception& error) {
    return OrtFailure(operation, error);
  }
}

}  // namespace tensora::inference::internal

#endif  // TENSORA_INFERENCE_ONNX_POLICY_H_
