#include <iostream>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "core/status.h"
#include "inference/onnx_policy.h"

namespace {

bool ExpectStatus(const tensora::Status& status,
                  ts_status_t expected,
                  const char* operation) {
  if (status.code() == expected) return true;
  std::cerr << operation << " expected status " << expected << ", got "
            << status.code() << ": " << status.message() << "\n";
  return false;
}

int CheckProviderResolution() {
  using tensora::inference::internal::ResolveProviderFromSnapshot;

  const std::vector<std::string> available = {
      "CPUExecutionProvider", "CUDAExecutionProvider"};
  const std::vector<std::string> configurable = {
      "CPUExecutionProvider", "CUDAExecutionProvider"};
  const std::vector<std::string> candidates = {
      "CUDAExecutionProvider", "CPUExecutionProvider"};

  std::string selected;
  if (!ExpectStatus(ResolveProviderFromSnapshot(
                        "auto", available, configurable, candidates, nullptr),
                    TS_INVALID_ARGUMENT, "provider null output"))
    return 1;
  if (!ExpectStatus(ResolveProviderFromSnapshot(
                        "UnknownProvider", available, configurable, candidates,
                        &selected),
                    TS_INVALID_ARGUMENT, "provider unknown"))
    return 2;
  if (!ExpectStatus(ResolveProviderFromSnapshot(
                        "auto", available, configurable, candidates, &selected),
                    TS_OK, "provider auto") ||
      selected != "CUDAExecutionProvider")
    return 3;
  if (!ExpectStatus(ResolveProviderFromSnapshot(
                        "auto", {"CPUExecutionProvider"},
                        {"CUDAExecutionProvider"}, candidates, &selected),
                    TS_UNSUPPORTED, "provider auto unavailable"))
    return 4;
  if (!ExpectStatus(ResolveProviderFromSnapshot(
                        "CUDAExecutionProvider", {"CPUExecutionProvider"},
                        configurable, candidates, &selected),
                    TS_UNSUPPORTED, "provider explicit unavailable"))
    return 5;
  if (!ExpectStatus(ResolveProviderFromSnapshot(
                        "CUDAExecutionProvider", available,
                        {"CPUExecutionProvider"}, candidates, &selected),
                    TS_UNSUPPORTED, "provider explicit disabled"))
    return 6;
  if (!ExpectStatus(ResolveProviderFromSnapshot(
                        "CPUExecutionProvider", available, configurable,
                        candidates, &selected),
                    TS_OK, "provider explicit CPU") ||
      selected != "CPUExecutionProvider")
    return 7;
  return 0;
}

int CheckTensorMetadataPolicies() {
  using tensora::inference::internal::ValidateFloatTensorType;
  using tensora::inference::internal::ValidateTensorElementCount;

  if (!ExpectStatus(
          ValidateFloatTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                  "onnx metadata"),
          TS_OK, "float metadata"))
    return 10;
  if (!ExpectStatus(
          ValidateFloatTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
                                  "onnx metadata"),
          TS_UNSUPPORTED, "double metadata"))
    return 11;
  if (!ExpectStatus(ValidateTensorElementCount(4, 4, "onnx output"), TS_OK,
                    "element count match"))
    return 12;
  if (!ExpectStatus(ValidateTensorElementCount(4, 3, "onnx output"),
                    TS_MODEL_ERROR, "element count mismatch"))
    return 13;
  return 0;
}

int CheckOrtErrorMapping() {
  using tensora::inference::internal::GuardOrt;
  using tensora::inference::internal::OrtFailure;

  const Ort::Exception no_file("missing", ORT_NO_SUCHFILE);
  const Ort::Exception invalid("invalid", ORT_INVALID_ARGUMENT);
  const Ort::Exception unsupported("unsupported", ORT_NOT_IMPLEMENTED);
  const Ort::Exception generic("generic", ORT_FAIL);

  if (!ExpectStatus(OrtFailure("open", no_file), TS_MODEL_ERROR,
                    "ORT no file"))
    return 20;
  if (!ExpectStatus(OrtFailure("argument", invalid), TS_INVALID_ARGUMENT,
                    "ORT invalid argument"))
    return 21;
  if (!ExpectStatus(OrtFailure("feature", unsupported), TS_UNSUPPORTED,
                    "ORT unsupported"))
    return 22;
  if (!ExpectStatus(OrtFailure("generic", generic), TS_MODEL_ERROR,
                    "ORT generic"))
    return 23;

  if (!ExpectStatus(
          GuardOrt("success", [] { return tensora::Status::Ok(); }), TS_OK,
          "ORT guard success"))
    return 24;
  if (!ExpectStatus(
          GuardOrt("guarded", []() -> tensora::Status {
            throw Ort::Exception("bad arg", ORT_INVALID_ARGUMENT);
          }),
          TS_INVALID_ARGUMENT, "ORT guard failure"))
    return 25;
  return 0;
}

}  // namespace

int main() {
  if (const int code = CheckProviderResolution(); code != 0) return code;
  if (const int code = CheckTensorMetadataPolicies(); code != 0) return code;
  if (const int code = CheckOrtErrorMapping(); code != 0) return code;
  return 0;
}
