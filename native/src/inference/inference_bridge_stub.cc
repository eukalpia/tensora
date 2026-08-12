#include "inference/inference_bridge.h"

namespace tensora::inference {
namespace {

Status Disabled(const char* operation) {
  return Unsupported(std::string(operation) +
                     ": ONNX Runtime backend is not enabled");
}

}  // namespace

Status IsAvailable(uint8_t* out_available) {
  if (out_available == nullptr) {
    return InvalidArgument("onnx_available: output pointer is null");
  }
  *out_available = 0;
  return Status::Ok();
}

Status ProviderCount(size_t* out_count) {
  if (out_count == nullptr) {
    return InvalidArgument("onnx_provider_count: output pointer is null");
  }
  *out_count = 0;
  return Status::Ok();
}

Status ProviderName(size_t, std::string* out_name) {
  if (out_name == nullptr) {
    return InvalidArgument("onnx_provider_name: output string pointer is null");
  }
  out_name->clear();
  return Disabled("onnx_provider_name");
}

Status SessionCreate(const std::string&,
                     bool,
                     const std::string&,
                     uint64_t* out_session) {
  if (out_session == nullptr) {
    return InvalidArgument("onnx_session_create: output handle pointer is null");
  }
  *out_session = 0;
  return Disabled("onnx_session_create");
}

Status SessionInputCount(uint64_t, size_t* out_count) {
  if (out_count == nullptr) {
    return InvalidArgument("onnx_session_input_count: output pointer is null");
  }
  *out_count = 0;
  return Disabled("onnx_session_input_count");
}

Status SessionOutputCount(uint64_t, size_t* out_count) {
  if (out_count == nullptr) {
    return InvalidArgument("onnx_session_output_count: output pointer is null");
  }
  *out_count = 0;
  return Disabled("onnx_session_output_count");
}

Status SessionInputName(uint64_t, size_t, std::string* out_name) {
  if (out_name == nullptr) {
    return InvalidArgument("onnx_session_input_name: output string pointer is null");
  }
  out_name->clear();
  return Disabled("onnx_session_input_name");
}

Status SessionOutputName(uint64_t, size_t, std::string* out_name) {
  if (out_name == nullptr) {
    return InvalidArgument("onnx_session_output_name: output string pointer is null");
  }
  out_name->clear();
  return Disabled("onnx_session_output_name");
}

Status SessionRun(uint64_t,
                  const std::vector<std::string>&,
                  const std::vector<std::shared_ptr<Tensor>>&,
                  const std::vector<std::string>&,
                  std::vector<std::shared_ptr<Tensor>>* out_tensors) {
  if (out_tensors == nullptr) {
    return InvalidArgument("onnx_session_run: output vector pointer is null");
  }
  out_tensors->clear();
  return Disabled("onnx_session_run");
}

Status SessionEndProfiling(uint64_t, std::string* out_path) {
  if (out_path == nullptr) {
    return InvalidArgument("onnx_session_end_profiling: output string pointer is null");
  }
  out_path->clear();
  return Disabled("onnx_session_end_profiling");
}

Status SessionRelease(uint64_t) { return Disabled("onnx_session_release"); }

}  // namespace tensora::inference
