#include "tensora.h"

#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "core/status.h"
#include "inference/inference_bridge.h"
#include "runtime/handle_registry.h"
#include "tensor/tensor.h"

namespace tensora {
namespace {

template <typename Function>
ts_status_t GuardedInferenceAbiCall(const char* operation,
                                    Function&& function) noexcept {
  ClearLastError();
  try {
    Status status = function();
    if (!status.ok()) {
      if (status.message().empty()) {
        status = Status(status.code(), std::string(operation) + ": failed");
      }
      SetLastError(status);
    }
    return status.code();
  } catch (const std::bad_alloc&) {
    const Status status =
        OutOfMemory(std::string(operation) + ": native allocation failed");
    SetLastError(status);
    return status.code();
  } catch (const std::exception& error) {
    const Status status =
        InternalError(std::string(operation) + ": " + error.what());
    SetLastError(status);
    return status.code();
  } catch (...) {
    const Status status =
        InternalError(std::string(operation) + ": unknown native exception");
    SetLastError(status);
    return status.code();
  }
}

Status CopyUtf8String(const std::string& value,
                      char* out_value,
                      size_t capacity,
                      size_t* out_required,
                      const char* operation) {
  if (out_required == nullptr) {
    return InvalidArgument(std::string(operation) +
                           ": output required-size pointer is null");
  }
  const size_t required = value.size() + 1;
  *out_required = required;
  if (capacity == 0) {
    return Status::Ok();
  }
  if (out_value == nullptr) {
    return InvalidArgument(std::string(operation) +
                           ": output buffer pointer is null");
  }
  if (capacity < required) {
    return InvalidArgument(std::string(operation) +
                           ": output buffer capacity is too small");
  }
  std::memcpy(out_value, value.c_str(), required);
  return Status::Ok();
}

Status LookupInferenceTensor(ts_tensor_t handle,
                             std::shared_ptr<Tensor>* out) {
  return HandleRegistry::Instance().Lookup<Tensor>(
      handle, HandleType::kTensor, out);
}

Status InsertInferenceTensor(std::shared_ptr<Tensor> tensor,
                             ts_tensor_t* out_handle) {
  if (out_handle == nullptr) {
    return InvalidArgument("onnx output: output handle pointer is null");
  }
  *out_handle = 0;
  return HandleRegistry::Instance().Insert(
      HandleType::kTensor, std::move(tensor), out_handle);
}

void ReleaseInsertedOutputs(const std::vector<ts_tensor_t>& handles) {
  for (ts_tensor_t handle : handles) {
    if (handle != 0) {
      HandleRegistry::Instance().Release(handle, HandleType::kTensor);
    }
  }
}

}  // namespace
}  // namespace tensora

extern "C" {

ts_status_t ts_onnx_available(uint8_t* out_available) {
  return tensora::GuardedInferenceAbiCall("onnx_available", [&] {
    return tensora::inference::IsAvailable(out_available);
  });
}

ts_status_t ts_onnx_provider_count(size_t* out_count) {
  return tensora::GuardedInferenceAbiCall("onnx_provider_count", [&] {
    return tensora::inference::ProviderCount(out_count);
  });
}

ts_status_t ts_onnx_provider_name(size_t index,
                                  char* out_name,
                                  size_t capacity,
                                  size_t* out_required) {
  return tensora::GuardedInferenceAbiCall("onnx_provider_name", [&] {
    std::string name;
    tensora::Status status = tensora::inference::ProviderName(index, &name);
    if (!status.ok()) return status;
    return tensora::CopyUtf8String(
        name, out_name, capacity, out_required, "onnx_provider_name");
  });
}

ts_status_t ts_onnx_session_create(const char* model_path,
                                    uint8_t enable_profiling,
                                    const char* profiling_prefix,
                                    ts_onnx_session_t* out_session) {
  return ts_onnx_session_create_with_provider(
      model_path, "auto", enable_profiling, profiling_prefix, out_session);
}

ts_status_t ts_onnx_session_create_with_provider(
    const char* model_path,
    const char* requested_provider,
    uint8_t enable_profiling,
    const char* profiling_prefix,
    ts_onnx_session_t* out_session) {
  return tensora::GuardedInferenceAbiCall(
      "onnx_session_create_with_provider", [&] {
        if (out_session == nullptr) {
          return tensora::InvalidArgument(
              "onnx_session_create_with_provider: output handle pointer is null");
        }
        *out_session = 0;
        if (model_path == nullptr) {
          return tensora::InvalidArgument(
              "onnx_session_create_with_provider: model path pointer is null");
        }
        if (requested_provider == nullptr) {
          return tensora::InvalidArgument(
              "onnx_session_create_with_provider: provider pointer is null");
        }
        const std::string prefix = profiling_prefix == nullptr
                                       ? std::string()
                                       : std::string(profiling_prefix);
        return tensora::inference::SessionCreate(
            std::string(model_path), std::string(requested_provider),
            enable_profiling != 0, prefix, out_session);
      });
}

ts_status_t ts_onnx_session_provider(ts_onnx_session_t session,
                                     char* out_provider,
                                     size_t capacity,
                                     size_t* out_required) {
  return tensora::GuardedInferenceAbiCall("onnx_session_provider", [&] {
    std::string provider;
    tensora::Status status =
        tensora::inference::SessionProvider(session, &provider);
    if (!status.ok()) return status;
    return tensora::CopyUtf8String(
        provider, out_provider, capacity, out_required,
        "onnx_session_provider");
  });
}

ts_status_t ts_onnx_session_input_count(ts_onnx_session_t session,
                                        size_t* out_count) {
  return tensora::GuardedInferenceAbiCall("onnx_session_input_count", [&] {
    return tensora::inference::SessionInputCount(session, out_count);
  });
}

ts_status_t ts_onnx_session_output_count(ts_onnx_session_t session,
                                         size_t* out_count) {
  return tensora::GuardedInferenceAbiCall("onnx_session_output_count", [&] {
    return tensora::inference::SessionOutputCount(session, out_count);
  });
}

ts_status_t ts_onnx_session_input_name(ts_onnx_session_t session,
                                       size_t index,
                                       char* out_name,
                                       size_t capacity,
                                       size_t* out_required) {
  return tensora::GuardedInferenceAbiCall("onnx_session_input_name", [&] {
    std::string name;
    tensora::Status status =
        tensora::inference::SessionInputName(session, index, &name);
    if (!status.ok()) return status;
    return tensora::CopyUtf8String(
        name, out_name, capacity, out_required, "onnx_session_input_name");
  });
}

ts_status_t ts_onnx_session_output_name(ts_onnx_session_t session,
                                        size_t index,
                                        char* out_name,
                                        size_t capacity,
                                        size_t* out_required) {
  return tensora::GuardedInferenceAbiCall("onnx_session_output_name", [&] {
    std::string name;
    tensora::Status status =
        tensora::inference::SessionOutputName(session, index, &name);
    if (!status.ok()) return status;
    return tensora::CopyUtf8String(
        name, out_name, capacity, out_required, "onnx_session_output_name");
  });
}

ts_status_t ts_onnx_session_run(ts_onnx_session_t session,
                                const char* const* input_names,
                                const ts_tensor_t* input_tensors,
                                size_t input_count,
                                const char* const* output_names,
                                size_t output_count,
                                ts_tensor_t* out_tensors,
                                size_t out_capacity,
                                size_t* out_written) {
  return tensora::GuardedInferenceAbiCall("onnx_session_run", [&] {
    if (out_written == nullptr) {
      return tensora::InvalidArgument(
          "onnx_session_run: output count pointer is null");
    }
    *out_written = 0;
    if (input_count > 0 && (input_names == nullptr || input_tensors == nullptr)) {
      return tensora::InvalidArgument(
          "onnx_session_run: input arrays are null");
    }
    if (output_count == 0 || output_names == nullptr) {
      return tensora::InvalidArgument(
          "onnx_session_run: output names are null or empty");
    }
    if (out_capacity < output_count) {
      return tensora::InvalidArgument(
          "onnx_session_run: output handle capacity is too small");
    }
    if (out_tensors == nullptr) {
      return tensora::InvalidArgument(
          "onnx_session_run: output handle array is null");
    }
    for (size_t index = 0; index < output_count; ++index) {
      out_tensors[index] = 0;
    }

    std::vector<std::string> input_name_values;
    std::vector<std::shared_ptr<tensora::Tensor>> inputs;
    input_name_values.reserve(input_count);
    inputs.reserve(input_count);
    for (size_t index = 0; index < input_count; ++index) {
      if (input_names[index] == nullptr) {
        return tensora::InvalidArgument(
            "onnx_session_run: an input name pointer is null");
      }
      input_name_values.emplace_back(input_names[index]);
      std::shared_ptr<tensora::Tensor> tensor;
      tensora::Status status =
          tensora::LookupInferenceTensor(input_tensors[index], &tensor);
      if (!status.ok()) return status;
      inputs.push_back(std::move(tensor));
    }

    std::vector<std::string> output_name_values;
    output_name_values.reserve(output_count);
    for (size_t index = 0; index < output_count; ++index) {
      if (output_names[index] == nullptr) {
        return tensora::InvalidArgument(
            "onnx_session_run: an output name pointer is null");
      }
      output_name_values.emplace_back(output_names[index]);
    }

    std::vector<std::shared_ptr<tensora::Tensor>> results;
    tensora::Status status = tensora::inference::SessionRun(
        session, input_name_values, inputs, output_name_values, &results);
    if (!status.ok()) return status;
    if (results.size() != output_count) {
      return tensora::InternalError(
          "onnx_session_run: bridge returned unexpected output count");
    }

    std::vector<ts_tensor_t> inserted(output_count, 0);
    for (size_t index = 0; index < results.size(); ++index) {
      status = tensora::InsertInferenceTensor(
          std::move(results[index]), &inserted[index]);
      if (!status.ok()) {
        tensora::ReleaseInsertedOutputs(inserted);
        return status;
      }
    }
    for (size_t index = 0; index < inserted.size(); ++index) {
      out_tensors[index] = inserted[index];
    }
    *out_written = inserted.size();
    return tensora::Status::Ok();
  });
}

ts_status_t ts_onnx_session_end_profiling(ts_onnx_session_t session,
                                          char* out_path,
                                          size_t capacity,
                                          size_t* out_required) {
  return tensora::GuardedInferenceAbiCall("onnx_session_end_profiling", [&] {
    std::string path;
    tensora::Status status =
        tensora::inference::SessionEndProfiling(session, &path);
    if (!status.ok()) return status;
    return tensora::CopyUtf8String(
        path, out_path, capacity, out_required,
        "onnx_session_end_profiling");
  });
}

ts_status_t ts_onnx_session_release(ts_onnx_session_t session) {
  return tensora::GuardedInferenceAbiCall("onnx_session_release", [&] {
    return tensora::inference::SessionRelease(session);
  });
}

ts_status_t ts_runtime_live_onnx_session_count(uint64_t* out_count) {
  return tensora::GuardedInferenceAbiCall(
      "runtime_live_onnx_session_count", [&] {
        if (out_count == nullptr) {
          return tensora::InvalidArgument(
              "runtime_live_onnx_session_count: output pointer is null");
        }
        *out_count = tensora::HandleRegistry::Instance().Count(
            tensora::HandleType::kInferenceSession);
        return tensora::Status::Ok();
      });
}

}  // extern "C"
