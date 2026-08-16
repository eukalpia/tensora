#include "tensora.h"

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace {

bool Check(ts_status_t status) { return status == TS_OK; }

bool ExpectStatus(ts_status_t actual,
                  ts_status_t expected,
                  const char* operation) {
  if (actual == expected) return true;
  std::fprintf(stderr, "%s expected status %d, got %d: %s\n", operation,
               expected, actual, ts_last_error_message());
  return false;
}

bool Close(float actual, float expected, float tolerance = 1e-5f) {
  return std::fabs(actual - expected) <= tolerance;
}

std::string ReadEnvironment(const char* name) {
  if (name == nullptr) return {};
#if defined(_WIN32)
  char* value = nullptr;
  size_t length = 0;
  if (_dupenv_s(&value, &length, name) != 0 || value == nullptr) {
    std::free(value);
    return {};
  }
  std::string result(value, length > 0 ? length - 1 : 0);
  std::free(value);
  return result;
#else
  const char* value = std::getenv(name);
  return value == nullptr ? std::string() : std::string(value);
#endif
}

std::string ReadName(ts_status_t (*function)(ts_onnx_session_t,
                                              size_t,
                                              char*,
                                              size_t,
                                              size_t*),
                     ts_onnx_session_t session,
                     size_t index) {
  size_t required = 0;
  if (!Check(function(session, index, nullptr, 0, &required)) || required == 0) {
    return {};
  }
  std::vector<char> buffer(required);
  size_t second_required = 0;
  if (!Check(function(session, index, buffer.data(), buffer.size(),
                      &second_required)) ||
      second_required != required) {
    return {};
  }
  return std::string(buffer.data());
}

bool RunReference(ts_onnx_session_t session,
                  ts_tensor_t input,
                  std::vector<float>* out_values) {
  const char* input_names[1] = {"X"};
  const char* output_names[1] = {"Y"};
  ts_tensor_t output = 0;
  size_t written = 0;
  const ts_status_t status = ts_onnx_session_run(
      session, input_names, &input, 1, output_names, 1, &output, 1, &written);
  if (status != TS_OK || output == 0 || written != 1) return false;

  uint64_t numel = 0;
  if (!Check(ts_tensor_numel(output, &numel)) || numel != 4) {
    ts_tensor_release(output);
    return false;
  }
  out_values->assign(4, 0.0f);
  size_t copied = 0;
  const bool ok = Check(ts_tensor_copy_to_host_f32(
      output, out_values->data(), out_values->size(), &copied));
  const bool released = Check(ts_tensor_release(output));
  return ok && released && copied == 4;
}

}  // namespace

int main() {
  const std::string model_path = ReadEnvironment("TENSORA_ONNX_TEST_MODEL");
  if (model_path.empty()) return 1;

  uint8_t available = 0;
  if (!Check(ts_onnx_available(&available)) || available != 1) return 2;
  if (ts_onnx_available(nullptr) != TS_INVALID_ARGUMENT) return 3;

  size_t provider_count = 0;
  if (!Check(ts_onnx_provider_count(&provider_count)) || provider_count == 0)
    return 4;
  if (!ExpectStatus(ts_onnx_provider_count(nullptr), TS_INVALID_ARGUMENT,
                    "provider count null"))
    return 5;

  bool cpu_provider = false;
  for (size_t index = 0; index < provider_count; ++index) {
    size_t required = 0;
    if (!Check(ts_onnx_provider_name(index, nullptr, 0, &required)) ||
        required == 0)
      return 6;
    std::vector<char> name(required);
    size_t second_required = 0;
    if (!Check(ts_onnx_provider_name(index, name.data(), name.size(),
                                     &second_required)) ||
        second_required != required)
      return 7;
    if (std::string(name.data()) == "CPUExecutionProvider") cpu_provider = true;
  }
  if (!cpu_provider) return 8;

  size_t provider_required = 0;
  if (!Check(ts_onnx_provider_name(0, nullptr, 0, &provider_required)) ||
      provider_required < 2)
    return 9;
  std::vector<char> provider_buffer(provider_required);
  if (!ExpectStatus(ts_onnx_provider_name(0, provider_buffer.data(),
                                          provider_buffer.size(), nullptr),
                    TS_INVALID_ARGUMENT, "provider required pointer null"))
    return 10;
  if (!ExpectStatus(ts_onnx_provider_name(0, nullptr, provider_required,
                                          &provider_required),
                    TS_INVALID_ARGUMENT, "provider output buffer null"))
    return 11;
  if (!ExpectStatus(ts_onnx_provider_name(0, provider_buffer.data(),
                                          provider_required - 1,
                                          &provider_required),
                    TS_INVALID_ARGUMENT, "provider output too small"))
    return 12;
  if (ts_onnx_provider_name(provider_count, nullptr, 0, &provider_count) !=
      TS_INVALID_ARGUMENT)
    return 13;

  ts_onnx_session_t missing = 123;
  if (!ExpectStatus(ts_onnx_session_create(nullptr, 0, nullptr, &missing),
                    TS_INVALID_ARGUMENT, "session null model path"))
    return 14;
  if (!ExpectStatus(ts_onnx_session_create("", 0, nullptr, &missing),
                    TS_INVALID_ARGUMENT, "session empty model path"))
    return 15;
  if (!ExpectStatus(ts_onnx_session_create(model_path.c_str(), 0, nullptr,
                                           nullptr),
                    TS_INVALID_ARGUMENT, "session null output handle"))
    return 16;
  if (!ExpectStatus(ts_onnx_session_create_with_provider(
                        model_path.c_str(), nullptr, 0, nullptr, &missing),
                    TS_INVALID_ARGUMENT, "session null provider"))
    return 17;
  if (!ExpectStatus(ts_onnx_session_create_with_provider(
                        model_path.c_str(), "DefinitelyUnknownProvider", 0,
                        nullptr, &missing),
                    TS_INVALID_ARGUMENT, "session unknown provider"))
    return 18;
  if (!ExpectStatus(ts_onnx_session_create_with_provider(
                        model_path.c_str(), "CUDAExecutionProvider", 0, nullptr,
                        &missing),
                    TS_UNSUPPORTED, "session unavailable CUDA provider"))
    return 19;

  if (ts_onnx_session_create("/definitely/missing/tensora-model.onnx", 0,
                             nullptr, &missing) != TS_MODEL_ERROR)
    return 20;
  if (missing != 0) return 21;

  uint64_t baseline_sessions = 0;
  uint64_t baseline_tensors = 0;
  if (!Check(ts_runtime_live_onnx_session_count(&baseline_sessions))) return 22;
  if (!Check(ts_runtime_live_tensor_count(&baseline_tensors))) return 23;
  if (!ExpectStatus(ts_runtime_live_onnx_session_count(nullptr),
                    TS_INVALID_ARGUMENT, "live session count null"))
    return 24;

  ts_onnx_session_t session = 0;
  if (!Check(ts_onnx_session_create(model_path.c_str(), 0, nullptr, &session)) ||
      session == 0)
    return 25;

  size_t provider_name_required = 0;
  if (!Check(ts_onnx_session_provider(session, nullptr, 0,
                                      &provider_name_required)) ||
      provider_name_required == 0)
    return 26;
  std::vector<char> selected_provider(provider_name_required);
  if (!Check(ts_onnx_session_provider(session, selected_provider.data(),
                                      selected_provider.size(),
                                      &provider_name_required)))
    return 27;
  if (std::string(selected_provider.data()).empty()) return 28;
  if (!ExpectStatus(ts_onnx_session_provider(session, nullptr,
                                             provider_name_required,
                                             &provider_name_required),
                    TS_INVALID_ARGUMENT, "session provider null buffer"))
    return 29;
  if (!ExpectStatus(ts_onnx_session_provider(session, selected_provider.data(),
                                             selected_provider.size(), nullptr),
                    TS_INVALID_ARGUMENT, "session provider null required"))
    return 30;
  if (!ExpectStatus(ts_onnx_session_provider(UINT64_C(999999999), nullptr, 0,
                                             &provider_name_required),
                    TS_INVALID_HANDLE, "session provider invalid handle"))
    return 31;

  size_t input_count = 0;
  size_t output_count = 0;
  if (!Check(ts_onnx_session_input_count(session, &input_count)) ||
      input_count != 1)
    return 32;
  if (!Check(ts_onnx_session_output_count(session, &output_count)) ||
      output_count != 1)
    return 33;
  if (!ExpectStatus(ts_onnx_session_input_count(session, nullptr),
                    TS_INVALID_ARGUMENT, "input count null"))
    return 34;
  if (!ExpectStatus(ts_onnx_session_output_count(session, nullptr),
                    TS_INVALID_ARGUMENT, "output count null"))
    return 35;
  if (!ExpectStatus(ts_onnx_session_output_count(UINT64_C(999999999),
                                                 &output_count),
                    TS_INVALID_HANDLE, "output count invalid session"))
    return 36;

  if (ReadName(ts_onnx_session_input_name, session, 0) != "X") return 37;
  if (ReadName(ts_onnx_session_output_name, session, 0) != "Y") return 38;
  if (ts_onnx_session_input_name(session, 1, nullptr, 0, &input_count) !=
      TS_INVALID_ARGUMENT)
    return 39;
  if (!ExpectStatus(ts_onnx_session_output_name(session, 1, nullptr, 0,
                                                &output_count),
                    TS_INVALID_ARGUMENT, "output name out of range"))
    return 40;

  size_t input_name_required = 0;
  if (!Check(ts_onnx_session_input_name(session, 0, nullptr, 0,
                                        &input_name_required)) ||
      input_name_required < 2)
    return 41;
  std::vector<char> input_name_buffer(input_name_required);
  if (!ExpectStatus(ts_onnx_session_input_name(
                        session, 0, input_name_buffer.data(),
                        input_name_buffer.size(), nullptr),
                    TS_INVALID_ARGUMENT, "input name null required"))
    return 42;
  if (!ExpectStatus(ts_onnx_session_input_name(session, 0, nullptr,
                                               input_name_required,
                                               &input_name_required),
                    TS_INVALID_ARGUMENT, "input name null buffer"))
    return 43;
  if (!ExpectStatus(ts_onnx_session_input_name(
                        session, 0, input_name_buffer.data(),
                        input_name_required - 1, &input_name_required),
                    TS_INVALID_ARGUMENT, "input name too small"))
    return 44;

  const int64_t dims[2] = {2, 2};
  const float values[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  ts_tensor_t input = 0;
  if (!Check(ts_tensor_from_f32(values, 4, dims, 2, &input))) return 45;

  std::vector<float> result;
  if (!RunReference(session, input, &result)) return 46;
  const float expected[4] = {3.0f, 5.0f, 7.0f, 11.0f};
  for (size_t index = 0; index < 4; ++index) {
    if (!Close(result[index], expected[index])) return 47;
  }

  const char* input_names[1] = {"X"};
  const char* output_names[1] = {"Y"};
  const char* wrong_input_names[1] = {"wrong"};
  const char* wrong_output_names[1] = {"wrong"};
  const char* null_names[1] = {nullptr};
  ts_tensor_t bad_output = 99;
  size_t bad_written = 99;

  const ts_status_t wrong_name_status = ts_onnx_session_run(
      session, wrong_input_names, &input, 1, output_names, 1, &bad_output, 1,
      &bad_written);
  if (wrong_name_status == TS_OK || bad_output != 0 || bad_written != 0)
    return 48;
  const ts_status_t wrong_output_status = ts_onnx_session_run(
      session, input_names, &input, 1, wrong_output_names, 1, &bad_output, 1,
      &bad_written);
  if (wrong_output_status == TS_OK || bad_output != 0 || bad_written != 0)
    return 49;

  if (!ExpectStatus(ts_onnx_session_run(session, input_names, &input, 1,
                                        output_names, 1, &bad_output, 1,
                                        nullptr),
                    TS_INVALID_ARGUMENT, "run null written"))
    return 50;
  if (!ExpectStatus(ts_onnx_session_run(session, nullptr, &input, 1,
                                        output_names, 1, &bad_output, 1,
                                        &bad_written),
                    TS_INVALID_ARGUMENT, "run null input names"))
    return 51;
  if (!ExpectStatus(ts_onnx_session_run(session, input_names, nullptr, 1,
                                        output_names, 1, &bad_output, 1,
                                        &bad_written),
                    TS_INVALID_ARGUMENT, "run null input tensors"))
    return 52;
  if (!ExpectStatus(ts_onnx_session_run(session, input_names, &input, 1,
                                        nullptr, 1, &bad_output, 1,
                                        &bad_written),
                    TS_INVALID_ARGUMENT, "run null output names"))
    return 53;
  if (!ExpectStatus(ts_onnx_session_run(session, input_names, &input, 1,
                                        output_names, 0, &bad_output, 1,
                                        &bad_written),
                    TS_INVALID_ARGUMENT, "run zero outputs"))
    return 54;
  if (!ExpectStatus(ts_onnx_session_run(session, input_names, &input, 1,
                                        output_names, 1, &bad_output, 0,
                                        &bad_written),
                    TS_INVALID_ARGUMENT, "run insufficient output capacity"))
    return 55;
  if (!ExpectStatus(ts_onnx_session_run(session, input_names, &input, 1,
                                        output_names, 1, nullptr, 1,
                                        &bad_written),
                    TS_INVALID_ARGUMENT, "run null output handles"))
    return 56;
  if (!ExpectStatus(ts_onnx_session_run(session, null_names, &input, 1,
                                        output_names, 1, &bad_output, 1,
                                        &bad_written),
                    TS_INVALID_ARGUMENT, "run null input name element"))
    return 57;
  if (!ExpectStatus(ts_onnx_session_run(session, input_names, &input, 1,
                                        null_names, 1, &bad_output, 1,
                                        &bad_written),
                    TS_INVALID_ARGUMENT, "run null output name element"))
    return 58;
  const ts_tensor_t invalid_input = UINT64_C(999999999);
  if (!ExpectStatus(ts_onnx_session_run(session, input_names, &invalid_input, 1,
                                        output_names, 1, &bad_output, 1,
                                        &bad_written),
                    TS_INVALID_HANDLE, "run invalid tensor handle"))
    return 59;
  if (!ExpectStatus(ts_onnx_session_run(UINT64_C(999999999), input_names,
                                        &input, 1, output_names, 1,
                                        &bad_output, 1, &bad_written),
                    TS_INVALID_HANDLE, "run invalid session handle"))
    return 60;

  const char* two_input_names[2] = {"X", "X"};
  const ts_tensor_t two_inputs[2] = {input, input};
  if (!ExpectStatus(ts_onnx_session_run(session, two_input_names, two_inputs, 2,
                                        output_names, 1, &bad_output, 1,
                                        &bad_written),
                    TS_INVALID_ARGUMENT, "run input count mismatch"))
    return 61;
  const char* two_output_names[2] = {"Y", "Y"};
  ts_tensor_t two_outputs[2] = {99, 99};
  if (!ExpectStatus(ts_onnx_session_run(session, input_names, &input, 1,
                                        two_output_names, 2, two_outputs, 2,
                                        &bad_written),
                    TS_INVALID_ARGUMENT, "run output count mismatch"))
    return 62;

  if (ts_onnx_session_run(session, nullptr, nullptr, 0, output_names, 1,
                          &bad_output, 1, &bad_written) != TS_INVALID_ARGUMENT)
    return 63;
  if (bad_output != 0 || bad_written != 0) return 64;

  const int64_t wrong_dims[2] = {1, 4};
  ts_tensor_t wrong_shape = 0;
  if (!Check(ts_tensor_from_f32(values, 4, wrong_dims, 2, &wrong_shape)))
    return 65;
  const ts_status_t wrong_shape_status = ts_onnx_session_run(
      session, input_names, &wrong_shape, 1, output_names, 1, &bad_output, 1,
      &bad_written);
  if (wrong_shape_status == TS_OK || bad_output != 0 || bad_written != 0)
    return 66;
  if (!Check(ts_tensor_release(wrong_shape))) return 67;

  if (ts_onnx_session_input_count(input, &input_count) != TS_INVALID_HANDLE)
    return 68;
  if (!ExpectStatus(ts_onnx_session_end_profiling(session, nullptr, 0,
                                                  &provider_required),
                    TS_INVALID_ARGUMENT, "end profiling disabled"))
    return 69;

  std::atomic<bool> concurrent_ok{true};
  std::vector<std::thread> threads;
  for (int thread_index = 0; thread_index < 8; ++thread_index) {
    threads.emplace_back([&] {
      for (int iteration = 0; iteration < 100; ++iteration) {
        std::vector<float> concurrent_result;
        if (!RunReference(session, input, &concurrent_result) ||
            concurrent_result.size() != 4 ||
            !Close(concurrent_result[0], 3.0f) ||
            !Close(concurrent_result[3], 11.0f)) {
          concurrent_ok.store(false);
          return;
        }
      }
    });
  }
  for (auto& thread : threads) thread.join();
  if (!concurrent_ok.load()) return 70;

  for (int iteration = 0; iteration < 10000; ++iteration) {
    std::vector<float> stress_result;
    if (!RunReference(session, input, &stress_result) ||
        !Close(stress_result[1], 5.0f))
      return 71;
  }

  if (!Check(ts_onnx_session_release(session))) return 72;
  if (ts_onnx_session_release(session) != TS_INVALID_HANDLE) return 73;
  if (!ExpectStatus(ts_onnx_session_release(input), TS_INVALID_HANDLE,
                    "release wrong handle type"))
    return 74;

  const std::filesystem::path profile_prefix =
      std::filesystem::temp_directory_path() / "tensora-ort-profile";
  ts_onnx_session_t profiling_session = 0;
  if (!Check(ts_onnx_session_create(model_path.c_str(), 1,
                                    profile_prefix.string().c_str(),
                                    &profiling_session)))
    return 75;
  std::vector<float> profile_result;
  if (!RunReference(profiling_session, input, &profile_result)) return 76;

  if (!ExpectStatus(ts_onnx_session_end_profiling(
                        profiling_session, nullptr, 0, nullptr),
                    TS_INVALID_ARGUMENT, "profiling null required"))
    return 77;

  size_t profile_required = 0;
  if (!Check(ts_onnx_session_end_profiling(profiling_session, nullptr, 0,
                                           &profile_required)) ||
      profile_required < 2)
    return 78;
  std::vector<char> profile_path(profile_required);
  if (!ExpectStatus(ts_onnx_session_end_profiling(
                        profiling_session, nullptr, profile_required,
                        &profile_required),
                    TS_INVALID_ARGUMENT, "profiling null buffer"))
    return 79;
  if (!ExpectStatus(ts_onnx_session_end_profiling(
                        profiling_session, profile_path.data(),
                        profile_required - 1, &profile_required),
                    TS_INVALID_ARGUMENT, "profiling buffer too small"))
    return 80;
  if (!Check(ts_onnx_session_end_profiling(
          profiling_session, profile_path.data(), profile_path.size(),
          &profile_required)) ||
      profile_required == 0)
    return 81;
  if (!std::filesystem::exists(std::filesystem::path(profile_path.data())))
    return 82;
  std::filesystem::remove(std::filesystem::path(profile_path.data()));
  if (!Check(ts_onnx_session_release(profiling_session))) return 83;

  if (!Check(ts_tensor_release(input))) return 84;

  uint64_t final_sessions = 0;
  uint64_t final_tensors = 0;
  if (!Check(ts_runtime_live_onnx_session_count(&final_sessions)) ||
      final_sessions != baseline_sessions)
    return 85;
  if (!Check(ts_runtime_live_tensor_count(&final_tensors)) ||
      final_tensors != baseline_tensors)
    return 86;

  return 0;
}
