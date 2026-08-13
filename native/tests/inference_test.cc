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
  bool cpu_provider = false;
  for (size_t index = 0; index < provider_count; ++index) {
    size_t required = 0;
    if (!Check(ts_onnx_provider_name(index, nullptr, 0, &required)) ||
        required == 0)
      return 5;
    std::vector<char> name(required);
    size_t second_required = 0;
    if (!Check(ts_onnx_provider_name(index, name.data(), name.size(),
                                     &second_required)) ||
        second_required != required)
      return 6;
    if (std::string(name.data()) == "CPUExecutionProvider") cpu_provider = true;
  }
  if (!cpu_provider) return 7;
  if (ts_onnx_provider_name(provider_count, nullptr, 0, &provider_count) !=
      TS_INVALID_ARGUMENT)
    return 8;

  ts_onnx_session_t missing = 123;
  if (ts_onnx_session_create("/definitely/missing/tensora-model.onnx", 0,
                             nullptr, &missing) != TS_MODEL_ERROR)
    return 9;
  if (missing != 0) return 10;

  uint64_t baseline_sessions = 0;
  uint64_t baseline_tensors = 0;
  if (!Check(ts_runtime_live_onnx_session_count(&baseline_sessions))) return 11;
  if (!Check(ts_runtime_live_tensor_count(&baseline_tensors))) return 12;

  ts_onnx_session_t session = 0;
  if (!Check(ts_onnx_session_create(model_path.c_str(), 0, nullptr, &session)) ||
      session == 0)
    return 13;
  size_t input_count = 0;
  size_t output_count = 0;
  if (!Check(ts_onnx_session_input_count(session, &input_count)) ||
      input_count != 1)
    return 14;
  if (!Check(ts_onnx_session_output_count(session, &output_count)) ||
      output_count != 1)
    return 15;
  if (ReadName(ts_onnx_session_input_name, session, 0) != "X") return 16;
  if (ReadName(ts_onnx_session_output_name, session, 0) != "Y") return 17;
  if (ts_onnx_session_input_name(session, 1, nullptr, 0, &input_count) !=
      TS_INVALID_ARGUMENT)
    return 18;

  const int64_t dims[2] = {2, 2};
  const float values[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  ts_tensor_t input = 0;
  if (!Check(ts_tensor_from_f32(values, 4, dims, 2, &input))) return 19;

  std::vector<float> result;
  if (!RunReference(session, input, &result)) return 20;
  const float expected[4] = {3.0f, 5.0f, 7.0f, 11.0f};
  for (size_t index = 0; index < 4; ++index) {
    if (!Close(result[index], expected[index])) return 21;
  }

  const char* wrong_input_names[1] = {"wrong"};
  const char* output_names[1] = {"Y"};
  ts_tensor_t bad_output = 99;
  size_t bad_written = 99;
  const ts_status_t wrong_name_status = ts_onnx_session_run(
      session, wrong_input_names, &input, 1, output_names, 1, &bad_output, 1,
      &bad_written);
  if (wrong_name_status == TS_OK || bad_output != 0 || bad_written != 0)
    return 22;

  if (ts_onnx_session_run(session, nullptr, nullptr, 0, output_names, 1,
                          &bad_output, 1, &bad_written) != TS_INVALID_ARGUMENT)
    return 23;
  if (bad_output != 0 || bad_written != 0) return 24;

  const int64_t wrong_dims[2] = {1, 4};
  ts_tensor_t wrong_shape = 0;
  if (!Check(ts_tensor_from_f32(values, 4, wrong_dims, 2, &wrong_shape)))
    return 25;
  const char* input_names[1] = {"X"};
  const ts_status_t wrong_shape_status = ts_onnx_session_run(
      session, input_names, &wrong_shape, 1, output_names, 1, &bad_output, 1,
      &bad_written);
  if (wrong_shape_status == TS_OK || bad_output != 0 || bad_written != 0)
    return 26;
  if (!Check(ts_tensor_release(wrong_shape))) return 27;

  if (ts_onnx_session_input_count(input, &input_count) != TS_INVALID_HANDLE)
    return 28;

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
  if (!concurrent_ok.load()) return 29;

  for (int iteration = 0; iteration < 10000; ++iteration) {
    std::vector<float> stress_result;
    if (!RunReference(session, input, &stress_result) ||
        !Close(stress_result[1], 5.0f))
      return 30;
  }

  if (!Check(ts_onnx_session_release(session))) return 31;
  if (ts_onnx_session_release(session) != TS_INVALID_HANDLE) return 32;

  const std::filesystem::path profile_prefix =
      std::filesystem::temp_directory_path() / "tensora-ort-profile";
  ts_onnx_session_t profiling_session = 0;
  if (!Check(ts_onnx_session_create(model_path.c_str(), 1,
                                    profile_prefix.string().c_str(),
                                    &profiling_session)))
    return 33;
  std::vector<float> profile_result;
  if (!RunReference(profiling_session, input, &profile_result)) return 34;
  std::vector<char> profile_path(4096);
  size_t profile_required = 0;
  if (!Check(ts_onnx_session_end_profiling(
          profiling_session, profile_path.data(), profile_path.size(),
          &profile_required)) ||
      profile_required == 0)
    return 35;
  if (!std::filesystem::exists(std::filesystem::path(profile_path.data())))
    return 36;
  std::filesystem::remove(std::filesystem::path(profile_path.data()));
  if (!Check(ts_onnx_session_release(profiling_session))) return 37;

  if (!Check(ts_tensor_release(input))) return 38;

  uint64_t final_sessions = 0;
  uint64_t final_tensors = 0;
  if (!Check(ts_runtime_live_onnx_session_count(&final_sessions)) ||
      final_sessions != baseline_sessions)
    return 39;
  if (!Check(ts_runtime_live_tensor_count(&final_tensors)) ||
      final_tensors != baseline_tensors)
    return 40;

  return 0;
}
