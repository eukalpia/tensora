#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "../src/inference/inference_bridge_onnx.cc"

namespace tensora::inference {

namespace {

bool ExpectStatus(const Status& status,
                  ts_status_t expected,
                  const char* operation) {
  if (status.code() == expected) return true;
  std::cerr << operation << " expected status " << expected << ", got "
            << status.code() << ": " << status.message() << "\n";
  return false;
}

int CheckProviderHelpers() {
  if (CanConfigureProvider("DefinitelyUnknownProvider")) return 1;

  try {
    (void)ResolveAutoProvider({});
    return 2;
  } catch (const Ort::Exception& error) {
    if (error.GetOrtErrorCode() != ORT_NOT_IMPLEMENTED) return 3;
  }

  try {
    (void)ResolveProvider("DefinitelyUnknownProvider");
    return 4;
  } catch (const Ort::Exception& error) {
    if (error.GetOrtErrorCode() != ORT_INVALID_ARGUMENT) return 5;
  }

  try {
    AppendProvider(nullptr, kCpuProvider);
    return 6;
  } catch (const Ort::Exception& error) {
    if (error.GetOrtErrorCode() != ORT_INVALID_ARGUMENT) return 7;
  }

  Ort::SessionOptions cpu_options;
  AppendProvider(&cpu_options, kCpuProvider);

  for (const char* provider :
       {kCudaProvider, kOpenVinoProvider, kMiGraphXProvider}) {
    Ort::SessionOptions options;
    try {
      AppendProvider(&options, provider);
    } catch (const Ort::Exception&) {
      // A CPU-only ONNX Runtime distribution may reject the provider. The
      // contract here is that provider-specific configuration is attempted
      // and any ORT failure remains contained by higher-level session APIs.
    }
  }

  try {
    Ort::SessionOptions options;
    AppendProvider(&options, "DefinitelyUnknownProvider");
    return 8;
  } catch (const Ort::Exception& error) {
    if (error.GetOrtErrorCode() != ORT_NOT_IMPLEMENTED) return 9;
  }

  try {
    (void)SessionState::MakeOptions("DefinitelyUnknownProvider", false, "");
    return 10;
  } catch (const Ort::Exception& error) {
    if (error.GetOrtErrorCode() != ORT_NOT_IMPLEMENTED) return 11;
  }
  return 0;
}

int CheckLocalOrtMapping() {
  if (!ExpectStatus(OrtFailure(
                        "missing", Ort::Exception("missing", ORT_NO_SUCHFILE)),
                    TS_MODEL_ERROR, "local ORT missing"))
    return 20;
  if (!ExpectStatus(OrtFailure(
                        "invalid",
                        Ort::Exception("invalid", ORT_INVALID_ARGUMENT)),
                    TS_INVALID_ARGUMENT, "local ORT invalid"))
    return 21;
  if (!ExpectStatus(OrtFailure(
                        "unsupported",
                        Ort::Exception("unsupported", ORT_NOT_IMPLEMENTED)),
                    TS_UNSUPPORTED, "local ORT unsupported"))
    return 22;
  if (!ExpectStatus(
          OrtFailure("generic", Ort::Exception("generic", ORT_FAIL)),
          TS_MODEL_ERROR, "local ORT generic"))
    return 23;
  return 0;
}

int CheckConversionContracts() {
  const int64_t dims[1] = {1};
  ShapeInfo shape;
  if (!ValidateShape(dims, 1, &shape).ok()) return 30;
  const float value = 1.0f;
  std::shared_ptr<CpuStorage> storage;
  if (!CpuStorage::FromData(&value, 1, &storage).ok()) return 31;
  Tensor non_float(std::move(shape), std::move(storage),
                   static_cast<DType>(TS_DTYPE_FLOAT16));

  std::vector<float> values;
  std::vector<int64_t> dimensions;
  if (!ExpectStatus(TensorToHost(non_float, &values, &dimensions),
                    TS_UNSUPPORTED, "input dtype"))
    return 32;
  if (!ExpectStatus(TensorToHost(non_float, nullptr, &dimensions),
                    TS_INVALID_ARGUMENT, "input null values"))
    return 33;

  std::shared_ptr<Tensor> output;
  if (!ExpectStatus(OrtValueToTensor(nullptr, &output), TS_INVALID_ARGUMENT,
                    "output null value"))
    return 34;

  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  double double_value = 2.0;
  int64_t output_dims[1] = {1};
  Ort::Value double_tensor = Ort::Value::CreateTensor<double>(
      memory_info, &double_value, 1, output_dims, 1);
  if (!ExpectStatus(OrtValueToTensor(&double_tensor, &output), TS_UNSUPPORTED,
                    "output dtype"))
    return 35;
  return 0;
}

}  // namespace

int RunWhiteboxContract() {
  if (const int code = CheckProviderHelpers(); code != 0) return code;
  if (const int code = CheckLocalOrtMapping(); code != 0) return code;
  if (const int code = CheckConversionContracts(); code != 0) return code;
  return 0;
}

}  // namespace tensora::inference

int main() { return tensora::inference::RunWhiteboxContract(); }
