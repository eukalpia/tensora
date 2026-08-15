#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "inference/inference_bridge.h"
#include "memory/cpu_storage.h"
#include "memory/tensor_storage.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

namespace {

using tensora::CpuStorage;
using tensora::ShapeInfo;
using tensora::Status;
using tensora::Tensor;

bool ExpectStatus(const Status& status,
                  ts_status_t expected,
                  const char* operation) {
  if (status.code() == expected) return true;
  std::cerr << operation << " expected status " << expected << ", got "
            << status.code() << ": " << status.message() << "\n";
  return false;
}

std::shared_ptr<Tensor> MakeInput() {
  const int64_t dims[2] = {2, 2};
  const float values[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  ShapeInfo shape;
  if (!tensora::ValidateShape(dims, 2, &shape).ok()) return nullptr;
  std::shared_ptr<CpuStorage> storage;
  if (!CpuStorage::FromData(values, 4, &storage).ok()) return nullptr;
  return std::make_shared<Tensor>(std::move(shape), std::move(storage));
}

class ShortWriteStorage final : public tensora::TensorStorage {
 public:
  tensora::StorageKind kind() const override {
    return tensora::StorageKind::kCpu;
  }

  uint64_t byte_size() const override { return 4 * sizeof(float); }

  Status CopyToHostF32(float*, size_t, size_t* out_written) const override {
    if (out_written != nullptr) *out_written = 3;
    return Status::Ok();
  }
};

std::shared_ptr<Tensor> MakeShortWriteInput() {
  const int64_t dims[2] = {2, 2};
  ShapeInfo shape;
  if (!tensora::ValidateShape(dims, 2, &shape).ok()) return nullptr;
  return std::make_shared<Tensor>(std::move(shape),
                                  std::make_shared<ShortWriteStorage>());
}

int Run(const std::string& model_path) {
  if (!ExpectStatus(tensora::inference::ProviderName(0, nullptr),
                    TS_INVALID_ARGUMENT, "provider name null"))
    return 1;
  if (!ExpectStatus(tensora::inference::SessionCreate(
                        model_path, "auto", false, "", nullptr),
                    TS_INVALID_ARGUMENT, "session create null"))
    return 2;

  uint64_t session = 0;
  if (!ExpectStatus(tensora::inference::SessionCreate(
                        model_path, "auto", false, "", &session),
                    TS_OK, "session create") ||
      session == 0)
    return 3;

  if (!ExpectStatus(tensora::inference::SessionProvider(session, nullptr),
                    TS_INVALID_ARGUMENT, "session provider null"))
    return 4;
  if (!ExpectStatus(tensora::inference::SessionInputName(session, 0, nullptr),
                    TS_INVALID_ARGUMENT, "input name null"))
    return 5;
  if (!ExpectStatus(tensora::inference::SessionOutputName(session, 0, nullptr),
                    TS_INVALID_ARGUMENT, "output name null"))
    return 6;
  if (!ExpectStatus(tensora::inference::SessionEndProfiling(session, nullptr),
                    TS_INVALID_ARGUMENT, "profiling output null"))
    return 7;

  auto input = MakeInput();
  auto short_input = MakeShortWriteInput();
  if (!input || !short_input) return 8;

  const std::vector<std::string> input_names = {"X"};
  const std::vector<std::string> output_names = {"Y"};
  const std::vector<std::shared_ptr<Tensor>> inputs = {input};
  std::vector<std::shared_ptr<Tensor>> outputs;

  if (!ExpectStatus(tensora::inference::SessionRun(
                        session, input_names, inputs, output_names, nullptr),
                    TS_INVALID_ARGUMENT, "run null outputs"))
    return 9;
  if (!ExpectStatus(tensora::inference::SessionRun(
                        session, input_names, {}, output_names, &outputs),
                    TS_INVALID_ARGUMENT, "run name count mismatch"))
    return 10;
  if (!ExpectStatus(tensora::inference::SessionRun(
                        session, input_names, {nullptr}, output_names, &outputs),
                    TS_INVALID_ARGUMENT, "run null input object"))
    return 11;
  if (!ExpectStatus(tensora::inference::SessionRun(
                        session, input_names, {short_input}, output_names,
                        &outputs),
                    TS_INTERNAL_ERROR, "run short storage write"))
    return 12;

  if (!ExpectStatus(tensora::inference::SessionRelease(session), TS_OK,
                    "session release"))
    return 13;
  return 0;
}

}  // namespace

int main() {
  const char* model_path = std::getenv("TENSORA_ONNX_TEST_MODEL");
  if (model_path == nullptr || *model_path == '\0') return 100;
  return Run(model_path);
}
