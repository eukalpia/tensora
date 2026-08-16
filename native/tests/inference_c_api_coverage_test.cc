#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "memory/cpu_storage.h"
#include "runtime/handle_registry.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

namespace coverage {
enum class RunMode { kWrongCount, kPartialInsertion };
RunMode run_mode = RunMode::kWrongCount;
std::shared_ptr<tensora::Tensor> result;
}

#include "../src/inference/inference_c_api.cc"

namespace tensora::inference {

Status SessionRun(uint64_t,
                  const std::vector<std::string>&,
                  const std::vector<std::shared_ptr<Tensor>>&,
                  const std::vector<std::string>&,
                  std::vector<std::shared_ptr<Tensor>>* out_tensors) {
  if (out_tensors == nullptr) {
    return InvalidArgument("coverage session run: output pointer is null");
  }
  out_tensors->clear();
  if (coverage::run_mode == coverage::RunMode::kWrongCount) {
    out_tensors->push_back(coverage::result);
  } else {
    out_tensors->push_back(coverage::result);
    out_tensors->push_back(nullptr);
  }
  return Status::Ok();
}

}  // namespace tensora::inference

namespace {

std::shared_ptr<tensora::Tensor> MakeTensor() {
  const int64_t dimensions[1] = {1};
  tensora::ShapeInfo shape;
  if (!tensora::ValidateShape(dimensions, 1, &shape).ok()) return nullptr;
  const float value = 1.0f;
  std::shared_ptr<tensora::CpuStorage> storage;
  if (!tensora::CpuStorage::FromData(&value, 1, &storage).ok()) return nullptr;
  return std::make_shared<tensora::Tensor>(std::move(shape), std::move(storage));
}

bool ExpectStatus(ts_status_t status,
                  ts_status_t expected,
                  const char* operation) {
  if (status == expected) return true;
  std::cerr << operation << " expected status " << expected << ", got "
            << status << '\n';
  return false;
}

}  // namespace

int main() {
  if (tensora::InsertInferenceTensor(nullptr, nullptr).code() !=
      TS_INVALID_ARGUMENT) {
    return 1;
  }

  coverage::result = MakeTensor();
  if (!coverage::result) return 2;
  const char* output_names[2] = {"Y0", "Y1"};
  ts_tensor_t outputs[2] = {0, 0};
  size_t written = 99;

  coverage::run_mode = coverage::RunMode::kWrongCount;
  if (!ExpectStatus(ts_onnx_session_run(123, nullptr, nullptr, 0, output_names,
                                        2, outputs, 2, &written),
                    TS_INTERNAL_ERROR, "wrong result count")) {
    return 3;
  }
  if (written != 0 || outputs[0] != 0 || outputs[1] != 0) return 4;

  const uint64_t before =
      tensora::HandleRegistry::Instance().Count(tensora::HandleType::kTensor);
  coverage::run_mode = coverage::RunMode::kPartialInsertion;
  if (!ExpectStatus(ts_onnx_session_run(123, nullptr, nullptr, 0, output_names,
                                        2, outputs, 2, &written),
                    TS_INVALID_ARGUMENT, "partial insertion rollback")) {
    return 5;
  }
  const uint64_t after =
      tensora::HandleRegistry::Instance().Count(tensora::HandleType::kTensor);
  if (after != before || written != 0 || outputs[0] != 0 || outputs[1] != 0) {
    return 6;
  }

  return 0;
}
