#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "autograd/autograd.h"
#include "backends/backend.h"
#include "core/allocation_guard.h"
#include "core/status.h"
#include "memory/cpu_storage.h"
#include "runtime/dispatcher.h"
#include "runtime/handle_registry.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"
#include "training/training_bridge.h"

namespace coverage {
thread_local bool fail_next_allocation = false;
}

void* operator new(std::size_t size) {
  if (coverage::fail_next_allocation) {
    coverage::fail_next_allocation = false;
    throw std::bad_alloc();
  }
  if (void* memory = std::malloc(size)) return memory;
  throw std::bad_alloc();
}

void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }

namespace std {

class TestOfstream {
 public:
  TestOfstream(const std::string& path, std::ios_base::openmode) {
    const std::string prefix = "coverage-fail-write-";
    if (path.rfind(prefix, 0) == 0) {
      fail_write_ = std::stoi(path.substr(prefix.size()));
    } else if (path == "coverage-fail-flush") {
      fail_flush_ = true;
    }
  }

  bool is_open() const { return true; }

  TestOfstream& write(const char*, std::streamsize) {
    ++write_count_;
    if (write_count_ == fail_write_) good_ = false;
    return *this;
  }

  bool good() const { return good_; }

  TestOfstream& flush() {
    if (fail_flush_) good_ = false;
    return *this;
  }

 private:
  int write_count_ = 0;
  int fail_write_ = -1;
  bool fail_flush_ = false;
  bool good_ = true;
};

}  // namespace std

#define ofstream TestOfstream
#include "../src/training/training_bridge_stub.cc"
#undef ofstream

namespace tensora::training {
namespace {

bool ExpectStatus(const Status& status,
                  ts_status_t expected,
                  const char* operation) {
  if (status.code() == expected) return true;
  std::cerr << operation << " expected status " << expected << ", got "
            << status.code() << ": " << status.message() << '\n';
  return false;
}

std::shared_ptr<Tensor> MakeTensor(const std::vector<float>& values,
                                   const std::vector<int64_t>& dimensions) {
  ShapeInfo shape;
  if (!ValidateShape(dimensions.empty() ? nullptr : dimensions.data(),
                     dimensions.size(), &shape)
           .ok()) {
    return nullptr;
  }
  std::shared_ptr<CpuStorage> storage;
  if (!CpuStorage::FromData(values.data(), values.size(), &storage).ok()) {
    return nullptr;
  }
  return std::make_shared<Tensor>(std::move(shape), std::move(storage));
}

int CheckInternalContracts() {
  auto matrix = MakeTensor({1.0f, 2.0f}, {1, 2});
  auto rank_one = MakeTensor({1.0f, 2.0f}, {2});
  auto bias = MakeTensor({1.0f, 2.0f}, {2});
  auto wide_bias = MakeTensor({1.0f, 2.0f, 3.0f}, {3});
  if (!matrix || !rank_one || !bias || !wide_bias) return 1;

  LinearState state;
  state.weight = matrix;
  bool allocation_failed = false;
  coverage::fail_next_allocation = true;
  try {
    (void)Parameters(state);
  } catch (const std::bad_alloc&) {
    allocation_failed = true;
  }
  coverage::fail_next_allocation = false;
  if (!allocation_failed) return 2;

  std::shared_ptr<Tensor> output;
  if (!ExpectStatus(MakeActivation(
                        *matrix, static_cast<autograd::Operation>(255), &output),
                    TS_INTERNAL_ERROR, "unsupported activation")) {
    return 3;
  }
  if (!ExpectStatus(AddBias(*matrix, *bias, nullptr), TS_INVALID_ARGUMENT,
                    "bias null output")) {
    return 4;
  }
  if (!ExpectStatus(AddBias(*rank_one, *bias, &output), TS_INTERNAL_ERROR,
                    "bias invalid ranks")) {
    return 5;
  }
  if (!ExpectStatus(AddBias(*matrix, *wide_bias, &output), TS_INTERNAL_ERROR,
                    "bias width mismatch")) {
    return 6;
  }

  std::ifstream input_stream;
  ShapeInfo expected_shape;
  const int64_t dimensions[1] = {1};
  if (!ValidateShape(dimensions, 1, &expected_shape).ok()) return 7;
  if (!ExpectStatus(ReadTensorPayload(input_stream, expected_shape, nullptr),
                    TS_INVALID_ARGUMENT, "load null values")) {
    return 8;
  }

  LinearState input_state;
  input_state.in_features = 1;
  if (!ExpectStatus(ValidateModuleInput(input_state, *matrix), TS_INVALID_SHAPE,
                    "module input feature mismatch")) {
    return 9;
  }
  if (!ExpectStatus(WithRequiresGrad(*matrix, true, nullptr),
                    TS_INVALID_ARGUMENT, "requires grad null output")) {
    return 10;
  }
  if (!ExpectStatus(CrossEntropyLoss(*matrix, *matrix, nullptr),
                    TS_INVALID_ARGUMENT, "cross entropy null output")) {
    return 11;
  }
  if (!ExpectStatus(CrossEntropyLoss(*matrix, *wide_bias, &output),
                    TS_INVALID_SHAPE, "cross entropy shape mismatch")) {
    return 12;
  }
  if (!ExpectStatus(CrossEntropyLoss(*rank_one, *rank_one, &output),
                    TS_INVALID_SHAPE, "cross entropy rank mismatch")) {
    return 13;
  }

  return 0;
}

int CheckCheckpointWriteFailures() {
  uint64_t module = 0;
  if (!ExpectStatus(LinearCreate(1, 1, false, &module), TS_OK,
                    "linear create")) {
    return 20;
  }

  for (const int write_call : {1, 6, 7, 9, 10}) {
    const std::string path = "coverage-fail-write-" +
                             std::to_string(write_call);
    if (!ExpectStatus(ModuleSave(module, path), TS_INTERNAL_ERROR,
                      "checkpoint write failure")) {
      ModuleRelease(module);
      return 21 + write_call;
    }
  }
  if (!ExpectStatus(ModuleSave(module, "coverage-fail-flush"),
                    TS_INTERNAL_ERROR, "checkpoint flush failure")) {
    ModuleRelease(module);
    return 40;
  }

  if (!ExpectStatus(ModuleRelease(module), TS_OK, "module release")) return 41;
  return 0;
}

int CheckOptimizerGradientMismatch() {
  uint64_t module = 0;
  if (!ExpectStatus(LinearCreate(1, 1, false, &module), TS_OK,
                    "optimizer module create")) {
    return 50;
  }

  uint64_t optimizer = 0;
  if (!ExpectStatus(SgdCreate(module, 0.1, 0.0, 0.0, &optimizer), TS_OK,
                    "sgd create")) {
    ModuleRelease(module);
    return 51;
  }

  std::shared_ptr<Tensor> parameter;
  if (!ExpectStatus(ModuleParameterAt(module, 0, &parameter), TS_OK,
                    "parameter lookup") ||
      !parameter) {
    OptimizerRelease(optimizer);
    ModuleRelease(module);
    return 52;
  }
  auto wrong_gradient = MakeTensor({1.0f, 2.0f}, {2});
  if (!wrong_gradient) {
    OptimizerRelease(optimizer);
    ModuleRelease(module);
    return 53;
  }

  auto meta = parameter->autograd_meta();
  if (!meta) {
    OptimizerRelease(optimizer);
    ModuleRelease(module);
    return 54;
  }
  {
    std::lock_guard<std::mutex> lock(meta->mutex);
    meta->gradient = wrong_gradient;
  }

  if (!ExpectStatus(OptimizerStep(optimizer), TS_INTERNAL_ERROR,
                    "optimizer gradient mismatch")) {
    OptimizerRelease(optimizer);
    ModuleRelease(module);
    return 55;
  }

  if (!ExpectStatus(OptimizerRelease(optimizer), TS_OK, "optimizer release")) {
    ModuleRelease(module);
    return 56;
  }
  if (!ExpectStatus(ModuleRelease(module), TS_OK, "module release")) return 57;
  return 0;
}

}  // namespace

int RunCoverageContracts() {
  if (const int code = CheckInternalContracts(); code != 0) return code;
  if (const int code = CheckCheckpointWriteFailures(); code != 0) return code;
  if (const int code = CheckOptimizerGradientMismatch(); code != 0) return code;
  return 0;
}

}  // namespace tensora::training

int main() { return tensora::training::RunCoverageContracts(); }
