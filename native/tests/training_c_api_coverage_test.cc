#include <memory>

#include "../src/training/training_c_api.cc"

int main() {
  const tensora::Status status = tensora::InsertTrainingTensor(nullptr, nullptr);
  return status.code() == TS_INVALID_ARGUMENT ? 0 : 1;
}
