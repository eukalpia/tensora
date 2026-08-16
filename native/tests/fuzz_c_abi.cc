#include "tensora.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

class Reader {
 public:
  Reader(const uint8_t* data, size_t size) : data_(data), size_(size) {}

  uint8_t Byte() {
    if (offset_ >= size_) return 0;
    return data_[offset_++];
  }

  uint64_t U64() {
    uint64_t value = 0;
    for (int index = 0; index < 8; ++index) {
      value |= static_cast<uint64_t>(Byte()) << (index * 8);
    }
    return value;
  }

 private:
  const uint8_t* data_;
  size_t size_;
  size_t offset_ = 0;
};

int64_t BoundedDimension(uint8_t value) {
  const int category = value % 10;
  if (category == 0) return 0;
  if (category == 1) return -static_cast<int64_t>((value % 8) + 1);
  return static_cast<int64_t>((value % 16) + 1);
}

void ReleaseIfLive(ts_tensor_t handle) {
  if (handle != 0) {
    const ts_status_t status = ts_tensor_release(handle);
    (void)status;
  }
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  Reader reader(data, size);

  const size_t rank = static_cast<size_t>(reader.Byte() % 10);
  std::vector<int64_t> dims(rank);
  uint64_t bounded_numel = 1;
  bool shape_could_be_valid = true;
  for (size_t index = 0; index < rank; ++index) {
    dims[index] = BoundedDimension(reader.Byte());
    if (dims[index] <= 0) {
      shape_could_be_valid = false;
    } else if (bounded_numel <= 256) {
      bounded_numel *= static_cast<uint64_t>(dims[index]);
      if (bounded_numel > 256) shape_could_be_valid = false;
    }
  }
  if (rank == 0) bounded_numel = 1;

  const size_t requested_length = static_cast<size_t>(reader.Byte() % 257);
  const size_t allocation_length = std::max<size_t>(requested_length, 1);
  std::vector<float> values(allocation_length);
  for (size_t index = 0; index < requested_length; ++index) {
    const uint32_t bits = static_cast<uint32_t>(reader.U64());
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    values[index] = value;
  }

  const bool use_null_data = (reader.Byte() & 1u) != 0;
  const bool use_null_dims = (reader.Byte() & 1u) != 0;
  const float* data_pointer = use_null_data ? nullptr : values.data();
  const int64_t* dims_pointer =
      (rank == 0 || use_null_dims) ? nullptr : dims.data();

  ts_tensor_t tensor = std::numeric_limits<ts_tensor_t>::max();
  const ts_status_t create_status = ts_tensor_from_f32(
      data_pointer, requested_length, dims_pointer, rank, &tensor);
  if (create_status != TS_OK && tensor != 0) {
    __builtin_trap();
  }

  if (create_status == TS_OK) {
    if (!shape_could_be_valid || tensor == 0) __builtin_trap();

    uint64_t numel = 0;
    if (ts_tensor_numel(tensor, &numel) != TS_OK || numel == 0 || numel > 256) {
      __builtin_trap();
    }

    std::vector<float> output(static_cast<size_t>(numel));
    size_t written = 0;
    if (ts_tensor_copy_to_host_f32(
            tensor, output.data(), output.size(), &written) != TS_OK ||
        written != output.size()) {
      __builtin_trap();
    }

    if (numel > 1) {
      size_t short_written = 99;
      if (ts_tensor_copy_to_host_f32(
              tensor, output.data(), output.size() - 1, &short_written) !=
              TS_INVALID_ARGUMENT ||
          short_written != 0) {
        __builtin_trap();
      }
    }

    const size_t reshape_rank = static_cast<size_t>(reader.Byte() % 8);
    std::vector<int64_t> reshape_dims(reshape_rank);
    for (size_t index = 0; index < reshape_rank; ++index) {
      reshape_dims[index] = BoundedDimension(reader.Byte());
    }
    ts_tensor_t reshaped = 0;
    const ts_status_t reshape_status = ts_tensor_reshape(
        tensor, reshape_dims.empty() ? nullptr : reshape_dims.data(),
        reshape_dims.size(), &reshaped);
    if (reshape_status != TS_OK && reshaped != 0) __builtin_trap();
    ReleaseIfLive(reshaped);

    ts_tensor_t copied = 0;
    const ts_status_t copy_status =
        ts_tensor_to_device(tensor, TS_DEVICE_CPU, 0, &copied);
    if (copy_status != TS_OK || copied == 0) __builtin_trap();
    ReleaseIfLive(copied);

    ReleaseIfLive(tensor);
  }

  const ts_tensor_t arbitrary_handle = reader.U64() | (uint64_t{1} << 63);
  uint64_t ignored_numel = 0;
  const ts_status_t invalid_status =
      ts_tensor_numel(arbitrary_handle, &ignored_numel);
  if (invalid_status == TS_OK) __builtin_trap();

  ts_tensor_t output_handle = 123;
  const ts_status_t invalid_device =
      ts_tensor_to_device(arbitrary_handle, 0xffffffffu, -1, &output_handle);
  if (invalid_device == TS_OK || output_handle != 0) __builtin_trap();

  return 0;
}
