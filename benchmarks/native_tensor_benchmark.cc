#include "tensora.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double percentile(std::vector<double> values, double p) {
  std::sort(values.begin(), values.end());
  if (values.empty()) return 0.0;
  const double index = p * static_cast<double>(values.size() - 1);
  const size_t lower = static_cast<size_t>(std::floor(index));
  const size_t upper = static_cast<size_t>(std::ceil(index));
  if (lower == upper) return values[lower];
  const double fraction = index - static_cast<double>(lower);
  return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

bool check(ts_status_t status, const char* operation) {
  if (status == TS_OK) return true;
  std::cerr << operation << " failed: " << ts_last_error_message() << "\n";
  return false;
}

bool benchmark_matmul(int64_t size, int warmup, int iterations) {
  const int64_t dims[2] = {size, size};
  const uint64_t numel = static_cast<uint64_t>(size) * size;
  std::vector<float> values(static_cast<size_t>(numel), 1.0f);

  ts_tensor_t a = 0;
  ts_tensor_t b = 0;
  if (!check(ts_tensor_from_f32(values.data(), values.size(), dims, 2, &a),
             "create a") ||
      !check(ts_tensor_from_f32(values.data(), values.size(), dims, 2, &b),
             "create b")) {
    return false;
  }

  for (int i = 0; i < warmup; ++i) {
    ts_tensor_t result = 0;
    if (!check(ts_tensor_matmul(a, b, &result), "warmup matmul") ||
        !check(ts_tensor_release(result), "release warmup result")) {
      return false;
    }
  }

  std::vector<double> samples;
  samples.reserve(static_cast<size_t>(iterations));
  for (int i = 0; i < iterations; ++i) {
    const auto start = Clock::now();
    ts_tensor_t result = 0;
    if (!check(ts_tensor_matmul(a, b, &result), "matmul")) return false;
    const auto end = Clock::now();
    if (!check(ts_tensor_release(result), "release result")) return false;

    const auto micros =
        std::chrono::duration<double, std::micro>(end - start).count();
    samples.push_back(micros);
  }

  std::cout << "native_cpu_matmul"
            << " size=" << size << "x" << size
            << " warmup=" << warmup << " iterations=" << iterations
            << " median_us=" << percentile(samples, 0.50)
            << " p95_us=" << percentile(samples, 0.95) << "\n";

  const bool released =
      check(ts_tensor_release(b), "release b") &&
      check(ts_tensor_release(a), "release a");
  return released;
}

}  // namespace

int main(int argc, char** argv) {
  const bool smoke = argc > 1 && std::string(argv[1]) == "--smoke";
  const int warmup = smoke ? 1 : 3;
  const int iterations = smoke ? 2 : 10;

  std::cout << "Tensora native CPU benchmark"
            << " abi=" << ts_abi_version()
            << " mode=" << (smoke ? "smoke" : "standard") << "\n";

  const std::vector<int64_t> sizes =
      smoke ? std::vector<int64_t>{64}
            : std::vector<int64_t>{64, 256, 1024};

  for (int64_t size : sizes) {
    if (!benchmark_matmul(size, warmup, iterations)) return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
