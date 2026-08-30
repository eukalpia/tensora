// Differential and contract tests for the Tensora CPU GEMM kernels.
//
// Every compiled microkernel is driven explicitly rather than only the one this
// host happens to select, so a machine with AVX-512 still proves the AVX2 and
// scalar paths. The naive triple loop in this file is the reference: it is the
// definition of the operation, and the optimized kernels must agree with it.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "kernels/cpu_features.h"
#include "kernels/gemm.h"

namespace {

using tensora::kernels::SimdLevel;
using tensora::kernels::SimdLevelName;

int g_failures = 0;

void Fail(const char* what, const char* detail) {
  std::printf("FAIL %s: %s\n", what, detail);
  ++g_failures;
}

// Deterministic pseudo-random values in [-1, 1). A fixed generator keeps a
// failure reproducible without recording input data.
class Values {
 public:
  explicit Values(uint64_t seed) : state_(seed | 1u) {}

  float Next() {
    state_ ^= state_ << 13;
    state_ ^= state_ >> 7;
    state_ ^= state_ << 17;
    const uint32_t bits = static_cast<uint32_t>(state_ >> 32);
    return static_cast<float>(bits) / 2147483648.0f - 1.0f;
  }

 private:
  uint64_t state_;
};

// Reference implementation. Deliberately the most obvious correct code.
void ReferenceGemm(int64_t m, int64_t n, int64_t k, float alpha,
                   const float* a, int64_t a_rs, int64_t a_cs, const float* b,
                   int64_t b_rs, int64_t b_cs, float beta, float* c,
                   int64_t c_rs, int64_t c_cs) {
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      double sum = 0.0;
      for (int64_t p = 0; p < k; ++p) {
        sum += static_cast<double>(a[i * a_rs + p * a_cs]) *
               static_cast<double>(b[p * b_rs + j * b_cs]);
      }
      float* target = &c[i * c_rs + j * c_cs];
      *target = static_cast<float>(alpha * static_cast<float>(sum) +
                                   beta * *target);
    }
  }
}

struct Layout {
  const char* name;
  bool transpose_a;
  bool transpose_b;
};

// Compares one kernel level against the reference for one shape and layout.
void CheckShape(SimdLevel level, int64_t m, int64_t n, int64_t k,
                const Layout& layout, float alpha, float beta) {
  Values values(static_cast<uint64_t>(m * 73856093 ^ n * 19349663 ^ k * 83492791));

  std::vector<float> a(static_cast<size_t>(m * k));
  std::vector<float> b(static_cast<size_t>(k * n));
  for (float& value : a) value = values.Next();
  for (float& value : b) value = values.Next();

  // A transposed operand is expressed by swapping strides, which is exactly
  // what Tensora's transpose view produces. No data is rearranged.
  const int64_t a_rs = layout.transpose_a ? 1 : k;
  const int64_t a_cs = layout.transpose_a ? m : 1;
  const int64_t b_rs = layout.transpose_b ? 1 : n;
  const int64_t b_cs = layout.transpose_b ? k : 1;

  std::vector<float> expected(static_cast<size_t>(m * n));
  std::vector<float> actual(static_cast<size_t>(m * n));
  for (size_t index = 0; index < expected.size(); ++index) {
    const float seed = values.Next();
    expected[index] = seed;
    actual[index] = seed;
  }

  ReferenceGemm(m, n, k, alpha, a.data(), a_rs, a_cs, b.data(), b_rs, b_cs,
                beta, expected.data(), n, 1);
  tensora::kernels::internal::SgemmAtLevel(level, m, n, k, alpha, a.data(),
                                           a_rs, a_cs, b.data(), b_rs, b_cs,
                                           beta, actual.data(), n, 1);

  // Tolerance scales with the reduction length: the kernels accumulate in a
  // different order than the reference, and float32 addition is not
  // associative. It is not widened beyond what reassociation can explain.
  const double tolerance = 1e-5 * static_cast<double>(k) + 1e-4;
  for (size_t index = 0; index < expected.size(); ++index) {
    const double difference =
        std::fabs(static_cast<double>(expected[index]) -
                  static_cast<double>(actual[index]));
    if (difference > tolerance) {
      char detail[256];
      std::snprintf(detail, sizeof(detail),
                    "level=%s layout=%s m=%lld n=%lld k=%lld index=%zu "
                    "expected=%.7f actual=%.7f",
                    SimdLevelName(level), layout.name,
                    static_cast<long long>(m), static_cast<long long>(n),
                    static_cast<long long>(k), index,
                    static_cast<double>(expected[index]),
                    static_cast<double>(actual[index]));
      Fail("gemm mismatch", detail);
      return;
    }
  }
}

// Verifies a non-unit output column stride, which takes the scatter path.
void CheckStridedOutput(SimdLevel level) {
  constexpr int64_t m = 9;
  constexpr int64_t n = 7;
  constexpr int64_t k = 11;
  constexpr int64_t c_cs = 3;
  constexpr int64_t c_rs = n * c_cs + 2;

  Values values(12345);
  std::vector<float> a(static_cast<size_t>(m * k));
  std::vector<float> b(static_cast<size_t>(k * n));
  for (float& value : a) value = values.Next();
  for (float& value : b) value = values.Next();

  const size_t span = static_cast<size_t>((m - 1) * c_rs + (n - 1) * c_cs + 1);
  std::vector<float> expected(span, 0.5f);
  std::vector<float> actual(span, 0.5f);

  ReferenceGemm(m, n, k, 1.0f, a.data(), k, 1, b.data(), n, 1, 0.25f,
                expected.data(), c_rs, c_cs);
  tensora::kernels::internal::SgemmAtLevel(level, m, n, k, 1.0f, a.data(), k, 1,
                                           b.data(), n, 1, 0.25f,
                                           actual.data(), c_rs, c_cs);

  for (size_t index = 0; index < span; ++index) {
    if (std::fabs(expected[index] - actual[index]) > 1e-3f) {
      char detail[128];
      std::snprintf(detail, sizeof(detail), "level=%s index=%zu",
                    SimdLevelName(level), index);
      Fail("strided output mismatch", detail);
      return;
    }
  }
}

// Degenerate shapes must leave the output defined by beta alone.
void CheckDegenerate(SimdLevel level) {
  std::vector<float> c(6, 2.0f);
  const std::vector<float> a(1, 1.0f);
  const std::vector<float> b(1, 1.0f);

  tensora::kernels::internal::SgemmAtLevel(level, 2, 3, 0, 1.0f, a.data(), 1, 1,
                                           b.data(), 1, 1, 0.0f, c.data(), 3,
                                           1);
  for (float value : c) {
    if (value != 0.0f) {
      Fail("degenerate k", "beta=0 must zero the output when k is zero");
      return;
    }
  }

  std::vector<float> keep(6, 2.0f);
  tensora::kernels::internal::SgemmAtLevel(level, 2, 3, 0, 1.0f, a.data(), 1, 1,
                                           b.data(), 1, 1, 1.0f, keep.data(), 3,
                                           1);
  for (float value : keep) {
    if (value != 2.0f) {
      Fail("degenerate k", "beta=1 must preserve the output when k is zero");
      return;
    }
  }
}

void RunLevel(SimdLevel level) {
  static const Layout kLayouts[] = {
      {"nn", false, false},
      {"tn", true, false},
      {"nt", false, true},
      {"tt", true, true},
  };

  // Sizes straddle the register block heights (4, 6), the vector widths
  // (4, 8, 16, 32) and the cache block boundaries, so packing edges and the
  // partial-tile path are all exercised.
  static const int64_t kSizes[] = {1, 2, 3, 5, 6, 7, 8, 15, 16, 17,
                                   31, 32, 33, 63, 64, 65, 96, 129};

  for (const Layout& layout : kLayouts) {
    for (int64_t size : kSizes) {
      CheckShape(level, size, size, size, layout, 1.0f, 0.0f);
    }
    CheckShape(level, 1, 129, 64, layout, 1.0f, 0.0f);
    CheckShape(level, 129, 1, 64, layout, 1.0f, 0.0f);
    CheckShape(level, 64, 64, 1, layout, 1.0f, 0.0f);
    CheckShape(level, 37, 53, 71, layout, 2.5f, -1.25f);
    CheckShape(level, 6, 32, 300, layout, 1.0f, 1.0f);
  }

  // Larger than one cache block in every dimension.
  CheckShape(level, 260, 260, 260, kLayouts[0], 1.0f, 0.0f);
  CheckShape(level, 300, 1100, 40, kLayouts[0], 1.0f, 0.0f);

  CheckStridedOutput(level);
  CheckDegenerate(level);
}

}  // namespace

int main() {
  const SimdLevel detected = tensora::kernels::DetectSimdLevel();
  std::printf("detected SIMD level: %s\n", SimdLevelName(detected));

  // Every level compiled into this binary is validated, not only the best one
  // available here. A kernel that is never selected on the build machine is
  // still shipped, so it is still tested.
  const SimdLevel levels[] = {SimdLevel::kScalar, SimdLevel::kAvx2,
                              SimdLevel::kAvx512};
  for (SimdLevel level : levels) {
    if (level > detected) {
      std::printf("skipping %s: not usable on this host\n",
                  SimdLevelName(level));
      continue;
    }
    std::printf("validating %s\n", SimdLevelName(level));
    RunLevel(level);
  }

  // The runtime override must actually change what Sgemm dispatches to.
  tensora::kernels::SetSimdLevelForTesting(SimdLevel::kScalar);
  if (tensora::kernels::CurrentSimdLevel() != SimdLevel::kScalar) {
    Fail("simd override", "SetSimdLevelForTesting did not take effect");
  }
  {
    const std::vector<float> a(4, 1.0f);
    const std::vector<float> b(4, 1.0f);
    std::vector<float> c(4, 0.0f);
    tensora::kernels::Sgemm(2, 2, 2, 1.0f, a.data(), 2, 1, b.data(), 2, 1, 0.0f,
                            c.data(), 2, 1);
    for (float value : c) {
      if (std::fabs(value - 2.0f) > 1e-6f) {
        Fail("dispatch", "Sgemm did not honour the overridden SIMD level");
        break;
      }
    }
  }

  if (g_failures != 0) {
    std::printf("%d GEMM check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("all GEMM checks passed\n");
  return 0;
}
