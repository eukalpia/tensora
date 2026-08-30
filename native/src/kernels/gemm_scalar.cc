#include <cstdint>

#include "kernels/gemm.h"
#include "kernels/gemm_blocking.h"

namespace tensora::kernels::internal {
namespace {

// Portable reference register block. It is the fallback on non-x86 targets and
// the differential-test oracle for the vector kernels: every SIMD result is
// compared against this one.
struct ScalarKernel {
  static constexpr int64_t kMr = 4;
  static constexpr int64_t kNr = 4;

  static void Micro(int64_t kc, const float* a, const float* b, float* c,
                    int64_t ldc) {
    float acc[kMr][kNr] = {};
    for (int64_t p = 0; p < kc; ++p) {
      const float* a_panel = a + p * kMr;
      const float* b_panel = b + p * kNr;
      for (int64_t i = 0; i < kMr; ++i) {
        const float a_value = a_panel[i];
        for (int64_t j = 0; j < kNr; ++j) {
          acc[i][j] += a_value * b_panel[j];
        }
      }
    }
    for (int64_t i = 0; i < kMr; ++i) {
      for (int64_t j = 0; j < kNr; ++j) {
        c[i * ldc + j] += acc[i][j];
      }
    }
  }
};

}  // namespace

void SgemmScalar(int64_t m, int64_t n, int64_t k, float alpha, const float* a,
                 int64_t a_rs, int64_t a_cs, const float* b, int64_t b_rs,
                 int64_t b_cs, float beta, float* c, int64_t c_rs,
                 int64_t c_cs) {
  RunSgemm<ScalarKernel>(m, n, k, alpha, a, a_rs, a_cs, b, b_rs, b_cs, beta, c,
                         c_rs, c_cs);
}

}  // namespace tensora::kernels::internal
