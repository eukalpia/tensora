#include <cstdint>

#include "kernels/gemm.h"
#include "kernels/gemm_blocking.h"

#if defined(TENSORA_KERNELS_HAVE_AVX2)
#include <immintrin.h>
#endif

namespace tensora::kernels::internal {

#if defined(TENSORA_KERNELS_HAVE_AVX2)

namespace {

// 6x16 register block: twelve accumulators of eight floats, leaving registers
// free for the two B vectors and the broadcast A scalar.
struct Avx2Kernel {
  static constexpr int64_t kMr = 6;
  static constexpr int64_t kNr = 16;

  static void Micro(int64_t kc, const float* a, const float* b, float* c,
                    int64_t ldc) {
    __m256 c00 = _mm256_setzero_ps();
    __m256 c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps();
    __m256 c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps();
    __m256 c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps();
    __m256 c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps();
    __m256 c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps();
    __m256 c51 = _mm256_setzero_ps();

    for (int64_t p = 0; p < kc; ++p) {
      const __m256 b0 = _mm256_loadu_ps(b);
      const __m256 b1 = _mm256_loadu_ps(b + 8);
      b += kNr;

      __m256 av = _mm256_broadcast_ss(a + 0);
      c00 = _mm256_fmadd_ps(av, b0, c00);
      c01 = _mm256_fmadd_ps(av, b1, c01);
      av = _mm256_broadcast_ss(a + 1);
      c10 = _mm256_fmadd_ps(av, b0, c10);
      c11 = _mm256_fmadd_ps(av, b1, c11);
      av = _mm256_broadcast_ss(a + 2);
      c20 = _mm256_fmadd_ps(av, b0, c20);
      c21 = _mm256_fmadd_ps(av, b1, c21);
      av = _mm256_broadcast_ss(a + 3);
      c30 = _mm256_fmadd_ps(av, b0, c30);
      c31 = _mm256_fmadd_ps(av, b1, c31);
      av = _mm256_broadcast_ss(a + 4);
      c40 = _mm256_fmadd_ps(av, b0, c40);
      c41 = _mm256_fmadd_ps(av, b1, c41);
      av = _mm256_broadcast_ss(a + 5);
      c50 = _mm256_fmadd_ps(av, b0, c50);
      c51 = _mm256_fmadd_ps(av, b1, c51);
      a += kMr;
    }

    StoreRow(c + 0 * ldc, c00, c01);
    StoreRow(c + 1 * ldc, c10, c11);
    StoreRow(c + 2 * ldc, c20, c21);
    StoreRow(c + 3 * ldc, c30, c31);
    StoreRow(c + 4 * ldc, c40, c41);
    StoreRow(c + 5 * ldc, c50, c51);
  }

 private:
  static void StoreRow(float* row, __m256 low, __m256 high) {
    _mm256_storeu_ps(row, _mm256_add_ps(_mm256_loadu_ps(row), low));
    _mm256_storeu_ps(row + 8, _mm256_add_ps(_mm256_loadu_ps(row + 8), high));
  }
};

}  // namespace

void SgemmAvx2(int64_t m, int64_t n, int64_t k, float alpha, const float* a,
               int64_t a_rs, int64_t a_cs, const float* b, int64_t b_rs,
               int64_t b_cs, float beta, float* c, int64_t c_rs,
               int64_t c_cs) {
  RunSgemm<Avx2Kernel>(m, n, k, alpha, a, a_rs, a_cs, b, b_rs, b_cs, beta, c,
                       c_rs, c_cs);
}

#else  // !TENSORA_KERNELS_HAVE_AVX2

void SgemmAvx2(int64_t m, int64_t n, int64_t k, float alpha, const float* a,
               int64_t a_rs, int64_t a_cs, const float* b, int64_t b_rs,
               int64_t b_cs, float beta, float* c, int64_t c_rs,
               int64_t c_cs) {
  SgemmScalar(m, n, k, alpha, a, a_rs, a_cs, b, b_rs, b_cs, beta, c, c_rs,
              c_cs);
}

#endif  // TENSORA_KERNELS_HAVE_AVX2

}  // namespace tensora::kernels::internal
