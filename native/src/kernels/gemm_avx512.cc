#include <cstdint>

#include "kernels/gemm.h"
#include "kernels/gemm_blocking.h"

#if defined(TENSORA_KERNELS_HAVE_AVX512)
#include <immintrin.h>
#endif

namespace tensora::kernels::internal {

#if defined(TENSORA_KERNELS_HAVE_AVX512)

namespace {

// 6x32 register block: twelve 16-wide accumulators. AVX-512 supplies 32 vector
// registers, so the two B vectors and the broadcast A scalar still fit without
// spilling.
struct Avx512Kernel {
  static constexpr int64_t kMr = 6;
  static constexpr int64_t kNr = 32;

  static void Micro(int64_t kc, const float* a, const float* b, float* c,
                    int64_t ldc) {
    __m512 c00 = _mm512_setzero_ps();
    __m512 c01 = _mm512_setzero_ps();
    __m512 c10 = _mm512_setzero_ps();
    __m512 c11 = _mm512_setzero_ps();
    __m512 c20 = _mm512_setzero_ps();
    __m512 c21 = _mm512_setzero_ps();
    __m512 c30 = _mm512_setzero_ps();
    __m512 c31 = _mm512_setzero_ps();
    __m512 c40 = _mm512_setzero_ps();
    __m512 c41 = _mm512_setzero_ps();
    __m512 c50 = _mm512_setzero_ps();
    __m512 c51 = _mm512_setzero_ps();

    for (int64_t p = 0; p < kc; ++p) {
      const __m512 b0 = _mm512_loadu_ps(b);
      const __m512 b1 = _mm512_loadu_ps(b + 16);
      b += kNr;

      __m512 av = _mm512_set1_ps(a[0]);
      c00 = _mm512_fmadd_ps(av, b0, c00);
      c01 = _mm512_fmadd_ps(av, b1, c01);
      av = _mm512_set1_ps(a[1]);
      c10 = _mm512_fmadd_ps(av, b0, c10);
      c11 = _mm512_fmadd_ps(av, b1, c11);
      av = _mm512_set1_ps(a[2]);
      c20 = _mm512_fmadd_ps(av, b0, c20);
      c21 = _mm512_fmadd_ps(av, b1, c21);
      av = _mm512_set1_ps(a[3]);
      c30 = _mm512_fmadd_ps(av, b0, c30);
      c31 = _mm512_fmadd_ps(av, b1, c31);
      av = _mm512_set1_ps(a[4]);
      c40 = _mm512_fmadd_ps(av, b0, c40);
      c41 = _mm512_fmadd_ps(av, b1, c41);
      av = _mm512_set1_ps(a[5]);
      c50 = _mm512_fmadd_ps(av, b0, c50);
      c51 = _mm512_fmadd_ps(av, b1, c51);
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
  static void StoreRow(float* row, __m512 low, __m512 high) {
    _mm512_storeu_ps(row, _mm512_add_ps(_mm512_loadu_ps(row), low));
    _mm512_storeu_ps(row + 16,
                     _mm512_add_ps(_mm512_loadu_ps(row + 16), high));
  }
};

}  // namespace

void SgemmAvx512(int64_t m, int64_t n, int64_t k, float alpha, const float* a,
                 int64_t a_rs, int64_t a_cs, const float* b, int64_t b_rs,
                 int64_t b_cs, float beta, float* c, int64_t c_rs,
                 int64_t c_cs) {
  RunSgemm<Avx512Kernel>(m, n, k, alpha, a, a_rs, a_cs, b, b_rs, b_cs, beta, c,
                         c_rs, c_cs);
}

#else  // !TENSORA_KERNELS_HAVE_AVX512

void SgemmAvx512(int64_t m, int64_t n, int64_t k, float alpha, const float* a,
                 int64_t a_rs, int64_t a_cs, const float* b, int64_t b_rs,
                 int64_t b_cs, float beta, float* c, int64_t c_rs,
                 int64_t c_cs) {
  SgemmAvx2(m, n, k, alpha, a, a_rs, a_cs, b, b_rs, b_cs, beta, c, c_rs, c_cs);
}

#endif  // TENSORA_KERNELS_HAVE_AVX512

}  // namespace tensora::kernels::internal
