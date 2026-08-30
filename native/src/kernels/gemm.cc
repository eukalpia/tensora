#include "kernels/gemm.h"

#include <cstdint>

#include "kernels/cpu_features.h"

namespace tensora::kernels {

namespace internal {

void SgemmAtLevel(SimdLevel level, int64_t m, int64_t n, int64_t k, float alpha,
                  const float* a, int64_t a_rs, int64_t a_cs, const float* b,
                  int64_t b_rs, int64_t b_cs, float beta, float* c,
                  int64_t c_rs, int64_t c_cs) {
  switch (level) {
    case SimdLevel::kAvx512:
      SgemmAvx512(m, n, k, alpha, a, a_rs, a_cs, b, b_rs, b_cs, beta, c, c_rs,
                  c_cs);
      return;
    case SimdLevel::kAvx2:
      SgemmAvx2(m, n, k, alpha, a, a_rs, a_cs, b, b_rs, b_cs, beta, c, c_rs,
                c_cs);
      return;
    case SimdLevel::kScalar:
      break;
  }
  SgemmScalar(m, n, k, alpha, a, a_rs, a_cs, b, b_rs, b_cs, beta, c, c_rs,
              c_cs);
}

}  // namespace internal

void Sgemm(int64_t m, int64_t n, int64_t k, float alpha, const float* a,
           int64_t a_row_stride, int64_t a_col_stride, const float* b,
           int64_t b_row_stride, int64_t b_col_stride, float beta, float* c,
           int64_t c_row_stride, int64_t c_col_stride) {
  internal::SgemmAtLevel(CurrentSimdLevel(), m, n, k, alpha, a, a_row_stride,
                         a_col_stride, b, b_row_stride, b_col_stride, beta, c,
                         c_row_stride, c_col_stride);
}

}  // namespace tensora::kernels
