#ifndef TENSORA_KERNELS_GEMM_H_
#define TENSORA_KERNELS_GEMM_H_

#include <cstdint>

#include "kernels/cpu_features.h"

namespace tensora::kernels {

/// Computes `C = alpha * A * B + beta * C` for row-major float32 matrices.
///
/// Every operand carries an explicit row and column stride, so a transposed or
/// otherwise strided view is consumed directly by the packing step instead of
/// being materialized into a temporary buffer first.
///
/// Shapes are `A[m, k]`, `B[k, n]`, `C[m, n]`. The implementation is selected
/// once per process from the usable SIMD level and is cache-blocked and
/// multithreaded; all variants are numerically validated against each other.
void Sgemm(int64_t m, int64_t n, int64_t k,
           float alpha,
           const float* a, int64_t a_row_stride, int64_t a_col_stride,
           const float* b, int64_t b_row_stride, int64_t b_col_stride,
           float beta,
           float* c, int64_t c_row_stride, int64_t c_col_stride);

namespace internal {

/// Per-level entry points. Exposed so tests can drive every compiled kernel on
/// one machine rather than only the one this host happens to select.
void SgemmScalar(int64_t m, int64_t n, int64_t k, float alpha,
                 const float* a, int64_t a_rs, int64_t a_cs,
                 const float* b, int64_t b_rs, int64_t b_cs,
                 float beta, float* c, int64_t c_rs, int64_t c_cs);
void SgemmAvx2(int64_t m, int64_t n, int64_t k, float alpha,
               const float* a, int64_t a_rs, int64_t a_cs,
               const float* b, int64_t b_rs, int64_t b_cs,
               float beta, float* c, int64_t c_rs, int64_t c_cs);
void SgemmAvx512(int64_t m, int64_t n, int64_t k, float alpha,
                 const float* a, int64_t a_rs, int64_t a_cs,
                 const float* b, int64_t b_rs, int64_t b_cs,
                 float beta, float* c, int64_t c_rs, int64_t c_cs);

/// Dispatches to the entry point for `level`, falling back to a compiled-in
/// lower level when the requested one is unavailable in this build.
void SgemmAtLevel(SimdLevel level, int64_t m, int64_t n, int64_t k, float alpha,
                  const float* a, int64_t a_rs, int64_t a_cs,
                  const float* b, int64_t b_rs, int64_t b_cs,
                  float beta, float* c, int64_t c_rs, int64_t c_cs);

}  // namespace internal

}  // namespace tensora::kernels

#endif  // TENSORA_KERNELS_GEMM_H_
