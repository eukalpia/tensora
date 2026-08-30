#ifndef TENSORA_KERNELS_GEMM_BLOCKING_H_
#define TENSORA_KERNELS_GEMM_BLOCKING_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>

#include "kernels/thread_pool.h"

namespace tensora::kernels::internal {

// Cache blocking shared by every microkernel.
//
// kBlockMc is a multiple of the supported register-block heights (4 and 6) and
// kBlockNc of the supported widths (4, 16 and 32), so a full block never ends
// on a partial register tile.
inline constexpr int64_t kBlockKc = 256;
inline constexpr int64_t kBlockMc = 240;
inline constexpr int64_t kBlockNc = 1024;
inline constexpr size_t kPackAlignment = 64;

// Reusable 64-byte-aligned scratch. Packing buffers are reused across calls so
// a sequence of small matmuls does not allocate on every invocation.
class AlignedScratch {
 public:
  AlignedScratch() = default;

  AlignedScratch(const AlignedScratch&) = delete;
  AlignedScratch& operator=(const AlignedScratch&) = delete;

  ~AlignedScratch() { Release(); }

  float* Reserve(size_t floats) {
    if (floats <= capacity_) return data_;
    Release();
    const size_t bytes = floats * sizeof(float);
    data_ = static_cast<float*>(
        ::operator new(bytes, std::align_val_t(kPackAlignment)));
    capacity_ = floats;
    return data_;
  }

 private:
  void Release() {
    if (data_ == nullptr) return;
    ::operator delete(data_, std::align_val_t(kPackAlignment));
    data_ = nullptr;
    capacity_ = 0;
  }

  float* data_ = nullptr;
  size_t capacity_ = 0;
};

// Packs a block of A into row panels of mr, folding alpha in during the copy.
// Rows past mc are zero filled so the microkernel always sees a full tile.
inline void PackA(const float* a, int64_t a_rs, int64_t a_cs, int64_t ic,
                  int64_t pc, int64_t mc, int64_t kc, int64_t mr, float alpha,
                  float* packed) {
  size_t offset = 0;
  for (int64_t panel = 0; panel < mc; panel += mr) {
    const int64_t rows = std::min<int64_t>(mr, mc - panel);
    for (int64_t p = 0; p < kc; ++p) {
      const int64_t k_index = pc + p;
      for (int64_t i = 0; i < rows; ++i) {
        const int64_t row = ic + panel + i;
        packed[offset + static_cast<size_t>(i)] =
            alpha * a[row * a_rs + k_index * a_cs];
      }
      for (int64_t i = rows; i < mr; ++i) {
        packed[offset + static_cast<size_t>(i)] = 0.0f;
      }
      offset += static_cast<size_t>(mr);
    }
  }
}

// Packs a block of B into column panels of nr, zero filling the trailing
// columns of the final panel.
inline void PackB(const float* b, int64_t b_rs, int64_t b_cs, int64_t pc,
                  int64_t jc, int64_t kc, int64_t nc, int64_t nr,
                  float* packed) {
  size_t offset = 0;
  for (int64_t panel = 0; panel < nc; panel += nr) {
    const int64_t cols = std::min<int64_t>(nr, nc - panel);
    for (int64_t p = 0; p < kc; ++p) {
      const int64_t k_index = pc + p;
      for (int64_t j = 0; j < cols; ++j) {
        const int64_t col = jc + panel + j;
        packed[offset + static_cast<size_t>(j)] =
            b[k_index * b_rs + col * b_cs];
      }
      for (int64_t j = cols; j < nr; ++j) {
        packed[offset + static_cast<size_t>(j)] = 0.0f;
      }
      offset += static_cast<size_t>(nr);
    }
  }
}

// Applies beta to C before accumulation begins.
inline void ScaleOutput(int64_t m, int64_t n, float beta, float* c,
                        int64_t c_rs, int64_t c_cs) {
  if (beta == 1.0f) return;
  for (int64_t i = 0; i < m; ++i) {
    float* row = c + i * c_rs;
    if (beta == 0.0f) {
      if (c_cs == 1) {
        std::memset(row, 0, static_cast<size_t>(n) * sizeof(float));
      } else {
        for (int64_t j = 0; j < n; ++j) row[j * c_cs] = 0.0f;
      }
      continue;
    }
    for (int64_t j = 0; j < n; ++j) row[j * c_cs] *= beta;
  }
}

// Cache-blocked, packed, multithreaded SGEMM accumulating into c.
//
// c must have unit column stride; callers with a strided output compute into a
// contiguous temporary and scatter afterwards. beta scaling is applied by the
// caller before this runs, so every k block accumulates uniformly.
template <typename Kernel>
void BlockedSgemmAccumulate(int64_t m, int64_t n, int64_t k, float alpha,
                            const float* a, int64_t a_rs, int64_t a_cs,
                            const float* b, int64_t b_rs, int64_t b_cs,
                            float* c, int64_t ldc) {
  constexpr int64_t kMr = Kernel::kMr;
  constexpr int64_t kNr = Kernel::kNr;

  static thread_local AlignedScratch b_scratch;
  const size_t b_panel_floats =
      static_cast<size_t>(kBlockKc) *
      static_cast<size_t>(((kBlockNc + kNr - 1) / kNr) * kNr);
  float* packed_b = b_scratch.Reserve(b_panel_floats);

  ThreadPool& pool = ThreadPool::Instance();

  for (int64_t jc = 0; jc < n; jc += kBlockNc) {
    const int64_t nc = std::min<int64_t>(kBlockNc, n - jc);
    for (int64_t pc = 0; pc < k; pc += kBlockKc) {
      const int64_t kc = std::min<int64_t>(kBlockKc, k - pc);
      PackB(b, b_rs, b_cs, pc, jc, kc, nc, kNr, packed_b);

      // Give each worker enough multiply-accumulates to cover the cost of
      // waking it. Below this threshold the hand-off dominates, and a small
      // matmul is measurably faster on one thread. Widening the grain is what
      // limits the worker count, since ParallelFor derives its chunk count from
      // it.
      constexpr int64_t kMinWorkPerWorker = int64_t{1} << 18;
      const int64_t work_per_row = nc * kc;
      int64_t grain = (kMinWorkPerWorker + work_per_row - 1) / work_per_row;
      grain = ((std::max<int64_t>(grain, kMr) + kMr - 1) / kMr) * kMr;

      const float* packed_b_view = packed_b;
      pool.ParallelFor(m, grain, [&](int64_t row_begin, int64_t row_end, int) {
        static thread_local AlignedScratch a_scratch;
        const size_t a_panel_floats =
            static_cast<size_t>(kBlockKc) *
            static_cast<size_t>(((kBlockMc + kMr - 1) / kMr) * kMr);
        float* packed_a = a_scratch.Reserve(a_panel_floats);

        alignas(64) float edge[static_cast<size_t>(kMr) *
                               static_cast<size_t>(kNr)];

        for (int64_t ic = row_begin; ic < row_end; ic += kBlockMc) {
          const int64_t mc = std::min<int64_t>(kBlockMc, row_end - ic);
          PackA(a, a_rs, a_cs, ic, pc, mc, kc, kMr, alpha, packed_a);

          for (int64_t jr = 0; jr < nc; jr += kNr) {
            const int64_t nr = std::min<int64_t>(kNr, nc - jr);
            const float* b_panel =
                packed_b_view +
                static_cast<size_t>((jr / kNr) * kc * kNr);

            for (int64_t ir = 0; ir < mc; ir += kMr) {
              const int64_t mr = std::min<int64_t>(kMr, mc - ir);
              const float* a_panel =
                  packed_a + static_cast<size_t>((ir / kMr) * kc * kMr);
              float* c_tile = c + (ic + ir) * ldc + (jc + jr);

              if (mr == kMr && nr == kNr) {
                Kernel::Micro(kc, a_panel, b_panel, c_tile, ldc);
                continue;
              }

              std::memset(edge, 0, sizeof(edge));
              Kernel::Micro(kc, a_panel, b_panel, edge, kNr);
              for (int64_t i = 0; i < mr; ++i) {
                for (int64_t j = 0; j < nr; ++j) {
                  c_tile[i * ldc + j] +=
                      edge[static_cast<size_t>(i * kNr + j)];
                }
              }
            }
          }
        }
      });
    }
  }
}

// Shared entry used by every per-ISA translation unit. Handles degenerate
// shapes, beta scaling, and outputs whose column stride is not one.
template <typename Kernel>
void RunSgemm(int64_t m, int64_t n, int64_t k, float alpha,
              const float* a, int64_t a_rs, int64_t a_cs,
              const float* b, int64_t b_rs, int64_t b_cs,
              float beta, float* c, int64_t c_rs, int64_t c_cs) {
  if (m <= 0 || n <= 0) return;
  if (k <= 0 || alpha == 0.0f) {
    ScaleOutput(m, n, beta, c, c_rs, c_cs);
    return;
  }

  if (c_cs == 1) {
    ScaleOutput(m, n, beta, c, c_rs, c_cs);
    BlockedSgemmAccumulate<Kernel>(m, n, k, alpha, a, a_rs, a_cs, b, b_rs,
                                   b_cs, c, c_rs);
    return;
  }

  static thread_local AlignedScratch c_scratch;
  const size_t elements = static_cast<size_t>(m) * static_cast<size_t>(n);
  float* temp = c_scratch.Reserve(elements);
  std::memset(temp, 0, elements * sizeof(float));
  BlockedSgemmAccumulate<Kernel>(m, n, k, alpha, a, a_rs, a_cs, b, b_rs, b_cs,
                                 temp, n);
  ScaleOutput(m, n, beta, c, c_rs, c_cs);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      c[i * c_rs + j * c_cs] += temp[static_cast<size_t>(i * n + j)];
    }
  }
}

}  // namespace tensora::kernels::internal

#endif  // TENSORA_KERNELS_GEMM_BLOCKING_H_
