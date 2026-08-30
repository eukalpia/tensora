#ifndef TENSORA_KERNELS_THREAD_POOL_H_
#define TENSORA_KERNELS_THREAD_POOL_H_

#include <cstddef>
#include <cstdint>
#include <functional>

namespace tensora::kernels {

/// Persistent worker pool used by Tensora CPU kernels.
///
/// Kernels receive a half-open index range plus a worker slot so they can index
/// per-thread scratch without allocating inside the parallel region.
using ParallelRangeFn =
    std::function<void(int64_t begin, int64_t end, int worker)>;

class ThreadPool {
 public:
  /// Process-wide pool. Sized from the hardware concurrency, overridable with
  /// the TENSORA_NUM_THREADS environment variable.
  static ThreadPool& Instance();

  /// Total workers including the calling thread, always at least one.
  int worker_count() const { return worker_count_; }

  /// Runs `body` over [0, count) split into at most `worker_count()` chunks,
  /// each a multiple of `grain` where possible, and blocks until all complete.
  ///
  /// Falls back to inline execution when the pool is unavailable, when the work
  /// is too small to pay for hand-off, or when called from inside another
  /// parallel region. Nested calls must not deadlock.
  void ParallelFor(int64_t count, int64_t grain, const ParallelRangeFn& body);

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

 private:
  ThreadPool();
  ~ThreadPool();

  class Impl;
  Impl* impl_;
  int worker_count_;
};

/// Resolves the configured worker count without constructing the pool.
int ConfiguredWorkerCount();

}  // namespace tensora::kernels

#endif  // TENSORA_KERNELS_THREAD_POOL_H_
