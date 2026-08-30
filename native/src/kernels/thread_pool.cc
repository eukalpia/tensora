#include "kernels/thread_pool.h"

#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace tensora::kernels {
namespace {

/// True while the calling thread is already inside a parallel region. Nested
/// parallelism runs inline instead of re-entering the pool, which keeps a
/// kernel that calls another kernel from deadlocking on its own workers.
thread_local bool t_inside_parallel_region = false;

int ReadEnvironmentWorkerCount() {
#if defined(_MSC_VER)
  size_t length = 0;
  char buffer[32] = {0};
  if (getenv_s(&length, buffer, sizeof(buffer), "TENSORA_NUM_THREADS") != 0) {
    return 0;
  }
  if (length == 0) return 0;
  const char* raw = buffer;
#else
  const char* raw = std::getenv("TENSORA_NUM_THREADS");
  if (raw == nullptr) return 0;
#endif
  errno = 0;
  char* end = nullptr;
  const long parsed = std::strtol(raw, &end, 10);
  if (end == raw || parsed <= 0 || parsed > 4096) return 0;
  return static_cast<int>(parsed);
}

}  // namespace

int ConfiguredWorkerCount() {
  const int configured = ReadEnvironmentWorkerCount();
  if (configured > 0) return configured;
  const unsigned int hardware = std::thread::hardware_concurrency();
  if (hardware == 0) return 1;
  return static_cast<int>(hardware);
}

class ThreadPool::Impl {
 public:
  explicit Impl(int workers) : extra_workers_(workers - 1) {
    if (extra_workers_ <= 0) return;
    threads_.reserve(static_cast<size_t>(extra_workers_));
    for (int index = 0; index < extra_workers_; ++index) {
      threads_.emplace_back([this, index] { WorkerLoop(index + 1); });
    }
  }

  ~Impl() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    wake_.notify_all();
    for (std::thread& thread : threads_) {
      if (thread.joinable()) thread.join();
    }
  }

  void Run(int chunks, const ParallelRangeFn& body,
           const std::vector<int64_t>& bounds) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      body_ = &body;
      bounds_ = &bounds;
      chunk_count_ = chunks;
      next_chunk_ = 1;  // slot 0 is executed by the calling thread
      outstanding_ = chunks - 1;
      failure_ = nullptr;
      ++generation_;
    }
    wake_.notify_all();

    RunChunk(0, 0);

    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [this] { return outstanding_ == 0; });
    std::exception_ptr failure = failure_;
    body_ = nullptr;
    bounds_ = nullptr;
    lock.unlock();
    if (failure) std::rethrow_exception(failure);
  }

 private:
  void RunChunk(int chunk, int worker) {
    const int64_t begin = (*bounds_)[static_cast<size_t>(chunk)];
    const int64_t end = (*bounds_)[static_cast<size_t>(chunk) + 1];
    if (begin >= end) return;
    (*body_)(begin, end, worker);
  }

  void WorkerLoop(int worker) {
    uint64_t seen_generation = 0;
    for (;;) {
      int chunk = -1;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        wake_.wait(lock, [this, &seen_generation] {
          return stopping_ ||
                 (generation_ != seen_generation && next_chunk_ < chunk_count_);
        });
        if (stopping_) return;
        seen_generation = generation_;
        if (next_chunk_ >= chunk_count_) continue;
        chunk = next_chunk_++;
      }

      std::exception_ptr failure;
      try {
        t_inside_parallel_region = true;
        RunChunk(chunk, worker);
      } catch (...) {
        failure = std::current_exception();
      }
      t_inside_parallel_region = false;

      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (failure && !failure_) failure_ = failure;
        --outstanding_;
        if (outstanding_ == 0) done_.notify_one();
      }
    }
  }

  std::vector<std::thread> threads_;
  int extra_workers_;

  std::mutex mutex_;
  std::condition_variable wake_;
  std::condition_variable done_;
  bool stopping_ = false;
  uint64_t generation_ = 0;
  int chunk_count_ = 0;
  int next_chunk_ = 0;
  int outstanding_ = 0;
  const ParallelRangeFn* body_ = nullptr;
  const std::vector<int64_t>* bounds_ = nullptr;
  std::exception_ptr failure_;
};

ThreadPool::ThreadPool()
    : impl_(nullptr), worker_count_(ConfiguredWorkerCount()) {
  impl_ = new Impl(worker_count_);
}

ThreadPool::~ThreadPool() { delete impl_; }

ThreadPool& ThreadPool::Instance() {
  static ThreadPool pool;
  return pool;
}

void ThreadPool::ParallelFor(int64_t count, int64_t grain,
                             const ParallelRangeFn& body) {
  if (count <= 0) return;
  if (grain < 1) grain = 1;

  const int64_t max_useful_chunks = (count + grain - 1) / grain;
  int chunks = static_cast<int>(
      std::min<int64_t>(max_useful_chunks, worker_count_));
  if (t_inside_parallel_region || chunks <= 1) {
    body(0, count, 0);
    return;
  }

  // Split into chunk boundaries aligned to the grain so a microkernel never
  // receives a partial register block in the middle of the range.
  std::vector<int64_t> bounds;
  bounds.reserve(static_cast<size_t>(chunks) + 1);
  const int64_t total_grains = (count + grain - 1) / grain;
  int64_t assigned = 0;
  bounds.push_back(0);
  for (int index = 0; index < chunks; ++index) {
    const int64_t remaining_chunks = chunks - index;
    const int64_t grains =
        (total_grains - assigned + remaining_chunks - 1) / remaining_chunks;
    assigned += grains;
    bounds.push_back(std::min<int64_t>(assigned * grain, count));
  }

  const bool previous = t_inside_parallel_region;
  t_inside_parallel_region = true;
  try {
    impl_->Run(chunks, body, bounds);
  } catch (...) {
    t_inside_parallel_region = previous;
    throw;
  }
  t_inside_parallel_region = previous;
}

}  // namespace tensora::kernels
