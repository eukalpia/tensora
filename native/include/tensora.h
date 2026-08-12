#ifndef TENSORA_H_
#define TENSORA_H_

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(TENSORA_NATIVE_BUILD)
#define TS_API __declspec(dllexport)
#else
#define TS_API __declspec(dllimport)
#endif
#else
#define TS_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define TS_ABI_VERSION 1u

typedef uint64_t ts_tensor_t;
typedef int32_t ts_status_t;
typedef uint32_t ts_dtype_t;
typedef uint32_t ts_device_t;

enum {
  TS_OK = 0,
  TS_INVALID_ARGUMENT = 1,
  TS_INVALID_SHAPE = 2,
  TS_OUT_OF_MEMORY = 3,
  TS_UNSUPPORTED = 4,
  TS_INVALID_HANDLE = 5,
  TS_INTERNAL_ERROR = 6
};

enum {
  TS_DTYPE_FLOAT32 = 1u
};

enum {
  TS_DEVICE_CPU = 1u
};

/*
 * ABI and error diagnostics.
 *
 * ts_last_error_message() returns a pointer to thread-local storage owned by
 * Tensora. The pointer remains valid until the next Tensora operation on the
 * same thread changes the diagnostic. Callers must copy it if they need longer
 * retention.
 */
TS_API uint32_t ts_abi_version(void);
TS_API const char* ts_last_error_message(void);
TS_API const char* ts_status_name(int32_t status);
TS_API ts_status_t ts_noop(void);

/*
 * Tensor creation.
 *
 * Successful creation returns one owned handle reference in out_tensor.
 * Callers must eventually release it with ts_tensor_release().
 *
 * Rank zero denotes a scalar. For rank > 0, dims must be non-NULL and every
 * dimension must be positive. data must contain exactly numel float32 values.
 */
TS_API ts_status_t ts_tensor_from_f32(const float* data,
                                      const int64_t* dims,
                                      size_t rank,
                                      ts_tensor_t* out_tensor);

TS_API ts_status_t ts_tensor_full_f32(const int64_t* dims,
                                      size_t rank,
                                      float value,
                                      ts_tensor_t* out_tensor);

/*
 * Metadata.
 *
 * All metadata functions are thread-safe for a valid live handle.
 * ts_tensor_shape writes exactly rank dimensions and requires capacity >= rank.
 */
TS_API ts_status_t ts_tensor_rank(ts_tensor_t tensor, size_t* out_rank);
TS_API ts_status_t ts_tensor_shape(ts_tensor_t tensor,
                                   int64_t* out_dims,
                                   size_t capacity,
                                   size_t* out_rank);
TS_API ts_status_t ts_tensor_dtype(ts_tensor_t tensor, uint32_t* out_dtype);
TS_API ts_status_t ts_tensor_device(ts_tensor_t tensor, uint32_t* out_device);
TS_API ts_status_t ts_tensor_numel(ts_tensor_t tensor, uint64_t* out_numel);

/*
 * Tensor operations.
 *
 * Each successful operation returns a new independent tensor handle with one
 * owned reference. Milestone 1 tensors are immutable, so read-only operations
 * may execute concurrently. Computation is not performed while holding the
 * handle-registry mutex.
 */
TS_API ts_status_t ts_tensor_reshape(ts_tensor_t tensor,
                                     const int64_t* dims,
                                     size_t rank,
                                     ts_tensor_t* out_tensor);
TS_API ts_status_t ts_tensor_transpose2d(ts_tensor_t tensor,
                                         ts_tensor_t* out_tensor);
TS_API ts_status_t ts_tensor_add(ts_tensor_t left,
                                 ts_tensor_t right,
                                 ts_tensor_t* out_tensor);
TS_API ts_status_t ts_tensor_multiply(ts_tensor_t left,
                                      ts_tensor_t right,
                                      ts_tensor_t* out_tensor);
TS_API ts_status_t ts_tensor_sum(ts_tensor_t tensor, ts_tensor_t* out_tensor);
TS_API ts_status_t ts_tensor_matmul(ts_tensor_t left,
                                    ts_tensor_t right,
                                    ts_tensor_t* out_tensor);

/*
 * Explicit native -> host copy.
 *
 * out_values must have capacity >= tensor numel. out_written receives the
 * number of float32 values copied.
 */
TS_API ts_status_t ts_tensor_copy_to_host_f32(ts_tensor_t tensor,
                                              float* out_values,
                                              size_t capacity,
                                              size_t* out_written);

/*
 * Lifetime.
 *
 * Retain/release is thread-safe. Handles are opaque identifiers, never native
 * object pointers. Released identifiers are not reused during process lifetime.
 * Releasing an already released/stale handle returns TS_INVALID_HANDLE.
 */
TS_API ts_status_t ts_tensor_retain(ts_tensor_t tensor);
TS_API ts_status_t ts_tensor_release(ts_tensor_t tensor);

/*
 * Runtime diagnostics used by lifecycle tests and profiling infrastructure.
 * These counters describe Tensora-owned Tensor objects and CPU storage.
 */
TS_API ts_status_t ts_runtime_live_tensor_count(uint64_t* out_count);
TS_API ts_status_t ts_runtime_live_storage_bytes(uint64_t* out_bytes);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* TENSORA_H_ */
