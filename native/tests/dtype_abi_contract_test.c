#include "tensora.h"

#define TS_STATIC_ASSERT(name, expression) \
  typedef char name[(expression) ? 1 : -1]

TS_STATIC_ASSERT(ts_abi_version_is_6, TS_ABI_VERSION == 6u);
TS_STATIC_ASSERT(ts_dtype_float32_is_1, TS_DTYPE_FLOAT32 == 1u);
TS_STATIC_ASSERT(ts_dtype_float16_is_2, TS_DTYPE_FLOAT16 == 2u);
TS_STATIC_ASSERT(ts_dtype_bfloat16_is_3, TS_DTYPE_BFLOAT16 == 3u);
TS_STATIC_ASSERT(ts_dtype_float64_is_4, TS_DTYPE_FLOAT64 == 4u);
TS_STATIC_ASSERT(ts_dtype_int8_is_5, TS_DTYPE_INT8 == 5u);
TS_STATIC_ASSERT(ts_dtype_uint8_is_6, TS_DTYPE_UINT8 == 6u);
TS_STATIC_ASSERT(ts_dtype_int16_is_7, TS_DTYPE_INT16 == 7u);
TS_STATIC_ASSERT(ts_dtype_int32_is_8, TS_DTYPE_INT32 == 8u);
TS_STATIC_ASSERT(ts_dtype_int64_is_9, TS_DTYPE_INT64 == 9u);
TS_STATIC_ASSERT(ts_dtype_bool_is_10, TS_DTYPE_BOOL == 10u);

int main(void) { return 0; }
