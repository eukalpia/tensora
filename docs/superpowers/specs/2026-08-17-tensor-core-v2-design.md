# Tensora Tensor Core V2 Design

## Purpose

P1 turns Tensora's dtype metadata into real, executable tensor semantics. The public API already names ten dtypes, but the native runtime currently stores only `float32`, exposes only `f32` host I/O, and restricts most core operations to equal-shape `float32` tensors. Tensor Core V2 closes that gap without weakening the P0/NN V2 high-assurance gates.

The phase is delivered as reviewable vertical slices. Every supported behavior must agree across Dart API, C ABI, native storage, kernels, host conversion, errors, autograd applicability, tests, coverage, and platform CI.

## Goals

1. Implement real CPU storage for:
   - `float16`
   - `bfloat16`
   - `float32`
   - `float64`
   - `int8`
   - `uint8`
   - `int16`
   - `int32`
   - `int64`
   - `bool`
2. Add exact typed host import/export and deterministic casting.
3. Add dtype promotion to native binary execution.
4. Add NumPy-style broadcasting and scalar arithmetic.
5. Add axis-aware reductions.
6. Add slicing/indexing, concat/stack, gather/scatter.
7. Preserve view/alias/version semantics and deterministic ownership.
8. Preserve `100%` owned native line coverage and all existing platform/sanitizer gates.

## Non-goals

- AMP and gradient scaling belong to the training phase after dtype/device qualification.
- CUDA/XPU/HIP dtype support is not inferred from CPU support.
- Integer and boolean autograd is not supported.
- Quantized inference formats are not introduced in this phase.
- Compiler graph capture and fusion remain P7 work.

## Public Dart API

### Creation and host materialization

Existing constructors become genuinely dtype-aware:

```dart
final ids = Tensor.fromList(
  [10, 20, 30],
  shape: Shape([3]),
  dtype: DType.int64,
);

final mask = Tensor.full(
  Shape([2, 4]),
  true,
  dtype: DType.boolean,
);
```

`Tensor.toList()` returns values using Dart-native representations:

- floating tensors -> `List<double>`
- integer tensors -> `List<int>`
- boolean tensors -> `List<bool>`

A typed low-level materialization API is also available for zero-ambiguity interop:

```dart
TypedData toTypedData();
```

`float16` and `bfloat16` return canonical `Uint16List` bit patterns from `toTypedData()`, while `toList()` returns decoded `double` values.

### Casting

```dart
final fp32 = ids.cast(DType.float32);
```

Casting is explicit and never silently changes the source tensor. Conversion failures throw structured Tensora exceptions with operation context.

### Arithmetic

```dart
final y = x + bias;
final z = x * 0.5;
final total = x.sum();
final rows = x.sum(axes: [1], keepDimensions: true);
```

Method forms remain available (`add`, `multiply`, `sum`) and operators delegate to the same native semantics.

### Indexing and composition

```dart
final window = x.slice([
  Slice.range(0, 4),
  Slice.all,
]);
final row = x.index([2]);
final joined = Tensor.concat([a, b], axis: 0);
final batched = Tensor.stack([a, b], axis: 0);
final picked = x.gather(indices, axis: 1);
final updated = x.scatter(indices, values, axis: 1);
```

All indexing APIs validate rank, axes, bounds, strides, duplicate scatter behavior, and output shape before native allocation.

## Native dtype model

The stable C ABI codes remain:

| DType | Code | Width | Canonical host representation |
|---|---:|---:|---|
| float32 | 1 | 4 | IEEE-754 binary32 |
| float16 | 2 | 2 | IEEE-754 binary16 bits |
| bfloat16 | 3 | 2 | bfloat16 bits, round-to-nearest-even |
| float64 | 4 | 8 | IEEE-754 binary64 |
| int8 | 5 | 1 | two's-complement signed byte |
| uint8 | 6 | 1 | unsigned byte |
| int16 | 7 | 2 | two's-complement signed 16-bit |
| int32 | 8 | 4 | two's-complement signed 32-bit |
| int64 | 9 | 8 | two's-complement signed 64-bit |
| bool | 10 | 1 | canonical byte `0` or `1` |

Native `DType` is expanded to all stable codes. Central helpers provide code validation, byte width, category checks, accumulator dtype, and promotion. No backend may duplicate or drift from those rules.

## CPU storage architecture

`CpuStorage` becomes dtype-aware while preserving the existing `float32` fast path required by autograd and NN V2.

The implementation uses a closed `std::variant` of typed vectors:

- `std::vector<uint16_t>` for `float16` and `bfloat16` bit patterns;
- `std::vector<float>` for `float32`;
- `std::vector<double>` for `float64`;
- typed integer vectors for integer dtypes;
- `std::vector<uint8_t>` for bool, always canonicalized to `0` or `1`.

The storage object owns its dtype and element count. Its byte size is derived from the actual typed allocation, not from shape metadata. Typed accessors return a status instead of throwing `std::bad_variant_access`.

Views continue sharing one storage object and one version/identity anchor. `storage_offset` and strides remain measured in elements, never bytes.

## C ABI v6

`TS_ABI_VERSION` becomes `6`. All ABI v5 symbols remain available and retain their behavior.

New functions:

```c
ts_status_t ts_tensor_from_host(
    const void* data,
    size_t data_bytes,
    uint32_t dtype,
    const int64_t* dims,
    size_t rank,
    ts_tensor_t* out_tensor);

ts_status_t ts_tensor_full(
    const void* scalar,
    size_t scalar_bytes,
    uint32_t dtype,
    const int64_t* dims,
    size_t rank,
    ts_tensor_t* out_tensor);

ts_status_t ts_tensor_copy_to_host(
    ts_tensor_t tensor,
    void* out_data,
    size_t capacity_bytes,
    size_t* out_written_bytes);

ts_status_t ts_tensor_cast(
    ts_tensor_t tensor,
    uint32_t target_dtype,
    ts_tensor_t* out_tensor);
```

The generic ABI transports canonical raw representations. It validates dtype codes, exact byte counts, multiplication overflow, shape element count, null pointers, bool canonicality, and output handles before reading memory or allocating.

Existing `*_f32` functions are wrappers over the generic implementation and remain ABI-compatible.

## Conversion semantics

### Floating point

- `float16` conversion follows IEEE-754 binary16 with round-to-nearest-even.
- `bfloat16` conversion uses round-to-nearest-even, preserving NaN classification.
- NaN and infinity are accepted for floating-to-floating casts.
- Floating-to-integer rejects NaN, infinity, and out-of-range values.
- Floating-to-integer truncates toward zero after validation.

### Integers

- Widening casts are exact.
- Narrowing casts reject values outside the destination range; no wraparound occurs.
- Integer-to-floating follows the destination IEEE representation. Loss of precision is allowed and deterministic.

### Boolean

- Host bool storage accepts only `0` and `1` bytes.
- Numeric-to-bool maps zero to false and non-zero to true.
- Bool-to-numeric maps false to zero and true to one.

All conversions are transactional: a failed element conversion leaves no published tensor handle.

## Promotion rules

Native promotion must match `DType.promote` in Dart exactly. The contract is centralized and tested exhaustively for every ordered dtype pair.

Key rules:

- equal dtype remains unchanged;
- bool promotes to the other operand;
- any `float64` operand produces `float64`;
- any `float32` operand produces `float32` unless `float64` is present;
- mixed `float16`/`bfloat16` promotes to `float32`;
- signed/unsigned `int8` mix promotes to `int16`;
- wider signed integer wins for the approved integer set.

## Operation support

### Dtype-generic structural operations

These preserve dtype and work for every CPU dtype:

- reshape
- transpose/view creation
- slice/index
- concat/stack
- gather/scatter
- host copy
- device metadata and lifetime operations

### Numerical operations

- add/multiply support all non-boolean numeric dtypes and native promotion;
- scalar arithmetic uses the same promotion path;
- sum uses `DType.reductionAccumulator` unless an explicit output dtype is supplied;
- matmul initially supports floating dtypes and promotes inputs before execution;
- bool arithmetic is rejected; bool logical operations are a later explicit API rather than accidental integer arithmetic.

### Autograd

- autograd remains supported only for floating tensors;
- initial native backward kernels remain qualified for `float32`;
- other floating dtypes may be stored and cast but `requiresGrad` fails explicitly until their backward kernels are qualified;
- integer and bool tensors always reject `requiresGrad`.

This explicit matrix prevents storage support from being misreported as training support.

## Broadcasting

Binary operations use trailing-dimension NumPy broadcasting:

- dimensions are compatible when equal or either is `1`;
- missing leading dimensions behave as `1`;
- zero-rank tensors are scalars;
- output shape is validated before allocation;
- index mapping uses checked integer arithmetic;
- incompatible shapes return `TS_INVALID_SHAPE`.

Backward reduces broadcasted gradients back to each parent shape. The implementation must preserve stale-alias/version checks.

## Reductions

Axis reduction accepts normalized, unique axes and supports negative Dart axes after front-end normalization. Empty axis lists return an identity view/copy according to the public contract. `keepDimensions` controls retained singleton dimensions. Accumulator dtype is explicit and testable.

## Indexing semantics

- Slice ranges are half-open `[start, stop)`.
- Positive strides are required in the initial P1 implementation; negative strides require a separately qualified view contract.
- Indexing removes indexed axes.
- Slice views share storage/version identity whenever representable by strides and offset.
- Materialization occurs only when an operation requires contiguous storage.
- Gather indices use `int64` and are bounds checked.
- Scatter returns a new tensor; source tensors are immutable.
- Duplicate scatter indices use deterministic last-write-wins order.

## Error and ownership guarantees

- Every failed ABI call clears output handles/counts before returning.
- Byte-count calculations use checked multiplication.
- No generic `void*` buffer is read before dtype and size validation.
- No C++ exception crosses the ABI.
- Cast, concat, gather, and scatter publish handles only after full validation and successful allocation.
- Live storage byte counters include the actual typed allocation size.
- Finalizers remain safety nets; normal Dart paths dispose deterministically.

## Testing and qualification

Each slice requires:

- native unit tests for every dtype and error path;
- exhaustive promotion matrix tests;
- round-trip host import/export golden vectors;
- half/bfloat edge vectors including zero, subnormal, infinity and NaN;
- cast boundary and overflow tests;
- Dart -> FFI -> native integration tests;
- view/alias/version tests for non-float32 storage;
- finite-difference tests for any newly qualified autograd dtype/operation;
- C ABI malformed-input and fuzz coverage;
- ASan/UBSan/TSan;
- Linux, macOS and Windows exact-SHA CI;
- owned native line coverage of `100%` without threshold reduction or broad exclusions.

## Delivery slices

### P1A — dtype storage, typed host I/O and cast

Implement the native dtype model, variant CPU storage, ABI v6 generic import/export/full/cast, Dart typed conversion, dtype-aware reshape/transpose, and explicit autograd/device restrictions.

### P1B — promotion, broadcasting and scalar arithmetic

Implement native promotion, add/multiply across numeric dtypes, scalar operators, broadcast planning, and broadcast-aware backward for qualified autograd dtypes.

### P1C — axis reductions

Implement axis/keep-dimension reduction planning, accumulator dtypes, `sum`, `mean`, `max`, `min`, and their supported backward rules.

### P1D — indexing and tensor composition

Implement slice/index views, concat/stack, gather/scatter, bounds/overflow checks, and alias/version qualification.

## P1 acceptance gate

P1 is complete only when all approved CPU dtypes have real storage and host I/O, every claimed operation follows the documented dtype/shape semantics, Dart and native promotion agree exhaustively, view/alias/version behavior is preserved, all malformed-input/lifetime tests pass, native owned-code line coverage is `100%`, and the full hosted matrix is green on one exact revision.
