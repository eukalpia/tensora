# Tensora Compatibility Policy

This document defines the support contract for Tensora platforms, backends, devices, providers, dtypes, public APIs, and the native ABI.

The central rule is:

> **Support is a tested contract, not a build flag or an upstream capability.**

Tensora is pre-1.0. Source-built functionality may be useful and well tested without constituting a published binary compatibility promise.

## 1. Support levels

### Beta

A Beta surface is implemented, covered by meaningful automated tests, and has no known blocker within its documented scope. Packaging, API details, performance characteristics, or the validated platform matrix may still evolve before 1.0.

### Experimental

An Experimental surface is implemented enough for evaluation but lacks one or more required qualification dimensions such as representative physical hardware, provider coverage, packaging validation, or a sufficiently broad operator/model matrix.

Experimental status is not a production compatibility commitment.

### Unsupported

The feature is not implemented as a supported Tensora contract. Upstream libraries theoretically supporting a capability does not change this status.

Tensora does not currently label any public runtime surface Stable.

## 2. Current source-build matrix

| Surface | State | Validated scope | Current limitations |
| --- | --- | --- | --- |
| Dart core package | Beta | Dart 3.7 minimum compatibility plus current stable Dart | Native runtime is still source-built/provided separately |
| C ABI | Beta | ABI v4, C11 consumer tests, typed opaque handles, fixed-width statuses/devices | Pre-1.0 ABI may evolve through explicit version changes |
| CPU tensor backend | Beta | Linux, macOS, Windows native + Dart FFI | `float32` only; compact eager operation set |
| `float32` | Beta | Tensor creation, transfer, transforms, elementwise ops, reduction, 2D matmul, training/inference paths where documented | No other public dtype yet |
| LibTorch CPU training | Beta | Linux, macOS, Windows native + Dart integration | Optional build dependency; not a packaged binary contract |
| Apple MPS training | Beta | Real Apple Silicon hosted CI: device transfer, matmul, module transfer, forward/loss/backward, optimizer steps, lifecycle checks | Validated operation surface is intentionally narrow |
| NVIDIA CUDA training | Experimental | Device/runtime integration and manual hardware qualification workflow | Physical NVIDIA qualification still required |
| Intel XPU training | Experimental | Device/runtime integration and manual hardware qualification workflow | Physical Intel qualification still required |
| AMD HIP/ROCm training identity | Experimental | Device/runtime integration and manual ROCm qualification workflow | Physical AMD qualification still required; runtime behavior follows the linked LibTorch/ROCm build |
| ONNX Runtime CPU / Linux | Beta | Native + Dart model load, metadata, inference, profiling, concurrency/lifecycle validation | Deterministic fixture, not a broad model-zoo compatibility claim |
| ONNX Runtime CPU / Windows | Beta | Pinned ORT 1.26 native + Dart validation with sidecar DLL resolution | Source-built runtime; no binary installer contract yet |
| CoreML inference / Apple Silicon | Beta | Real CoreML provider execution in hosted Apple Silicon CI | Current provider/operator coverage is limited to validated models |
| CUDA ONNX provider | Experimental | Provider selection path implemented | Physical provider qualification pending |
| DirectML ONNX provider | Experimental | Provider selection/session policy implemented | Physical DirectX 12 GPU qualification pending |
| OpenVINO ONNX provider | Experimental | Provider selection path implemented | Provider/hardware qualification pending |
| MIGraphX ONNX provider | Experimental | Provider selection path implemented | Provider/hardware qualification pending |
| Flutter/mobile runtime | Unsupported | None | Separate future runtime integration work |
| `.tmodel` | Unsupported | Design/roadmap only | No production parser/runtime/container implementation |
| Additional public dtypes | Unsupported | None | Public tensor API currently exposes only `float32` |

The matrix deliberately distinguishes **implemented** from **hardware-qualified**. A manual qualification workflow existing in the repository is not itself evidence that the target hardware has passed.

## 3. Platform validation

### Linux

Hosted CI validates:

- dependency-light native core;
- Dart FFI core behavior;
- LibTorch CPU training;
- ONNX Runtime CPU inference;
- sanitizers, fuzzing, concurrency and soak gates where applicable.

Vendor GPU paths require matching physical hardware qualification.

### macOS

Hosted Apple Silicon CI validates:

- dependency-light core;
- Dart FFI core behavior;
- LibTorch CPU training;
- real MPS training execution;
- real CoreML inference execution.

The Apple hardware evidence is tied to the exact runtime/provider versions exercised in CI and is not a promise for arbitrary historical macOS devices.

### Windows

Hosted CI validates:

- dependency-light core;
- Dart FFI core behavior;
- LibTorch CPU training;
- ONNX Runtime CPU inference.

Windows optional native dependencies are validated as **sidecar DLLs** beside `tensora_native.dll`. When the Dart bridge is given an explicit existing native-library path, it uses a restricted load policy that prioritizes dependencies from that runtime directory rather than depending on an unrelated system DLL or ambient PATH entry.

## 4. Device semantics

The public device vocabulary is:

```text
cpu
cuda:<index>
mps
xpu:<index>
hip:<index>
```

Rules:

- CPU uses index zero.
- MPS uses index zero.
- indexed accelerators reject negative indices.
- device discovery is explicit.
- requesting an unavailable device fails through a structured error.
- binary tensor operations require matching device kind **and** index.
- Tensora does not silently move mixed-device inputs to make an operation succeed.

`TensoraRuntime.availableDevices` reports visible devices for the loaded runtime. `preferredDevice` is a deterministic convenience query; it does not change default tensor construction. Tensor factories remain CPU-default unless the caller supplies `device:`.

Accelerator factory creation currently performs host staging followed by device transfer. This is observable data movement, not a zero-copy creation guarantee.

## 5. Training compatibility

Training is an optional LibTorch-backed build surface.

The validated public training subset includes:

- autograd leaf creation;
- gradient retrieval;
- scalar backward;
- ReLU, sigmoid, tanh;
- MSE and cross-entropy;
- `Linear`;
- parameters and buffers;
- train/eval mode;
- module device transfer;
- SGD, Adam, AdamW;
- checkpoint save/load.

A model/module must be moved to its target accelerator before creating the optimizer for the documented workflow.

The tested PyTorch baseline is a validation baseline, not a promise that all LibTorch versions are ABI-compatible.

## 6. ONNX provider semantics

Public provider preferences are:

```text
auto
CPUExecutionProvider
CUDAExecutionProvider
DmlExecutionProvider
CoreMLExecutionProvider
OpenVINOExecutionProvider
MIGraphXExecutionProvider
```

An **explicit provider request never silently falls back**. Session creation fails if the requested provider cannot be attached.

`auto` is the only policy that permits deterministic platform-aware fallback. The selected provider is exposed on the session and must remain observable.

Current portable input binding materializes dense float32 input data on the host before creating ONNX Runtime input values. Therefore an accelerated execution provider does not imply zero-copy GPU input binding.

DirectML sessions use provider-compatible session settings rather than inheriting CPU assumptions that conflict with DirectML execution.

## 7. DType compatibility

A dtype is supported only for the specific operation/backend/device combination that has executable coverage.

The current public dtype is:

```text
float32
```

Do not infer float16, bfloat16, integer, quantized, or mixed-precision support from upstream backend capabilities.

## 8. Public Dart API compatibility

Tensora is pre-1.0, but public API changes still require compatibility review.

Within a future stable major version:

- avoid removing public symbols;
- avoid changing method meaning silently;
- avoid behaviorally significant default changes;
- deprecate before removal where feasible;
- document migrations.

Device defaults, provider fallback, disposal semantics, and ownership behavior are compatibility-sensitive even when type signatures do not change.

## 9. Native ABI compatibility

The current native ABI is **version 4**.

The Dart bridge checks the loaded runtime ABI before creating native objects. ABI changes must define whether they are additive or breaking and increment the contract when required.

Public ABI rules:

- C-compatible fixed-width primitive types;
- opaque `uint64_t` handles;
- no C++ class layout across the boundary;
- output handles are zeroed before fallible creation work;
- native exceptions are contained and translated into structured statuses;
- invalid, stale, duplicate-release, and wrong-type handles are rejected.

Do not assume ABI compatibility because two native libraries happen to export similarly named symbols.

## 10. Native dependency compatibility

Optional native dependencies are runtime implementation details, but their tested versions matter.

Current integration baselines include:

- PyTorch/LibTorch 2.13.0 family used by training validation;
- ONNX Runtime 1.26.0 used by hosted inference validation.

These are **tested baselines**, not open-ended version ranges.

On Windows, backend DLLs required by a source-built `tensora_native.dll` should be distributed side-by-side with that DLL. CI intentionally validates this layout to prevent accidental binding to an incompatible system installation.

## 11. Backend fallback

Fallback is part of semantics and must be visible.

- Tensor/training device requests do not silently fall back to CPU.
- Explicit ONNX provider requests do not silently fall back.
- ONNX `auto` may fall back according to its deterministic provider policy.

Any future fallback mechanism must remain observable through diagnostics/profiling.

## 12. Model/operator compatibility

Successful loading of one ONNX model does not imply universal ONNX operator support across every provider.

A runtime should distinguish at least:

- model/format error;
- invalid input name/count/shape;
- unsupported dtype;
- provider unavailable;
- operator/provider incompatibility as reported by the backend;
- resource failure;
- runtime/ABI incompatibility.

The current CI fixture proves the complete load/run/lifetime path. Broader model families require their own compatibility evidence.

## 13. Ownership and lifetime compatibility

Creation follows this invariant:

```text
success => exactly one owned native reference escapes
failure => zero owned native references escape
```

Public wrappers expose deterministic `dispose()` behavior. Finalizers are backup cleanup, not the primary lifetime strategy.

Live tensor/storage/module/optimizer/session counters are test diagnostics for Tensora-owned wrapper lifetime. They do not claim to measure all allocator caches maintained internally by third-party runtimes.

## 14. Concurrency compatibility

Immutable tensor operations and the documented reusable ONNX session path have dedicated concurrency validation.

Mutable training objects are not automatically safe for arbitrary concurrent mutation. Threading guarantees must be established per mutable subsystem rather than inferred from the handle registry being synchronized.

## 15. Reproducibility

Identical seeds do not guarantee bitwise-identical floating-point results across:

- different hardware;
- different providers;
- different kernels;
- different native runtime versions.

Tests use mathematically known references and justified tolerances. Deterministic modes, when added, must document their exact scope and cost.

## 16. Hardware claims

Hardware-specific support requires actual execution evidence.

Compilation or provider discovery alone is insufficient. Hardware qualification must prove the selected device/provider is active and must reject CPU fallback where the test claims accelerator execution.

Current real-hardware automated evidence covers Apple MPS and Apple CoreML. Other vendor qualification remains pending until their manual hardware workflows complete on matching physical runners.

## 17. Binary distribution scope

Tensora currently validates **source-built** native runtimes. It does not yet promise that arbitrary compiler versions, OS revisions, CPU architectures, or third-party dependency layouts are binary-compatible.

A future published native artifact matrix must name:

- target OS and architecture;
- compiler/runtime ABI requirements;
- bundled or external dependencies;
- tested device/provider versions;
- supported loading/packaging layout.

## 18. Documentation rule

Never write broad claims such as:

```text
Supports GPU
Supports Windows GPU
Supports ONNX everywhere
```

without naming the validated device/provider/platform scope.

Prefer precise statements such as:

```text
Apple MPS training is hardware-validated on hosted Apple Silicon CI for the documented training subset.
```

## 19. Promotion criteria

A surface can move from Experimental to Beta only after its missing qualification dimensions are closed. For hardware/provider work that normally includes:

- representative physical execution;
- no unintended fallback;
- correctness reference;
- lifecycle stability;
- failure-path validation;
- documented dependency/runtime versions.

Promotion from Beta to Stable additionally requires a durable release and packaging contract appropriate for a post-1.0 compatibility promise.
