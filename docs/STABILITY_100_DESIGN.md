# Tensora 100% Stability and Coverage Design

## Goal

Raise the currently implemented Tensora runtime to a release-grade stability baseline before expanding the framework surface. The immediate quality target is **100.00% merged Dart production line coverage** for the core `tensora` package, backed by exact-head CI, native high-assurance validation, deterministic resource ownership, and explicit hardware qualification for accelerator claims.

Coverage is a quality gate, not a substitute for correctness. New tests must exercise real contracts: malformed inputs, error translation, ownership rollback, lifecycle, provider/device validation, loader failures, concurrency, and numerical correctness.

## Starting point

The integration baseline is commit `488ecab60e919f54e72cd7d1ea53ab9bbabfe7f8` from `feature/coverage-snapshot-20260814`.

The baseline already contains:

- native CPU tensor execution through the stable C ABI;
- optional LibTorch-backed training;
- optional ONNX Runtime inference;
- CPU, CUDA, MPS, XPU, and HIP device identities;
- hosted CPU, MPS, and CoreML validation paths;
- sanitizer, ThreadSanitizer, fuzz, lifecycle, training soak, workspace, Flutter, and coverage workflows;
- a merged Dart coverage snapshot of `843/932 = 90.4506%`.

Physical NVIDIA, Intel accelerator, and AMD accelerator qualification remains a hardware gate and must never be replaced by a mock, skipped assertion, silent CPU fallback, or conditional pass.

## Stability contract

The following invariants remain non-negotiable:

- C++ implementation types never cross the public C ABI;
- opaque typed handles are validated as untrusted input;
- successful creation returns exactly one owned native reference;
- failed creation/adoption leaks zero native references;
- explicit disposal is the primary lifetime mechanism;
- finalizers are best-effort fallback only;
- heavy computation never executes while holding the global handle-registry lock;
- shape, rank, element count, allocation size, device index, pointer, capacity, and handle values are treated as untrusted;
- recoverable user/runtime failures become structured status codes and typed Dart exceptions;
- no unsupported device or provider silently falls back;
- normal tensor-to-tensor operations keep numerical payloads in native memory;
- support documentation reflects only reproducibly validated capabilities.

## Coverage architecture

### Dart

The coverage workflow must merge every supported execution mode needed to reach production code:

- default native library discovery;
- missing-library diagnostics;
- dependency-light core runtime;
- core lifecycle soak;
- LibTorch-enabled training and training edge cases;
- ONNX Runtime inference and inference edge cases;
- wrapper/error-path tests.

The merged result must fail CI unless line coverage is exactly `100.00%` for owned production Dart sources included by the coverage contract. Coverage exclusions are allowed only for platform-specific code that is exercised by a dedicated real-platform CI job and whose exclusion is documented next to the code.

### Native

Add a separate native owned-code coverage report rather than mixing C++ and Dart percentages. Native coverage must be visible in CI and must focus on Tensora-owned runtime/ABI/backend code while excluding third-party libraries and generated compiler/runtime internals. The native gate is tightened only after unreachable platform/configuration branches are represented by real matrix jobs or explicitly justified exclusions.

## Implementation order

1. Close all existing Dart uncovered lines with contract-level tests.
2. Turn the coverage snapshot into an enforced `100.00%` merged gate.
3. Add native coverage instrumentation/reporting and publish uncovered owned-code paths.
4. Re-run the full exact-head software matrix.
5. Close real accelerator hardware qualification wherever a matching runner is available.
6. Only after the baseline is stable, expand Tensor Core through complete vertical slices: dtype semantics, views/strides, broadcasting, indexing, and production operators.
7. New Tensor Core code lands only with public contract, C ABI, native implementation, CPU correctness, accelerator parity where claimed, autograd where differentiable, error/lifetime tests, documentation, and benchmarks where relevant.

## Tensor Core expansion boundaries

The next core expansion must centralize dtype semantics before additional dtypes are exposed. Views must have explicit storage ownership and aliasing rules before transpose/reshape semantics are changed. Broadcasting and indexing must be defined once and shared by CPU and accelerator backends. Operator expansion is intentionally limited to operations that can be implemented and tested as complete vertical slices.

## Definition of done for this stability pass

The pass is complete only when:

- merged Dart production line coverage is `100.00%` on the exact final SHA;
- no line is made unreachable merely to satisfy coverage;
- all newly covered error paths assert typed error semantics and relevant native counters;
- native coverage reporting exists and produces a reproducible owned-code report;
- formatting, analysis, Dart tests, native tests, FFI integration, training, inference, sanitizer, ThreadSanitizer, fuzz, lifecycle/soak, workspace, Flutter, and benchmark gates are green where supported;
- physical hardware support is claimed only for hardware actually exercised on the exact qualifying revision;
- documentation and compatibility matrices match the validated implementation;
- no production path contains placeholders, silent fallbacks, swallowed errors, fake results, or unsupported claims.
