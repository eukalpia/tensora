# Tensora 100% Stability Implementation Plan

## Goal

Establish a release-grade exact-head baseline with 100.00% merged Dart production line coverage, native owned-code coverage reporting, and no weakening of existing correctness, ownership, security, or hardware-support contracts.

## Constraints

- Preserve the stable C ABI boundary and typed opaque handles.
- Preserve deterministic disposal and finalizer-as-fallback semantics.
- Do not add silent CPU/provider fallbacks.
- Do not claim hardware support without real hardware execution.
- Do not reduce sanitizer, fuzz, soak, concurrency, or benchmark coverage.
- Do not use coverage exclusions for ordinary reachable production code.
- New numerical features must be complete vertical slices.

## Task 1 — Close core runtime Dart coverage

**Production files**

- `packages/tensora/lib/src/native/native_runtime.dart`
- `packages/tensora/lib/src/errors/tensora_exception.dart`
- `packages/tensora/lib/src/tensor/tensor.dart`

**Tests**

- extend native-runtime edge tests for ABI/status/error translation, scalar shapes, metadata validation, device-index validation, copy-count validation, handle creation failure, and discovery/load diagnostics;
- extend Tensor tests for finalizer-safe release, adoption rollback, disposed access, and explicit transfer/device semantics;
- assert native live-handle/storage counters return to baseline after failure paths.

**Acceptance**

All currently reachable uncovered lines in these files are executed by meaningful assertions; the core-only test matrix remains green.

## Task 2 — Close training Dart coverage

**Production files**

- `packages/tensora/lib/src/training/training.dart`
- `packages/tensora/lib/src/native/native_training_runtime.dart`

**Tests**

- cover constructor validation and unsupported backend behavior;
- cover module/optimizer disposal and use-after-dispose;
- cover parameter/buffer adoption failure and rollback;
- cover optimizer/module wrong-handle and native error translation paths;
- verify zero leaked handles after every failed operation.

**Acceptance**

Training CPU integration and training soak remain green and all reachable training-library lines are covered.

## Task 3 — Close inference Dart coverage

**Production files**

- `packages/tensora/lib/src/inference/onnx.dart`
- `packages/tensora/lib/src/native/native_inference_runtime.dart`

**Tests**

- cover invalid provider selection, malformed input maps, unknown input/output names, metadata/profiling edge cases, session disposal, output adoption rollback, native count/capacity inconsistencies, and structured disabled-backend failures;
- verify session/tensor counters return to baseline after partial failures.

**Acceptance**

ONNX CPU reference execution remains numerically correct and all reachable inference-library lines are covered.

## Task 4 — Enforce exact 100.00% merged Dart coverage

**Workflow**

- modify `.github/workflows/coverage-snapshot.yml` to merge all LCOV reports deterministically;
- normalize duplicate source records by summing hit counts per line;
- print per-file and total coverage;
- fail unless every included production line is hit and total line coverage equals `100.00%`;
- upload merged LCOV plus a human-readable summary artifact.

**Acceptance**

A deliberately uncovered production line makes the workflow fail; the final exact SHA reports 100.00%.

## Task 5 — Add native owned-code coverage reporting

**Build/CI**

- add compiler coverage instrumentation for supported Linux Clang/GCC builds;
- collect only Tensora-owned `native/src` and public ABI implementation sources;
- exclude third-party/runtime-generated code;
- publish per-file uncovered lines and total line/branch coverage;
- keep this report separate from Dart coverage.

**Acceptance**

Native tests execute under instrumentation and CI publishes a reproducible owned-code coverage report without changing runtime semantics.

## Task 6 — Exact-head high-assurance validation

Run and require green, where supported:

- workspace format/analyze/tests on Dart stable and minimum Dart 3.7;
- Native CI Debug/Release on Linux/macOS/Windows;
- Dart FFI Linux/macOS/Windows;
- LibTorch CPU training;
- ONNX CPU inference;
- ASan/UBSan;
- ThreadSanitizer;
- C ABI fuzzing;
- core lifecycle soak;
- training soak;
- Flutter package tests;
- benchmark smoke;
- real MPS/CoreML paths already supported by hosted hardware.

Any failure is fixed at root cause; no required check is skipped or weakened.

## Task 7 — Hardware qualification

Run NVIDIA CUDA, Intel XPU, and AMD HIP/ROCm qualification only on matching physical runners. Each qualifying run must prove device transfer/allocation, real compute, training or inference as claimed, no CPU fallback, and lifecycle counters returning to baseline.

Unavailable hardware remains explicitly unqualified rather than being marked supported.

## Task 8 — Tensor Core expansion after the stability baseline

Implement in this order, each as a complete vertical slice with tests before implementation:

1. centralized dtype semantics and ABI representation;
2. storage metadata, strides, offsets, contiguity, views, and alias lifetime;
3. broadcasting shape inference and backend execution;
4. idiomatic indexing/slicing/select/narrow/gather/scatter/masked-select;
5. arithmetic, comparison, math, reduction, shape, composition, and matrix operators required by the project roadmap;
6. trusted-reference numerical/property tests and benchmarks for every operator family.

No dtype/operator is exported as supported until its Dart API, C ABI, native implementation, error paths, lifetime behavior, CPU correctness, accelerator parity where claimed, and documentation are complete.

## Current progress

The hard merged Dart coverage gate has been proven RED against the original integration baseline and the adversarial native-contract suite raised merged coverage from `843/932 = 90.4506%` to `913/932 = 97.9614%` while exposing a real ONNX output-adoption double-release attempt. That ownership defect is fixed and covered by a permanent fast native-contract CI gate.

The remaining Dart coverage work has been consolidated around deterministic finalizer callbacks, a shared device codec, testable platform library-name resolution, accelerator staging ownership, module tensor collection, and ONNX partial-adoption rollback. No ordinary reachable production line is being hidden solely to raise the metric.

## Final verification

The final SHA is acceptable only when the coverage artifacts, exact-head CI matrix, compatibility documentation, and runtime support claims all refer to the same revision and there are no known leaks, sanitizer defects, silent fallbacks, unsupported claims, or production placeholders.
