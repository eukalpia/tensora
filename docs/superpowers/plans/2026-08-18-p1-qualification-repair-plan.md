# P1 Tensor Core V2 Qualification Repair Plan

**Goal:** Restore a clean, cross-platform exact-SHA baseline for the existing P1A implementation before continuing P1B–P1D.

**Scope:** Qualification defects only. Do not add broadcasting, reductions, indexing or compiler functionality in this repair.

## Task 1 — Freeze evidence

1. Record pull-request head, base and merge SHA.
2. Preserve logs for every failed workflow.
3. Group cascading failures by root cause rather than workflow count.
4. Add minimal local reproductions for Linux native, public ABI and Dart minimum-SDK failures.

## Task 2 — Public ABI v6 contract

1. Make the standalone C contract require ABI v6.
2. Compile it as strict C11 with warnings as errors.
3. Verify stable dtype codes and function signatures.
4. Confirm malformed inputs clear outputs before returning.

## Task 3 — Windows linkage

1. Reproduce MSVC C4273 on the internal targets that compile tensor implementation sources directly.
2. Align export/import macros without weakening warnings-as-errors.
3. Prefer a build-structure fix over suppressing C4273.
4. Run Debug and Release Windows builds in hosted CI.

## Task 4 — Defensive status contracts

1. Reproduce the four `training_core_internal_tests` failures.
2. Trace each changed status to the first semantic-classification point.
3. Preserve established public status categories unless the approved design explicitly changes them.
4. Add or adjust focused tests first, then fix the source classification.
5. Run the full native CTest suite, ASan/UBSan and TSan.

## Task 5 — Dart 3.7 compatibility

1. Keep the package minimum at Dart 3.7.
2. Add explicit type casts only where old analyzer flow analysis requires them.
3. Preserve runtime validation before every cast.
4. Run formatter, strict analysis and package tests on Dart 3.7 and current stable.

## Task 6 — Coverage closure

1. Merge core, training and inference native traces.
2. Restore owned native line/function thresholds without exclusions.
3. Restore the Dart merged production threshold without test-only production branches.
4. Exercise real provider-backed paths required by the gate.

## Task 7 — Exact-SHA matrix

Require success for:

- Native Linux, macOS and Windows Debug and Release;
- ASan/UBSan and ThreadSanitizer;
- C ABI fuzz and public ABI contracts;
- Dart FFI on all desktop platforms;
- Dart 3.7 and current stable;
- CPU training, LibTorch policy, real Apple MPS, ONNX and CoreML;
- Flutter, workspace, coverage and high-assurance workflows;
- lifecycle and training soaks.

## Task 8 — Finish

1. Remove temporary source and diagnostic workflows that are not permanent product infrastructure.
2. Update the pull-request body with exact head, integration tree, workflow count and coverage.
3. Keep the pull request draft until every required check is green.
4. Merge only after no unresolved review thread remains.
