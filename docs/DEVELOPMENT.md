# Tensora Development Guide

This guide defines how engineering work should be performed in the Tensora repository.

The project is systems infrastructure spanning Dart, native code, device runtimes, model formats, and Flutter integration. A change is considered complete only when its behavior, ownership, failure modes, compatibility, tests, and relevant performance characteristics are understood.

## 1. Work from an explicit scope

Before changing code, define:

- the problem being solved;
- the affected package/runtime layer;
- public behavior;
- ownership/lifetime implications;
- device/backend implications;
- expected failure behavior;
- tests required for acceptance;
- whether an RFC is required.

Avoid opportunistic redesign of unrelated subsystems.

## 2. Prefer vertical slices

Implement end-to-end slices that can be validated.

Example:

```text
Dart Tensor API
  ↓
FFI binding
  ↓
C ABI entry point
  ↓
Native validation
  ↓
CPU implementation
  ↓
Correctness test
  ↓
Leak test
  ↓
Benchmark
```

A vertical slice proves the architecture. A directory full of declarations does not.

## 3. Local branch workflow

Create a focused branch from up-to-date `main`.

Examples:

```text
feature/cpu-tensor-storage
feature/onnx-session
fix/native-handle-release
perf/camera-frame-path
docs/model-format
```

Keep the branch limited to one coherent objective.

## 4. Public contract first

For a new subsystem, define the public or internal contract before filling in implementation details.

Review:

- type names;
- method semantics;
- async behavior;
- disposal behavior;
- invalid-state behavior;
- backend independence;
- extensibility;
- compatibility cost.

Do not stabilize an API merely because an implementation already exists.

## 5. Test before expanding breadth

The first implementation of a feature should include the tests needed to prove its fundamental behavior before adding variants.

For numerical code, include reference values or a trusted implementation comparison.

For native resource code, include repeated allocation/disposal stress.

For parsers, include malformed input.

For Flutter lifecycle behavior, include repeated mount/dispose and background/resume scenarios where test infrastructure allows.

## 6. Build modes

The repository should eventually provide reproducible development and release configurations for:

- Dart debug/test;
- native debug;
- native sanitizer builds;
- optimized native release;
- Flutter debug/profile/release as appropriate;
- CUDA-enabled builds when hardware/runtime are available.

Performance conclusions must come from appropriate optimized builds, not debug timings.

## 7. Dart quality gate

Before a Dart-facing change is considered ready:

```text
dart format
static analysis
unit tests
integration tests relevant to the change
```

Once concrete package commands exist, this document should list the exact repository commands rather than generic placeholders.

Warnings introduced by the change should be treated as defects unless explicitly justified.

## 8. Native quality gate

Native changes should validate:

- compilation with supported compilers;
- unit tests;
- integration through the C ABI;
- sanitizer coverage where applicable;
- ownership and cleanup;
- invalid input handling;
- thread-safety assumptions.

Do not allow C++ exceptions to escape the C ABI.

## 9. FFI rules

Every FFI entry point needs:

- exact ownership semantics;
- nullability rules;
- pointer/length validation;
- error return behavior;
- ABI-safe types;
- defined thread behavior;
- tests from Dart through the native boundary.

Avoid chatty scalar-level FFI APIs for tensor computation. Prefer coarse operations and graph/session execution where appropriate.

## 10. Native handles

Opaque handles must be validated before use.

The runtime should be able to detect, where practical:

- unknown handles;
- already released handles;
- wrong object type;
- invalid runtime/context ownership.

Never reinterpret arbitrary user-provided integers as trusted native pointers.

## 11. Memory engineering

For changes that allocate memory, document:

- allocator/provider;
- owner;
- release point;
- alias/view behavior;
- peak memory considerations;
- what happens on partial failure.

Prefer RAII in native internals.

Use explicit cleanup in Dart wrappers for expensive resources, with finalizers as a fallback only.

## 12. Device transfers

A change that moves tensor data between host and device must make that behavior explicit.

Profiler instrumentation should eventually record:

- source device;
- destination device;
- byte count;
- synchronization introduced;
- transfer duration where measurable.

Do not hide expensive transfers inside unrelated convenience APIs.

## 13. Concurrency

Before making a type shareable across threads or isolates, define:

- whether it is immutable;
- whether operations are reentrant;
- synchronization strategy;
- lifetime while requests are active;
- cancellation interaction;
- shutdown behavior.

Training state should be assumed mutable and non-shareable until explicitly designed otherwise.

## 14. Flutter development

Flutter-specific code belongs in Flutter-facing packages.

For any inference path verify:

- heavy work is outside the UI isolate;
- cancellation is safe;
- disposal releases native resources;
- queues are bounded;
- lifecycle transitions do not leak;
- large images/audio buffers are not copied through Dart without necessity.

## 15. Camera development

Realtime vision requires explicit backpressure.

The implementation must define:

- maximum queue depth;
- frame-dropping policy;
- ownership of a frame while inference is running;
- behavior when camera closes;
- behavior when model/session closes;
- cancellation behavior.

Never allow camera frames to accumulate without a fixed bound.

## 16. Model-format development

Model bundle parsing is security-sensitive.

Any change to `.tmodel` must include:

- schema update;
- format-version implications;
- malformed-input tests;
- resource-limit tests;
- path/archive safety tests;
- compatibility documentation;
- migration notes for breaking format changes.

## 17. Dependency policy

Before adding a dependency, document why Tensora should own the integration instead of implementing a smaller boundary itself.

Evaluate:

- license;
- platform availability;
- release cadence;
- security maintenance;
- ABI/API stability;
- binary size;
- transitive dependencies;
- packaging implications.

Native backend dependencies should be modular so unused large runtimes do not automatically ship to every application.

## 18. Performance workflow

Performance work follows this sequence:

1. establish a reproducible benchmark;
2. measure the current baseline;
3. identify the bottleneck with profiling;
4. change one relevant layer;
5. compare before/after;
6. verify correctness and memory behavior;
7. record hardware/runtime details.

Do not optimize from intuition alone when measurement is feasible.

## 19. Benchmark hygiene

Always distinguish:

- cold startup;
- warm execution;
- one-time compilation/provider initialization;
- steady-state latency;
- throughput;
- memory.

Report p50/p95/p99 where latency distributions matter.

Avoid using a single best run as representative performance.

## 20. Documentation workflow

When public behavior changes, update documentation in the same pull request.

Relevant documentation may include:

- README;
- architecture;
- model format;
- compatibility matrix;
- API docs;
- examples;
- benchmarks;
- release notes.

## 21. Review checklist

Before requesting review, verify:

- [ ] scope is focused;
- [ ] public contract is documented;
- [ ] no provider-specific type leaked into stable APIs;
- [ ] ownership is explicit;
- [ ] failure behavior is tested;
- [ ] relevant correctness tests pass;
- [ ] relevant stress/leak tests pass;
- [ ] compatibility impact is documented;
- [ ] benchmarks are present for performance claims;
- [ ] no placeholder code is presented as supported behavior;
- [ ] documentation is updated;
- [ ] no credentials or accidental large artifacts are committed.

## 22. Completion standard

Do not use “done” to mean “compiles locally.”

A Tensora change is complete when the supported behavior is demonstrably correct at the relevant boundaries and the repository contains enough evidence for another engineer to understand, reproduce, test, and maintain it.