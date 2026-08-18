# Tensora Core Superiority Execution Program

> This is a dependency program, not a single implementation branch. Every phase receives its own design, implementation plan, branch, pull request and exact-SHA qualification.

## Program invariants

- Never implement on `main`.
- Never combine two major phases merely to save pull-request overhead.
- Tests precede production changes.
- No silent provider or device fallback.
- No benchmark claim without correctness equivalence and raw artifacts.
- Existing P0 and NN V2 behavior remains green throughout.
- Native owned-code line/function gates and Dart high-assurance gates are not lowered.
- Every new ABI surface is additive until a separately approved breaking release.

## Dependency graph

```text
P1 Tensor Core V2 qualification
  -> P2 execution routing + compatibility observability
  -> P3 high-value operator core
  -> P4 Autograd V2
  -> P5 Tensora IR core
  -> P6 graph capture + compiler V1 + memory planner
  -> P7 CPU superbackend
  -> P8 CUDA superbackend
  -> P9 Metal / Apple backend
  -> P10 AMP + precision policy
  -> P11 quantization
  -> P12 data engine
  -> P13 distributed tensors
  -> P14 AutoParallel
  -> P15 edge runtime and NPU lowering
```

Independent model-format and product integrations may branch only after their required runtime contracts are stable.

## Phase gates

### P1 — Tensor Core V2

Deliver all approved CPU dtypes, typed host I/O, casting, promotion, broadcasting, reductions, slicing/indexing and composition. Preserve explicit float32 autograd boundaries until widened by P4.

Gate: all supported platforms, sanitizers, fuzz, Dart minimum SDK, coverage and P0/NN V2 regressions green on one exact SHA.

### P2 — Execution routing and compatibility observability

Introduce operation schemas, capability queries, tiered resolution, strict mode and profiling explanations without changing mathematical behavior.

Gate: deterministic routing tests, a no-silent-fallback adversarial matrix and zero regression in existing eager workloads.

### P3 — High-value operator core

Implement the operator families needed for reference MLP, CNN and Transformer graphs. Prefer composites first, then generated/native lowering based on profiles.

Gate: differential tests against trusted references and complete eager backward coverage for approved float32 paths.

### P4 — Autograd V2

Unify derivative registration, broadcast reductions, saved tensors, hooks, detach/no-grad/inference semantics and the initial forward/backward graph representation.

Gate: finite differences, mutation/alias adversarial tests, repeated-parent graphs, gradient accumulation and long training soaks.

### P5 — Tensora IR core

Add typed SSA-like values, operator schemas, symbolic shapes, effects, aliases, devices and serialization for diagnostics.

Gate: eager-to-IR semantic equivalence and deterministic round-trip tests. No performance claim yet.

### P6 — Compiler V1 and memory planner

Add capture, canonicalization, constant folding, dead-value elimination, initial fusion, buffer liveness and a compiled execution API.

Gate: compiled/eager/reference equivalence, bounded compilation, cache invalidation tests and published memory/latency baselines.

### P7–P9 — Hardware backends

Build backend capability contracts and progressively own high-value CPU, CUDA and Metal paths. Vendor libraries remain valid Tier 2 implementations.

Gate: real hardware, no CPU fallback, sanitizer/device checks where supported, raw performance artifacts and memory stability.

### P10–P11 — Precision and quantization

Introduce compiler-owned autocast, loss scaling, FP8 where hardware-qualified, INT8/INT4 policies and calibration.

Gate: accuracy budgets, deterministic policy recording and target-specific reference comparisons.

### P12 — Data engine

Add Dataset/DataLoader, batching, shuffling, samplers, isolate preprocessing, bounded prefetch and reproducibility.

Gate: deterministic epochs, cancellation, backpressure, bounded memory and training-throughput evidence.

### P13–P14 — Distributed tensor and AutoParallel

Add device mesh, placements, collectives, composable sharding and later automated plan search.

Gate: real multi-device correctness, failure-recovery contracts, scaling efficiency and inspectable plans.

### P15 — Edge runtime

Add lifecycle-safe Flutter execution, bounded media pipelines and NPU/backend lowering under explicit deployment constraints.

Gate: Android, iOS and desktop lifecycle matrices; UI-isolate safety; memory/thermal diagnostics; repeatable latency results.

## Benchmark ladder

Every phase maintains only benchmarks it can honestly qualify. The ladder grows from microbenchmarks to:

1. tensor creation, views, elementwise, reduction and matmul;
2. MLP forward/backward/update;
3. CNN training and inference;
4. Transformer-block training and inference;
5. full representative models;
6. multi-device scaling;
7. mobile and edge latency and energy proxies.

Reference versions and environment fingerprints are pinned in artifacts, not hard-coded forever in prose.

## Current execution point

P1 remains the only production implementation phase currently permitted. P2 begins only after P1 merges green.
