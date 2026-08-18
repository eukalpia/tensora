# Tensora Core Superiority + Compatibility Strategy

## Status

Approved strategic direction for Tensora after the P0 and NN V2 foundation.

This document defines how Tensora will compete with mature ML frameworks without becoming a slow, incomplete clone of their public namespaces.

## Objective

Tensora must become a general-purpose ML system that is measurably superior on the workloads and deployment targets it claims to support.

Success is not defined by feature-count parity. Success requires reproducible evidence across:

- eager and compiled execution latency;
- training and inference throughput;
- peak and steady-state memory;
- compile and startup latency;
- numerical correctness and determinism;
- single-device and distributed scaling;
- mobile and edge deployment cost;
- API complexity and required user code;
- crash, leak, ownership, and malformed-input safety.

No release may claim superiority from architecture diagrams or synthetic microbenchmarks alone.

## Chosen Strategy

Tensora uses **core superiority + compatibility**:

1. Own the critical execution stack:
   - Tensor semantics;
   - storage and views;
   - dtype promotion;
   - autograd;
   - graph capture;
   - Tensora IR;
   - optimization and memory planning;
   - runtime scheduling;
   - high-value CPU, CUDA, Metal, and accelerator kernels.
2. Cover the long tail through explicit compatibility paths:
   - ONNX Runtime providers;
   - vendor libraries;
   - backend lowering;
   - portable composite implementations;
   - narrowly scoped interop adapters.
3. Replace fallbacks in measured priority order, beginning with operations that dominate real workloads.

Fallback is never silent. Diagnostics must expose which execution tier handled every operation and why.

## Execution Tiers

Every supported operation belongs to one observable tier:

### Tier 0 — Hand-qualified Tensora kernel

Used for operations whose performance or semantics justify a specialized implementation.

Requirements:

- deterministic ownership;
- complete error contracts;
- reference correctness;
- supported-device qualification;
- benchmark evidence;
- sanitizer and lifecycle coverage.

### Tier 1 — Tensora-generated kernel

Generated from Tensora Kernel IR for a concrete dtype, layout, shape family, and target.

The compiler may fuse multiple logical operations into one generated kernel.

### Tier 2 — Vendor primitive

Lowered to a mature primitive such as BLAS, oneDNN, cuBLAS, cuDNN, MPSGraph, Metal Performance Shaders, or an approved NPU provider.

Vendor use must remain inspectable and must not introduce silent CPU fallback.

### Tier 3 — Portable composite

Expressed through qualified lower-level Tensora operations.

This tier prioritizes correctness and coverage while hot paths move toward generated or specialized kernels.

### Tier 4 — Compatibility backend

Executed through an explicit ONNX, provider, or interop boundary.

This tier is temporary for high-value paths and permanent only where owning the implementation would not improve performance, portability, reliability, or product control.

## Architectural Center

Tensora IR becomes the center of the system rather than an optional layer added after eager execution.

```text
Dart API
  ├─ eager execution
  └─ graph capture
          ↓
      Tensora IR
          ↓
  shape / dtype / alias analysis
          ↓
      Autograd IR
          ↓
 functionalization and mutation safety
          ↓
 fusion / layout / memory planning
          ↓
 device-aware scheduling
          ↓
 CPU / CUDA / Metal / NPU / compatibility backends
          ↓
 high-assurance Tensora runtime
```

Eager and compiled modes must share semantics. Compilation may change scheduling, layout, fusion, recomputation, and storage reuse, but not observable numerical or ownership contracts outside documented tolerance and determinism policies.

## Tensor and DType Contract

The Tensor Core must provide one consistent contract across Dart, the C ABI, native storage, execution, autograd applicability, errors, and host materialization.

The initial approved dtype set is:

- float16;
- bfloat16;
- float32;
- float64;
- int8;
- uint8;
- int16;
- int32;
- int64;
- bool.

CPU support does not imply accelerator support. Unsupported device/dtype combinations fail explicitly.

## Autograd Direction

Autograd evolves from a correct dynamic reverse-mode engine into a joint forward/backward optimization system.

The compiler must eventually reason about:

- saved-tensor lifetime;
- safe recomputation;
- automatic activation checkpointing;
- forward/backward fusion;
- gradient accumulation scheduling;
- early storage release;
- alias and mutation versions;
- mixed precision and gradient scaling;
- custom differentiation rules;
- higher-order differentiation only after first-order semantics are complete.

Integer and boolean autograd remain unsupported.

## Memory Superiority

Memory is a first-class optimization objective.

A compiled program may accept a memory budget and automatically choose among:

- buffer reuse;
- lifetime shortening;
- activation recomputation;
- activation checkpointing;
- layout changes;
- mixed precision;
- parameter, optimizer, or activation sharding;
- bounded host/device offload;
- fused optimizers;
- static allocation plans.

Every automatic choice must be explainable through diagnostics.

An out-of-memory failure must report the attempted plan, dominant allocations, and viable alternatives when they are known.

## Kernel and Autotuning Direction

Tensora does not rewrite every vendor primitive for ideological reasons.

The compiler selects among vendor primitives, generated kernels, specialized kernels, and portable compositions using correctness gates and measured cost models.

Autotuning keys include:

- hardware fingerprint;
- driver and runtime versions;
- operation graph;
- dtype;
- shape family;
- layout and strides;
- determinism policy;
- memory budget.

Tuning results are versioned and invalidated when any correctness-relevant key changes.

## Distributed Direction

Distributed execution is expressed through global tensors and device meshes rather than unrelated one-off wrappers.

The planner may compose:

- data parallelism;
- parameter and optimizer sharding;
- tensor parallelism;
- sequence or context parallelism;
- pipeline parallelism;
- expert parallelism.

Automatic planning is permitted only when the selected strategy is inspectable, reproducible, and constrained by explicit communication, memory, and topology models.

## Quantization Direction

Quantization belongs to the compiler and deployment planner.

Supported plans may mix:

- float32;
- float16;
- bfloat16;
- FP8 where qualified;
- INT8;
- INT4 weights;
- future target-specific formats.

A deployment search must preserve user-specified accuracy, latency, memory, and binary-size constraints. It may not silently choose a lower-accuracy configuration.

## Flutter and Edge Advantage

Tensora keeps one Dart-facing programming model across training, server inference, desktop, mobile, and edge deployment.

Heavy execution must remain outside the Flutter UI isolate.

Edge execution requires:

- bounded queues and explicit backpressure;
- predictable lifecycle release;
- zero-copy or bounded-copy camera and media paths;
- inspectable CPU/GPU/NPU placement;
- atomic model activation and rollback;
- no unexpected networking in core packages;
- target-aware quantization and memory planning.

## Reliability Contract

Existing high-assurance requirements remain non-negotiable:

- stable additive C ABI evolution;
- no native exception crossing the ABI;
- deterministic handle ownership;
- failure outputs cleared before return;
- no silent device or provider fallback;
- malformed metadata treated as untrusted;
- sanitizer, fuzz, race, leak, and lifecycle tests;
- exact-SHA qualification;
- hard native owned-code line and function coverage gates;
- documented compatibility boundaries.

Performance work may not weaken these contracts.

## Benchmark Contract

Tensora maintains a versioned comparison matrix against relevant baselines.

Required workload classes include:

- small and large MLPs;
- convolutional vision models;
- transformer encoder and decoder blocks;
- attention and KV-cache workloads;
- embedding and recommendation workloads;
- eager and compiled training;
- eager and compiled inference;
- CPU, CUDA, Apple, mobile, and supported NPU targets;
- single-device and multi-device scaling;
- cold start, warm start, and steady state.

Each result records:

- exact Tensora and baseline revisions;
- hardware, OS, driver, runtime, compiler, and library versions;
- shapes, dtype, batch, sequence length, and layout;
- warmup and measurement protocol;
- median, p95, throughput, peak memory, and compile time;
- correctness tolerance;
- determinism mode;
- execution tier and fallback usage.

A superiority claim requires multiple representative workloads and cannot be based on one favorable microbenchmark.

## Delivery Sequence

The immediate sequence remains vertical and gated:

1. P1A — real dtype storage, typed host I/O, casting, and explicit boundaries.
2. P1B — promotion, broadcasting, scalar arithmetic, and broadcast-aware backward.
3. P1C — axis reductions and supported backward rules.
4. P1D — indexing, slicing, concat, stack, gather, and scatter.
5. Operator Core.
6. Autograd V2.
7. NN V3 and production model primitives.
8. Tensora IR and graph capture.
9. Compiler, memory planner, and generated CPU kernels.
10. CUDA and Metal superbackends.
11. AMP, FP8, quantization, and autotuning.
12. Global tensors, distributed planning, and AutoParallel.
13. Edge runtime and accelerator deployment.

A later phase may be designed while an earlier one is being qualified, but production implementation does not cross a phase boundary before the preceding exact-SHA gate is green.

## Explicit Non-Goals

Tensora will not:

- copy thousands of APIs without workload evidence;
- claim parity from namespace size;
- replace cuBLAS, cuDNN, oneDNN, or ONNX Runtime without measurable benefit;
- hide compatibility execution;
- trade correctness or lifecycle safety for benchmark numbers;
- introduce distributed complexity before single-device execution is reliable;
- add a compiler whose value is not demonstrated on real workloads;
- promise support for hardware not qualified on physical devices.

## Acceptance Rule

A capability is complete only when its public contract, native implementation, correctness evidence, memory and ownership behavior, supported-platform matrix, benchmarks, diagnostics, documentation, and failure paths agree on one exact revision.
