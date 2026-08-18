# Tensora Core Superiority + Compatibility Design

## Status

Approved strategic direction for Tensora after the P0 and NN V2 foundations.

This document turns the project goal from “a Dart wrapper around mature ML runtimes” into a measurable architecture for a first-class tensor, training, compilation, and deployment system. Tensora owns the performance-critical core while preserving broad model and operator compatibility through explicit, inspectable fallback paths that are progressively replaced.

## Mission

Tensora must become the highest-efficiency end-to-end ML system for Dart and Flutter while remaining credible for server training and inference.

The project does not claim superiority from API resemblance or feature counts. A capability is superior only when reproducible qualification proves at least one material advantage without unacceptable regressions in the others:

- lower latency or higher throughput;
- lower peak or steady-state memory;
- shorter compilation or startup time;
- stronger determinism, safety, or lifecycle guarantees;
- simpler deployment or integration;
- equal or better numerical correctness.

Unsupported behavior must fail explicitly. No benchmark, compatibility adapter, or device transfer may silently fall back to a slower or different backend.

## Strategic choice

Tensora uses **core superiority + compatibility**:

1. Tensora owns Tensor semantics, dtype rules, shape rules, views, aliasing, storage, autograd, IR, compilation, scheduling, memory planning, diagnostics, and runtime policy.
2. Tensora-native optimized kernels are preferred for high-value workloads.
3. Generated kernels and mature vendor libraries extend performance coverage.
4. ONNX Runtime, CoreML, LibTorch, and later interop providers cover the long operator/model tail when a native path is not yet qualified.
5. Every fallback is visible in diagnostics and benchmark reports.
6. Hot fallbacks are progressively promoted into generated or native Tensora implementations.

Compatibility is therefore a migration layer, not the architectural center.

## Non-goals

- Reimplement thousands of low-value operators before Tensora is useful.
- Replace cuBLAS, cuDNN, oneDNN, or platform NPUs for ideological reasons.
- Report wins from mismatched precision, warmup, correctness, hardware, or synchronization.
- Add a backend that silently copies tensors through Dart collections.
- Accept lower safety or determinism merely to make a benchmark green.
- Begin distributed or compiler complexity before single-device semantics are exact and measurable.

## System architecture

```text
Dart / Flutter API
        |
        +--------------------+
        |                    |
   eager execution      graph capture/export
        |                    |
        +----------+---------+
                   |
             Tensora Program
                   |
       canonical Tensor/Shape/DType IR
                   |
      functionalization and alias analysis
                   |
       autograd transform / joint graph
                   |
 shape, dtype, device and effect inference
                   |
 optimization, fusion and layout selection
                   |
 memory planning and recomputation policy
                   |
 device scheduling and partitioning
                   |
  +----------------+-------------------+
  |                |                   |
 CPU backend    CUDA backend      Metal/NPU backends
  |                |                   |
 native / generated / vendor / compatibility kernels
                   |
          deterministic runtime
```

Eager and compiled execution must share the same public semantics, error model, dtype promotion, alias/version rules, and device policy.

## Operator resolution tiers

Each operator implementation is assigned one explicit tier.

### Tier 0 — Tensora native optimized

Hand-tuned or deeply specialized implementation owned by Tensora. Used for critical primitives and fused workload kernels.

### Tier 1 — Tensora generated

Generated from Tensora Kernel IR for the target shape, dtype, layout, and hardware. The compilation cache includes a hardware and compiler fingerprint.

### Tier 2 — qualified vendor backend

Dispatch to libraries such as BLAS, oneDNN, cuBLAS, cuDNN, platform Metal libraries, or NPU providers through a strict adapter.

### Tier 3 — portable Tensora composite

Correct implementation composed from already-qualified Tensora primitives. This prioritizes compatibility and reference correctness over maximum speed.

### Tier 4 — explicit interop fallback

Execution through an approved compatibility provider such as ONNX Runtime, CoreML, or LibTorch. The result records provider, copied bytes, device transitions, synchronization, and fallback reason.

Resolution rules:

1. Semantics are validated before backend selection.
2. A lower-numbered qualified tier wins unless policy explicitly requests another path.
3. No tier may change dtype, layout, device, determinism, gradients, or error behavior silently.
4. A fallback crossing device or process boundaries requires an inspectable cost estimate.
5. Strict mode may forbid selected tiers entirely.

## Tensor and storage core

The Tensor Core owns:

- stable dtype codes and deterministic promotion;
- checked shape, stride, offset, numel, and byte-size arithmetic;
- immutable metadata and explicit mutation epochs;
- view identity, alias tracking, and saved-tensor validation;
- per-device storage with deterministic lifetime;
- typed host import/export without hidden conversion;
- scalar and broadcasting semantics;
- reductions, indexing, composition, and copy rules;
- exact device transfer contracts.

The core must keep payloads native during tensor-to-tensor operations. Dart materialization is always explicit.

## Autograd

Autograd has two compatible modes:

### Eager reverse mode

- immediate graph construction;
- gradient accumulation;
- saved-tensor mutation validation;
- deterministic graph release;
- explicit `noGrad`, detach, and inference behavior;
- custom-gradient boundary with strict lifetime rules.

### IR differentiation

- transform forward IR into backward IR;
- optimize forward and backward jointly;
- choose saved activations versus recomputation;
- perform broadcast-gradient reductions explicitly;
- plan gradient buffer reuse;
- support checkpointing as a compiler decision under a memory budget.

Integer and boolean tensors never acquire gradients. Additional floating dtypes become differentiable only after numerical and device qualification.

## Compiler

The compiler is not an optional wrapper around eager execution. It is a first-class path with the following stages:

1. capture/export into stable program IR;
2. functionalize mutations while preserving observable semantics;
3. infer shapes, dtypes, devices, layouts, effects, and aliases;
4. eliminate dead work and fold constants;
5. partition by device and implementation tier;
6. fuse profitable regions;
7. select layouts and kernels;
8. jointly plan memory, recomputation, and transfers;
9. generate or lower executable artifacts;
10. cache by program, guard, hardware, backend, and compiler fingerprints.

Dynamic shapes use explicit guards and bounded specialization. Guard failure may recompile or return a structured policy error; it never silently executes a semantically different program.

## Memory superiority

Every executable plan exposes:

- parameter bytes;
- optimizer-state bytes;
- persistent buffer bytes;
- activation and gradient peak;
- temporary workspace peak;
- host/device transfer bytes;
- allocator fragmentation estimate;
- recomputed operations;
- offloaded tensors.

Given a memory budget, the planner may choose compatible combinations of:

- buffer reuse;
- activation checkpointing;
- selective recomputation;
- mixed precision;
- fused optimizer state;
- parameter/optimizer sharding;
- pinned-host or storage offload;
- layout changes;
- microbatching.

The chosen plan and rejected alternatives are inspectable. Out-of-memory errors include the planned and observed allocation state rather than a generic failure.

## Backend contract

A backend declares capabilities instead of being trusted by convention:

- supported devices and hardware generations;
- dtypes, layouts, ranks, and dynamic-shape limits;
- deterministic variants;
- synchronization and stream/event behavior;
- forward and backward coverage;
- workspace requirements;
- serialization and cache compatibility;
- numerical tolerances;
- known unsupported combinations.

Backend selection is fail-before-publish and transactional. Device fallback is forbidden unless the caller opts into a named policy.

## CPU direction

The CPU backend progresses from a correct portable reference to a measured superbackend:

- contiguous and strided vectorized loops;
- architecture dispatch for x86-64 and ARM64;
- SIMD-specialized elementwise and reduction kernels;
- BLAS/oneDNN integration where superior;
- cache-aware tiling;
- thread-pool scheduling without oversubscription;
- NUMA awareness for server workloads;
- generated fusion for elementwise/reduction regions;
- quantized kernels for deployment workloads.

Every specialization retains the portable path as a correctness oracle.

## CUDA direction

CUDA becomes a real Tensora device, not a synonym for LibTorch ownership:

- allocator and memory accounting;
- streams, events, synchronization, and error propagation;
- peer transfers and pinned host memory;
- cuBLAS/cuBLASLt and cuDNN adapters;
- generated elementwise, reduction, normalization, and fused kernels;
- Tensor Core layout and precision selection;
- graph capture where profitable;
- deterministic alternatives where available;
- physical hardware qualification with CPU fallback forbidden.

LibTorch may remain a compatibility provider, but it cannot define Tensora Tensor semantics.

## Apple and edge direction

Tensora treats Flutter deployment as a core advantage:

- identical Dart model/runtime concepts across desktop and mobile;
- Metal/CoreML/NPU adapters behind the backend contract;
- zero-copy or bounded-copy camera/image integration where platform APIs permit;
- asynchronous execution off the UI isolate;
- bounded realtime queues and backpressure;
- lifecycle-safe cancellation and disposal;
- compact AOT runtime and secure model bundles;
- no application-level JNI/Swift/C++ glue for ordinary supported workflows.

## Mixed precision and quantization

Precision policy is compiler-visible.

Training support grows through BF16/FP16, autocast, loss scaling, and later FP8 only after device-specific correctness gates. Deployment supports per-tensor, per-channel, and mixed quantization plans with calibration and accuracy validation.

A deployment request may specify latency, memory, energy, and minimum-accuracy constraints. The planner searches valid configurations and records the selected Pareto point.

## Distributed architecture

Distributed execution is built on global tensor placement rather than unrelated wrappers.

- logical device meshes;
- replicated, sharded, and partial placements;
- data, tensor, sequence, expert, and pipeline parallel composition;
- parameter, gradient, and optimizer-state sharding;
- communication represented in IR;
- overlap planned explicitly;
- checkpoint format independent of current world size;
- deterministic failure and recovery boundaries.

Auto-parallel search is introduced only after manually specified placements are correct and measurable.

## Compatibility and model ingestion

Compatibility paths include:

- ONNX sessions and graph import where semantics can be proven;
- CoreML and platform-provider execution;
- optional LibTorch interop for migration and unsupported training operators;
- safetensors and standard tokenizer/metadata formats;
- explicit conversion reports listing unsupported or changed semantics.

Imported models must pass golden-vector validation before promotion. Arbitrary custom code and automatic extension loading are forbidden by default.

## Observability

Each execution can expose a structured trace containing:

- operator and fused-region timing;
- selected implementation tier;
- compile and cache time;
- memory plan and observed peak;
- allocations, transfers, synchronizations, and copies;
- fallback reasons;
- graph breaks and specialization guards;
- dropped realtime frames;
- numerical and determinism mode.

Profiling must be usable from Dart and Flutter without parsing native console logs.

## Superiority benchmark contract

No comparison is published unless it uses:

- identical hardware and power policy;
- equivalent model, batch, shapes, precision, optimizer, and correctness target;
- explicit synchronization around timed accelerator work;
- documented warmup and sample counts;
- median, p95, variance, throughput, and peak memory;
- startup and compile time reported separately;
- environment and dependency fingerprints;
- raw machine-readable results;
- trusted output/gradient comparison;
- no hidden fallback.

Initial benchmark families:

- tensor microbenchmarks across dtype/layout/shape classes;
- MLP eager and compiled training;
- CNN training and inference;
- Transformer encoder and decoder blocks;
- attention forward/backward;
- embedding-heavy workloads;
- CPU and accelerator inference;
- compilation cold/warm cache;
- constrained-memory training;
- Flutter image and token-streaming deployment;
- multi-device scaling when available.

## Acceptance gates

A phase is complete only when one exact revision passes:

1. public contract and compatibility documentation;
2. reference numerical tests and gradient checks;
3. negative, malformed, overflow, alias, and lifetime tests;
4. supported platform and physical-device qualification;
5. sanitizers, races, fuzzing, and lifecycle soaks;
6. hard production coverage thresholds;
7. deterministic benchmark artifacts;
8. no silent fallback or placeholder production path;
9. no benchmark regression beyond the phase budget;
10. an explicit list of unsupported behavior.

## Delivery sequence

The strategic sequence is:

1. finish Tensor Core V2 and typed execution;
2. broaden operator and reduction coverage;
3. generalize eager autograd;
4. complete the high-value NN module set;
5. establish stable Tensor/Autograd IR;
6. add capture, compilation, fusion, and memory planning;
7. build CPU, CUDA, and Apple superbackends;
8. add mixed precision, quantization, and data pipelines;
9. add distributed tensor placement and auto-parallel planning;
10. specialize edge, vision, embeddings, and local-language-model workflows.

Each step remains a vertical slice. Parallel half-implementations are not promoted.

## Definition of strategic success

Tensora has achieved the intended position when representative workloads demonstrate, with reproducible artifacts, that:

- training and inference performance is competitive or superior on supported devices;
- peak memory is materially lower for constrained workloads;
- compilation and startup are predictable and inspectable;
- deployment from Dart/Flutter requires less glue and fewer format transitions;
- deterministic ownership and failure behavior remain stronger than compatibility providers;
- broad models continue to run through explicit fallbacks while hot paths migrate into Tensora-owned execution.

The project wins by controlling semantics and optimization end to end, not by pretending every operator was native on day one.
