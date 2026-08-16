# Tensora Unified Release-Line Design

## Purpose

Tensora currently has substantial validated work split across stacked and diverged feature branches. The goal is to converge that work into one high-assurance release line, preserve existing ABI/lifetime guarantees, and then complete the framework through validated vertical slices rather than a single unreviewable P0-P7 diff.

The current implementation phase is **P0 only**. P1-P7 remain sequenced follow-on phases and must not be started by weakening P0 gates.

## Source of truth

The implementation line starts from `feature/training-engine-v1` because it already contains the newest autonomous CPU training work and is ahead of the earlier GPU/coverage lines.

Useful work from diverged branches is reintroduced selectively, with tests proving semantic compatibility before adoption. In particular, dtype/ABI-assurance work must be ported without regressing the newer autograd, alias, view, inference, and lifecycle contracts.

`main` remains untouched until a release candidate satisfies the applicable acceptance gates.

## Global engineering rules

1. Every capability is implemented as a vertical slice: Dart API -> C ABI -> native implementation -> correctness proof -> lifetime/error proof -> coverage -> platform CI -> runnable example/benchmark where applicable.
2. No silent CPU fallback for explicitly requested accelerators.
3. No production capability is claimed from a package/class name alone.
4. ABI changes must be versioned, tested by C consumers, and validated against malformed inputs.
5. Native ownership remains deterministic; finalizers are safety nets, not primary lifecycle management.
6. Coverage thresholds may not be lowered to make a phase pass.
7. New executable lines are not hidden behind broad coverage exclusions.
8. Hardware support is promoted only with real hardware/provider evidence on an exact revision.
9. Large-scale compiler and distributed work starts only after eager correctness is proven on representative workloads.
10. `main` is not updated until the release line is green for the scope being promoted.

## P0 - Unified High-Assurance Core

### Objective

Create one coherent development line that combines the newest training engine with the strongest dtype/ABI and coverage guarantees, then make the exact head green across the required hosted matrix.

### Deliverables

- Preserve autonomous CPU reverse-mode autograd and optimizer/checkpoint behavior.
- Finish tensor-view and alias semantics, including mutation-version correctness.
- Reintroduce dtype semantic metadata and stable ABI dtype codes without claiming unsupported native storage.
- Reintroduce hardened native-library/finalizer/error-boundary behavior from the dtype/ABI-assurance line where it remains applicable.
- Require merged Dart production coverage >= 99.9%.
- Require owned native production line coverage = 100.0% for the defined P0 owned-code set.
- Keep ASan, UBSan, ThreadSanitizer, C ABI fuzzing, malformed-input tests, lifecycle soaks, and fault-runtime tests passing.
- Validate Linux, macOS, and Windows hosted builds on the exact revision.
- Keep CUDA/XPU/HIP qualification separate from hosted CPU CI.

### P0 acceptance gate

P0 is complete only when:

- all ordinary hosted CI workflows required by P0 complete successfully on the same exact head;
- Dart production coverage is >= 99.9% with no threshold reduction;
- native owned-code merged line coverage is 100.0%;
- view/alias/autograd numerical and stale-mutation tests pass;
- C ABI malformed-input/fault-runtime tests pass;
- leak/lifecycle counters return to baseline in stress tests;
- compatibility documentation matches the implementation; and
- no supported behavior depends on silent fallback or placeholder implementations.

## P1 - Tensor Core Expansion

Implement real native storage and execution semantics for the approved dtype set, then add casting/promotion, broadcasting, scalar arithmetic, axis reductions, slicing/indexing, concat/stack, gather/scatter, and the tensor operations required by higher-level neural-network primitives.

No dtype is considered supported merely because it appears in Dart metadata; storage, ABI, kernels, host conversion, promotion, errors, autograd applicability, tests, and coverage must agree.

## P2 - Neural-Network and Training Framework

Generalize modules and optimizers from the current narrow training slice to arbitrary parameter collections and nested module composition. Add parameter registration/state dictionaries, composition primitives, normalization, embedding, dropout, GELU/SiLU/SwiGLU, softmax/log-softmax, attention primitives, schedulers, gradient clipping, deterministic initialization, AMP/mixed precision, and gradient scaling where the underlying dtype/device support is qualified.

## P3 - Data and Trainer

Expand `Dataset`/`DataLoader` into deterministic production data pipelines with sampling, shuffling, batching/collation, prefetching and isolate-based CPU preprocessing where appropriate. Implement a real Trainer with train/eval loops, validation, metrics, callbacks, checkpoint/resume, gradient accumulation, reproducible resume, and failure-safe state restoration.

The product proof for P3 is a small Transformer/GPT trained end-to-end without LibTorch as the primary training engine, including save/load/resume reproducibility.

## P4 - .tmodel, Edge and Flutter Runtime

Implement `.tmodel` V1 as a secure versioned deployment bundle with strict manifest validation, integrity hashes, compatibility declarations, resource limits, golden vectors, safe unpacking, and CLI `pack`, `inspect`, and `verify` commands.

Add an explicit edge model registry with verified activation and rollback. Build real Flutter integration for model assets, asynchronous execution, cancellation, lifecycle transitions, native binary packaging, memory-pressure handling, and non-blocking UI-isolate behavior.

## P5 - Vision, Text and Embeddings

Implement production vision preprocessing and live-camera integration: image tensors, resize/crop, color conversion, normalization, NMS, detector helpers, camera adapters, bounded frame queues, backpressure, and transfer diagnostics.

Implement tokenizer adapters, batching/padding/masks, embedding model abstractions, normalized embeddings, similarity primitives, and deterministic batch inference.

## P6 - Transformers and Local LLM Runtime

Implement Transformer blocks, causal attention, KV-cache ownership, per-session generation state, greedy/top-k/top-p/temperature sampling, stop sequences, streaming tokens, cancellation, quantized inference integration, and memory diagnostics.

The acceptance proof is a Flutter/Dart application that loads a supported local model, streams generation, cancels safely, disposes session state, and repeats the workflow without unbounded memory growth.

## P7 - Compiler, Forge, Kernel DSL and Advanced Runtime

After eager correctness is demonstrated, implement graph capture, Tensora IR, shape/type propagation, constant folding, dead-node elimination, fusion justified by benchmarks, memory planning, backend partitioning, and compiled execution.

Forge/kernel-DSL work may introduce custom CPU/GPU kernels where profiling shows material benefit. Mature vendor primitives may remain backend building blocks; P7 does not require recreating cuBLAS/cuDNN for ideological completeness.

Distributed training/inference and more advanced compilation are follow-on acceptance levels built on the same tensor/autograd/checkpoint/device contracts.

## Error handling and safety model

- All public Dart failures are structured Tensora exceptions with operation context where applicable.
- The C ABI returns stable status codes, validates pointers/lengths/ranks/capacities/handles before access, clears failed output handles, and contains C++ exceptions.
- Partial native-handle adoption is transactional: failures release newly created native resources before propagating errors.
- Model/bundle parsers never execute arbitrary code and enforce resource limits before allocation/unpacking.
- Explicit device/provider requests fail rather than falling back silently.

## Testing strategy

Each phase must use the strongest applicable combination of:

- deterministic unit tests;
- Dart -> FFI -> C ABI -> native integration tests;
- trusted-reference numerical comparisons and finite differences;
- lifecycle and leak stress tests;
- ASan/UBSan/TSan;
- C ABI fuzzing and adversarial fault runtimes;
- exact-head Linux/macOS/Windows hosted validation;
- physical hardware qualification for accelerators;
- benchmark baselines for performance-sensitive work;
- release-time coverage merging with hard thresholds.

## Release discipline

P0-P7 are not one merge. Each phase produces an independently reviewable, testable release-line increment. `main` is updated only from a revision whose claimed capability set is fully supported by tests, coverage, documentation, and hardware evidence where required.
