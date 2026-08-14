# Tensora Training Engine V1 Implementation Plan

## Goal

Deliver a self-contained CPU neural-network training path whose tensor math, reverse-mode differentiation, module execution, loss calculation, optimizer update, checkpointing, and public ABI are owned by Tensora and run with `TENSORA_WITH_TORCH=OFF`.

## Global constraints

- Preserve existing C ABI compatibility.
- Do not reduce any coverage, sanitizer, fuzz, lifetime, or CI threshold.
- No silent fallback to LibTorch for the native CPU training proof.
- Hardware support is claimed only after physical qualification.
- Every new numerical primitive requires reference-value and malformed-input tests.
- Every new differentiable primitive requires analytical-gradient and finite-difference tests.
- Every owning native object requires deterministic release and leak/stress coverage.
- Keep public Dart operations isolate-local and preserve existing handle lifetime rules.

## Workstream 1 — Autograd core

### Files

- Create `native/src/autograd/autograd.h`
- Create `native/src/autograd/autograd.cc`
- Modify `native/src/tensor/tensor.h`
- Modify `native/src/tensor/tensor.cc`
- Create `native/tests/autograd_test.cc`
- Modify `native/CMakeLists.txt`

### Deliverable

A backend-independent reverse-mode engine supports leaf gradients, graph recording, reverse topological traversal, accumulation, zeroing, and deterministic graph release.

### Test sequence

1. Add failing tests for leaf state and scalar backward.
2. Add failing tests for add and multiply VJPs.
3. Add failing tests for sum VJP.
4. Add failing tests for matrix multiplication VJPs.
5. Add failing tests for repeated-parent gradient accumulation.
6. Add failing tests for graph release and repeated backward policy.
7. Implement only the interfaces required to satisfy each test.
8. Run core Debug/Release plus ASan/UBSan and ThreadSanitizer matrices.

## Workstream 2 — Record Tensor operations

### Files

- Modify `native/src/c_api.cc`
- Modify `native/src/training/training_bridge_stub.cc` or replace it with a Tensora CPU implementation
- Modify `native/src/backends/cpu/cpu_backend.cc` only for numerical primitives, not graph policy
- Extend `native/tests/tensor_core_test.cc`
- Extend `native/tests/c_abi_test.c`

### Deliverable

`withRequiresGrad`, `requiresGrad`, `backward`, `grad`, add, multiply, sum, matmul, reshape, transpose and ReLU function through the public C ABI with no LibTorch requirement.

### Acceptance

The same public training calls used by Dart work in a core-only native build.

## Workstream 3 — Native activations and losses

### Files

- Create `native/src/training/cpu_training_ops.h`
- Create `native/src/training/cpu_training_ops.cc`
- Create/extend `native/tests/native_training_test.cc`

### Deliverable

- ReLU;
- sigmoid;
- tanh;
- MSE;
- cross entropy with an eventual integer-target public contract;
- numerically stable implementations;
- VJPs for every operation.

### Acceptance

Analytical gradients match central finite differences within defined float32 tolerances.

## Workstream 4 — Native Module and Parameter runtime

### Files

- Create `native/src/training/module_registry.h`
- Create `native/src/training/module_registry.cc`
- Create `native/src/training/linear_module.h`
- Create `native/src/training/linear_module.cc`
- Modify core training bridge implementation
- Extend native training tests

### Deliverable

`Linear` works without LibTorch, parameters are Tensora tensors requiring gradients, bias semantics are explicit, module train/eval state is retained, parameter enumeration owns safe independent handles.

### Acceptance

Forward values match a reference calculation and both weight/bias gradients pass finite-difference checks.

## Workstream 5 — Optimizer runtime

### Files

- Create `native/src/training/optimizer_registry.h`
- Create `native/src/training/optimizer_registry.cc`
- Create `native/src/training/sgd_optimizer.h`
- Create `native/src/training/sgd_optimizer.cc`
- Later create Adam/AdamW implementations once SGD proof is green

### Deliverable

SGD supports learning rate, momentum, and weight decay; `zeroGrad` clears parameter gradients; `step` performs an atomic validated update.

### Acceptance

Closed-form one-step tests match reference values, momentum state is deterministic, and invalid hyperparameters never mutate parameters.

## Workstream 6 — First autonomous training proof

### Files

- Create `native/tests/cpu_training_convergence_test.cc`
- Create `packages/tensora/integration_test/cpu_training_engine_test.dart`
- Extend CI workflow coverage

### Deliverable

A deterministic small regression/classification task is trained through the same public ABI used by Dart.

### Acceptance

- `TENSORA_WITH_TORCH=OFF`;
- loss decreases by a fixed required ratio;
- parameters change;
- repeated training is memory-stable;
- all handles are released;
- Dart and native proofs agree.

## Workstream 7 — Checkpoint/resume

### Files

- Create `native/src/training/checkpoint.h`
- Create `native/src/training/checkpoint.cc`
- Extend module and optimizer registry interfaces
- Add corruption and rollback tests

### Deliverable

Versioned deterministic checkpoint save/load for parameters and optimizer state.

### Acceptance

Corrupt/truncated/oversized/incompatible input never partially mutates live state; resumed training matches uninterrupted training within tolerance.

## Workstream 8 — Tensor Core 2.0 layout model

### Files

- Create `native/src/tensor/layout.h`
- Create `native/src/tensor/layout.cc`
- Modify Tensor to store strides and storage offset
- Update CPU kernels to consume logical indexing
- Add view/alias tests

### Deliverable

Contiguous strides, arbitrary views, transpose/permute, reshape views where legal, slicing, contiguous materialization, alias/version safety.

### Acceptance

Views do not copy storage, logical reads/writes are correct for non-contiguous layouts, and autograd returns gradients in logical source layout.

## Workstream 9 — Broadcasting and operator expansion

### Deliverable

Broadcasting, dimension reductions, indexing/gather/scatter, concatenation/stack/split, exp/log/sqrt/pow, stable softmax/log-softmax/logsumexp, comparisons and masks.

### Acceptance

Forward and backward property tests cover scalar through rank-N shapes, zero/one dimensions according to shape policy, malformed broadcasting, and non-contiguous inputs.

## Workstream 10 — DType storage architecture

### Deliverable

Typed storage and dispatch for float32/float64/int32/int64/bool first; FP16/BF16 added with explicit accumulation rules before mixed-precision training.

### Acceptance

No dtype is advertised unless creation, storage, copy, operation dispatch, error handling, serialization and tests are implemented.

## Workstream 11 — Training ecosystem

### Deliverable

Adam, AdamW, schedulers, gradient clipping/accumulation, Dataset, DataLoader, deterministic shuffling, bounded prefetch, Trainer, richer Module tree, state dictionaries.

### Acceptance

End-to-end examples require no manual native glue and reproduce fixed-seed results within documented tolerance.

## Workstream 12 — Transformer training proof

### Deliverable

Embedding, LayerNorm/RMSNorm, RoPE, causal attention, GELU/SiLU/SwiGLU, stable cross entropy, tokenizer/data integration and a small decoder-only Transformer.

### Acceptance

A small GPT is initialized, trained, checkpointed, resumed and used for generation using Tensora-owned training semantics.

## Workstream 13 — Compiler foundation

### Deliverable

Graph capture, Tensora IR, type/shape/layout propagation, constant folding, dead-node elimination, fusion interfaces, memory planning and eager-vs-compiled equivalence tests.

### Acceptance

Compiled execution must demonstrate measured benefit on at least one real training workload while preserving numerical contracts.

## Workstream 14 — CUDA architecture and qualification

### CPU-verifiable work now

- device/layout/precision planning;
- allocation request validation;
- stream/event state-machine tests;
- kernel launch descriptor validation;
- code-generation or kernel-selection deterministic tests;
- C ABI error semantics when no CUDA device exists.

### Physical qualification later

- allocation/transfer numerical proof;
- CUDA kernel forward/backward proof;
- stream ordering;
- mixed precision;
- memory/leak stress;
- throughput benchmarks.

The absence of a current GPU runner is never converted into a false passing hardware claim.

## Workstream 15 — Distributed and large-model runtime

### Deliverable

Sharding metadata, distributed tensors/autograd, NCCL collectives, data/tensor/pipeline/context/expert parallel interfaces, compute/communication overlap, sharded optimizer/checkpoints and topology-aware planning.

### Acceptance

Single-node multi-GPU proof precedes multi-node qualification. Every distributed mode has numerical-equivalence, failure, recovery, and scaling benchmarks.

## Completion definition

Training Engine V1 is complete only when:

1. CPU training works without LibTorch;
2. Tensor Core 2.0 and supported dtypes have complete semantics;
3. native autograd covers the supported operator set;
4. Module/loss/optimizer/checkpoint contracts are Tensora-owned;
5. an end-to-end Dart training program converges;
6. a small Transformer can be trained and resumed;
7. all applicable CI, sanitizer, fuzz, lifetime, compatibility and coverage gates are green on one exact revision;
8. any CUDA claims are backed by physical-device evidence rather than configuration-only CI.