# Training Engine V1 — acceptance gates

Training Engine V1 is complete only when the exact revision satisfies every gate below without semantic fallbacks or reduced quality thresholds.

## Autonomous execution

- CPU tensor computation used by the training proof does not require LibTorch.
- Reverse-mode differentiation is implemented by Tensora.
- A Dart program can execute forward, loss, backward, and optimizer step against a core-only Tensora native runtime.
- A deterministic regression task demonstrates decreasing loss and parameter convergence.

## Numerical correctness

- analytical gradients are checked against central finite differences for representative unary, binary, reduction, matrix, activation, and loss operations;
- repeated-parent accumulation such as `x + x` is correct;
- view transformations preserve gradient connectivity;
- saved-tensor mutation is detected before backward publishes partial leaf gradients.

## Optimizers

- SGD, Adam, and AdamW have deterministic contract tests;
- Adam bias correction is validated;
- AdamW uses decoupled weight decay;
- zero-gradient behavior is explicit;
- optimizer updates increment parameter alias versions.

## Tensor layout

- strides and storage offsets are first-class native tensor metadata;
- contiguous reshape is zero-copy;
- supported transpose is zero-copy;
- host copies and CPU kernels respect logical view order;
- non-contiguous reshape materializes only when required;
- backing storage outlives individual alias handles safely.

## Checkpoints

- model parameters round-trip exactly;
- malformed, truncated, trailing-data, and incompatible checkpoints fail safely;
- failed loads do not partially mutate a live module;
- repeated save/load does not leak storage or handles.

## Runtime safety

- invalid and stale handles fail deterministically;
- failure output handles are cleared;
- Debug and Release tests pass;
- ASan/UBSan passes for the core-only runtime;
- ThreadSanitizer is required for shared runtime state before release;
- live tensor/module/optimizer/storage counters return to baseline after stress tests.

## Cross-platform

- the core-only native suite passes on supported Linux, macOS, and Windows CI configurations;
- Dart formatting, analysis, and integration tests pass against the same semantic contracts.

## Accelerator qualification

CPU success is not evidence of GPU success. CUDA is qualified only by a physical NVIDIA runner that passes device discovery, allocation, transfer, computation, backward/update, and leak/stress proofs. Hardware-independent CUDA planning may be tested on CPU CI, but it cannot promote a device to qualified status.
