# Training Engine V1 — verification state

This document records the current engineering state of the autonomous Tensora training vertical slice.

## Implemented

- native reverse-mode autograd independent of LibTorch;
- gradient accumulation and saved-version validation;
- CPU `add`, `multiply`, `sum`, `matmul`, reshape/transpose gradient propagation;
- ReLU, sigmoid, tanh, MSE, and cross-entropy backward paths;
- native `Linear` parameters and bias handling;
- native SGD, Adam, and AdamW updates;
- deterministic seeding;
- module checkpoint save/load with compatibility and corruption checks;
- core-only Dart training path with `TENSORA_WITH_TORCH=OFF`;
- handle-registry failure output clearing;
- tensor layout state for strides, storage offsets, shared aliases, and logical-order host copies;
- zero-copy contiguous reshape and rank-2 transpose design;
- stride-aware CPU elementwise, reduction, and matrix operations.

## P0 unified-line integration

The `work/unified-p0-high-assurance-20260815` line now carries the public dtype semantic table and stable ABI dtype codes while intentionally keeping native tensor allocation float32-only. Non-float32 creation is required to fail explicitly in both Debug and Release. This is a P0 compatibility contract, not a claim that additional dtype storage has been implemented; real multi-dtype storage remains P1 work.

## Verified gates reached before the tensor-view slice

The core-only training proof passed in Debug, Release, and ASan/UBSan configurations. The Dart training integration suite also passed against a native runtime built without LibTorch. Finite-difference checks validated analytical gradients for the implemented operations. Adam, AdamW, convergence, checkpoint restore, and checkpoint failure rollback proofs passed in the same core-only line.

## Current gate

Tensor views are under exact-head validation. The required contracts are:

- contiguous reshape and transpose share backing storage where their layouts permit it;
- non-contiguous operations consume logical tensor order rather than raw backing-storage order;
- reshaping a non-contiguous view materializes exactly when required;
- autograd remains numerically correct through views;
- all aliases observe one mutation version;
- stale saved aliases cause backward to fail before any leaf gradient is published.

The branch remains a development line until those contracts and the cross-platform native matrix are green on the exact revision.

## Hardware qualification

CUDA capability is not inferred from CPU emulation. Physical CUDA qualification remains a separate hardware gate that requires a runner with a visible NVIDIA device and successful device-side execution. CPU CI covers hardware-independent contracts only.
