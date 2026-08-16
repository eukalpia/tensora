# Tensora Training Engine V1 Design

## Objective

Tensora Training Engine V1 makes end-to-end neural-network training possible without requiring LibTorch for tensor execution, automatic differentiation, modules, losses, or optimizers on the supported CPU path.

The first release proof is intentionally strict:

```text
Dart model
  -> Tensora C ABI
  -> Tensora tensor runtime
  -> Tensora autograd
  -> Tensora module/loss/optimizer runtime
  -> CPU backend
  -> decreasing loss
```

LibTorch remains an optional compatibility backend during migration. It is not the semantic authority for V1.

## Design principles

1. Tensor semantics are backend-independent.
2. Autograd metadata belongs to Tensor values, not to one device backend.
3. Backward is reverse-mode VJP evaluation over an explicit dependency graph.
4. Graph ownership must not create permanent reference cycles.
5. Gradient accumulation is deterministic for a fixed execution order.
6. Training code must run with `TENSORA_WITH_TORCH=OFF`.
7. CPU correctness is the semantic reference for later CUDA and Metal implementations.
8. GPU support may not be claimed without physical-device validation.
9. No silent fallback may change dtype, device, shape, gradient, or optimizer semantics.
10. Every production execution path is covered by correctness, malformed-input, lifetime, and stress tests.

## Architecture

### Tensor value

A Tensor owns numerical storage and immutable logical metadata. V1 extends the value with autograd metadata while preserving existing handle ownership.

```text
Tensor
  storage
  shape
  dtype
  device
  device_index
  autograd_meta
```

`AutogradMeta` contains:

- `requires_grad`;
- accumulated gradient storage;
- optional backward node for non-leaf tensors;
- monotonically increasing version counter reserved for mutation safety.

The graph stores weak references to result values and strong references only to parent values required to evaluate a VJP. Releasing the user-visible result releases the graph when no descendant depends on it.

### Backward graph

Each differentiable operation records a node only when at least one input requires gradients.

```text
Tensor output
  -> GradNode
       -> parent[0]
       -> parent[1]
       -> saved forward values/metadata
```

A node exposes a single operation:

```cpp
Status Apply(const Tensor& upstream_gradient,
             std::vector<GradientContribution>* out);
```

A `GradientContribution` associates one parent tensor with one gradient tensor.

Backward performs:

1. scalar-root validation unless an explicit upstream gradient is supplied;
2. reverse topological traversal;
3. accumulation of all contributions targeting the same value;
4. leaf-gradient publication;
5. release of transient gradient buffers.

### V1 differentiable operations

The first executable slice covers:

- add;
- multiply;
- sum;
- matrix multiplication;
- ReLU;
- MSE loss;
- Linear forward constructed from Tensor operations.

Reshape and transpose preserve gradient connectivity. Sigmoid, tanh, cross entropy, broadcasting, slicing, and reductions by dimension follow immediately after the first CPU proof.

### Module runtime

V1 uses a native module registry independent from the tensor handle registry.

`LinearModule` owns:

- weight parameter `[in_features, out_features]`;
- optional bias `[out_features]`;
- training/evaluation state.

Parameters are ordinary Tensora tensors with `requires_grad=true`. Module parameter enumeration returns retained Tensor handles so callers own independent references.

### Optimizer runtime

The initial optimizer is SGD. An optimizer owns parameter references and state, not a module implementation.

SGD update semantics:

```text
parameter = parameter - learning_rate * (gradient + weight_decay * parameter)
```

Momentum state is allocated only when momentum is non-zero.

Adam and AdamW are added only after SGD has a complete correctness and checkpoint proof.

### Checkpoint format for V1

Training V1 uses a deterministic versioned binary state file for module parameters and optimizer state. It contains no executable code and validates:

- magic;
- format version;
- tensor count;
- dtype;
- rank/dimensions;
- payload sizes;
- finite-size limits;
- complete input consumption.

A corrupt or incompatible checkpoint fails without mutating live model state.

## Tensor Core 2.0 prerequisite

The long-term tensor representation is:

```text
Storage
  + Shape
  + Strides
  + storage_offset
  + DType
  + Device
```

Views share Storage. Contiguous operations may materialize only when required by a backend. The transition is incremental: V1 autograd is designed against logical Tensor operations so replacing copying reshape/transpose with true views does not change differentiation semantics.

Required Tensor Core 2.0 work:

- dtype storage dispatch;
- contiguous strides;
- arbitrary strides and offset;
- view-safe reshape;
- permute/transpose;
- broadcasting;
- slicing/indexing;
- reduction dimensions;
- contiguous materialization;
- alias/version safety.

## CUDA strategy without a current GPU runner

CUDA architecture is developed without pretending to validate hardware behavior.

CPU CI validates:

- CUDA C ABI parameter validation;
- device planner logic;
- dtype/layout/shape inference;
- launch descriptor construction;
- memory-planner invariants;
- kernel-source or kernel-selection contracts where deterministic;
- unsupported-device behavior.

A separate hardware workflow is required for claims about:

- allocation on a physical CUDA device;
- host/device transfer correctness;
- kernel numerical correctness;
- stream/event ordering;
- mixed precision;
- memory stability;
- throughput.

Until a GPU runner is available, physical CUDA jobs remain explicitly unqualified rather than silently skipped as proof.

## Compiler compatibility

Autograd nodes must expose enough operation metadata to later lower training graphs to Tensora IR. Eager V1 is therefore not a dead-end runtime.

The compiler boundary will consume:

- operation kind;
- tensor shape/dtype/device/layout;
- constant attributes;
- forward dependencies;
- backward/VJP definition;
- mutation/version information.

## Acceptance proofs

### Proof A — scalar/autograd correctness

Finite-difference checks for every V1 differentiable operation across deterministic random inputs.

### Proof B — MLP convergence

A two-layer network trained with `TENSORA_WITH_TORCH=OFF` must reduce loss by a deterministic required ratio on a fixed dataset.

### Proof C — lifetime

Repeated forward/backward/zero-grad/step cycles must return live Tensor/module/optimizer/storage counters to their expected steady state.

### Proof D — checkpoint resume

Training for N steps, saving, loading into a fresh model, and continuing must match uninterrupted training within float32 tolerance.

### Proof E — ABI

Dart exercises the complete CPU training loop through the public ABI with no internal C++ test-only entry point.

## Non-goals of the first vertical slice

The first CPU proof does not claim:

- distributed training;
- CUDA performance;
- FP16/BF16/FP8 training;
- full Transformer operator coverage;
- higher-order gradients;
- compiled graph execution.

These are subsequent V1 workstreams and are designed into the interfaces now rather than emulated with placeholders.