# Tensora NN V2 — Flutter-style Declarative Module System

Date: 2026-08-15
Status: approved design, pending implementation plan
Base: P0 exact green SHA `5aca4e5af508d51cc3f1905afabaec428edde3c2`

## 1. Goal

Tensora NN V2 replaces the current single-native-handle mental model with a composable, strongly typed, Flutter-inspired Dart API for building arbitrary neural-network module trees.

The public programming model must feel natural to a Flutter/Dart developer while remaining usable in pure Dart, CLI, server, desktop, and tests. `tensora_nn` must not depend on the Flutter SDK.

The primary success criterion is that users can define, inspect, move, train, save, restore, and optimize nested models without manually wiring native handles or registering parameters.

## 2. Public API direction

The canonical style is declarative construction with named parameters:

```dart
final model = Sequential(
  children: [
    Linear(inFeatures: 784, outFeatures: 512),
    GELU(),
    Linear(inFeatures: 512, outFeatures: 10),
  ],
);
```

Reusable architectures use a cached `Model.build()` contract:

```dart
class MLP extends Model {
  MLP({this.hiddenSize = 512});

  final int hiddenSize;

  @override
  Module build() {
    return Sequential(
      children: [
        Linear(inFeatures: 784, outFeatures: hiddenSize),
        GELU(),
        Linear(inFeatures: hiddenSize, outFeatures: 10),
      ],
    );
  }
}
```

`build()` is evaluated lazily once per `Model` instance and the materialized child tree is cached. Tensora must not rebuild trainable layers on every forward pass.

`Model` constructors are intentionally non-`const` in NN V2 because every instance owns mutable lifecycle/materialization state after first use. Flutter-like declarative syntax is a design influence, not a requirement to copy Flutter's immutable widget object model.

## 3. Core type hierarchy

The NN layer introduces these concepts:

- `Module`: executable composable node and lifecycle root.
- `Model`: declarative module whose `build()` returns a child `Module` tree.
- `Parameter`: a trainable tensor with stable identity and an owning registration path.
- `Buffer`: registered non-parameter model state.
- `Sequential`: ordered child container.
- `Identity`: no-op module.
- `ParameterGroup`: optimizer configuration over an explicit parameter collection.
- `StateDict`: deterministic named snapshot of parameters and persistent buffers.

`Module` is no longer defined as “one native module handle”. Leaf modules may own native handles, while composite modules may be pure Dart graph nodes containing other modules.

## 4. Module lifecycle and ownership

### 4.1 Materialization

A `Model` starts unmaterialized. The first operation that needs its tree (`call`, `parameters`, `stateDict`, `train`, `eval`, `to`, diagnostics) materializes `build()` exactly once.

Materialization is atomic. If `build()` or child validation fails, the model remains unmaterialized and no partially owned child tree is retained.

After successful materialization the root child identity is fixed for the lifetime of the `Model`; calling `build()` again is not part of public semantics.

### 4.2 Child ownership

A module tree has deterministic ownership semantics:

- a child may be registered under one owning path;
- shared parameter identity is allowed and deduplicated;
- shared module instances may be referenced only when doing so does not create ambiguous disposal ownership;
- cycles are rejected before execution with a typed Tensora error;
- parent disposal recursively disposes owned children exactly once;
- repeated `dispose()` is idempotent.

Composite constructors defensively copy child collections into an immutable internal order so later mutations of the caller's `List` cannot rewrite a materialized architecture.

### 4.3 Recursive mode and device propagation

`train()`, `eval()`, and `to(Device)` recurse through the materialized tree.

Mode propagation must preserve leaf-specific behavior such as dropout and normalization state.

Device moves are all-or-error at the public API level. Tensora must not silently leave part of a model on another device.

Where the current native backend cannot provide rollback for a partially completed multi-parameter device move, NN V2 must preflight all children/parameters before mutating any state; it must not advertise transactional device movement without that proof.

## 5. Parameter and buffer registration

Registration is explicit in the framework implementation but automatic for the user.

No reflection, mirrors, source generation, annotations, or field scanning are required for correctness.

Each built-in module registers its own children/parameters/buffers through internal typed registration primitives. Composite containers register children when constructed.

Public traversal APIs:

```dart
Iterable<Parameter> get parameters;
Iterable<NamedParameter> get namedParameters;
Iterable<Buffer> get buffers;
Iterable<NamedBuffer> get namedBuffers;
Iterable<Module> get modules;
Iterable<NamedModule> get namedModules;
```

Traversal returns deterministic immutable snapshots or snapshot-backed iterables; mutating the module tree during iteration is not supported.

Paths are deterministic and stable for an unchanged architecture, for example:

```text
0.weight
0.bias
2.weight
2.bias
```

For a reusable model, paths include semantic parent names where provided.

Shared parameters are yielded once by default using stable identity deduplication. An expert traversal option may expose aliases later, but alias expansion is not required for NN V2.

## 6. `Parameter`

`Parameter` wraps a native-backed Tensor identity intended for optimization.

Required behavior:

- exposes tensor metadata and gradient access without copying values into Dart;
- has stable identity independent of traversal order;
- can be frozen/unfrozen (`requiresGrad` semantic control);
- participates in state dict serialization;
- cannot outlive invalid ownership in a way that creates a dangling native handle;
- parameter views returned to users are independently safe wrappers or stable managed references according to the core ownership model.

Parameter collection must support arbitrary model sizes; it must not rely on the current native optimizer contract that takes one `Module` handle.

Stable parameter identity is native-handle/storage identity, not Dart object identity, so independently owned wrappers for the same underlying parameter deduplicate correctly.

## 7. Buffers

`Buffer` represents registered non-trainable state, including future running statistics and masks.

Buffers have:

- a deterministic name;
- device propagation;
- optional persistence in `StateDict`;
- no optimizer membership by default.

## 8. Sequential and model composition

Canonical container:

```dart
Sequential(
  children: [
    Linear(inFeatures: 64, outFeatures: 128),
    ReLU(),
    Linear(inFeatures: 128, outFeatures: 10),
  ],
)
```

`Sequential` validates non-disposed children, freezes their order at construction, and owns their ordered execution.

Future residual/branching modules must be implementable without changing the core registration model. NN V2 therefore must not bake “a model is a linear list” into native ABI or optimizer design.

A `Residual(child: ...)` module may be added in this stage only if required by acceptance examples; the core composition design must already support it.

## 9. Built-in layers and activations in this stage

Required public modules:

- `Linear`
- `Identity`
- `Sequential`
- `ReLU`
- `Sigmoid`
- `Tanh`
- `GELU`
- `SiLU`
- `SwiGLU`

Constructors use named arguments when there is more than one domain-significant numeric argument:

```dart
Linear(
  inFeatures: 784,
  outFeatures: 512,
  bias: true,
)
```

The old positional `Linear(int, int)` constructor must not remain the canonical documented API. If compatibility is retained temporarily, it must be clearly deprecated rather than silently diverging into two competing styles.

### 9.1 Exact activation semantics

`ReLU`, `Sigmoid`, and `Tanh` preserve their existing numerical semantics.

`SiLU(x)` is defined as `x * sigmoid(x)` and must expose the analytically correct autograd rule without materializing host values.

`GELU` defaults to the exact formulation `0.5 * x * (1 + erf(x / sqrt(2)))`. NN V2 does not add an approximate/tanh mode yet; that can be introduced later as an explicit named option without changing the default.

`SwiGLU` is a shape-changing gated activation. It requires the final input dimension to be positive and even, splits that final dimension into equal halves `a` and `b`, and returns `silu(a) * b`. Its output has the same leading dimensions and half the final dimension. Invalid/odd final dimensions throw a typed shape error in Release builds.

### 9.2 Native activation requirement

`GELU`, `SiLU`, and `SwiGLU` must execute through native tensor/autograd primitives. Implementations that call `Tensor.toList()`, compute in Dart, and reconstruct a Tensor are forbidden.

Each new primitive requires:

- native forward implementation;
- autograd rule;
- C ABI surface where needed;
- Dart binding/runtime method;
- CPU numerical reference tests;
- finite-difference gradient validation;
- device/backend parity policy tests;
- owned-source line coverage preservation.

## 10. Forward execution

Leaf modules may continue to use native module handles internally.

Composite module execution is orchestrated by the Dart NN layer unless/until the compiler/IR stage captures it. This keeps NN V2 independent from the future compiler and avoids prematurely freezing graph ABI.

Forward execution must:

- reject disposed inputs/modules;
- preserve autograd graph connectivity;
- never perform host copies as an implementation convenience;
- return native-backed tensors;
- retain explicit device mismatch errors with no silent CPU fallback.

Intermediate tensors created by a composite forward are owned by the operation chain and deterministically released when no longer needed, except values still referenced by autograd saved-tensor state. The implementation must not dispose a wrapper in a way that invalidates native storage retained by the graph.

## 11. Optimizer redesign

Current optimizers are module-bound. NN V2 changes the primary API to parameter collections:

```dart
final optimizer = AdamW(
  parameters: model.parameters,
  learningRate: 3e-4,
  weightDecay: 0.01,
);
```

Parameter groups:

```dart
final optimizer = AdamW.groups(
  groups: [
    ParameterGroup(
      parameters: model.parameters,
      learningRate: 3e-4,
      weightDecay: 0.01,
    ),
  ],
);
```

Required semantics:

- arbitrary parameter count;
- deterministic parameter ordering;
- duplicate parameter detection across groups by native parameter identity;
- empty collections are rejected;
- frozen parameters may be present but are skipped deterministically rather than becoming an error solely for being frozen;
- disposed/invalid parameters are rejected;
- group-specific hyperparameters;
- no hidden module ownership transfer;
- optimizer holds safe references to parameter identities for its lifetime;
- `zeroGrad`, `step`, and disposal remain deterministic;
- SGD, Adam, and AdamW all use the generalized collection model.

The native optimizer ABI must be extended to accept parameter collections/parameter groups rather than requiring a single module handle. Backward compatibility with the old module-bound native entry points may be retained internally during migration, but the new Dart API must not depend on it.

Parameter-group validation is atomic: all identities and hyperparameters are validated before a native optimizer object is published.

## 12. Loss modules

Keep static `Losses` compatibility temporarily, but add object-style APIs:

```dart
final criterion = CrossEntropyLoss();
final loss = criterion(logits, target);
```

Required:

- `MSELoss`
- `CrossEntropyLoss`

The object wrappers must remain zero-copy and call the existing/native loss implementation.

NN V2 does not change CrossEntropy's target representation; it continues to use the current equal-shaped one-hot float32 target contract until the tensor/dtype stage adds integer class targets.

## 13. State dict

Every materialized module exposes a deterministic state snapshot:

```dart
final state = model.stateDict();
final result = model.loadStateDict(state, strict: true);
```

State dict requirements:

- stable names;
- parameters plus persistent buffers;
- immutable map-like public view;
- no accidental host copy merely to enumerate state;
- snapshot entries own safe tensor references so the snapshot remains valid for its documented lifetime;
- strict mode reports missing keys, unexpected keys, shape mismatches, dtype mismatches, and device policy errors structurally;
- non-strict mode still returns a typed result describing missing/unexpected keys;
- load is transactional: all entries are validated before mutation and a validation failure must not partially mutate model state;
- shared parameters are restored once and aliases remain consistent.

`loadStateDict` returns a typed `StateLoadResult` in both strict and non-strict modes. Strict mode throws only for invalid/incompatible state after constructing structured mismatch context; successful strict load returns an empty-success result.

NN V2 state dict is an in-memory model-state abstraction. `.tmodel` packaging remains a later stage and is not redefined here.

## 14. Diagnostics and Flutter-like discoverability

`Module.toString()` and tree diagnostics must make architectures readable without debugging native handles.

Example:

```text
MLP
└── Sequential
    ├── 0: Linear(inFeatures: 784, outFeatures: 512, bias: true)
    ├── 1: GELU()
    └── 2: Linear(inFeatures: 512, outFeatures: 10, bias: true)
```

Required diagnostics:

- compact single-line `toString()` for leaf modules;
- indented tree rendering for composites;
- parameter count summary;
- train/eval state where relevant;
- device summary without forcing expensive device synchronization.

No provider implementation details belong in the normal tree representation.

Diagnostics must never materialize host tensor values and must not mutate/materialize a `Model` merely because an exception formatter touches it; explicit tree diagnostics may materialize because inspecting the architecture is itself a materializing operation.

## 15. Error handling

Use typed Tensora exceptions, not asserts, for public contract violations in Release builds.

NN V2 must structurally detect:

- module cycles;
- disposed child registration;
- duplicate/ambiguous ownership;
- duplicate optimizer parameter membership;
- incompatible state dict entries;
- forward use after disposal;
- invalid device propagation;
- invalid constructor dimensions;
- invalid SwiGLU shape;
- optimizer creation with an empty parameter set.

No silent fallback is introduced.

## 16. Backward compatibility

P0 behavior and ABI guarantees remain protected.

Migration policy:

- old static `Losses` remains available during NN V2;
- old module-bound optimizer constructors may remain deprecated for one development cycle if needed to avoid breaking existing acceptance tests, but all new documentation/tests use parameter collections;
- old positional `Linear` may receive a deprecated named compatibility factory only if Dart language constraints allow a clean unambiguous path; the canonical unnamed constructor becomes named-argument based;
- because Tensora is pre-1.0, if Dart cannot express both APIs without ambiguity, NN V2 makes the intentional breaking public Dart change and migration tests/documentation are updated in the same PR;
- native legacy entry points are not removed until their callers are migrated and contract tests prove the replacement.

API clarity takes priority over preserving accidental early syntax.

## 17. Package boundaries

`tensora` remains the low-level tensor/runtime/training foundation.

`tensora_nn` becomes the canonical high-level neural-network composition package and owns `Module`, `Model`, containers, activation modules, state-dict abstractions, `Parameter`, and `Buffer` at the public package layer.

Low-level native-handle adapters needed by `tensora_nn` remain internal capabilities exposed by `tensora` through non-user-facing bridge APIs, not public raw handles.

`tensora_optim` exports generalized optimizer APIs and depends on the parameter abstraction without depending on Flutter.

`tensora_flutter` remains an adapter package and does not become a dependency of NN/core packages.

Flutter widget classes never appear in NN/core signatures.

## 18. Testing strategy

Every public behavior is developed test-first.

### Dart tests

- lazy single materialization of `Model.build()`;
- materialization rollback on build failure;
- immutable child-order capture;
- deterministic traversal names/order;
- nested `Sequential` execution;
- recursive train/eval/to/dispose;
- cycle rejection;
- shared parameter deduplication by native identity;
- state dict snapshot lifetime;
- state dict strict/non-strict results;
- transactional state loading;
- diagnostics rendering;
- optimizer parameter groups and duplicate rejection;
- frozen-parameter behavior;
- lifecycle/finalizer fallback behavior;
- compatibility APIs where retained.

### Native tests

- generalized optimizer parameter collection ownership;
- SGD/Adam/AdamW arbitrary collections;
- new activation numerical tests;
- finite-difference gradients for GELU/SiLU/SwiGLU;
- SwiGLU even/odd/zero-rank shape contracts;
- invalid/null/overflow/adversarial ABI inputs;
- allocation/rollback paths;
- device mismatch behavior;
- leak/handle-count invariants.

### Cross-platform

The exact candidate SHA must preserve green qualification for:

- Linux Debug/Release;
- Windows Debug/Release;
- macOS Debug/Release;
- ASan + UBSan;
- ThreadSanitizer;
- C ABI fuzz;
- Dart minimum-version compatibility;
- real Apple MPS training acceptance where the existing runner supports it;
- existing ONNX/inference contracts.

## 19. Coverage and quality gates

NN V2 must not weaken P0 gates.

- native owned-source line coverage remains 100%;
- existing Dart high-assurance coverage threshold remains unchanged;
- new `tensora_nn` and `tensora_optim` production code receives a >=99.9% line target, with a 100% target for deterministic pure-Dart registration/traversal/state-dict code;
- no source exclusion is added merely to satisfy coverage;
- generated/vendor code remains the only legitimate exclusion class;
- formatting/analyze warnings are errors in CI where current policy requires them.

Branch coverage remains informational unless a separate release policy explicitly promotes it to a gate; NN V2 does not silently redefine the P0 line-coverage contract.

## 20. Acceptance examples

### 20.1 MLP

A two-layer MLP can be created entirely declaratively, trained with AdamW over `model.parameters`, snapshotted/restored with state dict, moved between supported devices, and disposed recursively.

### 20.2 Nested model

A custom `Model` may return nested `Sequential` modules and exposes stable named parameter paths without manual registration.

### 20.3 Shared parameter

Two computation paths may intentionally reference one shared parameter. Traversal and optimizer creation deduplicate it by native identity, and state restoration preserves sharing.

### 20.4 Training proof

A small multi-layer network must show loss reduction on a deterministic synthetic dataset using the generalized optimizer path and without relying on LibTorch as the CPU training implementation.

### 20.5 Activation proof

GELU, SiLU, and SwiGLU forward outputs match high-precision references within declared tolerances and their gradients pass finite-difference validation on CPU. Existing real-MPS qualification must remain green after adding their backend policy paths.

### 20.6 No-regression proof

All P0 exact-SHA gates remain green after NN V2 changes.

## 21. Explicit non-goals

This stage does not attempt to complete:

- full dtype storage/promotion matrix;
- arbitrary broadcasting/indexing tensor API;
- LayerNorm/Embedding/attention/Transformer blocks;
- AMP/GradScaler;
- DataLoader/Trainer;
- `.tmodel` packaging;
- graph capture/compiler/IR;
- distributed training.

The architecture must make those stages possible, but NN V2 does not pretend they are complete.

## 22. Definition of done

NN V2 is complete only when all of the following are true on one exact SHA:

1. Flutter-style declarative public API is implemented and documented.
2. `Model.build()` is lazy, cached, atomic, and lifecycle-safe.
3. Nested modules automatically expose deterministic parameters/buffers/modules.
4. Generalized SGD/Adam/AdamW optimize arbitrary parameter collections and groups.
5. `StateDict` snapshot/load is deterministic, lifetime-safe, and transactional on validation failure.
6. ReLU/Sigmoid/Tanh/GELU/SiLU/SwiGLU module APIs work with native autograd-backed execution.
7. MSE and CrossEntropy object loss APIs are available.
8. Module tree diagnostics are readable and stable.
9. Shared parameters, cycles, duplicate groups, frozen parameters, disposal, and failure rollback are tested.
10. Native line coverage remains 100%.
11. New deterministic Dart composition code reaches the declared high-assurance coverage target.
12. Linux/Windows/macOS, sanitizers, fuzz, Dart, training, inference, and real MPS qualification gates are green.
13. Existing P0 behavior is not weakened and no silent fallback is introduced.
