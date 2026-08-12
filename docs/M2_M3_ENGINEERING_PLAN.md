# Milestones 2 and 3 Engineering Plan

This document defines the implementation contract for the next two Tensora vertical slices:

- Milestone 2: native training with a mature tensor/autograd backend and CUDA support.
- Milestone 3: portable ONNX inference with reusable sessions and provider diagnostics.

The implementation extends the Milestone 1 ABI and ownership rules instead of replacing them. Existing CPU-only builds must continue to compile and pass without optional training or inference dependencies.

## 1. Dependency boundary

Tensora keeps one public Dart API and one stable C ABI. Third-party native runtimes remain implementation details.

Milestone 2 uses LibTorch/ATen for autograd, mature tensor kernels, module state, optimizers, and CUDA execution. Milestone 3 uses ONNX Runtime for ONNX graph loading and execution.

Both integrations are optional at native build time:

```text
TENSORA_WITH_TORCH=OFF       # default
TENSORA_WITH_ONNXRUNTIME=OFF # default
```

A core CPU-only build therefore preserves Milestone 1 portability and dependency size. Feature-enabled builds fail configuration clearly when a requested dependency is unavailable; there is no silent fallback that changes semantics.

Integration validation pins a tested baseline of PyTorch 2.13.0 and ONNX Runtime 1.29.0. These versions are validation baselines rather than ABI promises.

## 2. Tensor storage evolution

Milestone 1 stores CPU float32 values in `CpuStorage`. Milestone 2 introduces a narrow polymorphic storage contract so a `Tensor` can own either Tensora CPU storage or a backend-owned native tensor while preserving the existing opaque tensor handle.

Required properties:

- tensor handles remain opaque `uint64_t` identifiers;
- no C++ layout crosses the public ABI;
- host extraction stays explicit;
- native-to-host copies validate destination capacity;
- device transfer returns a new owned tensor;
- all tensor wrappers remain deterministic-release objects;
- disabled optional backends return structured `UNSUPPORTED` errors.

The public device model expands from `cpu` to `cuda:<index>` while keeping `Device.cpu` stable.

## 3. Milestone 2 — training surface

### 3.1 Device and tensor capabilities

The training-enabled runtime provides:

- CUDA availability and device-count diagnostics;
- `Tensor.to(Device)` transfer;
- float32 CPU and CUDA tensor storage in a LibTorch-enabled build;
- `requiresGrad` state;
- gradient retrieval;
- backward from a scalar loss;
- selected differentiable activations: ReLU, sigmoid, tanh;
- MSE and cross-entropy losses.

A CUDA request when LibTorch is not built or no requested CUDA device exists fails explicitly.

### 3.2 Module contract

The first native `Module` implementation is `Linear`.

A module owns native parameter state and exposes:

- forward execution;
- train/eval mode;
- parameter enumeration;
- buffer enumeration;
- device transfer before optimizer construction;
- checkpoint save/load.

Returned parameter tensors are ordinary Tensora tensor handles and obey normal release rules.

### 3.3 Optimizers

The first optimizer set is:

- SGD;
- Adam;
- AdamW.

Each optimizer owns one native handle and supports:

- `zeroGrad()`;
- `step()`;
- deterministic release.

Optimizers are created from a module after the module is on its target device.

### 3.4 Training proof

The deterministic acceptance workload is scalar linear regression:

```text
y = 2x + 1
```

The test must show:

- finite initial loss;
- decreasing loss over training steps;
- parameter values changing;
- checkpoint save/load reproducing model output;
- repeated training cycles without unbounded Tensora handle/storage growth.

Hosted CI validates the same training path on the CPU build of LibTorch. The CUDA acceptance workflow targets a Linux x64 self-hosted runner labelled `gpu` and `nvidia`, verifies CUDA availability before testing, and runs the same training proof on an actual CUDA device. CUDA is not declared fully validated until that hardware job succeeds.

## 4. Milestone 3 — ONNX inference surface

### 4.1 Session ownership

`OnnxSession` is a reusable native session represented by an opaque handle. It supports deterministic disposal and a finalizer fallback on the Dart side.

Session creation supports a model file path. Model loading failures are translated into structured Tensora model/runtime errors rather than raw backend exceptions.

### 4.2 Metadata and providers

The runtime exposes:

- available ONNX Runtime execution provider names;
- session input names;
- session output names;
- selected provider information where observable.

CPU execution is the required portable baseline. Discovery may report additional providers without implying they are automatically selected.

### 4.3 Execution

The initial inference contract supports float32 dense tensor inputs and outputs.

For each run:

1. input handles are validated;
2. values and validated dimensions are bound to ONNX Runtime;
3. the reusable session executes synchronously in native code;
4. output shapes and dtypes are validated;
5. each output becomes a new Tensora tensor handle.

Unsupported output types fail clearly instead of being reinterpreted.

### 4.4 Profiling and telemetry

Session options may enable ONNX Runtime profiling. Ending profiling returns the generated profile path through a capacity-checked ABI call.

Tensora disables ONNX Runtime telemetry immediately after creating the runtime environment. Core Tensora packages do not introduce implicit networking.

### 4.5 Inference proof

CI generates a small deterministic ONNX fixture and compares Tensora outputs with known reference values.

Acceptance covers:

- valid model load;
- malformed/missing model errors;
- input and output name inspection;
- correct float32 output values;
- incorrect input count/name/shape failures;
- repeated inference without unbounded Tensora handle/storage growth;
- concurrent read-only session runs with documented semantics;
- profiling lifecycle when enabled.

## 5. Error and lifetime model

The existing structured status model is extended only additively. Backend-specific exceptions never cross the C ABI.

New native object types use typed registry entries so tensor, module, optimizer, and inference-session handles cannot be confused.

Every creation API follows this rule:

```text
success => exactly one owned native reference
failure => zero owned references escape
```

All output handles are zeroed before work begins.

## 6. Security requirements

Native boundaries treat all external values as untrusted:

- device indices;
- model paths;
- module dimensions;
- optimizer hyperparameters;
- tensor handles;
- module/optimizer/session handles;
- input/output counts;
- name buffers and capacities;
- profiling buffers and capacities.

Model loading never registers custom operator libraries automatically. ONNX Runtime telemetry is disabled. Unsupported or malformed models fail without arbitrary extension loading.

Checkpoint loading is intended for checkpoints produced by the same Tensora module implementation. Secure distributable model bundles remain Milestone 4 scope.

## 7. CI gates

Milestones 2 and 3 add dedicated integration gates without weakening Milestone 1 gates.

Required hosted gates:

- existing native matrix;
- existing Dart/FFI matrix and coverage;
- LibTorch-enabled native build and tests on Linux;
- Dart training integration against the LibTorch-enabled native library;
- ONNX Runtime-enabled native build and tests on Linux;
- Dart ONNX integration against a generated deterministic fixture;
- warnings as errors;
- sanitizer validation for code paths that do not conflict with third-party runtime instrumentation.

Required hardware gate for full CUDA validation:

- Linux x64 self-hosted runner with `gpu` and `nvidia` labels;
- CUDA-visible LibTorch build;
- real device allocation and transfer;
- real forward/backward/optimizer execution;
- decreasing loss;
- no unbounded Tensora-owned memory or handle growth.

## 8. Definition of Done

Milestone 2 is complete when the training public contracts, CPU integration proof, checkpoint proof, lifetime proof, and real CUDA hardware proof all pass on one exact revision.

Milestone 3 is complete when ONNX session creation, metadata, provider discovery, inference correctness, failure handling, profiling, repeated-run memory stability, and concurrency tests all pass on the same exact revision.

The combined feature revision is not release-ready if either milestone relies on placeholders, skipped required behavior, silent backend fallback, or an unverified CUDA claim.
