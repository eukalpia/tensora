# Tensora Roadmap

This roadmap describes the intended engineering sequence for Tensora. It is deliberately organized around **validated vertical slices**, not feature-count targets.

Dates are intentionally omitted until the implementation team has measured the real cost of the foundation milestones. A milestone is complete only when its acceptance gates are satisfied on the exact revision being released.

## Guiding rule

Tensora must become useful quickly without pretending to own infrastructure it does not yet need to own.

The preferred evolution is:

```text
Phase 1
Dart API
  ↓
Stable Tensora ABI
  ↓
Mature native backends

Phase 2
Dart API
  ↓
Tensora graph/runtime
  ↓
Multiple native backends

Phase 3
Dart API
  ↓
Tensora IR/compiler
  ↓
External + custom optimized kernels

Phase 4
Unified Dart/Flutter ML ecosystem
  ↓
CPU / GPU / NPU / edge / server
```

## Milestone 0 — Foundation

### Goals

Define the system before committing to unstable implementation details.

### Deliverables

- monorepo layout and package dependency rules;
- public API design principles;
- stable C ABI direction;
- native handle and memory ownership model;
- structured error model;
- device abstraction;
- backend contract;
- threading and isolate model;
- Flutter execution model;
- `.tmodel` format draft;
- testing strategy;
- benchmark methodology;
- security boundaries;
- compatibility policy;
- RFC process;
- CI foundations.

### Acceptance gate

Milestone 0 is complete when the architecture contains no unresolved contradiction around:

- who owns native memory;
- how native resources are released;
- which APIs may block;
- what crosses isolates;
- what crosses the C ABI;
- how backends are selected;
- how unsupported functionality fails;
- how Flutter avoids UI-isolate blocking;
- how model bundles are validated;
- how additional backends can be added later.

No broad implementation should precede this gate.

---

## Milestone 1 — Tensor Core / CPU Vertical Slice

### Goals

Prove the public Dart API, native ABI, native storage model, and error/lifetime design using real computation.

### Minimum feature set

- `Tensor`;
- `Shape`;
- `DType`;
- `Device.cpu`;
- native tensor handles;
- explicit release;
- finalizer safety net;
- allocation accounting;
- tensor creation;
- reshape;
- transpose;
- slice;
- elementwise add/multiply;
- matrix multiplication;
- reductions required by tests;
- CPU backend;
- structured native-to-Dart errors.

### Required proof

```text
Dart
 ↓
Tensor API
 ↓
C ABI
 ↓
Native tensor
 ↓
CPU matmul
 ↓
validated result
```

### Acceptance gate

- mathematical correctness against a trusted reference;
- no known leaks under repeated create/execute/dispose stress tests;
- malformed shapes fail safely;
- overflow checks exist;
- no hidden bulk copies into Dart collections;
- FFI overhead benchmark exists;
- documentation includes a runnable example.

---

## Milestone 2 — Native Training / CUDA

### Goals

Prove that Dart can act as a serious training frontend while mature native infrastructure performs the numerical work.

### Planned scope

- CUDA device discovery;
- GPU tensor allocation;
- device transfers;
- autograd through a mature native backend;
- `Module`;
- parameters and buffers;
- `Linear`;
- selected activations;
- basic losses;
- SGD;
- Adam;
- AdamW;
- checkpoint basics.

### Required proof

A Dart program must perform real GPU-backed training:

```text
Dart model
 ↓
forward
 ↓
loss
 ↓
backward
 ↓
optimizer step
 ↓
CUDA execution
```

Evidence must include:

- decreasing loss on a deterministic small training task;
- model weights changing;
- GPU allocation/execution observed by diagnostics;
- checkpoint save/load reproducing state;
- repeated training without unbounded memory growth.

### Non-goal

Writing a complete custom CUDA kernel library.

---

## Milestone 3 — Portable Inference

### Goals

Make Tensora useful for real model deployment before the training ecosystem is complete.

### Planned scope

- ONNX model loading;
- input/output tensor binding;
- backend/provider discovery;
- synchronous native execution internally with asynchronous Dart-facing APIs where needed;
- reusable model sessions;
- profiling hooks;
- preprocessing/postprocessing pipeline primitives.

### Acceptance gate

- representative ONNX models match trusted reference outputs within dtype-appropriate tolerance;
- model load errors are structured;
- unsupported operators fail clearly;
- backend selection is inspectable;
- repeated inference is memory-stable;
- concurrent usage semantics are documented and tested.

---

## Milestone 4 — `.tmodel` V1

### Goal

Create a secure deployment bundle that packages model assets and execution metadata without inventing unnecessary replacements for standard formats.

### Planned contents

```text
model.tmodel
├── manifest.json
├── model.onnx and/or supported model artifact
├── weights.safetensors where applicable
├── tokenizer.json where applicable
├── labels.json where applicable
├── preprocessing.json
├── postprocessing.json
├── metadata.json
└── tests/
    ├── input-*.bin
    └── output-*.bin
```

### Required capabilities

- strict schema validation;
- integrity hashes;
- format versioning;
- runtime compatibility declaration;
- resource limits;
- golden sample validation;
- safe unpacking;
- no arbitrary code execution;
- CLI inspect/verify/pack operations.

### Acceptance gate

Corrupt, malicious, incompatible, oversized, or incomplete bundles are rejected safely and predictably.

---

## Milestone 5 — Flutter Runtime

### Goal

Make ordinary local-model execution in Flutter require no application-level native glue.

### Planned scope

- `tensora_flutter`;
- asset loading;
- asynchronous model execution;
- Flutter lifecycle integration;
- Android support;
- iOS support;
- desktop support as available;
- platform error translation;
- resource cleanup on lifecycle transitions.

### Acceptance gate

A Flutter application can load and execute a supported local model with Dart code only, while heavy execution stays off the UI isolate.

Required lifecycle tests:

- open/close;
- background/foreground;
- repeated widget disposal;
- model disposal;
- cancellation;
- memory pressure handling where available;
- device/backend failure paths.

---

## Milestone 6 — Vision and Live Camera

### Goal

Establish the first flagship product workflow: production live-camera AI.

### Planned scope

- image abstraction;
- camera frame adapters;
- native buffer integration;
- resize and color conversion;
- normalization;
- object detection helpers;
- NMS;
- optional Flutter overlays;
- bounded frame queue and backpressure policies.

### Required policy

Realtime camera processing must use bounded buffering. The default policy should favor the newest useful frame instead of accumulating stale work.

### Acceptance gate

A representative detector runs continuously without:

- UI-isolate blocking;
- unbounded queue growth;
- unbounded memory growth;
- unsafe lifecycle behavior.

Profiler output must expose latency, effective inference FPS, dropped frames, queue behavior, and transfers where observable.

---

## Milestone 7 — Text and Embeddings

### Scope

- tokenizer adapters;
- embedding model abstraction;
- batch embeddings;
- normalization utilities;
- similarity primitives;
- text preprocessing pipelines.

### Non-goal

Building a vector database into Tensora core.

---

## Milestone 8 — Local Language Models

### Goal

Provide a coherent Dart API for local generation while relying on appropriate native inference engines and optimized kernels.

### Planned scope

- language-model abstraction;
- immutable shared model weights;
- per-request/per-conversation sessions;
- KV-cache ownership;
- token streaming;
- cancellation;
- temperature, top-k, top-p, stop sequences;
- quantized model integration where supported;
- memory diagnostics.

### Acceptance gate

A Flutter application can:

- load a supported local model;
- start a generation session;
- stream tokens;
- cancel generation;
- dispose session state;
- repeat the workflow without memory growth.

---

## Milestone 9 — Production Edge Runtime

### Planned scope

- signed `.tmodel` bundles;
- local model registry;
- verified downloads as an optional package;
- atomic activation;
- rollback;
- execution policies;
- memory budgets;
- thermal/battery integration where reliable platform APIs exist;
- richer diagnostics and profiling.

### Principle

No remote networking should occur from core tensor/runtime packages unexpectedly. Model distribution belongs to an explicit optional subsystem.

---

## Milestone 10 — Training Ecosystem Expansion

### Planned scope

- `Dataset`;
- `DataLoader`;
- isolate-based CPU preprocessing where appropriate;
- mixed precision;
- gradient scaling;
- schedulers;
- richer checkpointing;
- trainer abstraction;
- transformer building blocks;
- reproducibility controls;
- distributed-ready interfaces.

### Non-goal

Multi-node distributed training before single-node training is reliable and measured.

---

## Milestone 11 — Graph Compiler

This milestone begins only after real application and training workloads reveal where compilation can produce material benefits.

### Planned components

- Tensora IR;
- graph capture;
- shape/type propagation;
- constant folding;
- dead-node elimination;
- fusion where justified;
- memory planning;
- backend partitioning;
- compiled execution API.

### Acceptance gate

Compiler work must demonstrate measurable benefit on real workloads. Architectural complexity without measurable value is not sufficient.

---

## Later opportunities

Possible future work, driven by evidence rather than roadmap pressure:

- custom fused CPU kernels;
- custom CUDA kernels;
- optimized attention kernels;
- Metal-specific optimizations;
- NPU providers;
- WebGPU;
- distributed training;
- model compilation for edge targets;
- Flutter DevTools integration;
- third-party backend SDK.

## Explicit early non-goals

The following should not delay initial usefulness:

- replacing cuBLAS;
- replacing cuDNN;
- creating a full CUDA kernel ecosystem from scratch;
- replacing ONNX Runtime for ideological reasons;
- supporting every ML operator;
- supporting every hardware accelerator;
- multi-node training;
- inventing proprietary weight formats without necessity;
- creating a cloud hosting product;
- creating a vector database;
- creating an unrelated general-purpose workflow platform;
- creating empty packages to make the project look complete.

## Release philosophy

A milestone becomes a release candidate only after:

1. its public contracts are documented;
2. correctness tests pass;
3. supported native/device integration tests pass on real hardware where applicable;
4. leak/stress tests pass;
5. relevant security tests pass;
6. benchmark data is recorded;
7. examples are runnable;
8. compatibility documentation is updated;
9. no production path relies on placeholders or silent semantic fallbacks.

See [docs/RELEASES.md](docs/RELEASES.md) for the release process.