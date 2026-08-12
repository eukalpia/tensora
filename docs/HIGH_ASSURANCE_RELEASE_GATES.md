# High-Assurance Release Gates

Tensora does not treat a green build as proof that a subsystem is defect-free. A release candidate must provide layered evidence that known failure classes have been exercised and that no known defect remains open in the release scope.

The project must not describe a milestone as error-free or assign a numerical probability to unknown bugs. Instead, it uses measurable reliability gates.

## Required evidence for Milestones 2 and 3

### Correctness

- deterministic reference tests for every public numerical operation;
- property/invariant tests for tensor metadata, transfers, loss functions, optimizer state, and repeated session execution;
- independent expected values for training and ONNX inference acceptance fixtures;
- checkpoint round-trip equivalence tests;
- CPU and CUDA results compared within documented float32 tolerances where the same operation is supported;
- no test may pass through a silent fallback to a different device or backend.

### Native boundary robustness

- every C ABI output pointer, handle, count, index, length, and capacity is treated as untrusted;
- null, stale, wrong-type, zero, negative, oversized, and inconsistent inputs have regression coverage;
- fuzz targets exercise public parsing/ABI validation surfaces that accept variable-size external data;
- AddressSanitizer and UndefinedBehaviorSanitizer remain clean for Tensora-owned native code;
- ThreadSanitizer is used on concurrency-focused core tests where third-party runtime instrumentation permits it;
- compiler warnings are errors on supported CI compilers.

### Memory and lifetime

- deterministic disposal is tested for Tensor, Module, Optimizer, and OnnxSession wrappers;
- double-dispose is safe at the Dart wrapper level;
- stale native handles fail deterministically;
- failed object creation leaks no owned native reference;
- stress tests return Tensora live-handle and storage counters to baseline;
- CUDA tests compare allocated-memory diagnostics before and after repeated training cycles with allocator synchronization where required;
- repeated ONNX session runs and session recreation show no unbounded process or Tensora-owned memory growth.

### Concurrency

- shared immutable tensor reads are stressed concurrently;
- reusable ONNX session concurrency is tested with multiple threads;
- registry lookup/release races are tested under load;
- no numerical work executes while holding the global handle-registry mutex;
- concurrency failures are never hidden by automatic retries.

### Coverage

- Dart library line coverage must remain at least 90% for Milestones 2 and 3 release candidates;
- critical ownership, error translation, native-handle, device-selection, checkpoint, and session-lifecycle branches require direct tests regardless of aggregate coverage;
- native coverage reports are recorded for the Linux reference build, but numerical coverage percentages never replace sanitizer, fuzz, stress, or integration evidence.

### Stress and soak

Before release-candidate status on one exact revision:

- at least 10,000 native core create/execute/release cycles remain green;
- at least 2,000 Dart public tensor lifecycle cycles remain green;
- at least 1,000 deterministic training-loop create/train/checkpoint/dispose cycles run on the hosted CPU training backend without Tensora-owned growth;
- at least 10,000 reusable-session inference calls run on the hosted ONNX Runtime backend without Tensora-owned growth;
- the real CUDA hardware gate runs repeated forward/backward/optimizer cycles and validates stable post-cleanup diagnostics;
- every soak gate fails on the first numerical mismatch, native error, leaked Tensora handle, or monotonic Tensora storage growth.

### Dependency and model safety

- optional native dependencies are version-pinned in validation workflows;
- model loading never enables arbitrary custom operator libraries implicitly;
- malformed, incompatible, missing, and unsupported ONNX models fail through structured errors;
- model paths and checkpoint paths are validated as ordinary data and are never interpreted as executable extensions;
- ONNX Runtime telemetry is disabled by the Tensora runtime environment;
- core Tensora packages introduce no implicit network access.

### Release discipline

A milestone can be called complete only when all mandatory gates for that milestone pass on the same exact revision. A later commit invalidates the evidence and requires the affected gates to run again.

Known failing tests, skipped mandatory tests, ignored sanitizer findings, unverified CUDA claims, placeholder implementations, or disabled required checks block release-candidate status.
