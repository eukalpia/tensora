# Tensora Testing Strategy

Tensora testing must prove more than compilation. The project spans numerical semantics, native memory, FFI, backends, devices, asynchronous execution, model parsing, and Flutter lifecycle behavior. Each layer requires evidence appropriate to its failure modes.

## 1. Test pyramid

Tensora should maintain these test classes:

```text
Dart unit tests
Native unit tests
C ABI tests
Dart ↔ native integration tests
Backend parity tests
Numerical reference tests
Autograd gradient checks
Serialization/model-format tests
Flutter integration tests
Real-device tests
Stress and leak tests
Failure-injection tests
Security regression tests
Fuzzing
Benchmarks/regression measurements
```

No single class is sufficient by itself.

## 2. Numerical correctness

Every mathematical operation should be checked against a trusted reference implementation or analytically known values.

Cover:

- representative shapes;
- rank edge cases;
- broadcasting;
- contiguous and supported non-contiguous layouts;
- zero-sized dimensions if supported;
- dtype conversions;
- overflow/invalid shape behavior;
- CPU/device parity for claimed combinations.

Floating-point comparisons must use dtype-appropriate absolute/relative tolerances.

A broad tolerance must be justified rather than used to hide errors.

## 3. Gradient correctness

Differentiable operators require gradient validation.

Use:

- analytical/reference gradient comparison;
- numerical finite-difference checks where practical;
- chained operations rather than only isolated primitives;
- accumulation tests;
- detach/no-grad tests;
- repeated backward behavior according to documented semantics.

Priority targets include:

- matmul;
- Linear;
- activations;
- normalization;
- losses;
- attention building blocks.

## 4. ABI tests

The C ABI is a compatibility boundary and requires dedicated tests.

Test:

- ABI version reporting;
- invalid/null handles;
- wrong handle type;
- duplicate release;
- use after release;
- pointer/length validation;
- invalid enum values;
- error retrieval;
- exception containment;
- concurrent calls for APIs documented as thread-safe.

A native exception must never escape the C ABI.

## 5. FFI integration

For each public native-backed Dart feature, at least one test should cross the real Dart → FFI → native → FFI → Dart path.

Mock-only coverage is insufficient for native behavior.

Tests should validate:

- correct result;
- error mapping;
- disposal;
- metadata conversion;
- repeated invocation;
- cancellation where supported.

## 6. Memory and lifetime testing

Resource lifetime is a first-class correctness property.

Run repeated cycles of:

```text
create
execute
release
```

at scales sufficient to expose leaks.

Include:

- partial initialization failure;
- model load failure;
- exception during execution;
- cancelled requests;
- view/storage lifetime;
- session/model shared ownership;
- repeated Flutter mount/dispose;
- repeated camera start/stop;
- GPU out-of-memory paths;
- runtime shutdown.

Track native and device memory where tooling allows.

## 7. Sanitizers

Native CI should use sanitizers on supported configurations where practical:

- AddressSanitizer;
- UndefinedBehaviorSanitizer;
- ThreadSanitizer in a compatible dedicated job.

Sanitizer failures are release blockers for affected supported code paths.

## 8. Backend parity

A feature claiming multiple backends must have parity tests across those backends.

Parity means compatible public semantics, not necessarily bit-identical floating-point results.

Test:

- outputs;
- dtype support;
- shape behavior;
- errors for unsupported combinations;
- transfer semantics;
- device selection.

Backend-specific deviations must be documented explicitly.

## 9. Model golden tests

Maintain small representative models with fixed inputs and expected outputs.

Initial goldens should eventually include:

- MLP;
- CNN;
- small Transformer;
- object detector or detection postprocessing fixture;
- embedding model.

Each supported inference backend should run compatible goldens.

Goldens should be small enough for routine CI where possible.

## 10. `.tmodel` tests

Model bundle tests must cover valid and hostile inputs.

Validate:

- correct manifest;
- missing files;
- duplicate entries;
- unsupported format version;
- incompatible runtime requirement;
- incorrect hashes;
- malformed JSON;
- path traversal;
- absolute paths;
- archive bombs/resource limits;
- extreme dimensions;
- oversized metadata;
- malformed tokenizer/labels metadata;
- corrupt embedded golden samples;
- signature validation when signatures are supported.

## 11. Fuzzing

Fuzz the smallest security-sensitive native/parser boundaries possible.

Targets should include:

- manifest parser;
- archive/container parser;
- shape metadata parser;
- tensor metadata parser;
- tokenizer metadata adapters;
- selected C ABI entry points accepting external metadata.

Every reproducible crash found by fuzzing should become a regression test.

## 12. Flutter tests

Flutter-facing features require lifecycle validation.

Scenarios:

- widget create/dispose;
- repeated model load/unload;
- application background/foreground;
- camera start/stop;
- audio interruption;
- cancellation during inference;
- platform permission failure;
- backend unavailable;
- memory pressure hooks where feasible;
- UI responsiveness during continuous inference.

Heavy compute must not execute synchronously on the UI isolate by default.

## 13. Camera stress tests

Live-camera pipelines must prove bounded behavior.

Test:

- frame producer faster than inference;
- queue at maximum capacity;
- latest/drop policy;
- camera shutdown while inference is active;
- model shutdown while frames are queued;
- device error;
- orientation/format changes where supported;
- long-running memory stability.

## 14. Local language model tests

For local generation APIs, test:

- model load/unload;
- session isolation;
- KV-cache ownership;
- streaming token order;
- cancellation;
- stop sequences;
- maximum token limits;
- repeated sessions;
- concurrent sessions if claimed supported;
- resource release after cancellation/failure.

Do not use probabilistic output quality as the only correctness test. Use deterministic settings/fixtures where possible for runtime behavior.

## 15. Failure injection

Critical runtime paths should be tested with injected failures where feasible:

- allocation failure;
- backend initialization failure;
- model provider failure;
- device unavailable/lost;
- invalid output from provider adapter;
- interrupted file write;
- failed atomic model activation;
- cancellation races.

The objective is to prove cleanup and error propagation, not merely happy-path functionality.

## 16. Platform/device matrix

A platform/backend is not stable merely because CI cross-compiles it.

Stable claims should include actual execution on representative hardware when the feature depends on real device behavior.

Examples:

- CUDA requires NVIDIA hardware execution;
- iOS runtime requires real/supported iOS validation;
- Android acceleration requires representative Android hardware/provider testing;
- platform camera pipelines require platform integration tests.

See `docs/COMPATIBILITY.md`.

## 17. Test determinism

Tests should minimize unnecessary nondeterminism.

Use fixed seeds where appropriate, while documenting that hardware/backend-level floating-point determinism may vary.

Avoid timing thresholds that are so tight they create flaky CI.

## 18. Performance regression tests

Performance gates should use stable benchmark infrastructure and historical baselines.

A performance regression should be investigated when it is:

- statistically meaningful;
- reproducible;
- relevant to a supported workload.

Do not fail CI on noisy one-off timings without enough evidence.

## 19. Security regressions

Any fixed vulnerability or crash involving malformed/untrusted input should receive a permanent regression test when disclosure constraints permit.

## 20. Release validation

Before release, validate the exact release candidate revision:

- formatting/static analysis;
- unit/integration suites;
- native tests;
- ABI tests;
- supported device tests;
- sanitizers where required;
- model-format security tests;
- stress/leak tests relevant to changed components;
- examples;
- compatibility docs;
- benchmark baseline.

A release should never rely on the assumption that earlier commits were green if the release candidate changed afterward.