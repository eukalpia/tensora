# Contributing to Tensora

Thank you for contributing to Tensora.

Tensora is systems infrastructure. Changes may affect numerical correctness, native memory safety, ABI compatibility, device behavior, Flutter lifecycle behavior, model security, and performance. Contributions are therefore expected to be small enough to review rigorously and complete enough to validate end to end.

## License of contributions

Tensora is distributed under the Apache License 2.0.

Unless explicitly stated otherwise, any contribution intentionally submitted for inclusion in Tensora is provided under the terms of the Apache License 2.0, consistent with the repository `LICENSE` file.

Do not contribute code, model artifacts, documentation, test data, generated sources, or other material that you do not have the right to submit under compatible terms.

## Before starting

For a small bug fix, documentation correction, test improvement, or narrowly scoped implementation, a pull request may be sufficient.

Open or discuss an RFC before implementing changes that materially affect:

- public Dart APIs;
- the stable C ABI;
- tensor semantics;
- memory ownership;
- serialization/model formats;
- backend contracts;
- threading or isolate behavior;
- Flutter lifecycle behavior;
- security boundaries;
- package boundaries;
- compatibility guarantees;
- release/versioning policy.

Raise these in the pull request that introduces the change, before the
implementation is broadly built out.

## Development model

Use short-lived topic branches based on current `main`.

Recommended naming:

```text
feature/tensor-slice
fix/cuda-device-cleanup
docs/backend-contract
test/model-parser-fuzz
perf/matmul-dispatch
```

Avoid combining unrelated refactoring and feature work in the same pull request.

## Pull-request size

Prefer a complete vertical slice over a broad partial implementation.

Good:

```text
Tensor reshape
+ native implementation
+ error handling
+ CPU tests
+ FFI integration test
+ documentation
```

Poor:

```text
50 operator declarations
+ mostly unimplemented backends
+ no tests
```

A large change should normally be split by stable internal boundary rather than by arbitrary file count.

## Required pull-request description

Every non-trivial pull request should explain:

1. **Problem** — what is missing, incorrect, unsafe, or inefficient?
2. **Scope** — exactly what this change modifies.
3. **Design** — important invariants and trade-offs.
4. **Validation** — tests and commands that were run.
5. **Compatibility** — public API/ABI/model-format implications.
6. **Performance** — benchmark impact when relevant.
7. **Security** — parser, native memory, model-input, or boundary implications when relevant.
8. **Remaining work** — intentionally deferred items, if any.

Do not describe a change as complete if required validation could not be run.

## Correctness requirements

Numerical changes must include a trusted reference comparison where practical.

Tests should cover:

- normal inputs;
- edge shapes;
- zero-sized behavior where supported;
- broadcasting rules where relevant;
- dtype behavior;
- unsupported combinations;
- error paths;
- overflow and invalid dimensions;
- device parity where a feature claims multiple devices.

Floating-point tests must use justified tolerances rather than arbitrary broad thresholds.

## Native code requirements

Native code must make ownership explicit.

Required expectations:

- prefer RAII internally;
- validate all data crossing the public ABI;
- do not expose C++ ABI types to Dart;
- check integer arithmetic used for sizes and offsets;
- never rely on Dart garbage collection as the only mechanism for releasing large native/GPU allocations;
- convert recoverable failures to structured Tensora errors;
- avoid process termination for user/model input errors;
- document thread-safety and lifetime assumptions.

Changes affecting native memory should include stress/leak validation.

## Dart API requirements

Public Dart APIs should be:

- idiomatic;
- strongly typed;
- backend-neutral;
- explicit about asynchronous behavior;
- explicit about ownership where a resource is disposable;
- useful outside Flutter unless the API is intentionally part of `tensora_flutter`.

Do not expose backend-specific implementation objects in stable APIs.

## Flutter requirements

Heavy model execution must not block Flutter's UI isolate by default.

Changes involving Flutter lifecycle must consider:

- disposal;
- background/foreground transitions;
- camera/audio interruption;
- cancellation;
- repeated initialization;
- memory pressure;
- backend/device failure.

Realtime pipelines must have bounded queues.

## Performance changes

Performance work must include measurements.

Report enough context to reproduce the result:

- hardware;
- operating system;
- build mode;
- backend/provider;
- model/operator;
- dtype;
- shape/batch size;
- warmup policy;
- sample count;
- latency/throughput statistic.

Do not report underlying backend performance as if it were a Tensora-specific optimization.

## Generated code

Generated code is acceptable when it reduces repetitive bindings or metadata maintenance, but:

- the generator must be versioned;
- output must be deterministic;
- generation instructions must be documented;
- CI should detect stale generated files;
- generated output should not be manually edited unless documented otherwise.

## Dependencies

New dependencies need a concrete justification.

Consider:

- license compatibility;
- maintenance quality;
- security history;
- binary size;
- transitive dependencies;
- supported platforms;
- ABI stability;
- whether the dependency becomes part of the public distribution.

Large native runtime dependencies must not be added to every target merely because one backend needs them.

## Commits

Write commits around coherent changes.

Recommended style:

```text
feat: add CPU tensor reshape
fix: release native session after failed model load
test: add malformed tmodel dimension cases
perf: reduce redundant host-device synchronization
docs: document backend capability contract
```

Keep commit messages factual and implementation-focused.

## Review expectations

Review should evaluate more than style.

Depending on the change, reviewers should verify:

- architectural boundary correctness;
- mathematical semantics;
- ownership and cleanup;
- concurrency;
- error propagation;
- compatibility;
- security;
- performance evidence;
- tests;
- documentation.

Unresolved correctness, ownership, security, or compatibility questions block merge.

## Merge requirements

A pull request is ready to merge when:

- required CI checks pass on the exact head revision;
- requested review changes are resolved;
- no known production path is a placeholder;
- public behavior is documented;
- compatibility documents are updated if required;
- benchmark/security evidence is present when relevant;
- commits contain no secrets, credentials, large accidental binaries, or unrelated generated artifacts.

## Breaking changes

Do not introduce a breaking public API, ABI, or model-format change casually.

A breaking change must:

- have a documented reason;
- define the migration path;
- follow the versioning policy;
- update compatibility documentation;
- use an RFC when architectural.

## Security issues

Do not publicly disclose an unpatched vulnerability through a normal issue if doing so would materially increase user risk.

Follow `SECURITY.md`.

## Documentation is part of the change

A feature is not complete when only the code works.

Update relevant:

- API documentation;
- architecture documentation;
- examples;
- compatibility matrix;
- benchmark documentation;
- release notes.

## Definition of a strong contribution

A strong Tensora contribution is understandable in isolation, preserves architectural boundaries, has explicit ownership semantics, has evidence for its claims, and leaves the repository in a more verifiable state than before.