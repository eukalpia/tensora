# Training Engine V1 — scope boundary

Training Engine V1 establishes a self-owned eager training path before graph compilation or distributed execution becomes a release dependency.

In scope:

- tensor storage/layout semantics needed by training;
- reverse-mode differentiation;
- parameter and module ownership;
- core losses and optimizers;
- deterministic CPU training;
- checkpoint/restart safety;
- Dart frontend integration;
- numerical, lifetime, sanitizer, and cross-platform validation;
- an end-to-end small language-model training proof after the required tensor and module primitives exist.

Not sufficient by itself for completion:

- an API wrapper around another training framework;
- tests that only assert successful return codes;
- device names without physical device execution;
- compiler directories without executable graph/IR correctness proof;
- performance claims without same-workload measurements.

LibTorch may remain available as a compatibility/reference backend, but Training Engine V1 acceptance requires a complete core-only path that does not depend on it for tensor differentiation or parameter updates.
