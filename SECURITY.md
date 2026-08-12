# Tensora Security Policy

Tensora includes native code, model parsers, device runtimes, FFI boundaries, and application-facing model execution. Security issues in these areas can affect process integrity, memory safety, availability, or confidentiality of application data.

Security reports are taken seriously.

## Supported versions

Until Tensora begins publishing supported releases, there is no stable supported-version table.

Once releases are available, this file will identify supported release lines and security-fix policy explicitly.

## Reporting a vulnerability

If you believe you have found a vulnerability that could put users at meaningful risk, please avoid publishing exploit details in a normal public issue before maintainers have had an opportunity to assess and address the problem.

Use GitHub's private security reporting mechanism for this repository when enabled. If private reporting is not available, contact the repository maintainers through a private channel identified in the repository profile/project documentation rather than posting weaponizable details publicly.

A useful report includes:

- affected component;
- affected revision/version if known;
- platform/backend;
- prerequisites;
- reproduction steps;
- expected behavior;
- observed behavior;
- impact assessment;
- proof-of-concept or crash input where safe to share;
- suggested mitigation if available.

Do not include secrets or unrelated private data in a report.

## Security-sensitive areas

The following are especially sensitive.

### Native memory and FFI

Risks include:

- use-after-free;
- double-free;
- invalid handle reuse;
- buffer overflow;
- integer overflow;
- out-of-bounds access;
- unsafe pointer/length handling;
- race conditions;
- exceptions escaping ABI boundaries.

### Model and bundle parsing

Treat model files and metadata as untrusted input.

Risks include:

- path traversal;
- archive bombs;
- malformed graphs;
- malformed tensors;
- dimension/numel overflow;
- excessive allocation;
- corrupted tokenizer data;
- malicious metadata;
- parser crashes.

### Model updates/distribution

Future model distribution features must consider:

- integrity verification;
- signature verification;
- rollback safety;
- atomic activation;
- transport security;
- cache poisoning;
- downgrade policy.

### Flutter/mobile runtime

Risks include:

- unsafe native buffer lifetime;
- application lifecycle races;
- camera/audio data retained after disposal;
- unintended network transfer;
- platform permission misuse.

### Custom native extensions

Untrusted `.tmodel` bundles must never automatically load arbitrary native libraries or executable scripts.

Custom operators/extensions require explicit installation and application opt-in.

## Local-first privacy

Core Tensora runtime packages should not send model input, prompts, images, audio, embeddings, model outputs, or telemetry to remote services by default.

Any optional networking/distribution subsystem must be explicit and documented.

Logging must avoid recording sensitive model inputs by default.

## Resource exhaustion

Availability is part of security.

Where external model files or dimensions influence allocation, enforce configurable hard limits before dangerous allocation where practical.

Potential limits include:

- model bundle size;
- expanded archive size;
- file count;
- manifest/tokenizer metadata size;
- tensor rank;
- individual dimension;
- tensor element count;
- memory budget;
- inference queue depth.

## Secure native engineering

Native code should:

- prefer RAII internally;
- validate untrusted lengths and indices;
- use checked arithmetic for allocation sizes;
- avoid raw owning pointers where safer ownership types are possible;
- return structured errors across the C ABI;
- be tested under sanitizers where supported;
- receive fuzz coverage for parser/ABI boundaries.

## Fuzzing

Security-sensitive parsers and ABI boundaries should be fuzzed continuously or regularly depending on CI cost.

Crashes found through fuzzing should become regression tests after remediation when disclosure constraints permit.

Priority fuzz targets include:

- `.tmodel` container parser;
- manifest parser;
- tensor/shape metadata;
- tokenizer metadata;
- selected C ABI functions accepting external buffers/metadata.

## Dependency security

Third-party dependencies must be reviewed for:

- license compatibility;
- maintenance status;
- known security posture;
- platform implications;
- binary distribution requirements.

Security updates to critical native dependencies should be assessed promptly rather than deferred solely to normal feature release cadence.

## Vulnerability handling

A typical security fix should:

1. reproduce/validate the issue privately;
2. identify affected versions/paths;
3. design the smallest correct fix;
4. add regression coverage where safe;
5. run relevant sanitizers/fuzz/tests;
6. prepare supported-version fixes;
7. publish remediation before or together with detailed disclosure;
8. document impact accurately without exaggeration.

## Security release priority

Confirmed issues involving memory corruption, arbitrary code execution, sandbox escape, unsafe automatic native extension loading, or practical compromise of sensitive local data should receive release priority appropriate to their impact.

## No security-through-obscurity claims

Do not describe an unsafe behavior as acceptable merely because it is difficult to trigger or because a model file is “normally trusted.”

Trust boundaries must be explicit.

## Scope of guarantees

Tensora can validate its own runtime and container behavior, but it cannot guarantee that every third-party model is safe, unbiased, legally usable, or suitable for a particular application. Applications remain responsible for model provenance, product-level authorization, privacy policy, and domain-specific safety controls.