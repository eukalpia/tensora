# Tensora Model Bundle (`.tmodel`)

This document defines the design direction for the Tensora deployment bundle.

`.tmodel` is intended to package model artifacts, metadata, preprocessing/postprocessing definitions, validation data, and compatibility requirements into one secure deployment unit.

It is **not** intended to replace established weight/model formats without a clear technical reason.

## 1. Goals

A `.tmodel` bundle should allow an application to answer, before execution:

- what model this is;
- which task it performs;
- which inputs/outputs it expects;
- which runtime version it requires;
- which backends/devices it can use;
- how inputs are preprocessed;
- how outputs are postprocessed;
- what resources it is expected to consume;
- whether the bundle is intact;
- whether the bundle is compatible with the current runtime;
- whether embedded golden samples pass.

## 2. Non-goals

`.tmodel` should not become:

- a new universal tensor weight format;
- an archive capable of executing arbitrary scripts;
- a package manager;
- a dependency installer;
- an opaque binary format that cannot be inspected;
- a mechanism for loading arbitrary native libraries.

## 3. Proposed bundle layout

A bundle may contain:

```text
vehicle_detector.tmodel
├── manifest.json
├── model.onnx
├── weights.safetensors
├── tokenizer.json
├── labels.json
├── preprocessing.json
├── postprocessing.json
├── metadata.json
├── signature.json
└── tests/
    ├── input-001.bin
    ├── output-001.bin
    ├── input-002.bin
    └── output-002.bin
```

Only files required by the declared model/task need to exist.

## 4. Manifest

`manifest.json` is the authoritative bundle description.

Planned top-level fields include:

```text
format_version
model_id
name
version
task
architecture
minimum_tensora_version
artifacts
inputs
outputs
backends
dtypes
quantization
resources
preprocessing
postprocessing
tokenizer
integrity
license
metadata
```

The exact schema will be versioned and published before `.tmodel` is considered stable.

## 5. Format version

Example concept:

```json
{
  "format_version": 1
}
```

The runtime must reject unsupported future format versions rather than guessing.

Backward-compatible additive changes within a format version should be narrowly defined. Breaking schema/semantic changes require a new format version.

## 6. Runtime compatibility

A model should declare the minimum Tensora runtime it requires.

Example concept:

```json
{
  "minimum_tensora_version": "0.4.0"
}
```

The loader validates this before model initialization.

## 7. Artifact declarations

Every packaged model artifact should be declared explicitly with:

- path;
- artifact type;
- format/version where relevant;
- hash;
- size;
- optional role.

Example concept:

```json
{
  "artifacts": [
    {
      "path": "model.onnx",
      "type": "onnx",
      "sha256": "...",
      "size": 12345678
    }
  ]
}
```

The loader must not trust archive contents that are absent from or inconsistent with the manifest where strict mode applies.

## 8. Inputs and outputs

Model interfaces should declare:

- symbolic name;
- dtype;
- rank/shape constraints;
- dynamic dimensions;
- semantic description where useful;
- layout where relevant.

Example concept:

```json
{
  "inputs": [
    {
      "name": "images",
      "dtype": "float32",
      "shape": [1, 3, 640, 640],
      "layout": "NCHW"
    }
  ]
}
```

Do not accept dimensions that overflow configured runtime limits.

## 9. Backend declarations

A bundle may describe compatible or preferred execution providers, but the runtime remains responsible for validation.

Example concepts:

```text
preferred backends
required capabilities
known unsupported providers
```

A bundle must not claim that a backend is usable merely because the manifest names it.

## 10. Quantization metadata

If quantized, record enough information to understand deployment requirements.

Potential fields:

- scheme;
- weight dtype;
- activation dtype;
- group size;
- calibration metadata where appropriate;
- backend requirements.

Quantization metadata must not silently change model semantics during load.

## 11. Resource estimates

The bundle may declare expected:

- RAM;
- VRAM;
- model file size;
- temporary workspace;
- context/KV-cache behavior for language models.

These are estimates unless explicitly guaranteed.

The runtime should enforce independent safety limits rather than trusting model-provided estimates.

## 12. Preprocessing

`preprocessing.json` should describe a restricted, typed pipeline rather than arbitrary executable code.

Possible operations:

- decode image;
- resize;
- crop;
- color conversion;
- normalization;
- channel/layout conversion;
- tokenizer selection/configuration;
- audio resampling;
- supported spectrogram transforms.

Each operation must have a versioned schema and bounded parameters.

## 13. Postprocessing

Potential operations:

- softmax;
- top-k;
- thresholding;
- non-maximum suppression;
- bounding-box conversion;
- embedding normalization;
- token decoding.

Again, the model package describes data, not arbitrary source code.

## 14. Golden tests

A bundle may include small golden input/output fixtures.

Uses:

- post-download verification;
- backend compatibility check;
- regression diagnostics;
- confirming preprocessing/postprocessing parity.

Golden tests must be size-limited.

Validation uses dtype-appropriate tolerance rather than requiring bitwise equality for all floating-point providers.

## 15. Integrity

At minimum, stable `.tmodel` should support cryptographic hashes for declared artifacts.

The bundle loader verifies integrity before execution.

Corruption must produce a structured model-format/integrity error.

## 16. Signatures

A later production milestone should support signed bundles.

Conceptual flow:

```text
download
 ↓
parse bounded container
 ↓
validate manifest
 ↓
verify artifact hashes
 ↓
verify signature/trust policy
 ↓
validate compatibility
 ↓
run selected golden checks
 ↓
activate atomically
```

Signature support must distinguish cryptographic validity from application trust policy.

## 17. Archive safety

If `.tmodel` uses an archive/container internally, extraction must defend against:

- `../` traversal;
- absolute paths;
- symlinks/hardlinks if unsafe for the implementation;
- duplicate paths;
- case-collision problems on case-insensitive filesystems;
- compressed-size bombs;
- total uncompressed size overflow;
- excessive file count;
- oversized filenames/metadata.

Prefer streaming validation and bounded extraction.

## 18. Resource limits

The runtime/application should be able to configure limits such as:

```text
maximum bundle size
maximum expanded size
maximum file count
maximum manifest size
maximum tokenizer size
maximum tensor rank
maximum dimension
maximum total tensor elements
maximum model-declared memory
```

Limits are checked before dangerous allocation where possible.

## 19. Paths

Paths inside the bundle are logical relative paths.

Rules should include:

- normalized separators;
- no absolute paths;
- no parent traversal;
- no empty components where ambiguous;
- canonical duplicate detection.

## 20. Arbitrary code prohibition

A `.tmodel` file must never automatically load:

- dynamic libraries;
- scripts;
- shell commands;
- Dart source;
- Python source;
- native plugins.

Custom operators require explicit application/runtime installation outside the untrusted model package.

## 21. Licensing metadata

A bundle should support descriptive model licensing metadata, but Tensora must not assert that metadata is legally sufficient for all models.

Applications/distributors remain responsible for satisfying third-party model licenses.

## 22. CLI behavior

Planned commands:

```text
tensora pack
tensora unpack
tensora inspect
tensora verify
```

`inspect` should not execute the model.

`verify` should support integrity/schema checks independently from full inference.

## 23. Atomic model activation

When used with future model distribution features:

1. download to a candidate location;
2. validate fully;
3. optionally run golden tests;
4. fsync/persist where appropriate;
5. atomically switch active model reference;
6. retain previous known-good version for rollback.

A failed update must not corrupt the active model.

## 24. Compatibility discipline

The format should evolve conservatively.

Before changing a stable schema field, define:

- old semantics;
- new semantics;
- backward compatibility;
- forward compatibility;
- converter/migration requirements;
- runtime error behavior.

## 25. V1 completion gate

`.tmodel` V1 is not stable until:

- formal schema exists;
- parser is strict and bounded;
- compatibility behavior is documented;
- hash verification exists;
- malicious archive/path tests exist;
- fuzzing exists for parser boundaries;
- CLI inspect/verify is implemented;
- at least one Flutter and one server inference example consume the same bundle;
- format-version tests are in CI.