# P1A Qualification Candidate

## Scope

This record identifies the first complete P1A candidate for Tensor Core V2.

P1A provides:

- real CPU storage for the approved ten-dtype set;
- ABI v6 typed host import and export;
- deterministic dtype casting;
- dtype-correct reshape and transpose views;
- public Dart typed construction and materialization;
- explicit float32-only autograd boundaries;
- explicit unsupported device/dtype failures;
- preserved NN V2, training, inference, ownership, and lifecycle contracts.

## Candidate ancestry

The qualified code parent is:

`0bff2c59629a0de0175c4821eae15c6759ae0556`

The current revision adds only this qualification record and is the hosted exact-SHA candidate.

## Pre-hosted verification

Before publishing the code parent, an isolated qualification workflow required all of the following to pass:

- Dart stable dependency resolution;
- Dart stable formatting for every modified Dart source;
- fatal-info and fatal-warning analysis for `tensora` and `tensora_train` production, test, and integration surfaces;
- dependency-light native Debug build;
- all 12 native CTest suites;
- Clang AddressSanitizer and UndefinedBehaviorSanitizer build;
- all 12 native CTest suites under ASan/UBSan with leak detection and halt-on-error behavior.

The qualification workflow committed the candidate only after those gates completed successfully and removed its temporary bootstrap workflow in the same commit.

## Hosted gate

P1A is not promoted by this document alone.

Promotion requires the complete repository pull-request matrix to pass on the exact current revision, including:

- Linux, macOS, and Windows native builds;
- Dart 3.7 and current stable compatibility;
- Dart FFI and Flutter;
- native and Dart coverage thresholds;
- ThreadSanitizer and C ABI fuzzing;
- core-only CPU training;
- LibTorch CPU training;
- real Apple MPS training;
- ONNX Runtime and CoreML inference;
- lifecycle and training soak tests;
- policy, ABI, dtype, alias, workspace, and high-assurance gates.

Any failure remains a release blocker and must be corrected without reducing thresholds, exclusions, or safety contracts.
