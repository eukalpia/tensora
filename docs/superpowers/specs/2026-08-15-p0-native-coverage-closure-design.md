# P0 Native Coverage Closure Design

## Goal

Close the remaining native owned-code line coverage gap at 100% without lowering thresholds, excluding production lines, or weakening defensive behavior.

## Approach

Use coverage-only white-box contract executables that compile the existing implementation translation units into dedicated test binaries. This follows the repository's existing `onnx_whitebox_contract_test.cc` pattern and lets tests exercise anonymous-namespace helpers and defensive rollback/error paths that the public API intentionally prevents callers from reaching.

Production behavior remains unchanged. Test doubles are scoped to coverage executables only. For training checkpoint I/O, a coverage-only stream double deterministically fails at selected write/flush stages. For ONNX, coverage-only wrappers control provider snapshots and selected session outcomes while delegating normal execution to real ONNX Runtime. For LibTorch, tests construct internal state and register a real test buffer on `torch::nn::Linear` to exercise the otherwise-unused buffer conversion path.

## Coverage targets

- `native/src/inference/inference_bridge_onnx.cc`: provider resolution/configuration tails, dtype rejection, profiling, output invariants, ORT and allocation guards.
- `native/src/inference/inference_c_api.cc`: insertion helper null contract, partial-output rollback, bridge result-count invariant.
- `native/src/training/training_bridge_stub.cc`: activation/bias defensive contracts, checkpoint write failures, null load payload, optimizer gradient mismatch, vector-return cleanup path.
- `native/src/training/training_bridge_torch.cc`: internal optimizer output contract and real module-buffer conversion.
- `native/src/training/training_c_api.cc`: internal tensor insertion null-output contract.

## Verification

The existing `Native Coverage` workflow remains authoritative and keeps `--fail-under-line 100`. After it reaches 100%, open a draft PR from the work branch so `pull_request`-only platform workflows run against the exact same head SHA, including Training CI on Linux/macOS/MPS/Windows and CPU Training Engine Debug/Release/ASan+UBSan qualification.
