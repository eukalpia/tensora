# Native Coverage 100% Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reach 100% native owned-code line coverage and qualify the exact head SHA across the repository's platform workflows without weakening production contracts.

**Architecture:** Add isolated white-box coverage executables that include selected implementation translation units and inject deterministic test-only collaborators where real public APIs cannot reach defensive branches. Keep runtime source semantics and the `--fail-under-line 100` gate unchanged. Use a draft PR only after native coverage is green so pull-request-only platform workflows qualify the same SHA.

**Tech Stack:** C++20, CMake, gcov/gcovr, ONNX Runtime 1.26.0, LibTorch/PyTorch 2.13, GitHub Actions.

## Global Constraints

- Do not lower native line coverage below 100%.
- Do not add gcov/gcovr exclusion markers for owned production code.
- Do not remove defensive runtime checks solely for coverage.
- Coverage-only fakes must live under `native/tests/` and must not alter shipped library behavior.
- Final qualification must use the exact same commit SHA for native coverage and PR platform workflows.

---

### Task 1: Core training white-box coverage

**Files:**
- Create: `native/tests/training_stub_coverage_test.cc`
- Modify: `.github/workflows/native-coverage.yml`

**Interfaces:**
- Consumes: anonymous-namespace helpers and public bridge functions from `native/src/training/training_bridge_stub.cc`.
- Produces: deterministic execution of the currently uncovered activation, bias, checkpoint, payload-null, parameter cleanup, and optimizer mismatch lines.

- [ ] **Step 1: Add a coverage-only stream double and direct internal contract tests**

Compile `training_bridge_stub.cc` into the test translation unit after pre-including its headers. Provide a coverage-only `std::TestOfstream` selected through a preprocessor rename so `ModuleSave` can fail at specific write call numbers and on `flush()` without changing production I/O.

The test must exercise:

```cpp
MakeActivation(input, static_cast<autograd::Operation>(255), &output);
AddBias(matrix, bias, nullptr);
AddBias(rank_one, bias, &output);
AddBias(matrix, wrong_width_bias, &output);
ReadTensorPayload(stream, expected_shape, nullptr);
```

It must also run `ModuleSave` with deterministic failure modes for checkpoint header, tensor rank, tensor shape, tensor element count, tensor payload, and final flush, then create an SGD optimizer, inject a wrong-sized gradient through the parameter's `AutogradMeta`, and call `OptimizerStep` expecting `TS_INTERNAL_ERROR`.

- [ ] **Step 2: Wire the executable into the core coverage build**

Add a manual compile/run command beside the existing core bridge contract executable:

```bash
c++ -std=c++20 -Wall -Wextra -Wpedantic -Werror --coverage -O0 -g \
  -Inative/include -Inative/src \
  native/tests/training_stub_coverage_test.cc \
  -Lbuild/native-coverage-core -ltensora_native \
  -Wl,-rpath,"$GITHUB_WORKSPACE/build/native-coverage-core" \
  --coverage \
  -o build/native-coverage-core/tensora_training_stub_coverage_tests
./build/native-coverage-core/tensora_training_stub_coverage_tests
```

- [ ] **Step 3: Run Native Coverage and confirm the training stub gaps disappear**

Expected: no uncovered lines remain in `training_bridge_stub.cc` or `training_c_api.cc` after Task 3 below.

---

### Task 2: LibTorch bridge white-box coverage

**Files:**
- Create: `native/tests/training_torch_bridge_coverage_test.cc`
- Modify: `.github/workflows/native-coverage.yml`

**Interfaces:**
- Consumes: internal `LinearState`, `CreateOptimizerState`, `ModuleBufferAt`, and the real Torch storage conversion path from `native/src/training/training_bridge_torch.cc`.
- Produces: execution of the two currently uncovered LibTorch bridge lines.

- [ ] **Step 1: Write the white-box test**

Include `training_bridge_torch.cc` into the test translation unit. Verify:

```cpp
CreateOptimizerState(nullptr, nullptr, nullptr)
```

returns `TS_INVALID_ARGUMENT`. Then create `std::make_shared<LinearState>(1, 1, true)`, register a real float32 buffer on `state->module`, insert the state as `HandleType::kModule`, and call `ModuleBufferAt(handle, 0, &out)` expecting `TS_OK` and a non-null tensor.

- [ ] **Step 2: Compile with the existing LibTorch compile/link environment**

Use the training build's `compile_commands.json` to reuse the exact compiler flags used for `training_bridge_torch.cc`, replace the source with the coverage test, then reuse the existing `tensora_training_backend_internal_tests` link command with the test object substituted, mirroring the current torch backend policy test mechanism.

- [ ] **Step 3: Execute before collecting `training.json`**

Expected: `training_bridge_torch.cc` reaches 100% line coverage.

---

### Task 3: C API rollback and insertion coverage

**Files:**
- Create: `native/tests/inference_c_api_coverage_test.cc`
- Create: `native/tests/training_c_api_coverage_test.cc`
- Modify: `.github/workflows/native-coverage.yml`

**Interfaces:**
- Consumes: `InsertInferenceTensor`, `ReleaseInsertedOutputs`, `InsertTrainingTensor`, and `ts_onnx_session_run` from the implementation translation units.
- Produces: direct validation of null-output contracts and partial-output cleanup.

- [ ] **Step 1: Training C API helper contract**

Include `training_c_api.cc` and call:

```cpp
InsertTrainingTensor(nullptr, nullptr)
```

Expect `TS_INVALID_ARGUMENT`.

- [ ] **Step 2: Inference C API helper and rollback contracts**

Include `inference_c_api.cc`. Directly validate `InsertInferenceTensor(nullptr, nullptr)`. Provide a test-local definition of `tensora::inference::SessionRun` that can return either the wrong result count or `{valid_tensor, nullptr}` while returning `TS_OK`.

Call `ts_onnx_session_run` with a valid registered input tensor. In wrong-count mode expect `TS_INTERNAL_ERROR`. In partial-insertion mode expect the second insert to fail and verify `HandleRegistry::Count(HandleType::kTensor)` returns to its pre-call value, proving `ReleaseInsertedOutputs` ran.

- [ ] **Step 3: Compile/run in the appropriate core/inference coverage phases**

Expected: both C API files reach 100% line coverage and rollback is behaviorally proven.

---

### Task 4: ONNX bridge defensive and profiling coverage

**Files:**
- Create: `native/tests/onnx_bridge_coverage_test.cc`
- Modify: `native/tests/generate_onnx_fixture.py`
- Modify: `.github/workflows/native-coverage.yml`

**Interfaces:**
- Consumes: internal helpers and `SessionState` from `inference_bridge_onnx.cc`, real ONNX Runtime for ordinary execution, and coverage-only wrappers for provider/session outcomes that CPU-only ORT cannot produce.
- Produces: execution of all remaining ONNX bridge lines while retaining production provider and exception logic.

- [ ] **Step 1: Generate dtype fixtures**

Extend the fixture generator so the requested reference model is still created and sibling models are also emitted for unsupported float64 input and unsupported float64 output metadata.

- [ ] **Step 2: Add provider/session wrappers scoped to the coverage executable**

Pre-include ONNX headers, define coverage state, then preprocessor-rename `Ort::SessionOptions`, `Ort::Session`, and `Ort::GetAvailableProviders` only while including `inference_bridge_onnx.cc`. The wrappers must delegate normal operations to real ORT but support deterministic modes for:

```text
provider list with an available-but-not-configurable provider
successful CUDA/OpenVINO/MIGraphX append calls
GetAvailableProviders throwing Ort::Exception
Session::Run returning the wrong output count
Session::EndProfilingAllocated throwing Ort::Exception
```

- [ ] **Step 3: Exercise remaining contracts**

Use the generated dtype fixtures to verify unsupported input/output model metadata. Run a real profiled CPU session once to cover profile setup and successful end-profiling, then use the wrapper failure modes for provider enumeration, runtime output-count invariant, and end-profiling exception mapping. Use a `TensorStorage` test double whose `CopyToHostF32` throws `std::bad_alloc` to exercise the outer `SessionRun` allocation catch.

- [ ] **Step 4: Run the white-box executable before `inference.json` collection**

Expected: `inference_bridge_onnx.cc` reaches 100% line coverage.

---

### Task 5: Native gate and exact-SHA platform qualification

**Files:**
- No production source changes expected.

**Interfaces:**
- Consumes: final branch head and existing GitHub Actions workflows.
- Produces: evidence that the exact SHA is fully qualified.

- [ ] **Step 1: Run Native Coverage on the final head**

Expected summary:

```text
lines: 100.0%
Native Coverage: success
```

- [ ] **Step 2: Confirm the other P0 workflows on the same SHA are green**

Required: Native Contract CI, DType Contract CI, ONNX Policy CI, Torch Backend Policy CI, High Assurance CI, Native Coverage.

- [ ] **Step 3: Open a draft pull request from `work/unified-p0-high-assurance-20260815` to `main`**

The PR event must trigger workflows that do not run on `work/**` pushes, especially Training CI and CPU Training Engine CI.

- [ ] **Step 4: Verify platform jobs on the exact PR head SHA**

Required evidence includes Linux LibTorch CPU training, macOS Apple Silicon MPS training, Windows LibTorch CPU training, core-only Debug and Release, and ASan+UBSan jobs with zero failures.

- [ ] **Step 5: Compare the verified PR head SHA with the Native Coverage head SHA**

They must be identical before declaring P0 complete.
