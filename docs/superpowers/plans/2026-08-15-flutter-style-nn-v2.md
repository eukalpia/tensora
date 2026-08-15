# Tensora NN V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Tensora's Flutter-style declarative neural-network API with composable modules, generalized parameter optimizers, native GELU/SiLU/SwiGLU autograd, deterministic state dictionaries, and high-assurance lifecycle semantics.

**Architecture:** Keep `tensora` as the native tensor/training foundation and expose narrow bridge objects instead of raw handles. Build user-facing composition in `tensora_nn`, and generalized optimizer orchestration in `tensora_optim`; composite forwards remain Dart-orchestrated while leaf execution stays native. Extend the additive C ABI for activation primitives, arbitrary parameter collections, native parameter identity, and transactional state assignment without removing legacy P0 entry points.

**Tech Stack:** Dart 3.7+ / Flutter-compatible pure Dart packages, dart:ffi, C++17, Tensora C ABI, custom CPU autograd, LibTorch policy backend, CMake/CTest, GitHub Actions.

## Global Constraints

- Base implementation is P0 exact green SHA `5aca4e5af508d51cc3f1905afabaec428edde3c2`.
- `tensora_nn` must not depend on Flutter SDK.
- No reflection, mirrors, source generation, annotations, or field scanning for module registration.
- No `Tensor.toList()` implementation path for GELU, SiLU, or SwiGLU.
- No silent device/provider fallback.
- Legacy native ABI entry points remain available while new entry points are additive.
- Native owned-source line coverage remains 100%.
- Existing Dart high-assurance threshold is not lowered; deterministic new composition code targets 100% line coverage.
- CPU training acceptance must remain independent from LibTorch.
- Existing real Apple MPS qualification remains a required regression gate.

---

### Task 1: RED public NN V2 contract tests

**Files:**
- Create: `packages/tensora_nn/test/module_v2_contract_test.dart`
- Create: `packages/tensora_nn/test/model_materialization_test.dart`
- Create: `packages/tensora_nn/test/state_dict_test.dart`
- Create: `packages/tensora_optim/test/optimizer_groups_test.dart`
- Modify: `packages/tensora_nn/test/public_api_test.dart`
- Modify: `packages/tensora_optim/test/public_api_test.dart`

**Interfaces:**
- Produces desired public signatures used by all later tasks:
  - `abstract base class Module`
  - `abstract base class Model extends Module { Module build(); }`
  - `final class Sequential extends Module { Sequential({required List<Module> children}); }`
  - `final class Linear extends Module { Linear({required int inFeatures, required int outFeatures, bool bias = true}); }`
  - `Parameter`, `Buffer`, `NamedParameter`, `NamedBuffer`, `NamedModule`
  - `StateDict`, `StateLoadResult`
  - `ParameterGroup`
  - `SGD({required Iterable<Parameter> parameters, ...})`, `Adam`, `AdamW`, and `.groups(...)` factories.

- [ ] **Step 1: Write compile-level and behavior tests first.** Tests construct `Sequential(children: [...])`, a custom cached `Model.build()`, verify deterministic names (`0.weight`, `0.bias`, ...), immutable child capture, duplicate/cycle rejection, readable tree diagnostics, state-dict strict/non-strict mismatch reporting, and optimizer group duplicate detection.
- [ ] **Step 2: Add activation/loss API tests.** Require `ReLU`, `Sigmoid`, `Tanh`, `GELU`, `SiLU`, `SwiGLU`, `MSELoss`, and `CrossEntropyLoss` object APIs.
- [ ] **Step 3: Commit tests only.**
- [ ] **Step 4: Use GitHub Actions on the draft PR to verify RED.** Expected failures are undefined NN V2 symbols/constructors, never syntax mistakes in the tests.

### Task 2: Native activation primitives and gradients

**Files:**
- Modify: `native/include/tensora.h`
- Modify: `native/src/training/training_bridge.h`
- Modify: `native/src/training/training_bridge_stub.cc`
- Modify: `native/src/training/training_bridge_torch.cc`
- Modify: `native/src/training/training_c_api.cc`
- Modify: `native/src/autograd/autograd.h`
- Modify: `packages/tensora/lib/src/native/native_training_bindings.dart`
- Modify: `packages/tensora/lib/src/native/native_training_runtime.dart`
- Modify: `packages/tensora/lib/src/tensor/tensor.dart`
- Modify: `native/tests/autograd_finite_difference_test.cc`
- Modify: `native/tests/training_test.cc`
- Modify: `native/tests/training_core_internal_test.cc`

**Interfaces:**
- Adds `ts_tensor_gelu`, `ts_tensor_silu`, `ts_tensor_swiglu`.
- Adds `Tensor.gelu()`, `Tensor.silu()`, `Tensor.swiglu()`.
- `GELU(x) = 0.5*x*(1+erf(x/sqrt(2)))`.
- `SiLU(x) = x*sigmoid(x)`.
- `SwiGLU(x)` splits the final even dimension into `a,b` and returns `silu(a)*b` with halved final dimension.

- [ ] **Step 1: Add native numerical and finite-difference tests for forward and backward, including SwiGLU rank-zero/odd/empty-last-dimension failures.**
- [ ] **Step 2: Run Actions and observe RED due to missing C ABI symbols/operations.**
- [ ] **Step 3: Extend `autograd::Operation` with `kGelu`, `kSilu`, `kSwiGlu`; record saved parents and implement analytical gradients in `ApplyNode`.** GELU derivative is `0.5*(1+erf(x/sqrt(2))) + x*exp(-x*x/2)/sqrt(2*pi)`. SiLU derivative is `sigmoid(x)*(1+x*(1-sigmoid(x)))`. SwiGLU backward reconstructs a full-shape gradient with first-half derivative `upstream*b*silu'(a)` and second-half derivative `upstream*silu(a)`.
- [ ] **Step 4: Implement CPU and LibTorch bridge operations and C ABI wrappers.** Torch uses `torch::gelu(input, "none")`, `torch::silu(input)`, and final-dimension chunk/slice composition without host copies.
- [ ] **Step 5: Bind FFI and expose Tensor methods.**
- [ ] **Step 6: Verify activation/native/autograd tests GREEN before moving on.**

### Task 3: Generalized native optimizers over tensor collections

**Files:**
- Modify: `native/include/tensora.h`
- Modify: `native/src/training/training_bridge.h`
- Modify: `native/src/training/training_bridge_stub.cc`
- Modify: `native/src/training/training_bridge_torch.cc`
- Modify: `native/src/training/training_c_api.cc`
- Modify: `packages/tensora/lib/src/native/native_training_bindings.dart`
- Modify: `packages/tensora/lib/src/native/native_training_runtime.dart`
- Create: `native/tests/optimizer_parameter_collection_test.cc`
- Modify: `native/tests/optimizer_checkpoint_test.cc`

**Interfaces:**
- Adds `ts_sgd_create_for_tensors(const ts_tensor_t* parameters, size_t count, ...)`.
- Adds matching `ts_adam_create_for_tensors` and `ts_adamw_create_for_tensors`.
- Legacy `ts_*_create(ts_module_t, ...)` delegates to the same internal collection implementation.
- Dart runtime methods take immutable `List<int>` tensor handles and allocate a temporary `Uint64` FFI array.

- [ ] **Step 1: Add native RED tests for 1, 2, N parameters, duplicate handles, empty/null arrays, frozen parameters, invalid handles, lifetime retention after caller releases the input handle, and SGD/Adam/AdamW updates.**
- [ ] **Step 2: Refactor core-only `OptimizerState` to own `vector<shared_ptr<Tensor>> parameters` instead of one `LinearState`; keep a dedicated optimizer mutex.**
- [ ] **Step 3: Implement common collection validation:** reject null/empty, invalid handles, duplicate underlying tensor identity; filter frozen parameters deterministically while rejecting a collection that has no trainable parameters after filtering.
- [ ] **Step 4: Adapt zero-grad/step to iterate the retained parameter vector and preserve moment state by index.**
- [ ] **Step 5: Implement the equivalent LibTorch collection creation and keep legacy module-based factories delegating through extracted module tensors.**
- [ ] **Step 6: Add C ABI and Dart runtime/binding support; verify native collection tests GREEN.**

### Task 4: Stable parameter identity and state assignment bridge

**Files:**
- Modify: `native/include/tensora.h`
- Modify: `native/src/training/training_bridge.h`
- Modify: `native/src/training/training_bridge_stub.cc`
- Modify: `native/src/training/training_bridge_torch.cc`
- Modify: `native/src/training/training_c_api.cc`
- Modify: `packages/tensora/lib/src/native/native_training_bindings.dart`
- Modify: `packages/tensora/lib/src/native/native_training_runtime.dart`
- Create: `native/tests/state_assignment_test.cc`

**Interfaces:**
- `ts_tensor_identity(ts_tensor_t tensor, uint64_t* out_identity)` returns stable native Tensor object identity suitable for parameter deduplication; it is not a raw memory address exposed to user APIs.
- `ts_tensor_clone_detached(ts_tensor_t tensor, ts_tensor_t* out_tensor)` creates a native snapshot without Dart host copies.
- `ts_tensor_assign_many(const ts_tensor_t* targets, const ts_tensor_t* sources, size_t count)` validates all pairs before any mutation and applies state copies; CPU path stages values first and increments mutation epochs after commit.

- [ ] **Step 1: Add RED native contract tests for stable identity across retained wrappers, detached clone independence, all-or-no-mutation validation failures, shape/dtype/device mismatch, aliases, null pointers, and invalid handles.**
- [ ] **Step 2: Implement identity from shared Tensor object ownership rather than storage pointer; wrappers created by `module_parameter_at` must resolve the same identity because they retain the same shared Tensor object.**
- [ ] **Step 3: Implement clone-detached and batch assignment in CPU backend using staged `vector<vector<float>>` payloads; perform complete validation/allocation before mutation.**
- [ ] **Step 4: Implement Torch assignment using preflight metadata/device checks and staged clones, restoring from backups on provider exceptions before returning error.**
- [ ] **Step 5: Bind Dart runtime calls and verify state assignment tests GREEN.**

### Task 5: Core bridge objects for `Parameter` and native leaf modules

**Files:**
- Create: `packages/tensora/lib/src/training/nn_bridge.dart`
- Modify: `packages/tensora/lib/src/training/training.dart`
- Modify: `packages/tensora/lib/tensora.dart`
- Create: `packages/tensora/test/nn_bridge_test.dart`

**Interfaces:**
- Public-user-safe bridge types do not expose raw handles.
- `NativeLeafModule` owns one native module handle and exposes `forward`, `parameterTensors`, `bufferTensors`, `setTraining`, `to`, and deterministic disposal.
- `NativeParameterRef` owns an independently retained Tensor wrapper plus `identity`, `requiresGrad`, `gradient`, `snapshot`, and internal optimizer-access capability.
- Existing low-level behavior remains accessible for legacy tests until migration is complete.

- [ ] **Step 1: Write RED tests proving bridge wrappers retain native ownership safely, reject use-after-dispose, report stable identity, and do not leak module/tensor handles.**
- [ ] **Step 2: Split native-handle lifecycle from the old public `Module` mental model without exposing integer handles.**
- [ ] **Step 3: Add capability-token based methods consumed only by workspace packages for optimizer/state operations.**
- [ ] **Step 4: Verify core Dart tests and existing FFI contracts GREEN.**

### Task 6: `tensora_nn` declarative composition system

**Files:**
- Create: `packages/tensora_nn/lib/src/module.dart`
- Create: `packages/tensora_nn/lib/src/model.dart`
- Create: `packages/tensora_nn/lib/src/parameter.dart`
- Create: `packages/tensora_nn/lib/src/state_dict.dart`
- Create: `packages/tensora_nn/lib/src/layers.dart`
- Create: `packages/tensora_nn/lib/src/activations.dart`
- Create: `packages/tensora_nn/lib/src/losses.dart`
- Create: `packages/tensora_nn/lib/src/diagnostics.dart`
- Modify: `packages/tensora_nn/lib/tensora_nn.dart`
- Modify: `packages/tensora_nn/README.md`
- Use tests from Task 1 plus focused unit tests under `packages/tensora_nn/test/`.

**Interfaces:**
- `Module` owns ordered child/parameter/buffer registration maps and public immutable traversal snapshots.
- `Model.build()` materializes exactly once; build failure leaves the model unmaterialized.
- `Sequential(children: ...)` defensively copies and freezes the list.
- `Linear` wraps `NativeLeafModule` and registers `weight` then optional `bias` from native parameter refs.
- Activation modules call native Tensor activation primitives.
- Loss objects call existing native loss primitives.

- [ ] **Step 1: Make Task 1 module/model tests GREEN with the minimal registration/lifecycle implementation.**
- [ ] **Step 2: Add ownership validation and cycle detection using DFS identity sets before accepting a materialized tree.**
- [ ] **Step 3: Add deterministic traversal with parameter dedup by native identity and stable dot-separated paths.**
- [ ] **Step 4: Implement recursive `train`, `eval`, `to`, and idempotent `dispose`; leaf modules perform native operations, composites recurse once.**
- [ ] **Step 5: Implement native-backed activation and loss modules.**
- [ ] **Step 6: Implement compact leaf `toString()` and explicit `toTreeString()` without host copies.**
- [ ] **Step 7: Run package tests and workspace analyze/format gates GREEN.**

### Task 7: StateDict and transactional model restoration

**Files:**
- Implement/modify: `packages/tensora_nn/lib/src/state_dict.dart`
- Modify: `packages/tensora_nn/lib/src/module.dart`
- Modify tests: `packages/tensora_nn/test/state_dict_test.dart`

**Interfaces:**
- `StateDict` is an immutable `Map<String, Tensor>`-like snapshot owning detached native tensors and implementing `dispose()`.
- `StateLoadResult` exposes immutable `missingKeys`, `unexpectedKeys`, and an `isSuccess` getter.
- `Module.stateDict()` snapshots parameters and persistent buffers with no Dart host-value copies.
- `Module.loadStateDict(StateDict state, {bool strict = true})` fully validates names/shape/dtype/device before calling one native `assignMany` operation.

- [ ] **Step 1: Make snapshot lifetime and strict/non-strict mismatch tests GREEN.**
- [ ] **Step 2: Validate all target/source metadata in Dart for clear structured errors, then let native batch assignment revalidate atomically at the ABI boundary.**
- [ ] **Step 3: Ensure shared parameters are emitted/assigned once by native identity.**
- [ ] **Step 4: Add lifecycle tests proving disposing a StateDict cannot invalidate live model parameters and model disposal cannot invalidate independent state snapshots.**

### Task 8: `tensora_optim` parameter and group API

**Files:**
- Create: `packages/tensora_optim/lib/src/parameter_group.dart`
- Create: `packages/tensora_optim/lib/src/optimizer.dart`
- Modify: `packages/tensora_optim/lib/tensora_optim.dart`
- Modify: `packages/tensora_optim/pubspec.yaml`
- Modify: `packages/tensora_optim/README.md`
- Use tests: `packages/tensora_optim/test/optimizer_groups_test.dart`

**Interfaces:**
- `ParameterGroup` stores an immutable deduplicated parameter snapshot plus nullable group hyperparameter overrides.
- Each public Optimizer may own one native optimizer handle per group; `zeroGrad` and `step` iterate handles in stable order.
- All groups are validated before any native handle is created. If later handle creation fails, already-created handles are released before rethrow.

- [ ] **Step 1: Make simple `AdamW(parameters: model.parameters, ...)` RED tests GREEN.**
- [ ] **Step 2: Implement `.groups(...)` factories for SGD/Adam/AdamW with duplicate identity rejection across groups and group-specific hyperparameters.**
- [ ] **Step 3: Skip frozen parameters deterministically and reject empty effective optimizer sets.**
- [ ] **Step 4: Add rollback tests for partial native optimizer creation and idempotent disposal.**

### Task 9: End-to-end arbitrary MLP training proof and documentation

**Files:**
- Create: `examples/declarative_mlp/pubspec.yaml`
- Create: `examples/declarative_mlp/bin/main.dart`
- Create: `packages/tensora/integration_test/nn_v2_training_integration_test.dart`
- Modify: `README.md`
- Modify: `docs/API_DESIGN.md`
- Modify: `ROADMAP.md`

**Interfaces:**
- Example uses only declarative NN V2 + generalized optimizer public APIs.
- Integration test trains a two-layer MLP on a deterministic synthetic mapping and asserts loss reduction over fixed steps on the core-only CPU backend.

- [ ] **Step 1: Write RED integration test using `Sequential(children: [...])` and `AdamW(parameters: model.parameters, ...)` with no LibTorch requirement.**
- [ ] **Step 2: Make training proof GREEN and ensure all temporary tensors/modules/optimizer/state snapshots are deterministically disposed.**
- [ ] **Step 3: Update documentation so named constructors and declarative model trees are the canonical API; keep legacy API only in migration notes.**

### Task 10: High-assurance coverage closure and exact-SHA qualification

**Files:**
- Modify only tests/workflows required to exercise new code; do not lower thresholds.
- Likely: `.github/workflows/native-coverage.yml`, `.github/workflows/high-assurance-ci.yml`, `.github/workflows/workspace-ci.yml` to include new package tests if current workspace discovery does not already cover them.

**Interfaces:**
- Final acceptance is one exact commit SHA on `feature/flutter-style-nn-v2`.

- [ ] **Step 1: Run/inspect all PR workflows and identify only real uncovered owned-source lines or platform failures.**
- [ ] **Step 2: Add tests for missing defensive/error paths until native owned-source line coverage is exactly 100%; do not add metric-only production exclusions.**
- [ ] **Step 3: Reach >=99.9% Dart coverage for new production code and 100% for deterministic pure-Dart registration/traversal/state-dict components where measurable by the current coverage pipeline.**
- [ ] **Step 4: Verify Linux Debug/Release, Windows Debug/Release, macOS Debug/Release, ASan+UBSan, TSan, fuzz, Dart FFI, workspace, training, inference, ONNX policy, dtype, alias, soak, and real Apple MPS qualification all succeed on the same candidate SHA.**
- [ ] **Step 5: Keep the PR draft until the user explicitly asks to integrate/merge.**
