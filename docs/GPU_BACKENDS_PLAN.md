# Multi-vendor accelerator backends

## Goal

Provide explicit, capability-driven acceleration across desktop platforms while keeping public Dart contracts independent of one vendor runtime.

This work is a cross-cutting expansion of Tensora's tensor/training and ONNX inference layers. It does **not** redefine the `.tmodel` milestone numbering in `ROADMAP.md`.

## Current backend matrix

| Platform | Vendor | Tensor/training path | ONNX inference path | Current qualification |
| --- | --- | --- | --- | --- |
| Linux | CPU | Tensora core / LibTorch CPU | ONNX Runtime CPU | Hosted Beta evidence |
| Windows | CPU | Tensora core / LibTorch CPU | ONNX Runtime CPU | Hosted Beta evidence |
| macOS Apple Silicon | Apple | MPS | CoreML | Real hosted hardware Beta evidence |
| Linux | NVIDIA | CUDA | CUDA provider | Experimental; physical qualification pending |
| Windows | NVIDIA | CUDA | CUDA or DirectML provider where built | Experimental; physical qualification pending |
| Windows | Intel | XPU | OpenVINO or DirectML provider where built | Experimental; physical qualification pending |
| Linux | Intel | XPU where provided by linked runtime | OpenVINO provider where built | Experimental; physical qualification pending |
| Linux | AMD | HIP/ROCm identity over ROCm LibTorch | MIGraphX provider where built | Experimental; physical qualification pending |
| Windows | AMD | no validated LibTorch training contract | DirectML provider | Experimental; physical qualification pending |

Implementation breadth must never be reported as qualification breadth.

## Public device contract

Tensora exposes:

```text
Device.cpu
Device.cuda(index)
Device.mps
Device.xpu(index)
Device.hip(index)
```

Device behavior is explicit:

1. discovery reports only devices visible to the loaded runtime;
2. negative indexed-device values are rejected;
3. unavailable transfers fail with structured errors;
4. explicit training-device requests do not silently fall back to CPU;
5. binary operations require the same device kind and device index;
6. returned tensors preserve the requested/selected device identity.

`TensoraRuntime.availableDevices` exposes visible devices. `preferredDevice` is a deterministic convenience query, not an implicit global default. Tensor factories remain CPU-default unless `device:` is supplied.

## Tensor creation and transfer

Accelerator factory creation currently follows:

```text
Dart values
  ↓
CPU staging tensor
  ↓
explicit native device transfer
  ↓
accelerator tensor
  ↓
deterministic staging release
```

The implementation has failure-path tests requiring tensor/storage counters to return to baseline if the requested accelerator is unavailable.

This is not a zero-copy import contract.

## Training backend

Training uses the optional LibTorch backend.

The device mapper handles:

- CPU;
- CUDA;
- MPS;
- XPU;
- HIP/ROCm identity.

For ROCm builds, upstream PyTorch commonly exposes the physical AMD device through its CUDA-style tensor API while reporting a HIP runtime. Tensora keeps the public device identity as HIP/ROCm so application code does not need to encode that upstream implementation detail.

A module is moved to its target device before optimizer construction in the documented workflow.

## ONNX provider contract

Supported public provider preferences are:

```text
auto
CPUExecutionProvider
CUDAExecutionProvider
DmlExecutionProvider
CoreMLExecutionProvider
OpenVINOExecutionProvider
MIGraphXExecutionProvider
```

Rules:

1. explicit provider requests never silently fall back;
2. `auto` is the only controlled fallback policy;
3. the selected provider is exposed on the session;
4. provider discovery alone is not hardware support evidence;
5. DirectML uses provider-compatible session settings;
6. accelerated provider execution does not imply device-zero-copy input binding.

The current portable ONNX bridge materializes Tensora input data on the host before creating ONNX Runtime inputs.

## Apple qualification

Hosted Apple Silicon CI provides real device/provider evidence.

### MPS training

The MPS acceptance path requires PyTorch to execute an actual Metal operation and then runs Tensora through:

- accelerator tensor creation;
- matrix multiplication;
- `Linear.to(Device.mps)`;
- parameter identity checks;
- forward pass;
- MSE loss;
- backward;
- 120 SGD steps;
- material loss reduction;
- strict tensor/storage/module/optimizer lifecycle checks.

CPU fallback fails the test.

### CoreML inference

The CoreML acceptance path:

- requires Apple Silicon;
- builds against a pinned ONNX Runtime distribution;
- creates a session with explicit CoreML preference;
- verifies the selected provider;
- executes a deterministic model;
- validates output values and resource lifetime.

## Windows dependency contract

Windows optional-backend builds use sidecar dependencies:

```text
runtime-directory/
  tensora_native.dll
  backend DLLs
```

The Dart runtime securely loads an explicit existing `tensora_native.dll` while prioritizing dependency resolution from that same directory.

Hosted CI validates this with:

- pinned LibTorch sidecars for training;
- pinned ONNX Runtime sidecars for inference;
- a machine image that also contains an older unrelated system ONNX Runtime DLL.

The test must still bind the runtime's pinned dependency rather than the unrelated system installation.

Manual Windows GPU qualification must use the same sidecar layout. PATH is not the deployment contract.

## NVIDIA qualification

NVIDIA paths remain Experimental until a matching physical runner completes the hardware workflow.

Qualification must prove:

- NVIDIA hardware is visible;
- the linked PyTorch build is CUDA rather than ROCm;
- CUDA device count is non-zero;
- real device computation succeeds;
- Tensora tensor/module/loss/optimizer objects stay on CUDA;
- loss decreases;
- lifecycle counters return to baseline.

Linux and Windows are separate qualification targets.

## Intel qualification

Intel XPU remains Experimental until a matching physical Windows/Intel runner completes the XPU acceptance path.

Qualification must require:

- an Intel display adapter;
- `torch.xpu` availability;
- a real XPU kernel probe;
- Tensora `Device.xpu(0)` identity throughout the training path;
- no CPU fallback;
- lifecycle stability.

OpenVINO inference is a separate provider qualification and must not be inferred from XPU training success.

## AMD qualification

### ROCm / Linux

The ROCm target requires:

- AMD hardware visible through ROCm tooling;
- a PyTorch build reporting HIP;
- a real ROCm tensor kernel;
- Tensora public `Device.hip(0)` identity;
- end-to-end training acceptance;
- no CPU fallback.

### DirectML / Windows

DirectML qualification requires:

- AMD display hardware;
- a concrete ONNX Runtime distribution that contains the DirectML provider;
- its DLLs staged beside `tensora_native.dll`;
- explicit DirectML session selection;
- selected-provider verification;
- actual deterministic inference.

DirectML is an inference provider and is not represented as LibTorch tensor storage.

## Failure semantics

Accelerator work must preserve explicit failure behavior.

Examples:

- unavailable device → structured unsupported failure;
- invalid device index → structured invalid/unsupported failure as defined by the ABI;
- explicit unavailable ONNX provider → session creation failure;
- mixed-device binary op → rejection rather than implicit transfer;
- failed factory transfer → no leaked staging tensor/storage;
- missing Windows sidecar dependency → clear runtime-load failure rather than accidental system binding.

## Validation layers

Hosted gates prove cross-platform source portability and CPU/Apple behavior:

- Native CI;
- Dart FFI CI;
- Training CI;
- Inference CI;
- sanitizers;
- ThreadSanitizer;
- C ABI fuzzing;
- lifecycle/training soak;
- high-assurance regression gates.

Physical vendor qualification is intentionally separate and manually targeted so absence of a self-hosted GPU runner does not create a misleading permanent PR failure.

## Promotion rule

An accelerator path moves from Experimental to Beta only after the missing physical/provider evidence is complete on an exact revision and includes:

- real hardware/provider execution;
- correctness reference;
- explicit selected device/provider evidence;
- no hidden CPU fallback;
- lifecycle stability;
- documented dependency/runtime versions;
- appropriate platform packaging validation.

Compilation, source mapping, or a workflow definition alone is insufficient.
