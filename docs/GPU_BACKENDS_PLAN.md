# Multi-vendor GPU backends

## Goal

Provide explicit, capability-driven hardware acceleration across supported desktop platforms without pretending that one vendor runtime works everywhere.

## Backend matrix

| Platform | Vendor | Tensor/training backend | ONNX inference backend |
|---|---|---|---|
| Windows | NVIDIA | CUDA | CUDA, then DirectML fallback |
| Windows | Intel | XPU | OpenVINO, then DirectML fallback |
| Windows | AMD | hardware-specific training remains separate from the LibTorch path | MIGraphX when installed, then DirectML fallback |
| macOS | Apple | MPS | CoreML |
| Linux | NVIDIA | CUDA | CUDA |
| Linux | Intel | XPU | OpenVINO |
| Linux | AMD | HIP/ROCm | MIGraphX |

## Contracts

1. Public device values are vendor-neutral runtime contracts: CPU, CUDA, MPS, XPU, and HIP.
2. Device discovery is explicit. Unsupported or unavailable devices return zero count and device transfers fail with a structured unsupported error.
3. Training uses the LibTorch device backend compiled into the native runtime. No silent CPU fallback is allowed after the caller selects a GPU device.
4. ONNX inference accepts an explicit execution-provider preference and exposes the provider selected for the session.
5. `auto` inference selection is deterministic and platform-aware. It may fall back to CPU only when no requested or preferred accelerator is available.
6. DirectML is an inference path on Windows and is not represented as tensor storage in the LibTorch training runtime.
7. CoreML is an inference path on macOS and MPS is the tensor/training path.
8. Existing CPU-only builds remain dependency-light and continue to compile on Linux, macOS, and Windows.
9. Every hardware path has a self-hosted validation workflow that proves the selected device/provider is active and rejects CPU fallback.

## Test order

1. Add failing Dart value/device tests for MPS, XPU, HIP and generic device counts.
2. Add failing native ABI tests for new device constants and generic discovery.
3. Implement ABI and Dart mappings.
4. Add failing LibTorch tests for device mapping and availability semantics.
5. Implement CUDA/MPS/XPU/HIP mapping behind platform/build guards.
6. Add failing ONNX provider-selection tests.
7. Implement explicit provider selection and selected-provider diagnostics.
8. Add platform hardware validation workflows for NVIDIA CUDA, Apple MPS/CoreML, Intel XPU/OpenVINO and Windows DirectML/vendor inference.
9. Run hosted CPU compatibility, sanitizers, coverage and all available hardware gates on the exact revision.
