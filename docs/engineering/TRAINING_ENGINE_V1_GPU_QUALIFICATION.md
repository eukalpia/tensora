# Training Engine V1 — accelerator qualification

Accelerator support is divided into hardware-independent contracts and physical qualification.

## Hardware-independent CI

CPU-hosted CI may validate:

- device-code parsing;
- unsupported-device behavior;
- shape, dtype, layout, and dispatch planning;
- serialization and checkpoint contracts;
- graph/autograd transformations that do not claim device execution;
- compilation of accelerator-specific source where a suitable toolchain is available.

These checks do not qualify an accelerator.

## Physical CUDA gate

CUDA is qualified only on a runner where all of the following are true:

1. `nvidia-smi` succeeds and identifies a physical NVIDIA device.
2. The runtime reports a non-zero CUDA device count.
3. Tensor allocation is observed on the selected device.
4. Host-to-device and device-to-host transfers preserve values.
5. Device-side tensor computation produces the expected result.
6. Forward, loss, backward, and optimizer update execute with device-resident parameters.
7. Mixed-precision behavior is checked for supported dtypes.
8. Repeated training and inference loops show no unbounded device-memory growth.
9. Concurrent stream/event behavior passes its explicit synchronization contracts.
10. The exact revision under qualification is recorded with the hardware and software environment.

A missing physical runner leaves CUDA in an unqualified state; it must never be replaced by a silent CPU fallback or a simulated success result.
