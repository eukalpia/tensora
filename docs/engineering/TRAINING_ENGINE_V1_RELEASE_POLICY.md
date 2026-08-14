# Training Engine V1 — release policy

The development branch may contain incomplete later slices, but a release candidate is cut only from an exact revision where every declared V1 capability has evidence.

Required evidence includes:

- native Debug and Release correctness;
- sanitizer results for core-owned code;
- finite-difference gradient validation;
- deterministic convergence proof;
- optimizer first-step and convergence proof;
- checkpoint round-trip and transactional failure proof;
- tensor view/alias correctness;
- Dart integration against a core-only native runtime;
- cross-platform native tests on supported operating systems;
- physical hardware proof for every accelerator advertised as qualified;
- benchmark records for performance claims;
- no placeholder path or silent semantic fallback in a declared capability.

A failing required gate cannot be converted into a release success by reducing its threshold, suppressing the failing semantic case, or substituting an unqualified backend.
