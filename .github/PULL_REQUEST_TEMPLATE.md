## Problem

<!-- What problem does this change solve? State the current behavior and why it is insufficient. -->

## Scope

<!-- What is included? What is intentionally not included? -->

## Design

<!-- Describe important architecture, ownership, lifecycle, concurrency, or API decisions. Link an RFC when required. -->

## Validation

<!-- List the exact tests, analysis, builds, sanitizer runs, device tests, or manual validation performed. -->

- [ ] Formatting/static analysis relevant to the change passes.
- [ ] Unit/integration tests relevant to the change pass.
- [ ] Native/FFI tests were run when native boundaries changed.
- [ ] Real device/backend validation was run for hardware-specific support claims.
- [ ] Stress/leak validation was run for resource-lifecycle changes.

## Correctness

<!-- For numerical/model changes: what trusted reference or golden output validates the result? -->

## Ownership and concurrency

<!-- Who owns new native/device resources? How are they released? Is the code thread-safe/reentrant? -->

## Compatibility

- [ ] No stable public API, C ABI, `.tmodel`, checkpoint, or provider compatibility behavior changes.
- [ ] Compatibility changes are documented below and follow the version/RFC policy.

Details:

## Security

<!-- Does this parse untrusted data, handle native buffers, change model loading, add dependencies, or alter a trust boundary? -->

- [ ] No new security-sensitive boundary is introduced.
- [ ] Relevant malformed-input/resource-limit/security regression tests are included.

Details:

## Performance

<!-- Required when the change makes a performance claim or affects a hot path. Include hardware, backend, dtype, shape/model, build mode, and before/after measurements. -->

## Documentation

- [ ] Public behavior is documented.
- [ ] Examples are updated where relevant.
- [ ] Compatibility/support documentation is updated where relevant.

## Remaining work

<!-- List explicitly deferred follow-up work. Do not hide required completion work here. -->

## Merge readiness

- [ ] The pull request is focused on one coherent objective.
- [ ] No production path is a placeholder presented as supported behavior.
- [ ] No unrelated generated artifacts, credentials, or large files are included.
- [ ] Required validation corresponds to the current pull-request head revision.