# Training Engine V1 — contribution rules

Changes to the training engine must preserve the following rules:

- add or update a failing semantic test before changing numerical behavior;
- do not replace a failing capability with a silent fallback;
- keep output handles cleared on error paths;
- keep alias mutation semantics explicit and testable;
- retain finite-difference coverage for differentiable primitives;
- add sanitizer coverage for new core-owned memory/lifetime paths;
- keep device qualification separate from device enumeration;
- record measurable evidence before making performance claims;
- avoid introducing compiler or distributed abstractions that bypass eager correctness contracts.

A change that improves performance but breaks numerical equivalence, lifetime safety, checkpoint rollback, or alias consistency is not accepted as an optimization.
