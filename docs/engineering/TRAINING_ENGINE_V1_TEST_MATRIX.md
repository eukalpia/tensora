# Training Engine V1 — test matrix

The core training line uses layered tests so failures identify the broken contract rather than only the final symptom.

| Layer | Required evidence |
| --- | --- |
| Tensor/ABI | malformed inputs, stale handles, output clearing, lifetime counters |
| Layout | zero-copy eligible views, logical-order copies, materialization accounting |
| Autograd | exact small derivatives, repeated-edge accumulation, finite differences, stale-alias rejection |
| Module | parameter enumeration, forward shape/value correctness, training/eval state, device errors |
| Optimizer | SGD/Adam/AdamW contracts, first-step checks, convergence, zero-grad behavior |
| Checkpoint | exact round-trip, corrupt/truncated/trailing/incompatible rejection, rollback |
| Dart | forward/loss/backward/update against core-only runtime |
| Runtime | Debug, Release, ASan/UBSan, TSan where supported |
| Platform | Linux, macOS, Windows native contracts |
| Accelerator | physical hardware discovery, compute, training, memory stability |

A test may appear in more than one CI workflow when it validates both a focused subsystem gate and the ordinary release matrix. Focused workflows do not replace the release matrix.
