# Training Engine V1 — numerical validation contract

Numerical correctness is evaluated independently from API success.

## Gradient validation

For differentiable operations with a stable neighborhood, analytical reverse-mode gradients are compared with central finite differences:

`df/dx ≈ (f(x + ε) - f(x - ε)) / (2ε)`

The suite uses values away from known non-differentiable boundaries when testing operations such as ReLU. Absolute and relative tolerances must be appropriate for float32 and documented in the test.

## Accumulation

Graphs where one value contributes through more than one edge must accumulate every contribution exactly once. Repeated-parent cases are explicit regression tests.

## Mutation safety

Each saved forward value is associated with the mutation version of its backing alias set. Backward must validate all saved versions before publishing any new leaf gradient. A stale alias therefore fails transactionally instead of producing a mixture of gradients from different parameter states.

## Optimizer validation

Optimizer tests include closed-form first-step checks where practical, convergence checks, and state/lifetime checks. Adam validates moment updates and bias correction. AdamW validates decoupled weight decay separately from the gradient update.

## Checkpoint validation

Checkpoint restoration is judged by exact parameter round-trip for the stored dtype. Failure cases must leave the live module unchanged.
