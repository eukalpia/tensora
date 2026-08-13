# Monorepo CI Gates Implementation Plan

**Goal:** Make Workspace CI and High Assurance CI pass without lowering existing quality thresholds.

**Approach:**

1. Apply the canonical Dart formatter output to the four files reported by both Dart 3.7 and stable.
2. Run the existing native runtime default-discovery and structured missing-library tests as isolated coverage processes.
3. Merge both new LCOV reports into the existing 90% gate.
4. Verify all workflows on one exact commit SHA before opening a pull request into `feature/monorepo-packages`.

**Constraints:** Preserve runtime behavior, C ABI, typed handles, deterministic resource release, structured errors, and the 90.00% coverage threshold. Do not add production dependencies or silent fallbacks.
