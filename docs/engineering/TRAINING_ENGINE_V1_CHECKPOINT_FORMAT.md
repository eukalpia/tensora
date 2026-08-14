# Training Engine V1 — checkpoint invariants

The current native module checkpoint is a deliberately narrow V1 format for deterministic module-state validation while the wider model format evolves.

Required invariants:

- fixed magic and explicit format version;
- explicit parameter count and per-parameter shape metadata;
- exact float32 payload length validation;
- no acceptance of trailing bytes;
- no arbitrary executable content;
- read and validate the complete candidate state before mutating the live module;
- incompatible parameter structure fails without partial application;
- corrupt or incomplete input fails deterministically;
- successfully loaded parameter mutations advance alias versions.

Optimizer-state persistence and richer module trees are later extensions and must be versioned explicitly rather than changing V1 interpretation in place.
