# Training Engine V1 — performance evidence contract

Correctness is required before performance claims. Optimization work must retain an unoptimized reference path or golden comparison sufficient to detect numerical regressions.

## Metrics

Training benchmarks record at least:

- examples or tokens per second;
- forward, backward, and optimizer latency;
- peak and steady-state resident memory;
- allocation count and storage bytes where observable;
- materialized copies caused by layout changes;
- thread count and CPU utilization for CPU runs;
- device utilization, memory bandwidth, kernel launches, and communication time for qualified accelerator runs.

## Comparison rules

- benchmark the same model, batch, sequence length, dtype, optimizer, and convergence target;
- separate compile/warm-up time from steady-state execution;
- report median and tail behavior over repeated runs;
- never compare a numerically weaker configuration without labeling the difference;
- do not claim acceleration from an unsupported or silently different backend;
- record exact revision and environment.

## Tensor layout requirement

A reshape or transpose that can be represented as a view must not allocate a full data copy. Materialization of a non-contiguous layout must be observable in tests and profiling. This requirement exists because memory traffic is a first-order cost for future attention and large-model workloads.
