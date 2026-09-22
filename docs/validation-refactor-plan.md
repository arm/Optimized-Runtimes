# Benchmark context refactor

Performance and accuracy are measured by the existing Ultralytics benchmark.
The internal validation package records the Raspberry Pi state needed to
interpret those results.

## Ownership

- `benchmarks/ultralytics/benchmark.py` owns the dataset, validation workload, and
  Ultralytics output.
- `record_benchmark` captures host state immediately before and after that
  unchanged command and records its process outcome.
- Pull-request review compares the current Ultralytics result and context with
  the chosen reference result and context.

Compiler acceptance remains with the compiler, runtime behavior remains with
the native and public-interface tests, and native builds remain with CMake.

## Recorded context

The context report contains hardware and kernel identity, CPU affinity and
load, frequency policy, temperature, throttling flags, peak child-process RSS,
available memory, filesystem cache counters, swap state, major page faults, and
swap traffic. Each dynamic value is captured before and after the benchmark.

`Cached` and `SReclaimable` describe the relevant system cache footprint.
`MemAvailable`, swap usage, major-fault deltas, and swap-I/O deltas show whether
RAM pressure plausibly affected the run. The recorder observes these values; it
does not clear caches or alter system configuration.

## Code-quality standard

- Extend the established Ultralytics workflow instead of creating another
  benchmark workload.
- Keep operating-system collection separate from benchmark execution.
- Use typed records internally and serialize untyped data only at the JSON
  boundary.
- Document the source and interpretation of recorded values.
- Exercise the recorder through its command interface with a real child
  process.
- Keep the package in Ruff, strict Mypy, and dead-code checks.

## Implementation checklist

- [x] Remove the synthetic image benchmark and its runtime test path.
- [x] Remove the candidate/baseline comparison command and terminology.
- [x] Wrap the existing Ultralytics Make target without modifying its harness.
- [x] Capture before-and-after CPU, thermal, memory, cache, fault, and swap
  state.
- [x] Preserve benchmark output and exit status.
- [x] Record child-process peak RSS and write context atomically.
- [x] Document the workflow affirmatively and define every recorded category.
- [x] Pass the portable behavior and static-analysis suites.
- [x] Run the recorder around a real Ultralytics benchmark and inspect the
  resulting context alongside the benchmark output.
