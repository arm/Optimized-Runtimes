# Recording Ultralytics benchmark evidence

This package records each Ultralytics benchmark together with the runtime,
model, source revision, and Raspberry Pi host conditions needed to interpret
the result.

## Usage

Reporting runs automatically through the Make target. See the
[Ultralytics benchmark documentation](../ultralytics/README.md) for benchmark
options.

By default, each invocation creates a UTC-timestamped directory under
`benchmarks/reporting/output/`. Set
`BENCHMARK_OUTPUT=/path/to/new-directory` to choose a specific new location.
The recorder never replaces an existing result.

## Recorded data

Each evidence directory contains:

- `benchmark.log` - Combined Ultralytics standard output and standard error,
  including accuracy metrics and stage timings.
- `results.json` - The command outcome, source revision, runtime identity,
  complete model manifest, and host observations from before and after the
  benchmark.

The host observations cover CPU policy, temperature, throttling, available
memory, filesystem cache, major page faults, and swap activity.

Before validation, the recorder observes one idle second and checks the model,
runtime, and local dataset paths. It stops rather than recording a distorted
performance result when those paths are RAM-backed or when the host shows low
available memory, excessive dirty/writeback data, file-cache refaults, active
reclaim scanning, or swap I/O. The recorder does not clear caches or change CPU
policy; preparing those conditions remains the benchmark operator's
responsibility. Successful preflight observations are included in
`results.json`.

Every performance comparison must name its dataset and record the dataset
Hub revision, validation-configuration SHA-256, image count, and instance
count. The accepted baseline uses the pinned, unpublished candidate revision
of `coco-128-by-sa` intended for publication. It is supplied explicitly during
qualification and is not the benchmark target's default dataset. The exact
revision is recorded in [`baseline.md`](baseline.md).

The accepted baseline is recorded in [`baseline.md`](baseline.md). Update it
only when review establishes a new accepted baseline result.

For a performance-sensitive pull request, summarize the reference and current
evidence with [`performance-report-template.md`](performance-report-template.md)
and attach both evidence directories to the pull request.
