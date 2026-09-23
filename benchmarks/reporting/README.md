# Recording Ultralytics benchmark evidence

This package records Raspberry Pi host state immediately before and after the
existing Ultralytics benchmark. The benchmark continues to select the dataset,
run validation, and report accuracy and stage timings in its normal format. Its
output is retained with the source revision and measured runtime and model
identities so a review does not depend on terminal history.

Run the benchmark with a new evidence directory for each result:

```sh
make ultralytics-benchmark \
  RUNTIME=/path/to/libarm-yolo26n.so.0 \
  MODEL=/path/to/converted-model \
  DATA=/path/to/dataset.yaml
```

By default, Make writes each run to a new UTC-timestamped directory under
`benchmarks/reporting/output/`, with the source revision included in its name.
Set `BENCHMARK_OUTPUT=/path/to/new-directory` when a specific evidence location
is required. The recorder never replaces an existing result.

The Ultralytics benchmark command passes its unchanged validation process to
`record_benchmark`. The recorder continues to display that output while
publishing:

- `benchmark.log`, containing the combined Ultralytics standard output and
  standard error;
- `model-manifest.json`, identifying the model and source checkpoint;
- `revision.txt`, containing the measured source revision; and
- `context.json`, containing the command outcome and host observations.

The context contains:

- the executed command, runtime identity, timestamps, exit status, and child-process peak RSS;
- Raspberry Pi model, kernel, architecture, CPU affinity, and load average;
- CPU frequency policy, thermal-zone readings, and firmware throttling flags;
- `MemAvailable`, `Cached`, and `SReclaimable` from `/proc/meminfo`;
- configured and free swap; and
- major page faults and swap input/output counters from `/proc/vmstat`.

The before and after values show whether memory availability, filesystem cache,
swap activity, CPU state, or temperature changed during the benchmark. Record
the reference and current revisions separately, then summarize them using
[`performance-report-template.md`](performance-report-template.md). The report
presents evidence for human review and does not impose an acceptance threshold.

For a performance-sensitive pull request, put the completed comparison in the
pull request and attach both evidence directories. Do not add a separate report
file for each change.

Once established, the repository keeps one accepted baseline at
`benchmarks/reporting/baseline.md`. Update it only when review establishes a
new reference for a changed runtime or model; changes that leave the measured
artifact unchanged leave the baseline unchanged. Git history preserves earlier
baselines, while closed pull requests preserve their individual comparisons.
