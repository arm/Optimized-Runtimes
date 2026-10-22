# YOLO26n performance baseline

This is the accepted performance reference for the runtime, converted model,
and `coco-128-by-sa` dataset identified below. Future performance-sensitive
changes must use the same dataset and validation configuration before comparing
their results with this measurement.

The measurement uses the published Hugging Face dataset revision identified
below. Benchmark operators must download that revision and supply its local
validation configuration explicitly with `DATA=/path/to/data.yaml`.

## Measurement

| Item | Value |
| --- | --- |
| Source revision | `3d0f4123b5546a58aca9dfbd80534c1125d0a647` |
| Runtime library SHA-256 | `6b9978243ed1fd325e5b7b6a2bc616280862ed01edc1f653ee55b088658b60b2` |
| Model manifest SHA-256 | `3723a972dc64782fe86577eabde3d02539e10c778ae5ef77b0c5b9bc9f0d1f54` |
| Dataset | `coco-128-by-sa` |
| Dataset repository | [Arm/coco-128-by-sa](https://huggingface.co/datasets/Arm/coco-128-by-sa) |
| Dataset Hub revision | [`97d6ea181e9dd0784db880b705e426ad23d424e1`](https://huggingface.co/datasets/Arm/coco-128-by-sa/tree/97d6ea181e9dd0784db880b705e426ad23d424e1) |
| Dataset images | 128 |
| Dataset instances | 1,025 |
| Validation configuration SHA-256 | `accc0ff00b364d39c5f209eeb7e53609db634541551b7d4a392b7aa7e1e5f62b` |
| Selection metadata SHA-256 | `18205722c822b59635d4b6c911cef4495b5d688b4b23d84fde6d4e4396cec61f` |
| License manifest SHA-256 | `a56e45ab8e72a21906332dcfcfe8ba4ece9039c2c0cdc2409952042c57675b96` |
| COCO annotations SHA-256 | `009ef7e4cd3b533774d8755dbf8a3aa788d5af401b8283e87e27b2cad58cda57` |
| Ultralytics | 8.4.108 |
| Ultralytics options | `workers=0 plots=False` |
| Cache state | Cold; host filesystem caches dropped immediately before each run |
| Evidence directories | `20261022T-rebaseline-official-coco-cold1`, `20261022T-rebaseline-official-coco-cold2`, and `20261022T-rebaseline-official-coco-cold3` |
| Recorded | 2026-10-22 18:33–18:34 UTC |

`workers=0` keeps Ultralytics dataset loading in the benchmark process instead
of starting data-loader subprocesses. It does not change the native runtime's
four-worker execution configuration. `plots=False` prevents Ultralytics from
writing validation plots; it does not change detector scoring.

## Results

| Measurement | Value |
| --- | ---: |
| Precision | 0.7613 |
| Recall | 0.5805 |
| mAP50 | 0.6584 |
| mAP75 | 0.5302 |
| mAP50-95 | 0.4863 |
| Preprocess | 2.3 ms/image |
| Inference | 38.3 ms/image |
| Postprocess | 3.8 ms/image |

## Host context

The measurement ran on a Raspberry Pi 5 Model B Rev 1.1 with Linux
6.12.75+rpt-rpi-2712. All four cores were available, and the CPU frequency
policy remained on the `performance` governor at 2.4 GHz.

The host filesystem caches were dropped immediately before each benchmark.
For the median Arm Runtime run, cached and reclaimable data totaled 0.71 GiB
at the initial observation and 1.02 GiB afterward. Cold reads caused 1,289
major page faults; swap input and output did not change during the run.

The CPU temperature increased from 48.5 °C to 52.4 °C. The firmware reported
no current or historical throttling flags before or after the run. Available
memory decreased from 6.93 GiB to 6.87 GiB, and peak benchmark-process-tree RSS
was 444 MiB.
