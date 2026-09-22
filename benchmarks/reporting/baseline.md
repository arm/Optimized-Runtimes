# YOLO26n performance baseline

This is the accepted performance reference for the runtime and converted model
identified below. Future performance-sensitive changes can compare their
results with this measurement. The comparison and its acceptance remain review
decisions; this result is not an automated threshold.

## Measurement

| Item | Value |
| --- | --- |
| Source revision | `d41c7dc6dee5e605380d4bb8dae2b5882f94b69e` |
| Runtime library SHA-256 | `b8bb69c0f3b84fc6a0204c2821dec41f5e6aeb78a1ffb3e8c9e5fd8b827c46c5` |
| Model manifest SHA-256 | `3723a972dc64782fe86577eabde3d02539e10c778ae5ef77b0c5b9bc9f0d1f54` |
| Dataset | `coco128.yaml` (128 images, 929 instances) |
| Ultralytics | 8.4.108 |
| Ultralytics options | `workers=0 plots=False` |
| Recorded | 2026-09-22 00:25 UTC |

`workers=0` keeps Ultralytics dataset loading in the benchmark process instead
of starting data-loader subprocesses. It does not change the native runtime's
four-worker execution configuration. `plots=False` prevents Ultralytics from
writing validation plots; it does not change detector scoring.

## Results

| Measurement | Value |
| --- | ---: |
| Precision | 0.738 |
| Recall | 0.538 |
| mAP50 | 0.645 |
| mAP50-95 | 0.479 |
| Preprocess | 2.3 ms/image |
| Inference | 38.4 ms/image |
| Postprocess | 2.6 ms/image |

## Host context

The measurement ran on a Raspberry Pi 5 Model B Rev 1.1 with Linux
6.12.47+rpt-rpi-2712. All four cores were available, and the CPU frequency
policy remained on the `performance` governor at 2.4 GHz.

The CPU temperature increased from 58.95 °C to 71.05 °C. The firmware reported
no current throttling before or after the run. Filesystem cache was approximately
1.75 GB, available memory decreased from 15.53 GB to 15.40 GB, and the run
caused no major page faults or swap input/output. Peak benchmark-process-tree
RSS was 467 MB.

This baseline was recorded after merging
[the build and artifact layout change](https://github.com/Arm-Debug/yolo-raspi-research/pull/24).
