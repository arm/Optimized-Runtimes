# YOLO26n performance baseline

This is the accepted performance reference for the runtime, converted model,
and `coco-128-by-sa` dataset identified below. Future performance-sensitive
changes must use the same dataset and validation configuration before comparing
their results with this measurement.

The dataset is not yet published. Until it is available through a stable public
location, benchmark operators must supply its local validation configuration
explicitly with `DATA=/path/to/data-validation-all.yaml`. The default
`coco128.yaml` benchmark is a convenience workflow and is not the accepted
performance workload.

## Measurement

| Item | Value |
| --- | --- |
| Source revision | `7ee806b34ca7cf06e1c90b0a99160a5ad4b14350` |
| Runtime library SHA-256 | `6b9978243ed1fd325e5b7b6a2bc616280862ed01edc1f653ee55b088658b60b2` |
| Model manifest SHA-256 | `3723a972dc64782fe86577eabde3d02539e10c778ae5ef77b0c5b9bc9f0d1f54` |
| Dataset | `coco-128-by-sa` |
| Dataset images | 128 |
| Dataset instances | 1,113 |
| Dataset source-tree identity | `6c690e5f4fbab7295ae47be2e6730129236aee18` |
| Validation configuration SHA-256 | `771a999dce2c4752b6506164640765ab099e21be48249f9fee00b9ce7b185cdc` |
| Selection metadata SHA-256 | `8b9076231e554966413db4a6fe9ddd070e2d9852223349cfedb06f58617b25a3` |
| License manifest SHA-256 | `695761f930ec77c416b8b8453f549fe38016fda158dc5e50af59c8b5c0bd0c8e` |
| Ultralytics | 8.4.108 |
| Ultralytics options | `workers=0 plots=False` |
| Cache state | Cold; host filesystem caches dropped immediately before the run |
| Evidence directory | `20260924T074313100023210Z-7ee806b34ca7` |
| Recorded | 2026-09-24 07:43 UTC |

`workers=0` keeps Ultralytics dataset loading in the benchmark process instead
of starting data-loader subprocesses. It does not change the native runtime's
four-worker execution configuration. `plots=False` prevents Ultralytics from
writing validation plots; it does not change detector scoring.

## Results

| Measurement | Value |
| --- | ---: |
| Precision | 0.752 |
| Recall | 0.544 |
| mAP50 | 0.633 |
| mAP50-95 | 0.468 |
| Preprocess | 2.1 ms/image |
| Inference | 36.4 ms/image |
| Postprocess | 4.1 ms/image |

## Host context

The measurement ran on a Raspberry Pi 5 Model B Rev 1.1 with Linux
6.12.75+rpt-rpi-2712. All four cores were available, and the CPU frequency
policy remained on the `performance` governor at 2.4 GHz.

The host filesystem caches were dropped immediately before starting the
benchmark. At the benchmark's initial observation, cached and reclaimable data
totaled 0.40 GB; it increased to 0.70 GB during the run. The resulting cold
reads caused 1,079 major page faults, while swap input and output did not
change.

The CPU temperature increased from 47.4 °C to 52.9 °C. The firmware reported
no current or historical throttling flags before or after the run. Available
memory decreased from 7.70 GB to 7.61 GB, and peak benchmark-process-tree RSS
was 465 MB.
