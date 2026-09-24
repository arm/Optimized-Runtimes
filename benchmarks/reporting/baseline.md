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
| Source revision | `00cbd4412db2af127d6d84cef8f82ad10ede22de` |
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
| Evidence directory | `20260924T073434514455149Z-00cbd4412db2` |
| Recorded | 2026-09-24 07:34 UTC |

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
| Inference | 37.7 ms/image |
| Postprocess | 3.3 ms/image |

## Host context

The measurement ran on a Raspberry Pi 5 Model B Rev 1.1 with Linux
6.12.75+rpt-rpi-2712. All four cores were available, and the CPU frequency
policy remained on the `performance` governor at 2.4 GHz.

The CPU temperature increased from 47.95 °C to 54.55 °C. The firmware reported
no current or historical throttling flags before or after the run. Available
memory decreased from 7.66 GB to 7.57 GB, while filesystem cache remained about
3.52 GB. Swap was configured and already occupied, but the run caused no swap
input/output or major page faults. Peak benchmark-process-tree RSS was 474 MB.

## Legacy Ultralytics COCO128 result

The previous baseline used Ultralytics `coco128.yaml`, not
`coco-128-by-sa`. It contained 128 images and 929 instances, while the
accepted dataset contains 1,113 instances and only 31 image filenames overlap.
The results are retained for historical context but are not the reference for
new performance comparisons.

| Item | Legacy value |
| --- | --- |
| Source revision | `d41c7dc6dee5e605380d4bb8dae2b5882f94b69e` |
| Runtime library SHA-256 | `b8bb69c0f3b84fc6a0204c2821dec41f5e6aeb78a1ffb3e8c9e5fd8b827c46c5` |
| Model manifest SHA-256 | `3723a972dc64782fe86577eabde3d02539e10c778ae5ef77b0c5b9bc9f0d1f54` |
| Dataset | Ultralytics `coco128.yaml` (128 images, 929 instances) |
| Precision / recall | 0.738 / 0.538 |
| mAP50 / mAP50-95 | 0.645 / 0.479 |
| Preprocess / inference / postprocess | 2.3 / 38.4 / 2.6 ms/image |
| Recorded | 2026-09-22 00:25 UTC |
