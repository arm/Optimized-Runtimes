# YOLO26n performance baseline

This is the accepted performance reference for the runtime, converted model,
and `coco-128-by-sa` dataset identified below. Future performance-sensitive
changes must use the same dataset and validation configuration before comparing
their results with this measurement.

The measurement uses the unpublished Hugging Face dataset revision identified
below, which is the candidate intended for publication. Benchmark operators
with access must download that revision and supply its local validation
configuration explicitly with `DATA=/path/to/data.yaml`.

## Measurement

| Item | Value |
| --- | --- |
| Source revision | `c87e781ba97b353c99cd8fac2dd1995e0d2e43b6` |
| Runtime library SHA-256 | `6b9978243ed1fd325e5b7b6a2bc616280862ed01edc1f653ee55b088658b60b2` |
| Model manifest SHA-256 | `3723a972dc64782fe86577eabde3d02539e10c778ae5ef77b0c5b9bc9f0d1f54` |
| Dataset | `coco-128-by-sa` |
| Dataset repository | [Arm/coco-128-by-sa](https://huggingface.co/datasets/Arm/coco-128-by-sa) |
| Dataset Hub revision | [`f8af0a317c0b54c3dbd38c70b9757dccb8b02460`](https://huggingface.co/datasets/Arm/coco-128-by-sa/tree/f8af0a317c0b54c3dbd38c70b9757dccb8b02460) |
| Dataset images | 128 |
| Dataset instances | 1,050 |
| Validation configuration SHA-256 | `accc0ff00b364d39c5f209eeb7e53609db634541551b7d4a392b7aa7e1e5f62b` |
| Selection metadata SHA-256 | `042d13fd4cfe7d670a09604f4e437829420982effd12a55d91efa7f1d9fb28a4` |
| License manifest SHA-256 | `38e5daf4bd373f4dc887bb1817ddae0cbe607c788bb46195592a5d9bfacd521c` |
| COCO annotations SHA-256 | `3c756fbb846ccfcc71218b13b3fa9e274e87764e015346cc143e8a36c08b7388` |
| Ultralytics | 8.4.108 |
| Ultralytics options | `workers=0 plots=False` |
| Cache state | Cold; host filesystem caches dropped immediately before each run |
| Evidence directory | `20261020T182235078652000Z-c87e781ba97b` |
| Recorded | 2026-10-20 18:22 UTC |

`workers=0` keeps Ultralytics dataset loading in the benchmark process instead
of starting data-loader subprocesses. It does not change the native runtime's
four-worker execution configuration. `plots=False` prevents Ultralytics from
writing validation plots; it does not change detector scoring.

## Results

| Measurement | Value |
| --- | ---: |
| Precision | 0.769 |
| Recall | 0.577 |
| mAP50 | 0.6587 |
| mAP75 | 0.5290 |
| mAP50-95 | 0.4866 |
| Preprocess | 2.2 ms/image |
| Inference | 36.8 ms/image |
| Postprocess | 3.9 ms/image |

## Host context

The measurement ran on a Raspberry Pi 5 Model B Rev 1.1 with Linux
6.12.75+rpt-rpi-2712. All four cores were available, and the CPU frequency
policy remained on the `performance` governor at 2.4 GHz.

The host filesystem caches were dropped immediately before each benchmark.
For the recorded Arm Runtime run, cached and reclaimable data totaled 0.56 GB
at the initial observation and 0.87 GB afterward. Cold reads caused 1,119
major page faults; swap input and output did not change during the run.

The CPU temperature increased from 50.2 °C to 51.8 °C. The firmware reported
no current or historical throttling flags before or after the run. Available
memory decreased from 7.40 GB to 7.35 GB, and peak benchmark-process-tree RSS
was 694 MB.
