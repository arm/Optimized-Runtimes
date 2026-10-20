# YOLO26n performance baseline

This is the accepted performance reference for the runtime, converted model,
and `coco-128-by-sa` dataset identified below. Future performance-sensitive
changes must use the same dataset and validation configuration before comparing
their results with this measurement.

The measurement uses the unpublished Hugging Face dataset revision identified
below, which is the candidate intended for publication. Benchmark operators
with access must download that revision and supply its local validation
configuration explicitly with `DATA=/path/to/data.yaml`. The default
`coco128.yaml` benchmark is a convenience workflow and is not the accepted
performance workload.

## Measurement

| Item | Value |
| --- | --- |
| Source revision | `fefc40c532702f13052b584c8a83232a0896439d` |
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
| Cache state | Warm; no major page faults during the run |
| Evidence directory | `20261020T181348287318000Z-fefc40c53270` |
| Recorded | 2026-10-20 18:13 UTC |

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
| Preprocess | 2.4 ms/image |
| Inference | 43.5 ms/image |
| Postprocess | 3.4 ms/image |

Three consecutive Arm Runtime validation runs measured 43.04, 41.61, and
43.54 ms/image, for a median of 43.04 ms/image. The comparison table in the
root README reports that median; the evidence directory above contains the
last run.

## Runtime comparison

Each format was exported from the same standard YOLO26n checkpoint and
validated with Ultralytics 8.4.108 using the dataset and validation settings
above. The reported inference time excludes preprocessing and postprocessing.

| Runtime | Version | Inference (ms/image) | mAP50-95 |
| --- | --- | ---: | ---: |
| Arm Runtime | 0.1.0 | 43.04 | 0.4866 |
| NCNN | 1.0.20260526 | 80.72 | 0.4899 |
| MNN | 3.6.1 | 99.96 | 0.4711 |
| OpenVINO | 2026.4.1 | 111.75 | 0.4787 |
| ONNX Runtime | 1.30.0 | 143.05 | 0.4787 |
| ExecuTorch | 1.3.1+cpu | 158.24 | 0.4893 |

## Host context

The measurement ran on a Raspberry Pi 5 Model B Rev 1.1 with Linux
6.12.75+rpt-rpi-2712. All four cores were available, and the CPU frequency
policy remained on the `performance` governor at 2.4 GHz.

The host filesystem caches could not be dropped in this environment. Cached
and reclaimable data totaled 6.27 GB before and after the run. No major page
faults occurred, and swap input and output did not change during the run.

The CPU temperature increased from 51.3 °C to 54.0 °C. The firmware reported
no current or historical throttling flags before or after the run. Available
memory remained about 7.36 GB, and peak benchmark-process-tree RSS was 468 MB.
