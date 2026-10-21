---
license: apache-2.0
pipeline_tag: object-detection
base_model: yolo26n
tags:
  - arm
  - arm-optimized
  - edge-linux
  - raspberry-pi-5
  - xnnpack
---

# Arm Runtime for YOLO26 on Raspberry Pi 5

## About

This runtime runs YOLO26n on Raspberry Pi 5 with Raspberry Pi OS Trixie. It is
a fixed-graph implementation that uses model-specific memory planning, custom
operators, and operator fusion to maximize throughput on the Raspberry Pi CPU.

The runtime supports the standard YOLO26n model and compatible fine-tuned
variants at square resolutions of 320, 480, 640, 800, 960, and 1280 pixels.

## Performance

The following YOLO26n results were measured on Raspberry Pi 5 at 640x640,
batch 1, against the 128-image, 1,050-instance
[`coco-128-by-sa`](https://huggingface.co/datasets/Arm/coco-128-by-sa/tree/e7a3ecd065a304caddd2f170b8d41aeda66735b5)
dataset revision selected for publication. Inference time excludes
preprocessing and postprocessing.

| Runtime | Inference (ms/image) | mAP50-95 |
| --- | ---: | ---: |
| Arm Runtime (this project) | 36.80 | 0.4866 |
| NCNN | 66.23 | 0.4899 |
| MNN | 90.38 | 0.4711 |
| OpenVINO | 102.56 | 0.4787 |
| ONNX Runtime | 127.29 | 0.4787 |
| ExecuTorch | 137.90 | 0.4893 |

See the [performance baseline](benchmarks/reporting/baseline.md) for the Arm
Runtime measurement and host context.

## Quick start

### Prerequisites

On Raspberry Pi OS Trixie, install the required packages:

```bash
sudo apt update
sudo apt install --yes --no-install-recommends \
  ca-certificates \
  curl \
  git \
  make \
  python3 \
  python3-venv \
  tar \
  gzip
```

Clone the repository:

```bash
git clone https://github.com/arm/Optimized-Runtimes.git
cd Optimized-Runtimes/yolo26-raspberry-pi-5
```

### Run an example

Run object detection on a sample video using the standard YOLO26n checkpoint:

```bash
make example-video
```

For a fine-tuned checkpoint, your own video, or live camera input, see
[Usage](#usage).

### Run the benchmark

Run the [Ultralytics benchmark harness](https://docs.ultralytics.com/guides/raspberry-pi/#raspberry-pi-5-yolo26-benchmarks)
using this runtime:

```bash
make ultralytics-benchmark
```

See the [Ultralytics benchmark documentation](benchmarks/ultralytics/README.md)
for configuration options.

For now, the no-argument command uses Ultralytics COCO128 as an integration
check; it does not reproduce the performance table above. Those measurements
use the pinned `coco-128-by-sa` revision, supplied explicitly while that
dataset is not publicly available.

## Usage

### Acquiring a runtime

A prebuilt runtime library is available from this repository's releases. The
example and benchmark targets download the release matching the checked-out
project version when no runtime exists under `dist/runtime/`.

To build the runtime from source, install the native build prerequisites:

```bash
sudo apt update
sudo apt install --yes --no-install-recommends \
  build-essential \
  ca-certificates \
  cmake \
  git \
  patch \
  python3 \
  python3-venv \
  tar \
  gzip
```

Then run:

```bash
make runtime
```

The resulting runtime is installed under `dist/runtime/`, where the examples
and benchmark consume it automatically.

### Acquiring a model

The examples and benchmark require a YOLO26n checkpoint converted for this
runtime. If the default model is absent, their Make targets download the pinned
standard YOLO26n checkpoint and convert it automatically.

The benchmark normally uses the standard model because its classes match the
default COCO dataset. To use a compatible fine-tuned checkpoint with the video
or camera examples, convert it first:

```bash
make model CHECKPOINT=/path/to/your-checkpoint.pt
```

The default conversion uses 640x640 input and writes the model under
`dist/models/`. Select another supported resolution with, for example:

```bash
make model CHECKPOINT=/path/to/your-checkpoint.pt RESOLUTION=320
```

## Examples and benchmarks

The examples use the [Python API](src/runtime/api/python/arm_yolo26n/README.md).
Their Make targets prepare missing default runtime and model artifacts
automatically.

- [Object detection on prerecorded video](examples/video/README.md)
- [Object detection on live video from a Raspberry Pi Camera](examples/picamera2/README.md)
- [Ultralytics benchmark](benchmarks/ultralytics/README.md)

## Other information

Run `make help` for other useful commands.

### Project layout

- `src/converter/` - Converts compatible YOLO26n PyTorch checkpoints into the runtime model format.
- `src/runtime/` - Implements the fixed YOLO26n runtime.
- `src/runtime/patches/` - Contains project changes to XNNPACK and pthreadpool.
- `src/runtime/api/python/arm_yolo26n/` - Provides the Python API.
- `examples/` - Contains prerecorded-video and live-camera examples.
- `benchmarks/ultralytics/` - Integrates the runtime with the Ultralytics benchmark.
