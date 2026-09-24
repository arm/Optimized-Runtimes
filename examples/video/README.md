# Prerecorded-video example

This example performs object detection on prerecorded video. Video decoding and
resizing complete before detection is timed, and annotated-video rendering runs
afterward so neither is included in the reported model performance.

The example writes:

- detections and per-frame timings to `examples/video/output/detections.json`;
- an annotated video to `examples/video/output/rendered.mp4`.

## Usage

### Prerequisites

Install the required packages:

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

### Using the sample video

The following command downloads a sample video and the pinned standard YOLO26n
checkpoint, converts the model, and runs detection on the first 200 frames:

```bash
make example-video
```

> The input is MediaPipe's object-detection `test_video.mp4`, Copyright 2019
> The MediaPipe Authors, licensed under Apache-2.0.

### Using your own video and model

The video must contain at least 200 decodable frames. Convert your model as
described in [Acquiring a model](../../README.md#acquiring-a-model), then run:

```bash
make example-video VIDEO=/path/to/video.mp4
```

To use a converted model outside the default `dist/models/` location, set
`MODEL` explicitly:

```bash
make example-video \
  VIDEO=/path/to/video.mp4 \
  MODEL=/path/to/converted-model
```
