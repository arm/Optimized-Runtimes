# Prerecorded-video example

This example measures YOLO26n throughput on repeatable prerecorded footage.
Use it with your own video to estimate model performance without camera capture
overhead. The repository also uses the same workload for controlled internal
validation.

On a Raspberry Pi 5 running 64-bit Raspberry Pi OS Trixie, a repository
checkout is enough to run the complete example:

```bash
make examples
make example-video
```

`make examples` creates the example environment, obtains a runtime and the
official YOLO26n checkpoint, and converts the checkpoint locally. It does not
run an example. `make example-video` obtains the test video, runs inference,
and writes `examples/video/output/detections.json`. Neither command compiles
native code or installs Torch or Ultralytics.

The commands below expose the individual video preparation, detection, and
rendering stages when you want to substitute your own runtime, model, or footage.

The input is MediaPipe's object-detection `test_video.mp4`, Copyright 2019 The
MediaPipe Authors, licensed under Apache-2.0. The `example-video` Make target
downloads the pinned revision and verifies its SHA-256 identity. To use your
own video, provide its path explicitly:

```bash
make example-video VIDEO=/path/to/video.mp4
```

Run a converted model with the installed runtime:

```bash
examples/.venv/bin/python -m examples.video.detect \
  --runtime dist/runtime/lib/libarm-yolo26n.so.0 \
  --model dist/models/yolo26n-640 \
  --video downloads/media/test-video.mp4 \
  --output examples/video/output/detections.json \
  --frames 200
```

Decode and resize complete before timing, and one inference warms each runtime.
The example derives frame width from the selected model rather than
accepting an independent size override. Its summary reports decode-and-resize
time separately from preprocessing, inference, and postprocessing. JSON
serialization and optional rendering remain outside the model interval.

```bash
examples/.venv/bin/python -m examples.video.render \
  --video downloads/media/test-video.mp4 \
  --detections examples/video/output/detections.json \
  --output examples/video/output/detections.mp4
```

The renderer reports video decode, overlay rendering, and video encoding as
separate per-frame measurements. This keeps model throughput comparable across
workloads without hiding the costs of producing a viewable result.
