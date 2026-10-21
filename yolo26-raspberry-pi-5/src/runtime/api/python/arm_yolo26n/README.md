# Arm YOLO26n Python APIs

This package provides two intentionally separate interfaces to the fixed
YOLO26n runtime:

- `Yolo26nTensorSession` is the low-level interface for callers that already
  own preprocessing and postprocessing. It accepts and returns FP32 NumPy
  arrays.
- `Yolo26Detector` is the high-level image interface. It accepts uint8 NumPy
  images and returns class-labelled detections in source-image coordinates.

Both interfaces accept the runtime library and converted model directory as
separate, explicit inputs. The package validates the model and tensor contract
before creating the native model handle. It reads `model.bin` once, verifies
those bytes against the manifest, and passes them to the native runtime.

## Raw tensor execution

Use `Yolo26nTensorSession` when integrating with an existing tensor pipeline:

```python
import numpy as np

from arm_yolo26n import Yolo26nTensorSession

with Yolo26nTensorSession(
    runtime="/path/to/libarm-yolo26n.so.0",
    model="/path/to/converted-model",
) as session:
    model_input = np.random.default_rng().random(session.input_shape, dtype=np.float32)
    raw_prediction = session.execute(model_input)
```

Inputs must be contiguous or convertible to contiguous FP32 NCHW arrays with
the exact shape recorded by the model. Outputs are FP32 NumPy arrays with the
model's validated YOLO26n output shape.

## Image detection

Use `Yolo26Detector` when starting with an image:

```python
from arm_yolo26n import DetectorConfig, Yolo26Detector

config = DetectorConfig(confidence=0.25, iou=0.45)
with Yolo26Detector(
    runtime="/path/to/libarm-yolo26n.so.0",
    model="/path/to/converted-model",
    config=config,
) as detector:
    result = detector.detect(frame_bgr, pixel_format="BGR")
    for detection in result.detections:
        print(detection.class_name, detection.confidence, detection.box)
```

The detector handles letterboxing, channel conversion, normalization, native
execution, non-maximum suppression, and scaling boxes back to source-image
coordinates. It accepts interleaved `BGR`, `RGB`, and four-channel `BGRX`
arrays. The `BGRX` path lets Picamera2 `XRGB8888` buffers be consumed without
repacking them into a second three-channel image.

Sessions and detectors own native resources and are intentionally
non-reentrant. Use them as context managers, and use independent processes
when concurrent isolation is required.

See the [prerecorded-video example](../../../../../examples/video/README.md) for repeatable
workload benchmarking and the [Picamera2 example](../../../../../examples/picamera2/README.md)
for live camera integration.
