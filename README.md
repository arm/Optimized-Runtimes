# Arm Runtime for YOLO26 on Raspberry Pi 5

This repository contains the source for the [Arm Runtime for YOLO26 on Raspberry Pi 5](arm.github.io/Arm-Runtime-for-YOLO26-on-Raspberry-Pi-5) **CHECK LINK BEFORE PUBLISHING**

## Quick Start
### Examples and Benchmarking
* **TODO**: list the steps to get the ultralytics benchmark
* **TODO**: List the steps to run the video benchmark
* * **TODO**: List the steps to run the picamera2 example

### Build from source
On Raspberry Pi 5:
```bash
make yolo26n-export \
  YOLO26N_RESOLUTION=640 \
  YOLO26N_CHECKPOINT=/path/to/yolo26n.pt
make yolo26n-build
make yolo26n-bundle YOLO26N_RESOLUTION=640
```
> For resolution choose one of `320`, `480`, `640`, `800`, `960`, or `1280`

The resulting verified static-resolution bundle is written to `dist/yolo26n-<resolution>/`

## About
For general information see [Arm Runtime for YOLO26 on Raspberry Pi 5](arm.github.io/Arm-Runtime-for-YOLO26-on-Raspberry-Pi-5)

## Layout
The project consists of:
- `compiler/` - A YOLO26n PyTorch checkpoint compiler, which validates and converts provided YOLO26n checkpoints into the format required by the runtime
- [needs folder] An XNNPACK-backed native runtime which executes the fixed YOLO26n graph
- [needs folder] Python tensor-level API
- [needs folder] Python image-level API
- `examples/video` - Python example which performs object detection on pre-recorded video
- `examples/picamera2` - Python example which performs object detection on live video from a Pi Camera
- `ultralytics/` - source and scripts to execute the Ultralytics benchmark

## Run through Ultralytics (this should probably get moved to the ultralytics folder)

Assemble the built YOLO26n bundle into the Arm Raspberry Pi 5
Ultralytics package:

```bash
make ultralytics-package
```

Install the pinned Python environment and repository-owned
`ArmYolo26Pi5Backend`, then benchmark the package directly:

```bash
make ultralytics-install \
  WORKSPACE=dist/arm-yolo26n-pi5-ultralytics
make ultralytics-benchmark \
  WORKSPACE=dist/arm-yolo26n-pi5-ultralytics
```

The installer verifies every package artifact, installs the source-tracked
native runtime wrapper, and registers the source-tracked backend with
Ultralytics 8.4.108. The backend is selected by the model-specific package
suffix. YOLO26n resolves the verified YOLO26n bundle from its descriptor. No
runtime environment variables are required.
Additional Ultralytics validation arguments may be appended to the benchmark
command.
