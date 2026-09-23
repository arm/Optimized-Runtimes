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

## Run through Ultralytics

Run validation with a selected runtime and converted model:

```bash
make ultralytics-benchmark \
  RUNTIME="$PWD/dist/runtime/lib/libarm-yolo26n.so.0" \
  MODEL="$PWD/dist/models/yolo26n-640"
```

The command prepares the pinned Ultralytics environment and repository-owned
backend when needed, then records benchmark evidence under
`benchmarks/reporting/output/`. Additional Ultralytics validation arguments may
be supplied through `YOLO_ARGS`.
