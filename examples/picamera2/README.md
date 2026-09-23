# Picamera2 examples

These programs connect a Raspberry Pi camera to the YOLO26n runtime.
`detect.py` is the bounded headless diagnostic. `live.py` adds a
hardware-accelerated DRM/KMS or QtGL preview with detections, rolling detector
throughput, and mean inference latency.

The application owns camera configuration, capture scheduling, and display.
`Yolo26Detector` owns model-specific preprocessing, native execution,
postprocessing, and class metadata. Both programs read the static square input
size from the converted model and request a 4:3 camera frame that needs only
padding, so a 480 model requests 480x360 and a 640 model requests 640x480.

## Raspberry Pi setup

Install the camera bindings on 64-bit Raspberry Pi OS Trixie:

```bash
sudo apt update
sudo apt install -y \
  python3-picamera2 python3-venv rpicam-apps

make examples
```

`make examples` obtains the released native runtime and converts the official
checkpoint locally without a native build. Picamera2 comes from Raspberry Pi OS
so its bindings match libcamera. Confirm camera access with
`rpicam-hello --list-cameras`.

## Headless detection

```bash
make example-camera CAMERA_ARGS="--frames 100 --print-detections"
```

Without `CAMERA_ARGS`, the example runs until interrupted. Select a different
camera with `CAMERA=1`.

## Live preview

On a Raspberry Pi desktop, use QtGL:

```bash
examples/.venv/bin/python -m examples.picamera2.live \
  --runtime "$PWD/dist/runtime/lib/libarm-yolo26n.so.0" \
  --model "$PWD/dist/models/yolo26n-640" \
  --preview qtgl
```

For an SSH process, forward the active desktop environment if needed:

```bash
DISPLAY=:0 WAYLAND_DISPLAY=wayland-0 XDG_RUNTIME_DIR="/run/user/$(id -u)" \
  examples/.venv/bin/python -m examples.picamera2.live \
  --runtime "$PWD/dist/runtime/lib/libarm-yolo26n.so.0" \
  --model "$PWD/dist/models/yolo26n-640" \
  --preview qtgl
```

Use `--preview auto` to let Picamera2 choose, or stop the graphical desktop and
use `--preview drm` for direct console display. Preview dimensions are UI
settings and do not change the model's detector resolution.

The status panel reports detector throughput, not the independently running
camera refresh rate:

```text
Detect <live> FPS  |  Inference <live> ms
```

Additional camera patterns are in the [official Picamera2 examples](https://github.com/raspberrypi/picamera2/tree/main/examples)
and [Picamera2 manual](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf).
