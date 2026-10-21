# Picamera2 example

This example performs object detection on live video from a connected Raspberry
Pi Camera. Its reported pipeline FPS includes camera capture and detection. Both
headless and preview modes request a 30 FPS camera stream by default.

The camera examples run until interrupted and do not record video. Pass
`CAMERA_ARGS="--frames 100"` to stop after a fixed number of detections.

## Usage

Run these commands from the `yolo26-raspberry-pi-5/` runtime directory.

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
  python3-picamera2 \
  rpicam-apps \
  tar \
  gzip
```

Confirm camera access:

```bash
rpicam-hello --list-cameras
```

To use your own compatible model, see [Acquiring a model](../../README.md#acquiring-a-model).
Otherwise, the pinned standard YOLO26n checkpoint is downloaded and converted
automatically.

### Live camera without a preview

Run detection without rendering a preview:

```bash
make example-camera
```

This mode avoids display and overlay-rendering overhead. Select another camera
with `CAMERA=1`, and pass additional application options through `CAMERA_ARGS`:

```bash
make example-camera CAMERA=1 CAMERA_ARGS="--frames 100 --print-detections"
```

### Live camera preview

Run from a terminal in an active desktop session:

```bash
make example-camera CAMERA_PREVIEW=qtgl
```

Rendering the annotated preview adds performance overhead. Preview dimensions
do not change the model's detector resolution.

When connecting over SSH to an existing desktop session, the desktop user and
SSH user must be the same:

```bash
DISPLAY=:0 WAYLAND_DISPLAY=wayland-0 XDG_RUNTIME_DIR="/run/user/$(id -u)" \
  make example-camera CAMERA_PREVIEW=qtgl
```

Use `CAMERA_PREVIEW=auto` to let Picamera2 select a preview backend. Stop the
graphical desktop and use `CAMERA_PREVIEW=drm` for direct console display.
