#!/usr/bin/env python3
"""Display live YOLO26n detections over a Picamera2 preview."""

from __future__ import annotations

import argparse
import os
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from arm_yolo26n import DetectorConfig, Yolo26Detector
from examples.picamera2.overlay import (
    LatestOverlayUploader,
    OverlayRenderer,
    RollingPerformance,
)
from picamera2 import MappedArray, Picamera2, Preview

_PREVIEWS = {"auto": True, "drm": Preview.DRM, "qtgl": Preview.QTGL}


@dataclass(frozen=True, slots=True)
class _Arguments:
    runtime: Path
    model: Path
    camera: int
    framerate: float
    preview: str
    preview_width: int
    preview_height: int
    fps_window: float
    hide_confidence: bool
    frame_limit: int


@dataclass(slots=True)
class _Metrics:
    frames: int = 0
    capture_wait_seconds: float = 0.0
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    pipeline_seconds: float = 0.0
    overlay_render_mean_ms: float = 0.0
    overlay_upload_mean_ms: float = 0.0
    overlay_uploads: int = 0

    def report(self) -> str:
        return (
            f"frames={self.frames} "
            f"capture_wait_mean_ms={self.capture_wait_seconds * 1000 / self.frames:.2f} "
            f"preprocess_mean_ms={self.preprocess_ms / self.frames:.2f} "
            f"inference_mean_ms={self.inference_ms / self.frames:.2f} "
            f"postprocess_mean_ms={self.postprocess_ms / self.frames:.2f} "
            f"overlay_render_mean_ms={self.overlay_render_mean_ms:.2f} "
            f"overlay_upload_mean_ms={self.overlay_upload_mean_ms:.2f} "
            f"overlay_uploads={self.overlay_uploads} "
            f"pipeline_fps={self.frames / self.pipeline_seconds:.2f}"
        )


def _parse_args() -> tuple[_Arguments, DetectorConfig]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--framerate", type=float, default=30.0)
    parser.add_argument("--preview", choices=tuple(_PREVIEWS), default="auto")
    parser.add_argument("--preview-width", type=int, default=1280)
    parser.add_argument("--preview-height", type=int, default=960)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--fps-window", type=float, default=1.5)
    parser.add_argument("--hide-confidence", action="store_true")
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="stop after this many detections; 0 runs until interrupted",
    )
    args = parser.parse_args()
    if args.frames < 0:
        parser.error("--frames must be non-negative")
    if args.framerate <= 0:
        parser.error("--framerate must be positive")
    if args.preview_width <= 0 or args.preview_height <= 0:
        parser.error("preview dimensions must be positive")
    if args.fps_window <= 0:
        parser.error("--fps-window must be positive")
    try:
        detector_config = DetectorConfig(
            confidence=args.confidence,
            iou=args.iou,
            max_detections=args.max_detections,
        )
    except ValueError as error:
        parser.error(str(error))
    return (
        _Arguments(
            runtime=args.runtime,
            model=args.model,
            camera=args.camera,
            framerate=args.framerate,
            preview=args.preview,
            preview_width=args.preview_width,
            preview_height=args.preview_height,
            fps_window=args.fps_window,
            hide_confidence=args.hide_confidence,
            frame_limit=args.frames,
        ),
        detector_config,
    )


def _configure_camera(camera: Picamera2, args: _Arguments, resolution: int) -> tuple[int, int]:
    capture_size = (
        resolution,
        resolution * 3 // 4,
    )
    camera.configure(
        camera.create_video_configuration(
            # A 4:3 frame needs padding rather than resizing. Native
            # preprocessing ignores the XRGB padding byte.
            main={"size": capture_size, "format": "XRGB8888"},
            controls={"FrameRate": args.framerate},
            # Wait for the requested frame rather than a retained older frame.
            queue=False,
        )
    )
    return capture_size


def _uses_texture_overlay(preview: str) -> bool:
    return preview == "qtgl" or (preview == "auto" and os.getenv("DISPLAY", "").startswith(":"))


def _install_frame_overlay(camera: Picamera2, renderer: OverlayRenderer) -> None:
    def draw_overlay(request: object) -> None:
        # capture_array copies the frame before Picamera2 invokes this callback,
        # so annotations never become detector input.
        with MappedArray(request, "main") as mapped:
            renderer.draw(mapped.array)

    camera.post_callback = draw_overlay


def _run_detection_loop(
    camera: Picamera2,
    detector: Yolo26Detector,
    renderer: OverlayRenderer,
    uploader: LatestOverlayUploader | None,
    args: _Arguments,
) -> _Metrics:
    metrics = _Metrics()
    performance = RollingPerformance(args.fps_window)
    started_at = perf_counter()
    try:
        capture_job = camera.capture_array("main", wait=False)
        while True:
            wait_started_at = perf_counter()
            frame_bgrx = camera.wait(capture_job)
            metrics.capture_wait_seconds += perf_counter() - wait_started_at
            final_frame = bool(args.frame_limit and metrics.frames + 1 >= args.frame_limit)
            if not final_frame:
                capture_job = camera.capture_array("main", wait=False)
            result = detector.detect(frame_bgrx, pixel_format="BGRX")
            performance.add(perf_counter(), result.timings.inference_ms)
            metrics.preprocess_ms += result.timings.preprocess_ms
            metrics.inference_ms += result.timings.inference_ms
            metrics.postprocess_ms += result.timings.postprocess_ms
            metrics.frames += 1
            renderer.update(result.detections, performance)
            if uploader is not None:
                uploader.request()
            if final_frame:
                break
    except KeyboardInterrupt:
        pass
    finally:
        metrics.pipeline_seconds = perf_counter() - started_at
        metrics.overlay_render_mean_ms = renderer.render_mean_ms
        if uploader is not None:
            metrics.overlay_upload_mean_ms = uploader.upload_mean_ms
            metrics.overlay_uploads = uploader.upload_count
    return metrics


def _run(args: _Arguments, detector_config: DetectorConfig) -> _Metrics:
    with ExitStack() as resources:
        detector = resources.enter_context(
            Yolo26Detector(runtime=args.runtime, model=args.model, config=detector_config)
        )
        camera = Picamera2(args.camera)
        resources.callback(camera.close)
        capture_width, capture_height = _configure_camera(camera, args, detector.input_shape[2])
        camera.start_preview(
            _PREVIEWS[args.preview],
            width=args.preview_width,
            height=args.preview_height,
        )
        resources.callback(camera.stop_preview)
        renderer = OverlayRenderer(
            width=capture_width,
            height=capture_height,
            show_confidence=not args.hide_confidence,
        )
        texture_overlay = _uses_texture_overlay(args.preview)
        if not texture_overlay:
            _install_frame_overlay(camera, renderer)
        camera.start()
        resources.callback(camera.stop)
        uploader = None
        if texture_overlay:
            uploader = LatestOverlayUploader(
                renderer,
                args.preview_width,
                args.preview_height,
                camera.set_overlay,
            )
            resources.callback(uploader.close)
        return _run_detection_loop(camera, detector, renderer, uploader, args)


def main() -> int:
    args, detector_config = _parse_args()
    metrics = _run(args, detector_config)
    if metrics.frames and metrics.pipeline_seconds > 0.0:
        print(metrics.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
