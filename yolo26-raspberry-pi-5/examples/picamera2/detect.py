#!/usr/bin/env python3
"""Run headless YOLO26n detection on frames captured through Picamera2."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from arm_yolo26n import DetectionResult, Yolo26Detector
from picamera2 import MappedArray, Picamera2


@dataclass(frozen=True, slots=True)
class _Arguments:
    runtime: Path
    model: Path
    camera: int
    framerate: float
    frame_limit: int
    print_detections: bool


@dataclass(slots=True)
class _Metrics:
    frames: int = 0
    detections: int = 0
    capture_wait_seconds: float = 0.0
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    pipeline_seconds: float = 0.0

    def add(self, result: DetectionResult) -> None:
        self.frames += 1
        self.detections += len(result.detections)
        self.preprocess_ms += result.timings.preprocess_ms
        self.inference_ms += result.timings.inference_ms
        self.postprocess_ms += result.timings.postprocess_ms

    def report(self) -> str:
        return (
            f"frames={self.frames} detections={self.detections} "
            f"capture_wait_mean_ms={self.capture_wait_seconds * 1000 / self.frames:.2f} "
            f"preprocess_mean_ms={self.preprocess_ms / self.frames:.2f} "
            f"inference_mean_ms={self.inference_ms / self.frames:.2f} "
            f"postprocess_mean_ms={self.postprocess_ms / self.frames:.2f} "
            f"pipeline_fps={self.frames / self.pipeline_seconds:.2f}"
        )


def _parse_args() -> _Arguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--framerate",
        type=float,
        default=30.0,
        help="camera frame rate",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="stop after this many frames; 0 runs until interrupted",
    )
    parser.add_argument("--print-detections", action="store_true")
    args = parser.parse_args()
    if args.frames < 0:
        parser.error("--frames must be non-negative")
    if args.framerate <= 0:
        parser.error("--framerate must be positive")
    return _Arguments(
        runtime=args.runtime,
        model=args.model,
        camera=args.camera,
        framerate=args.framerate,
        frame_limit=args.frames,
        print_detections=args.print_detections,
    )


def _configure_camera(camera: Picamera2, resolution: int, framerate: float) -> None:
    capture_size = (
        resolution,
        resolution * 3 // 4,
    )
    camera.configure(
        camera.create_video_configuration(
            # A 4:3 frame needs padding rather than resizing. Picamera2's
            # RGB888 NumPy representation is BGR byte order.
            main={"size": capture_size, "format": "RGB888"},
            controls={"FrameRate": framerate},
            # Wait for the requested frame rather than a retained older frame.
            queue=False,
        )
    )


def _detect_request(
    detector: Yolo26Detector,
    request: object,
) -> DetectionResult:
    with ExitStack() as resources:
        resources.callback(request.release)
        mapped = resources.enter_context(MappedArray(request, "main"))
        return detector.detect(
            mapped.array,
            pixel_format="BGR",
            on_frame_consumed=resources.close,
        )


def _run_pipeline(
    camera: Picamera2,
    detector: Yolo26Detector,
    args: _Arguments,
) -> _Metrics:
    metrics = _Metrics()
    camera.start()
    started_at = perf_counter()
    try:
        # Keep exactly one request pending so capture overlaps inference without
        # creating an application-side frame queue.
        capture_job = camera.capture_request(wait=False)
        while True:
            wait_started_at = perf_counter()
            request = camera.wait(capture_job)
            metrics.capture_wait_seconds += perf_counter() - wait_started_at
            final_frame = bool(args.frame_limit and metrics.frames + 1 >= args.frame_limit)
            if not final_frame:
                capture_job = camera.capture_request(wait=False)
            result = _detect_request(detector, request)
            metrics.add(result)
            if args.print_detections:
                print(result.detections)
            if final_frame:
                break
    except KeyboardInterrupt:
        pass
    finally:
        metrics.pipeline_seconds = perf_counter() - started_at
        camera.stop()
    return metrics


def main() -> int:
    args = _parse_args()
    with Yolo26Detector(runtime=args.runtime, model=args.model) as detector:
        camera = Picamera2(args.camera)
        try:
            _configure_camera(camera, detector.input_shape[2], args.framerate)
            metrics = _run_pipeline(camera, detector, args)
        finally:
            camera.close()
    if metrics.frames and metrics.pipeline_seconds > 0.0:
        print(metrics.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
