#!/usr/bin/env python3
"""Run YOLO26n detection on prerecorded video."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from arm_yolo26n import DetectionResult, Yolo26Detector


@dataclass(frozen=True, slots=True)
class _Arguments:
    runtime: Path
    model: Path
    video: Path
    output: Path
    frame_count: int


@dataclass(frozen=True, slots=True)
class _PreloadedVideo:
    frames: tuple[np.ndarray, ...]
    source_width: int
    source_height: int
    source_fps: float
    decode_resize_ms: float


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _parse_args() -> _Arguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=200)
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")
    if not args.runtime.is_file():
        parser.error(f"--runtime is not a file: {args.runtime}")
    if not args.model.is_dir():
        parser.error(f"--model is not a directory: {args.model}")
    if not args.video.is_file():
        parser.error(f"--video is not a file: {args.video}")
    if args.output.resolve() == args.video.resolve():
        parser.error("--output must not overwrite --video")
    return _Arguments(args.runtime, args.model, args.video, args.output, args.frames)


def _preload(video: Path, frame_count: int, input_width: int) -> _PreloadedVideo:
    started_at = perf_counter()
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video}")
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if source_width <= 0 or source_height <= 0 or source_fps <= 0.0:
        capture.release()
        raise RuntimeError(
            "video must report positive width, height, and frame rate: "
            f"{source_width}x{source_height} at {source_fps} fps"
        )
    input_height = round(source_height * input_width / source_width)
    frames = []
    try:
        while len(frames) < frame_count:
            available, frame = capture.read()
            if not available:
                break
            frames.append(
                cv2.resize(frame, (input_width, input_height), interpolation=cv2.INTER_AREA)
            )
    finally:
        capture.release()
    if len(frames) != frame_count:
        raise RuntimeError(
            f"video provided {len(frames)} decodable frames; --frames requested {frame_count}"
        )
    return _PreloadedVideo(
        frames=tuple(frames),
        source_width=source_width,
        source_height=source_height,
        source_fps=source_fps,
        decode_resize_ms=(perf_counter() - started_at) * 1000,
    )


def _detection_record(result: DetectionResult) -> list[dict[str, object]]:
    return [
        {
            "box": list(detection.box),
            "confidence": detection.confidence,
            "class_id": detection.class_id,
            "class_name": detection.class_name,
        }
        for detection in result.detections
    ]


def _run_model(
    detector: Yolo26Detector,
    video: _PreloadedVideo,
) -> tuple[list[dict[str, object]], dict[str, float | int]]:
    detector.detect(video.frames[0], pixel_format="BGR")
    records = []
    model_seconds = 0.0
    preprocess_ms = 0.0
    inference_ms = 0.0
    postprocess_ms = 0.0
    for index, frame in enumerate(video.frames):
        started_at = perf_counter()
        result = detector.detect(frame, pixel_format="BGR")
        model_seconds += perf_counter() - started_at
        preprocess_ms += result.timings.preprocess_ms
        inference_ms += result.timings.inference_ms
        postprocess_ms += result.timings.postprocess_ms
        records.append(
            {
                "frame_index": index,
                "timestamp_seconds": index / video.source_fps,
                "completed_seconds": model_seconds,
                "timings": {
                    "preprocess_ms": result.timings.preprocess_ms,
                    "inference_ms": result.timings.inference_ms,
                    "postprocess_ms": result.timings.postprocess_ms,
                },
                "detections": _detection_record(result),
            }
        )
    frame_count = len(records)
    return records, {
        "frames": frame_count,
        "model_fps": frame_count / model_seconds,
        "preprocess_mean_ms": preprocess_ms / frame_count,
        "inference_mean_ms": inference_ms / frame_count,
        "postprocess_mean_ms": postprocess_ms / frame_count,
        "model_mean_ms": model_seconds * 1000 / frame_count,
        "decode_resize_ms": video.decode_resize_ms,
    }


def _write_results(
    args: _Arguments,
    video: _PreloadedVideo,
    records: list[dict[str, object]],
    summary: dict[str, float | int],
) -> None:
    input_height, input_width = video.frames[0].shape[:2]
    document = {
        "version": 1,
        "coordinate_space": "inference_frame_pixels_xyxy",
        "source": {
            "video": str(args.video.resolve()),
            "sha256": _digest(args.video),
            "width": video.source_width,
            "height": video.source_height,
            "fps": video.source_fps,
        },
        "inference_frame": {"width": input_width, "height": input_height},
        "runtime_library": str(args.runtime.resolve()),
        "model_directory": str(args.model.resolve()),
        "runtime": "yolo26n",
        "measurement": {
            "decode_and_resize": "preloaded_before_model_timing",
            "warmup_frames": 1,
            "serialization": "outside_model_timing",
        },
        "summary": summary,
        "frames": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    with Yolo26Detector(runtime=args.runtime, model=args.model) as detector:
        video = _preload(args.video, args.frame_count, detector.input_shape[3])
        records, summary = _run_model(detector, video)
    _write_results(args, video, records, summary)
    print(
        "Detection performance:\n"
        f"  throughput: {summary['model_fps']:.2f} FPS\n"
        f"  detection time: {summary['model_mean_ms']:.2f} ms/frame\n"
        f"  frames: {summary['frames']}\n"
        "  mean timing: "
        f"preprocess {summary['preprocess_mean_ms']:.2f} ms, "
        f"inference {summary['inference_mean_ms']:.2f} ms, "
        f"postprocess {summary['postprocess_mean_ms']:.2f} ms\n"
        f"  video decode and resize: {summary['decode_resize_ms']:.2f} ms total"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
