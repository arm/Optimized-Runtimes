#!/usr/bin/env python3
"""Render detections recorded by the prerecorded-video example."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import cast

import cv2
import numpy as np
from examples.picamera2.overlay import OverlayRenderer, RollingPerformance


@dataclass(frozen=True, slots=True)
class _Arguments:
    video: Path
    detections: Path
    output: Path


@dataclass(frozen=True, slots=True)
class _RecordedDetection:
    box: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


@dataclass(frozen=True, slots=True)
class _DetectionDocument:
    source_sha256: str
    source_width: int
    source_height: int
    source_fps: float
    inference_width: int
    inference_height: int
    frames: tuple[Mapping[str, object], ...]


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise RuntimeError(f"detections document has invalid {name}")
    return cast(Mapping[str, object], value)


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise RuntimeError(f"detections document has invalid {name}")
    return value


def _positive_number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise RuntimeError(f"detections document has invalid {name}")
    return float(value)


def _parse_args() -> _Arguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error(f"--video is not a file: {args.video}")
    if not args.detections.is_file():
        parser.error(f"--detections is not a file: {args.detections}")
    if args.output.resolve() in (args.video.resolve(), args.detections.resolve()):
        parser.error("--output must not overwrite an input")
    return _Arguments(args.video, args.detections, args.output)


def _load_document(path: Path) -> _DetectionDocument:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid detections JSON: {path}") from error
    document = _mapping(value, "top-level object")
    if document.get("version") != 1:
        raise RuntimeError("unsupported detections document version")
    source = _mapping(document.get("source"), "source record")
    inference = _mapping(document.get("inference_frame"), "inference frame")
    source_sha256 = source.get("sha256")
    frame_values = document.get("frames")
    if not isinstance(source_sha256, str) or len(source_sha256) != 64:
        raise RuntimeError("detections document has invalid source SHA-256")
    if not isinstance(frame_values, list) or not frame_values:
        raise RuntimeError("detections document must contain at least one frame")
    frames = tuple(_mapping(frame, "frame record") for frame in frame_values)
    return _DetectionDocument(
        source_sha256=source_sha256,
        source_width=_positive_int(source.get("width"), "source width"),
        source_height=_positive_int(source.get("height"), "source height"),
        source_fps=_positive_number(source.get("fps"), "source frame rate"),
        inference_width=_positive_int(inference.get("width"), "inference width"),
        inference_height=_positive_int(inference.get("height"), "inference height"),
        frames=frames,
    )


def _detections(record: Mapping[str, object]) -> tuple[_RecordedDetection, ...]:
    values = record.get("detections")
    if not isinstance(values, list):
        raise RuntimeError("frame record has invalid detections")
    detections = []
    for value in values:
        detection = _mapping(value, "detection record")
        box = detection.get("box")
        confidence = detection.get("confidence")
        class_id = detection.get("class_id")
        class_name = detection.get("class_name")
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(not isinstance(coordinate, (int, float)) for coordinate in box)
            or not isinstance(confidence, (int, float))
            or type(class_id) is not int
            or not isinstance(class_name, str)
        ):
            raise RuntimeError("detections document has invalid detection record")
        x1, y1, x2, y2 = (float(coordinate) for coordinate in box)
        detections.append(
            _RecordedDetection(
                box=(x1, y1, x2, y2),
                confidence=float(confidence),
                class_id=class_id,
                class_name=class_name,
            )
        )
    return tuple(detections)


def _composite(frame: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    alpha = overlay[..., 3:4].astype(np.float32) / 255.0
    overlay_bgr = overlay[..., 2::-1].astype(np.float32)
    return (overlay_bgr * alpha + frame.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)


def _render(args: _Arguments, document: _DetectionDocument) -> tuple[float, float, float]:
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {args.video}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        document.source_fps,
        (document.source_width, document.source_height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not create video: {args.output}")

    renderer = OverlayRenderer(width=document.inference_width, height=document.inference_height)
    performance = RollingPerformance()
    overlay = np.empty((document.source_height, document.source_width, 4), dtype=np.uint8)
    decode_seconds = 0.0
    render_seconds = 0.0
    encode_seconds = 0.0
    try:
        for expected_index, record in enumerate(document.frames):
            decode_started_at = perf_counter()
            available, frame = capture.read()
            decode_seconds += perf_counter() - decode_started_at
            if not available:
                raise RuntimeError(f"video ended before frame {expected_index}")
            if record.get("frame_index") != expected_index:
                raise RuntimeError("detections must be contiguous from frame zero")
            timings = _mapping(record.get("timings"), "frame timings")
            performance.add(
                float(record.get("completed_seconds", 0.0)),
                float(timings.get("inference_ms", 0.0)),
            )
            render_started_at = perf_counter()
            renderer.update(_detections(record), performance)
            renderer.draw_rgba(overlay)
            composited = _composite(frame, overlay)
            render_seconds += perf_counter() - render_started_at
            encode_started_at = perf_counter()
            writer.write(composited)
            encode_seconds += perf_counter() - encode_started_at
    finally:
        capture.release()
        writer.release()
    return decode_seconds, render_seconds, encode_seconds


def main() -> int:
    args = _parse_args()
    document = _load_document(args.detections)
    if _digest(args.video) != document.source_sha256:
        raise RuntimeError("video content does not match the benchmark detections")
    decode_seconds, render_seconds, encode_seconds = _render(args, document)
    frame_count = len(document.frames)
    print(
        f"frames={frame_count} "
        f"decode_mean_ms={decode_seconds * 1000 / frame_count:.2f} "
        f"render_mean_ms={render_seconds * 1000 / frame_count:.2f} "
        f"encode_mean_ms={encode_seconds * 1000 / frame_count:.2f} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
