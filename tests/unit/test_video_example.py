"""Public prerecorded-video rendering behavior.

The real renderer and OpenCV consume a detection document at the benchmark's
published file boundary. Model execution is covered by the end-to-end suite.
"""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from tests.support import run_module


def _write_video(path: Path, frames: int = 2) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48))
    if not writer.isOpened():
        raise RuntimeError("OpenCV cannot create the test video")
    try:
        for index in range(frames):
            writer.write(np.full((48, 64, 3), index * 20, dtype=np.uint8))
    finally:
        writer.release()


def _read_video(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV cannot open the test video: {path}")
    frames = []
    try:
        while True:
            available, frame = capture.read()
            if not available:
                return frames
            frames.append(frame)
    finally:
        capture.release()


class VideoRenderCommandTests(unittest.TestCase):
    def test_renderer_writes_a_playable_video_from_benchmark_results(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "input.avi"
            detections = root / "detections.json"
            output = root / "rendered.mp4"
            _write_video(video)
            detections.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "source": {
                            "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                            "width": 64,
                            "height": 48,
                            "fps": 30.0,
                        },
                        "inference_frame": {"width": 64, "height": 48},
                        "frames": [
                            {
                                "frame_index": index,
                                "completed_seconds": index / 30.0,
                                "timings": {"inference_ms": 1.0},
                                "detections": [
                                    {
                                        "box": [8, 8, 32, 32],
                                        "confidence": 0.9,
                                        "class_id": 0,
                                        "class_name": "object",
                                    }
                                ],
                            }
                            for index in range(2)
                        ],
                    }
                ),
                encoding="utf-8",
            )
            completed = run_module(
                "examples.video.render",
                "--video",
                video,
                "--detections",
                detections,
                "--output",
                output,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

            source_frames = _read_video(video)
            rendered_frames = _read_video(output)
            self.assertEqual(len(rendered_frames), len(source_frames))
            self.assertTrue(all(frame.shape == (48, 64, 3) for frame in rendered_frames))
            for source, rendered in zip(source_frames, rendered_frames, strict=True):
                # The published detection occupies this border. Comparing the
                # rendered result with the decoded source verifies that the
                # command did more than transcode a playable video.
                border_difference = cv2.absdiff(source[8:10, 8:33], rendered[8:10, 8:33])
                self.assertGreater(float(border_difference.mean()), 30.0)


if __name__ == "__main__":
    unittest.main()
