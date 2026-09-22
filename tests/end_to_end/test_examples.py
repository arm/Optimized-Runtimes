"""End-to-end public video and Picamera2 example behavior on Linux AArch64.

The real detector, native runtime, rendering code, and OpenCV are exercised.
Only the camera device API is replaced because physical capture is the external
boundary of these deterministic tests.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import sys
import types
import unittest
from contextlib import AbstractContextManager, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from tests.support import configured_model, configured_runtime, run_module


class _Request:
    def __init__(self, image: np.ndarray) -> None:
        self.image = image
        self.released = False

    def release(self) -> None:
        self.released = True


class _MappedArray(AbstractContextManager["_MappedArray"]):
    def __init__(self, request: _Request, _stream: str) -> None:
        self.array = request.image

    def __enter__(self) -> _MappedArray:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class _Camera:
    instances: list[_Camera] = []

    def __init__(self, _index: int) -> None:
        self.configuration: dict[str, object] = {}
        self.closed = False
        self.post_callback = None
        self.requests: list[_Request] = []
        self.preview_frames: list[np.ndarray] = []
        self.texture_overlays: list[np.ndarray] = []
        self.instances.append(self)

    def create_video_configuration(self, **configuration: object) -> dict[str, object]:
        return configuration

    def configure(self, configuration: dict[str, object]) -> None:
        self.configuration = configuration

    def _shape(self, channels: int) -> tuple[int, int, int]:
        main = self.configuration["main"]
        assert isinstance(main, dict)
        width, height = main["size"]
        return height, width, channels

    def capture_request(self, *, wait: bool) -> object:
        assert not wait
        return "request"

    def capture_array(self, _stream: str, *, wait: bool) -> object:
        assert not wait
        return "array"

    def wait(self, job: object) -> _Request | np.ndarray:
        if job == "request":
            request = _Request(np.zeros(self._shape(3), dtype=np.uint8))
            self.requests.append(request)
            return request
        request = _Request(np.zeros(self._shape(4), dtype=np.uint8))
        # Picamera2 draws its preview callback on the completed camera request
        # before returning a copied array to the detection loop.
        if self.post_callback is not None:
            self.post_callback(request)
            self.preview_frames.append(request.image.copy())
        return request.image.copy()

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def start_preview(self, *_args: object, **_kwargs: object) -> None:
        pass

    def stop_preview(self) -> None:
        pass

    def set_overlay(self, overlay: np.ndarray) -> None:
        self.texture_overlays.append(overlay.copy())


def _camera_module() -> types.ModuleType:
    module = types.ModuleType("picamera2")
    module.MappedArray = _MappedArray
    module.Picamera2 = _Camera
    module.Preview = SimpleNamespace(DRM="drm", QTGL="qtgl")
    return module


def _import_camera_example(module: str) -> types.ModuleType:
    sys.modules.pop(module, None)
    previous = sys.modules.get("picamera2")
    sys.modules["picamera2"] = _camera_module()
    try:
        return importlib.import_module(module)
    finally:
        # Restore only the unavailable device module. Replacing the entire
        # module table would remove real dependencies imported by the example.
        if previous is None:
            sys.modules.pop("picamera2", None)
        else:
            sys.modules["picamera2"] = previous


def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48))
    if not writer.isOpened():
        raise RuntimeError("OpenCV cannot create the test video")
    try:
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
        writer.write(np.full((48, 64, 3), 32, dtype=np.uint8))
    finally:
        writer.release()


class VideoDetectionBehaviorTests(unittest.TestCase):
    def test_command_processes_real_detector_results(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "input.avi"
            output = root / "detections.json"
            _write_video(video)
            completed = run_module(
                "examples.video.detect",
                "--runtime",
                configured_runtime(),
                "--model",
                configured_model(),
                "--video",
                video,
                "--output",
                output,
                "--frames",
                2,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            document = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(document["version"], 1)
            self.assertEqual(document["coordinate_space"], "inference_frame_pixels_xyxy")
            self.assertEqual(document["runtime_library"], str(configured_runtime()))
            self.assertEqual(document["model_directory"], str(configured_model()))
            self.assertEqual(document["source"]["video"], str(video.resolve()))
            self.assertEqual(
                document["source"]["sha256"], hashlib.sha256(video.read_bytes()).hexdigest()
            )
            self.assertEqual((document["source"]["width"], document["source"]["height"]), (64, 48))
            self.assertEqual(document["summary"]["frames"], 2)
            self.assertEqual([frame["frame_index"] for frame in document["frames"]], [0, 1])
            self.assertGreater(document["summary"]["inference_mean_ms"], 0)
            for frame in document["frames"]:
                self.assertGreaterEqual(frame["timestamp_seconds"], 0)
                self.assertGreater(frame["completed_seconds"], 0)
                self.assertLessEqual(
                    {"preprocess_ms", "inference_ms", "postprocess_ms"},
                    set(frame["timings"]),
                )
                self.assertIsInstance(frame["detections"], list)
                for detection in frame["detections"]:
                    self.assertLessEqual(
                        {"box", "confidence", "class_id", "class_name"},
                        set(detection),
                    )


class PicameraExampleBehaviorTests(unittest.TestCase):
    def setUp(self) -> None:
        _Camera.instances.clear()

    def test_headless_example_processes_a_camera_frame_and_closes(self) -> None:
        app = _import_camera_example("examples.picamera2.detect")
        output = io.StringIO()
        with (
            patch.object(
                sys,
                "argv",
                [
                    "detect.py",
                    "--runtime",
                    str(configured_runtime()),
                    "--model",
                    str(configured_model()),
                    "--frames",
                    "1",
                ],
            ),
            redirect_stdout(output),
        ):
            self.assertEqual(app.main(), 0)
        camera = _Camera.instances[0]
        self.assertTrue(camera.closed)
        self.assertEqual(len(camera.requests), 1)
        self.assertTrue(camera.requests[0].released)
        self.assertIn("frames=1", output.getvalue())

    def test_live_example_renders_a_camera_frame_and_closes(self) -> None:
        app = _import_camera_example("examples.picamera2.live")
        output = io.StringIO()
        with (
            patch.object(
                sys,
                "argv",
                [
                    "live.py",
                    "--runtime",
                    str(configured_runtime()),
                    "--model",
                    str(configured_model()),
                    "--preview",
                    "drm",
                    "--frames",
                    "2",
                ],
            ),
            redirect_stdout(output),
        ):
            self.assertEqual(app.main(), 0)
        camera = _Camera.instances[0]
        self.assertTrue(camera.closed)
        self.assertEqual(len(camera.preview_frames), 2)
        self.assertTrue(np.any(camera.preview_frames[-1]))
        self.assertIn("frames=2", output.getvalue())

    def test_live_example_uploads_a_texture_overlay_and_closes(self) -> None:
        app = _import_camera_example("examples.picamera2.live")
        output = io.StringIO()
        with (
            patch.object(
                sys,
                "argv",
                [
                    "live.py",
                    "--runtime",
                    str(configured_runtime()),
                    "--model",
                    str(configured_model()),
                    "--preview",
                    "qtgl",
                    "--frames",
                    "2",
                ],
            ),
            redirect_stdout(output),
        ):
            self.assertEqual(app.main(), 0)
        camera = _Camera.instances[0]
        self.assertTrue(camera.closed)
        self.assertTrue(camera.texture_overlays)
        self.assertTrue(np.any(camera.texture_overlays[-1]))
        self.assertIn("frames=2", output.getvalue())


if __name__ == "__main__":
    unittest.main()
