"""End-to-end public runtime and detector behavior on Linux AArch64.

These tests require `YOLO26N_E2E_RUNTIME` and `YOLO26N_E2E_MODELS`. They exercise the runtime,
NumPy, OpenCV, and repository Python code. No local component is replaced.
"""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from arm_yolo26n import DetectorConfig, Yolo26Detector, Yolo26nError, Yolo26nTensorSession
from tests.support import (
    configured_model,
    configured_models,
    configured_runtime,
    read_json,
)


def _models_by_resolution() -> dict[int, Path]:
    return {
        read_json(model / "manifest.json")["model"]["input_shape"][2]: model
        for model in configured_models()
    }


class TensorSessionBehaviorTests(unittest.TestCase):
    def test_session_executes_each_model_and_owns_its_lifecycle(self) -> None:
        for resolution, model in _models_by_resolution().items():
            with self.subTest(resolution=resolution):
                session = Yolo26nTensorSession(runtime=configured_runtime(), model=model)
                try:
                    output = session.execute(np.zeros(session.input_shape, dtype=np.float32))
                    self.assertEqual(tuple(output.shape), session.output_shape)
                    self.assertTrue(np.isfinite(output).all())
                    self.assertEqual(len(session.class_names), session.output_shape[1] - 4)
                finally:
                    session.close()

                session.close()
                with self.assertRaises(Yolo26nError):
                    session.execute(np.zeros(session.input_shape, dtype=np.float32))

    def test_session_rejects_incompatible_public_inputs_and_model(self) -> None:
        with Yolo26nTensorSession(
            runtime=configured_runtime(), model=configured_model()
        ) as session:
            wrong_shape = (1, 3, session.input_shape[2] - 1, session.input_shape[3])
            cases = (
                np.zeros(wrong_shape, dtype=np.float32),
                np.zeros(session.input_shape, dtype=np.float16),
            )
            for tensor in cases:
                with self.subTest(shape=tuple(tensor.shape), dtype=tensor.dtype):
                    with self.assertRaises((TypeError, ValueError)):
                        session.execute(tensor)

        with TemporaryDirectory() as directory:
            damaged = Path(directory) / "model"
            shutil.copytree(configured_model(), damaged)
            (damaged / "model.bin").write_bytes(b"damaged")
            with self.assertRaisesRegex(
                Yolo26nError, "model artifact 'model.bin' does not match its manifest"
            ):
                Yolo26nTensorSession(runtime=configured_runtime(), model=damaged)


class DetectorBehaviorTests(unittest.TestCase):
    def test_equivalent_bgr_rgb_and_bgrx_frames_produce_equivalent_results(self) -> None:
        with Yolo26Detector(runtime=configured_runtime(), model=configured_model()) as detector:
            resolution = detector.input_shape[2]
            rng = np.random.default_rng(7)
            shapes = (
                (resolution * 3 // 4, resolution),  # Vertical padding.
                (resolution, resolution - 13),  # Horizontal padding and odd width.
                (resolution * 2 // 3 + 1, resolution - 13),  # Resize and odd dimensions.
            )
            for height, width in shapes:
                with self.subTest(shape=(height, width)):
                    bgr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
                    rgb = np.ascontiguousarray(bgr[..., ::-1])
                    storage = np.zeros((height, width + 4, 4), dtype=np.uint8)
                    storage[:, :width, :3] = bgr
                    bgrx = storage[:, :width]

                    bgr_result = detector.detect(bgr, pixel_format="BGR")
                    rgb_result = detector.detect(rgb, pixel_format="RGB")
                    bgrx_result = detector.detect(bgrx, pixel_format="BGRX")

                    self.assertEqual(bgr_result.source_shape, bgr.shape[:2])
                    self.assertEqual(bgr_result.detections, rgb_result.detections)
                    self.assertEqual(bgr_result.detections, bgrx_result.detections)

    def test_detector_applies_thresholds_and_reports_source_coordinates(self) -> None:
        """Check the public filtering contract; COCO validation measures accuracy."""
        rng = np.random.default_rng(7)
        resolution = read_json(configured_model() / "manifest.json")["model"]["input_shape"][2]
        height = resolution * 3 // 4
        frame = rng.integers(0, 256, (height, resolution, 3), dtype=np.uint8)
        # This fixed frame has observed detections above 0.001 but none above
        # 0.01, so both threshold outcomes are observable without accepting
        # arbitrary zero-confidence boxes.
        with Yolo26Detector(
            runtime=configured_runtime(),
            model=configured_model(),
            config=DetectorConfig(confidence=0.001, max_detections=5),
        ) as detector:
            detections = detector.detect(frame, pixel_format="BGR").detections

        self.assertTrue(detections)
        self.assertLessEqual(len(detections), 5)
        for detection in detections:
            x1, y1, x2, y2 = detection.box
            self.assertTrue(0 <= x1 < x2 <= resolution)
            self.assertTrue(0 <= y1 < y2 <= height)
            self.assertGreaterEqual(detection.confidence, 0.001)
            self.assertEqual(detection.class_name, detector.class_names[detection.class_id])

        with Yolo26Detector(
            runtime=configured_runtime(),
            model=configured_model(),
            config=DetectorConfig(confidence=0.01),
        ) as detector:
            self.assertEqual(detector.detect(frame, pixel_format="BGR").detections, ())

    def test_detector_reports_public_metadata_callback_and_rejections(self) -> None:
        consumed: list[bool] = []
        with Yolo26Detector(
            runtime=configured_runtime(),
            model=configured_model(),
            config=DetectorConfig(confidence=0.25, iou=0.7, max_detections=20),
        ) as detector:
            resolution = detector.input_shape[2]
            frame = np.zeros((resolution, resolution, 3), dtype=np.uint8)
            baseline = detector.detect(frame.copy(), pixel_format="BGR")

            def reuse_frame() -> None:
                consumed.append(True)
                frame.fill(255)

            result = detector.detect(
                frame,
                pixel_format="BGR",
                on_frame_consumed=reuse_frame,
            )
            self.assertEqual(consumed, [True])
            self.assertEqual(result.detections, baseline.detections)
            self.assertTrue(np.all(frame == 255))
            self.assertEqual(result.source_shape, (resolution, resolution))
            self.assertEqual(len(detector.class_names), detector.output_shape[1] - 4)
            with self.assertRaises(TypeError):
                detector.detect(
                    np.zeros((resolution, resolution, 3), dtype=np.float32),
                    pixel_format="BGR",
                )


if __name__ == "__main__":
    unittest.main()
