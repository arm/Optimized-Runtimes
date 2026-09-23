"""Public Python runtime admission behavior that does not require Pi hardware."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from arm_yolo26n import Yolo26nError, Yolo26nTensorSession


def _write_model(directory: Path, payload: bytes) -> None:
    directory.mkdir()
    (directory / "model.bin").write_bytes(payload)
    manifest = {
        "kind": "yolo26n-exported-model",
        "version": 1,
        "model": {
            "checkpoint_sha256": "0" * 64,
            "class_names": ["first", "second"],
            "input_shape": [1, 3, 320, 320],
            "output_shape": [1, 6, 2100],
            "target": {
                "version": 1,
                "target": "linux-aarch64-raspberry-pi-5",
                "input_shape": [1, 3, 320, 320],
                "input_dtype": "fp32",
                "output_dtype": "fp32",
                "thread_count": 4,
            },
        },
        "artifact": {
            "path": "model.bin",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class RuntimeAdmissionTests(unittest.TestCase):
    def test_source_install_requires_a_runtime_override(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "bundled YOLO26n runtime library"):
                Yolo26nTensorSession(model=root / "missing-model")

    def test_session_accepts_a_runtime_override_for_source_development(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "runtime library"):
                Yolo26nTensorSession(model=root / "missing-model", runtime=root / "missing.so")

            runtime = root / "libarm-yolo26n.so.0"
            runtime.write_bytes(b"not loaded because model admission fails first")
            with self.assertRaisesRegex(FileNotFoundError, "model directory"):
                Yolo26nTensorSession(model=root / "missing-model", runtime=runtime)

    def test_session_rejects_a_model_that_does_not_match_its_manifest(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "libarm-yolo26n.so.0"
            runtime.write_bytes(b"not loaded because model admission fails first")
            model = root / "model"
            _write_model(model, b"model")
            (model / "model.bin").write_bytes(b"damaged")

            with self.assertRaisesRegex(Yolo26nError, "'model.bin' does not match its manifest"):
                Yolo26nTensorSession(runtime=runtime, model=model)


if __name__ == "__main__":
    unittest.main()
