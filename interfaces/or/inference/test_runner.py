# SPDX-License-Identifier: Apache-2.0
"""Exercise the real OR client against the bundled upstream mock, without GTest."""

import json
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "model.bin").write_bytes(b"LRTC\x01")
        (self.root / "first.bin").write_bytes(struct.pack("4f", 1, -2, 3, 4))
        (self.root / "inputs.txt").write_text("first.bin\n")

    def run_client(self, *extra, environment=None):
        return subprocess.run([str(HERE / "or-run"), str(HERE / "test_runtime.so"), str(self.root / "model.bin"),
                               str(self.root / "inputs.txt"), str(self.root / "output"), *extra],
                              env=dict(os.environ, **(environment or {})), capture_output=True, text=True)

    def test_actual_output_and_multiple_inputs(self):
        (self.root / "second.bin").write_bytes(struct.pack("4f", 5, 6, 7, 8))
        (self.root / "inputs.txt").write_text("first.bin\nsecond.bin\n")
        result = self.run_client()
        self.assertEqual(result.returncode, 0, result.stderr)
        output = self.root / "output"
        self.assertEqual(struct.unpack("4f", (output / "0-0.bin").read_bytes()), (2, -4, 6, 8))
        self.assertEqual(struct.unpack("4f", (output / "1-0.bin").read_bytes()), (10, 12, 14, 16))
        metadata = json.loads((output / "0.json").read_text())
        self.assertEqual(metadata["outputs"][0]["shape"], [4])
        self.assertEqual(metadata["outputs"][0]["byte_strides"], [4])
        self.assertEqual(metadata["outputs"][0]["dtype"], 1)
        self.assertEqual(json.loads((output / "run.json").read_text())["inputs"], 2)

    def test_warmed_measurement_and_no_overwrite(self):
        result = self.run_client("2", "5")
        self.assertEqual(result.returncode, 0, result.stderr)
        metadata = json.loads((self.root / "output/0.json").read_text())
        self.assertEqual(len(metadata["samples_ms"]), 5)
        self.assertGreaterEqual(metadata["p50_ms"], 0)
        self.assertNotEqual(self.run_client().returncode, 0)

    def test_bounded_dynamic_output_uses_actual_shape(self):
        result = self.run_client(environment={"TEST_DYNAMIC_OUTPUT": "1"})
        self.assertEqual(result.returncode, 0, result.stderr)
        output = self.root / "output"
        self.assertEqual(json.loads((output / "0.json").read_text())["outputs"][0]["shape"], [4])
        self.assertEqual(struct.unpack("4f", (output / "0-0.bin").read_bytes()), (2, -4, 6, 8))

    def test_short_input_fails_without_success_receipt(self):
        (self.root / "first.bin").write_bytes(b"short")
        result = self.run_client()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Input byte size", result.stderr)
        self.assertFalse((self.root / "output/run.json").exists())

    def test_unsupported_memory_profile(self):
        result = self.run_client(environment={"OR_MOCK_MEMORY_PROFILE": "external-input"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No runtime-owned direct host-pointer", result.stderr)
        self.assertNotIn("cleanup failed", result.stderr)

    def test_terminal_execution_failure_is_not_success(self):
        result = self.run_client(environment={"TEST_EXECUTION_FAILURE": "1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("OR status 7", result.stderr)
        self.assertNotIn("cleanup failed", result.stderr)
        self.assertFalse((self.root / "output/run.json").exists())

    def test_bad_iterations_and_empty_input_list(self):
        self.assertNotEqual(self.run_client("0", "-1").returncode, 0)
        self.assertNotEqual(self.run_client("0", "0").returncode, 0)
        (self.root / "inputs.txt").write_text("")
        self.assertNotEqual(self.run_client().returncode, 0)
        self.assertFalse((self.root / "output/run.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
