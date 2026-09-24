"""Behavior of the command that records evidence around an external benchmark.

The test executes a real child process. Hardware readings remain the local
operating system's responsibility and are asserted only through the recorder's
public JSON output.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.support import read_json, run_module


class BenchmarkRecorderCommandTests(unittest.TestCase):
    @staticmethod
    def _inputs(directory: Path) -> tuple[Path, Path]:
        runtime = directory / "libarm-yolo26n.so.0"
        runtime.write_bytes(b"runtime")
        manifest = directory / "manifest.json"
        manifest.write_text('{"kind":"yolo26n-exported-model","version":1}\n')
        return runtime, manifest

    def test_recorder_publishes_review_evidence_and_preserves_command_output(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "evidence"
            runtime, manifest = self._inputs(root)
            completed = run_module(
                "benchmarks.reporting.record_benchmark",
                "--output-dir",
                output,
                "--runtime-library",
                runtime,
                "--model-manifest",
                manifest,
                "--revision",
                "0123456789abcdef",
                "--",
                sys.executable,
                "-c",
                "print('benchmark output')",
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("benchmark output", completed.stdout)
            self.assertIn("benchmark output", (output / "benchmark.log").read_text())
            self.assertEqual(
                {path.name for path in output.iterdir()}, {"benchmark.log", "results.json"}
            )

            report = read_json(output / "results.json")
            self.assertEqual(report["kind"], "yolo26n-ultralytics-benchmark-result")
            self.assertEqual(report["exit_code"], 0)
            self.assertEqual(report["source_revision"], "0123456789abcdef")
            self.assertEqual(report["model_manifest"]["path"], str(manifest))
            self.assertEqual(
                report["model_manifest"]["document"],
                {"kind": "yolo26n-exported-model", "version": 1},
            )
            self.assertEqual(report["runtime_library"]["path"], str(runtime))
            self.assertGreaterEqual(report["child_peak_rss_bytes"], 0)
            for phase in ("before", "after"):
                snapshot = report["host"][phase]
                self.assertIn("MemAvailable", snapshot["memory_bytes"])
                self.assertIn("Cached", snapshot["memory_bytes"])
                self.assertIn("SReclaimable", snapshot["memory_bytes"])
                self.assertIn("pgmajfault", snapshot["virtual_memory_events"])

    def test_recorder_returns_the_benchmark_failure_after_writing_context(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "evidence"
            runtime, manifest = self._inputs(root)
            completed = run_module(
                "benchmarks.reporting.record_benchmark",
                "--output-dir",
                output,
                "--runtime-library",
                runtime,
                "--model-manifest",
                manifest,
                "--revision",
                "fedcba9876543210",
                "--",
                sys.executable,
                "-c",
                "raise SystemExit(7)",
            )
            self.assertEqual(completed.returncode, 7)
            self.assertEqual(read_json(output / "results.json")["exit_code"], 7)


if __name__ == "__main__":
    unittest.main()
