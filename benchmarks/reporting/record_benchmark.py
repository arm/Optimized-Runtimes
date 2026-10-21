#!/usr/bin/env python3
"""Record review evidence around an unchanged Ultralytics benchmark command.

Ultralytics remains responsible for loading the dataset, running validation,
and reporting accuracy and stage timings. This command retains that output with
the measured runtime and model identities, source revision, and Raspberry Pi state needed to
interpret the result, then exits with the benchmark's status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from .host_metrics import HostPreflight, capture_host_preflight, capture_host_snapshot

__all__ = ["record_benchmark"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(
    command: Sequence[str], log_path: Path, preflight: HostPreflight | None
) -> tuple[int, dict[str, object]]:
    """Run once, preserving output and descriptive state from both boundaries."""
    started_at = datetime.now(UTC)
    before = capture_host_snapshot()
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("benchmark output stream is unavailable")
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return_code = process.wait()
    after = capture_host_snapshot()
    finished_at = datetime.now(UTC)
    child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    report: dict[str, object] = {
        "version": 1,
        "kind": "yolo26n-ultralytics-benchmark-result",
        "command": list(command),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "exit_code": return_code,
        # Linux reports ru_maxrss in KiB. The fresh recorder process has only
        # one child command, so this is the benchmark process tree's peak.
        "child_peak_rss_bytes": child_usage.ru_maxrss * 1024,
        "host": {"before": before.to_json(), "after": after.to_json()},
    }
    if preflight is not None:
        report["preflight"] = preflight.to_json()
    return return_code, report


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def record_benchmark(
    output: Path,
    runtime_library: Path,
    model_manifest: Path,
    revision: str,
    command: Sequence[str],
    benchmark_paths: Sequence[Path] = (),
) -> int:
    """Publish one complete measurement directory without replacing prior evidence."""
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if not runtime_library.is_file():
        raise FileNotFoundError(f"runtime library does not exist: {runtime_library}")
    if not model_manifest.is_file():
        raise FileNotFoundError(f"model manifest does not exist: {model_manifest}")
    manifest_document = json.loads(model_manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest_document, Mapping):
        raise ValueError("model manifest must be a JSON object")

    preflight = None
    if benchmark_paths:
        preflight = capture_host_preflight(tuple(benchmark_paths))
        print(preflight.summary(), flush=True)
        if preflight.problems:
            raise RuntimeError(
                "benchmark stopped because host pressure could distort performance:\n- "
                + "\n- ".join(preflight.problems)
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        return_code, result = _run(command, staging / "benchmark.log", preflight)
        result["source_revision"] = revision
        result["runtime_library"] = {
            "path": str(runtime_library),
            "bytes": runtime_library.stat().st_size,
            "sha256": _sha256(runtime_library),
        }
        result["model_manifest"] = {
            "path": str(model_manifest),
            "sha256": _sha256(model_manifest),
            "document": manifest_document,
        }
        _write_json(staging / "results.json", result)
        staging.rename(output)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return return_code


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-library", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--benchmark-path", action="append", default=[], type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a benchmark command is required after --")
    revision = args.revision.strip()
    if not revision or "\n" in revision:
        parser.error("--revision must be one non-empty line")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")

    return_code = record_benchmark(
        output,
        args.runtime_library.expanduser().resolve(),
        args.model_manifest.expanduser().resolve(),
        revision,
        command,
        tuple(path.expanduser().resolve() for path in args.benchmark_path),
    )
    print(f"benchmark evidence: {output}")
    if return_code:
        raise SystemExit(return_code)


if __name__ == "__main__":
    try:
        _main()
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from None
