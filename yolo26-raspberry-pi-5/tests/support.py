"""Helpers for invoking public commands and locating external test fixtures."""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import tempfile
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE = tempfile.TemporaryDirectory(prefix="yolo26n-tests-")
atexit.register(_WORKSPACE.cleanup)
WORKSPACE = Path(_WORKSPACE.name)


def run_module(module: str, *arguments: object) -> subprocess.CompletedProcess[str]:
    """Run a Python command exactly as a user invokes it from the repository."""
    return subprocess.run(
        [sys.executable, "-m", module, *(str(argument) for argument in arguments)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


@cache
def checkpoint() -> Path:
    """Return the checkpoint supplied to the converter functional-test target."""
    # The Make target supplies this external input without making checkpoint
    # discovery part of the converter API.
    configured = os.environ.get("YOLO26N_CHECKPOINT")
    if configured:
        return Path(configured).resolve()
    raise RuntimeError("YOLO26N_CHECKPOINT is required for converter functional tests")


def exported_model(resolution: int = 320) -> Path:
    """Convert the shared checkpoint through the supported command."""
    return _exported_model(resolution)


@cache
def _exported_model(resolution: int) -> Path:
    output = WORKSPACE / f"export-{resolution}"
    completed = run_module(
        "converter.export_yolo26n",
        "--checkpoint",
        checkpoint(),
        "--resolution",
        resolution,
        "--output-dir",
        output,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr)
    return output


def read_json(path: Path) -> dict[str, object]:
    """Read a JSON object emitted by a public command."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"expected JSON object: {path}")
    return value


@cache
def configured_runtime() -> Path:
    """Return the native library supplied to the end-to-end tests."""
    value = os.environ.get("YOLO26N_E2E_RUNTIME")
    if not value:
        raise RuntimeError("YOLO26N_E2E_RUNTIME is required for end-to-end tests")
    runtime = Path(value).resolve()
    if not runtime.is_file():
        raise RuntimeError(f"YOLO26N_E2E_RUNTIME is not a file: {runtime}")
    return runtime


def configured_model() -> Path:
    """Return one converted model for example behavior tests."""
    return configured_models()[0]


@cache
def configured_models() -> tuple[Path, ...]:
    """Return the six converted models required by the end-to-end tests."""
    value = os.environ.get("YOLO26N_E2E_MODELS")
    if not value:
        raise RuntimeError("YOLO26N_E2E_MODELS is required for end-to-end tests")

    models_by_resolution: dict[int, Path] = {}
    for item in value.split(os.pathsep):
        model_directory = Path(item).resolve()
        if not model_directory.is_dir():
            raise RuntimeError(f"YOLO26N_E2E_MODELS contains a non-directory: {model_directory}")
        manifest = read_json(model_directory / "manifest.json")
        model = manifest.get("model")
        if not isinstance(model, dict):
            raise RuntimeError(f"converted model has no model record: {model_directory}")
        shape = model.get("input_shape")
        if not isinstance(shape, list) or len(shape) != 4 or type(shape[2]) is not int:
            raise RuntimeError(f"converted model has no supported input shape: {model_directory}")
        models_by_resolution[shape[2]] = model_directory

    expected = {320, 480, 640, 800, 960, 1280}
    if set(models_by_resolution) != expected:
        raise RuntimeError(
            "YOLO26N_E2E_MODELS must contain exactly one model for each supported "
            f"resolution; received {sorted(models_by_resolution)}"
        )
    return tuple(models_by_resolution[resolution] for resolution in sorted(models_by_resolution))
