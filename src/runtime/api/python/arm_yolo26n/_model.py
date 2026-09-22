"""Load and validate one converted YOLO26n model directory."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .errors import Yolo26nError

# These values define the converted-model/runtime contract. They describe the
# only model family and native target this purpose-built runtime accepts.
_MODEL_KIND = "yolo26n-exported-model"
_MODEL_VERSION = 1
_TARGET = "linux-aarch64-raspberry-pi-5"

# The Raspberry Pi 5 runtime owns a fixed four-worker pool, matching its four
# CPU cores, so the thread count is part of model/runtime compatibility.
_TARGET_THREAD_COUNT = 4
_BOX_COORDINATE_CHANNELS = 4

# Graph sizing and memory planning are implemented only for these profiles.
_SUPPORTED_RESOLUTIONS = (320, 480, 640, 800, 960, 1280)

# YOLO26n predicts at its P3, P4, and P5 feature levels.
_DETECTION_STRIDES = (8, 16, 32)


@dataclass(frozen=True, slots=True)
class _LoadedModel:
    """Validated metadata and serialized weights required by a session."""

    input_shape: tuple[int, int, int, int]
    output_shape: tuple[int, int, int]
    class_names: tuple[str, ...]
    model_bytes: bytes


@dataclass(frozen=True, slots=True)
class _ArtifactRecord:
    relative_path: Path
    size: int
    sha256: str

    @classmethod
    def from_manifest(cls, value: object) -> _ArtifactRecord:
        """Parse the model artifact without accepting unknown fields."""
        record = _require_mapping(value, "artifact")
        if set(record) != {"bytes", "path", "sha256"}:
            raise Yolo26nError("invalid artifact in YOLO26n model manifest")

        path_value = record["path"]
        size = record["bytes"]
        digest = record["sha256"]
        if (
            not isinstance(path_value, str)
            or type(size) is not int
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise Yolo26nError("invalid artifact in YOLO26n model manifest")

        relative_path = Path(path_value)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise Yolo26nError("invalid artifact path in YOLO26n model manifest")
        return cls(relative_path=relative_path, size=size, sha256=digest)

    def load(self, model_directory: Path) -> bytes:
        """Read the model bytes and verify their manifest record."""
        path = model_directory / self.relative_path
        try:
            model_bytes = path.read_bytes()
        except OSError as error:
            raise Yolo26nError(
                f"YOLO26n model artifact {self.relative_path.as_posix()!r} cannot be read"
            ) from error
        if len(model_bytes) != self.size or hashlib.sha256(model_bytes).hexdigest() != self.sha256:
            raise Yolo26nError(
                f"YOLO26n model artifact {self.relative_path.as_posix()!r} "
                "does not match its manifest"
            )
        return model_bytes


def _load_model(model_directory: str | Path) -> _LoadedModel:
    """Validate one converted model before native code receives its weights."""
    directory = Path(model_directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"YOLO26n model directory does not exist: {directory}")

    manifest = _read_manifest(directory / "manifest.json")
    if (manifest.get("kind"), manifest.get("version")) != (_MODEL_KIND, _MODEL_VERSION):
        raise Yolo26nError("unsupported YOLO26n model manifest kind or version")

    model = _require_mapping(manifest.get("model"), "model record")
    input_shape = _require_shape(model.get("input_shape"), "input shape")
    output_shape = _require_shape(model.get("output_shape"), "output shape")
    resolution = input_shape[2] if len(input_shape) == 4 else None
    if (
        len(input_shape) != 4
        or input_shape[:2] != (1, 3)
        or input_shape[2] != input_shape[3]
        or resolution not in _SUPPORTED_RESOLUTIONS
    ):
        raise Yolo26nError("unsupported YOLO26n input shape")
    if (
        len(output_shape) != 3
        or output_shape[0] != 1
        or output_shape[1] <= _BOX_COORDINATE_CHANNELS
        or output_shape[2] != _detection_positions(resolution)
    ):
        raise Yolo26nError("unsupported YOLO26n output shape")

    expected_target = {
        "version": 1,
        "target": _TARGET,
        "input_shape": list(input_shape),
        "input_dtype": "fp32",
        "output_dtype": "fp32",
        "thread_count": _TARGET_THREAD_COUNT,
    }
    if model.get("target") != expected_target:
        raise Yolo26nError("unsupported YOLO26n target contract")

    names = model.get("class_names")
    if (
        not isinstance(names, list)
        or len(names) != output_shape[1] - _BOX_COORDINATE_CHANNELS
        or any(not isinstance(name, str) or not name for name in names)
    ):
        raise Yolo26nError("invalid class names in YOLO26n model manifest")

    artifact = _ArtifactRecord.from_manifest(manifest.get("artifact"))
    return _LoadedModel(
        input_shape=input_shape,
        output_shape=output_shape,
        class_names=tuple(cast(list[str], names)),
        model_bytes=artifact.load(directory),
    )


def _read_manifest(path: Path) -> Mapping[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"YOLO26n model manifest does not exist: {path}") from None
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Yolo26nError(f"invalid YOLO26n model manifest: {path}") from error
    return _require_mapping(value, "manifest")


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise Yolo26nError(f"invalid {name} in YOLO26n model manifest")
    return cast(Mapping[str, object], value)


def _require_shape(value: object, name: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(type(dimension) is not int or dimension <= 0 for dimension in value)
    ):
        raise Yolo26nError(f"invalid {name} in YOLO26n model manifest")
    return tuple(value)


def _detection_positions(resolution: int) -> int:
    return sum((resolution // stride) ** 2 for stride in _DETECTION_STRIDES)
