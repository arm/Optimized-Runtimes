#!/usr/bin/env python3
"""Export a supported YOLO26n checkpoint for the Raspberry Pi 5 runtime."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .checkpoint_intake import (
    decode_checkpoint,
    read_checkpoint,
)
from .converter_io import JsonObject, RejectedInput, atomic_write, canonical_bytes, sha256_bytes
from .model_contract import ModelResolution
from .model_format import pack_model
from .weight_lowering import lower_yolo26n_weights

_MODEL_NAME = "model.bin"
_MANIFEST_VERSION = 1


@dataclass(frozen=True)
class ExportManifest:
    """Typed model and artifact facts serialized as ``manifest.json``."""

    checkpoint_sha256: str
    class_names: tuple[str, ...]
    input_shape: tuple[int, int, int, int]
    output_shape: tuple[int, int, int]
    target: JsonObject
    artifact_bytes: int
    artifact_sha256: str

    def to_json(self) -> JsonObject:
        """Create the external JSON structure at the serialization boundary."""
        return {
            "kind": "yolo26n-exported-model",
            "version": _MANIFEST_VERSION,
            "model": {
                "checkpoint_sha256": self.checkpoint_sha256,
                "class_names": self.class_names,
                "input_shape": self.input_shape,
                "output_shape": self.output_shape,
                "target": self.target,
            },
            "artifact": {
                "path": _MODEL_NAME,
                "bytes": self.artifact_bytes,
                "sha256": self.artifact_sha256,
            },
        }


@dataclass(frozen=True)
class ExportedModel:
    """The serialized runtime model and its typed external description."""

    payload: bytes
    manifest: ExportManifest


def export_model(
    checkpoint_path: Path,
    resolution: int,
) -> ExportedModel:
    """Validate a checkpoint and serialize the runtime's model data."""
    checkpoint = read_checkpoint(checkpoint_path)
    checkpoint_sha256 = sha256_bytes(checkpoint)
    model = decode_checkpoint(checkpoint)
    try:
        model_resolution = ModelResolution(resolution)
        model_payload = pack_model(
            lower_yolo26n_weights(model),
            model_resolution.resolution,
            model.class_count,
        )
    except ValueError as error:
        raise RejectedInput(f"could not prepare YOLO26n runtime weights: {error}") from error

    manifest = ExportManifest(
        checkpoint_sha256=checkpoint_sha256,
        class_names=model.class_names,
        input_shape=model_resolution.input_shape,
        output_shape=model_resolution.output_shape(model.class_count),
        target=model_resolution.target_manifest(),
        artifact_bytes=len(model_payload),
        artifact_sha256=sha256_bytes(model_payload),
    )
    return ExportedModel(model_payload, manifest)


def write_export(output_dir: Path, exported: ExportedModel) -> None:
    """Publish a complete export directory without exposing partial output."""
    if output_dir.exists():
        raise RejectedInput(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        atomic_write(temporary / _MODEL_NAME, exported.payload)
        atomic_write(temporary / "manifest.json", canonical_bytes(exported.manifest.to_json()))
        os.rename(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    exported = export_model(
        args.checkpoint,
        args.resolution,
    )
    write_export(args.output_dir, exported)


if __name__ == "__main__":
    try:
        _main()
    except (OSError, RejectedInput) as error:
        raise SystemExit(f"YOLO26n export rejected: {error}") from None
