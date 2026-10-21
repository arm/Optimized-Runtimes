#!/usr/bin/env python3
"""Read a checkpoint and expose only validated YOLO26n model facts."""

from __future__ import annotations

from pathlib import Path

from .checkpoint_decoder import DecodedModule, decode_checkpoint_archive
from .converter_io import RejectedInput
from .model_contract import ValidatedYolo26n
from .model_validation import validate_yolo26n_model


def read_checkpoint(path: Path) -> bytes:
    """Read the checkpoint once so validation and export use identical bytes."""
    return path.read_bytes()


def _read_model_from_decoded_checkpoint(decoded: object) -> DecodedModule:
    """End the untyped pickle boundary by extracting its one required field."""
    if not isinstance(decoded, dict):
        raise RejectedInput(
            "checkpoint root must be a mapping containing a YOLO26n model; "
            f"received {type(decoded).__name__}"
        )
    model = decoded.get("model")
    if not isinstance(model, DecodedModule):
        raise RejectedInput(
            "checkpoint model must be ultralytics.nn.tasks.DetectionModel; "
            f"received {type(model).__name__!r}"
        )
    return model


def decode_checkpoint(checkpoint: bytes) -> ValidatedYolo26n:
    """Decode a Torch archive into the complete model contract used by export."""
    decoded = decode_checkpoint_archive(checkpoint)
    return validate_yolo26n_model(_read_model_from_decoded_checkpoint(decoded))
