#!/usr/bin/env python3
"""Check whether a checkpoint can be exported at the selected input resolution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .converter_io import JsonObject, RejectedInput, atomic_write, canonical_bytes, sha256_bytes
from .export_yolo26n import ExportedModel, export_model

_VALIDATION_VERSION = 1


def _validation_result(exported: ExportedModel) -> JsonObject:
    """Describe the successfully exported artifact without retaining model data."""
    return {
        "version": _VALIDATION_VERSION,
        "status": "accepted",
        "artifact": {
            "bytes": len(exported.payload),
            "sha256": sha256_bytes(exported.payload),
        },
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = _validation_result(export_model(args.checkpoint, args.resolution))
    atomic_write(args.output, canonical_bytes(result))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    try:
        _main()
    except (OSError, RejectedInput) as error:
        raise SystemExit(f"YOLO26n checkpoint rejected: {error}") from None
