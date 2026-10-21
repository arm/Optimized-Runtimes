"""Standard-library JSON, hashing, and atomic writes for converted artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

type JsonScalar = None | bool | int | float | str
type JsonValue = JsonScalar | list[JsonValue] | tuple[JsonValue, ...] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


class RejectedInput(RuntimeError):
    """A caller-controlled converter input failed validation."""


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 identity used in artifact manifests."""
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value: JsonValue) -> bytes:
    """Encode a JSON value with one stable representation for hashing."""
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (encoded + "\n").encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    """Replace one file only after its complete contents reach storage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
