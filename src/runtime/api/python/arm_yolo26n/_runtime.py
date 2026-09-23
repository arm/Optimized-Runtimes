"""Locate the runtime library shipped with the application package."""

from __future__ import annotations

from pathlib import Path

_LIBRARY_NAME = "libarm-yolo26n.so.0"


def _runtime_path(override: str | Path | None) -> Path:
    """Return an explicit development runtime or the installed package runtime."""
    if override is None:
        path = Path(__file__).with_name(_LIBRARY_NAME)
        description = "bundled YOLO26n runtime library"
    else:
        path = Path(override).expanduser().resolve()
        description = "YOLO26n runtime library"
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    return path
