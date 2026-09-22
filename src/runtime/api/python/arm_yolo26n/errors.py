"""Exceptions raised by the YOLO26n Python APIs."""

__all__ = ["Yolo26nError"]


class Yolo26nError(RuntimeError):
    """A YOLO26n operation could not be completed."""
