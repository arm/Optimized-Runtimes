"""Collect Raspberry Pi state that can explain benchmark variation.

The collector takes descriptive snapshots. It does not change governors, clear
filesystem caches, or decide whether a run is acceptable. Missing optional
Linux and Raspberry Pi interfaces are represented by absent values so the
benchmark can still run and the report shows what could not be observed.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

__all__ = ["HostSnapshot", "capture_host_snapshot"]

# MemAvailable estimates memory usable without swapping. Cached and
# SReclaimable show the size of reclaimable filesystem/slab caches implicated
# in the observed RAM-pressure issue. Swap totals distinguish configured swap
# from swap space that was actually consumed.
_MEMORY_FIELDS = (
    "MemTotal",
    "MemAvailable",
    "Cached",
    "SReclaimable",
    "SwapTotal",
    "SwapFree",
)

# Deltas between snapshots reveal disk-backed faults and swap traffic during
# the benchmark without trying to prescribe a policy for those observations.
_VM_FIELDS = ("pgmajfault", "pswpin", "pswpout")


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").rstrip("\x00\n")
    except OSError:
        return None


def _read_integer(path: Path) -> int | None:
    value = _read_text(path)
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class _FrequencyPolicy:
    """One Linux CPU-frequency policy at the instant of the snapshot."""

    name: str
    governor: str | None
    current_khz: int | None
    maximum_khz: int | None

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "governor": self.governor,
            "current_khz": self.current_khz,
            "maximum_khz": self.maximum_khz,
        }


@dataclass(frozen=True, slots=True)
class _ThermalZone:
    """One Linux thermal-zone reading and its kernel-provided type."""

    name: str
    kind: str | None
    millidegrees_celsius: int | None

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "type": self.kind,
            "millidegrees_celsius": self.millidegrees_celsius,
        }


@dataclass(frozen=True, slots=True)
class _ThrottleState:
    """Current and historical Raspberry Pi firmware limit flags."""

    raw: str
    value: int

    def to_json(self) -> dict[str, object]:
        return {
            "raw": self.raw,
            "value": self.value,
            "currently_limited": bool(self.value & 0xF),
            "previously_limited": bool(self.value & 0xF0000),
        }


@dataclass(frozen=True, slots=True)
class HostSnapshot:
    """Host facts captured immediately before or after an Ultralytics run.

    Memory values use the names and meanings published by ``/proc/meminfo``.
    Virtual-memory counters use the cumulative kernel values from
    ``/proc/vmstat``; consumers determine activity during the run by subtracting
    the before snapshot from the after snapshot.
    """

    model: str | None
    kernel: str
    machine: str
    cpu_affinity: tuple[int, ...]
    load_average: tuple[float, float, float]
    frequency: tuple[_FrequencyPolicy, ...]
    temperature: tuple[_ThermalZone, ...]
    throttling: _ThrottleState | None
    memory_bytes: dict[str, int]
    virtual_memory_events: dict[str, int]

    def to_json(self) -> dict[str, object]:
        return {
            "model": self.model,
            "kernel": self.kernel,
            "machine": self.machine,
            "cpu_affinity": list(self.cpu_affinity),
            "load_average": list(self.load_average),
            "frequency": [item.to_json() for item in self.frequency],
            "temperature": [item.to_json() for item in self.temperature],
            "throttling": self.throttling.to_json() if self.throttling else None,
            "memory_bytes": self.memory_bytes,
            "virtual_memory_events": self.virtual_memory_events,
        }


def _frequency_policies() -> tuple[_FrequencyPolicy, ...]:
    root = Path("/sys/devices/system/cpu/cpufreq")
    return tuple(
        _FrequencyPolicy(
            name=path.name,
            governor=_read_text(path / "scaling_governor"),
            current_khz=_read_integer(path / "scaling_cur_freq"),
            maximum_khz=_read_integer(path / "scaling_max_freq"),
        )
        for path in sorted(root.glob("policy*"))
    )


def _thermal_zones() -> tuple[_ThermalZone, ...]:
    return tuple(
        _ThermalZone(
            name=path.name,
            kind=_read_text(path / "type"),
            millidegrees_celsius=_read_integer(path / "temp"),
        )
        for path in sorted(Path("/sys/class/thermal").glob("thermal_zone*"))
    )


def _throttle_state() -> _ThrottleState | None:
    if shutil.which("vcgencmd") is None:
        return None
    try:
        completed = subprocess.run(
            ["vcgencmd", "get_throttled"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    raw = completed.stdout.strip()
    match = re.fullmatch(r"throttled=0x([0-9a-fA-F]+)", raw)
    return _ThrottleState(raw, int(match.group(1), 16)) if match else None


def _proc_values(path: Path, wanted: tuple[str, ...], *, scale: int = 1) -> dict[str, int]:
    """Read selected integer counters from a Linux procfs key-value file."""
    values: dict[str, int] = {}
    text = _read_text(path)
    if text is None:
        return values
    for line in text.splitlines():
        fields = line.replace(":", "").split()
        if len(fields) >= 2 and fields[0] in wanted:
            try:
                values[fields[0]] = int(fields[1]) * scale
            except ValueError:
                continue
    return values


def capture_host_snapshot() -> HostSnapshot:
    """Read the hardware, scheduling, thermal, memory, and cache state now."""
    return HostSnapshot(
        model=_read_text(Path("/proc/device-tree/model")),
        kernel=platform.release(),
        machine=platform.machine(),
        cpu_affinity=tuple(sorted(os.sched_getaffinity(0))),
        load_average=os.getloadavg(),
        frequency=_frequency_policies(),
        temperature=_thermal_zones(),
        throttling=_throttle_state(),
        memory_bytes=_proc_values(Path("/proc/meminfo"), _MEMORY_FIELDS, scale=1024),
        virtual_memory_events=_proc_values(Path("/proc/vmstat"), _VM_FIELDS),
    )
