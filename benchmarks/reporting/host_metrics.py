"""Collect and qualify Raspberry Pi state around benchmark execution.

The collector never changes governors or clears filesystem caches. It records
descriptive snapshots and can reject active memory pressure or RAM-backed
benchmark paths before a performance run. Missing optional Linux and Raspberry
Pi interfaces remain absent so unsupported observations do not block a run.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Self

__all__ = ["HostPreflight", "HostSnapshot", "capture_host_preflight", "capture_host_snapshot"]

_MIB = 1024 * 1024

# MemAvailable estimates memory usable without swapping. Cached and
# SReclaimable show the size of reclaimable filesystem/slab caches implicated
# in the observed RAM-pressure issue. Swap totals distinguish configured swap
# from swap space that was actually consumed.
_MEMORY_FIELDS = (
    "MemTotal",
    "MemAvailable",
    "Cached",
    "SReclaimable",
    "Shmem",
    "Dirty",
    "Writeback",
    "SwapTotal",
    "SwapFree",
)

# Deltas between snapshots reveal disk-backed faults and swap traffic during
# the benchmark without trying to prescribe a policy for those observations.
_VM_FIELDS = (
    "pgmajfault",
    "pswpin",
    "pswpout",
    "workingset_refault_file",
    "pgscan_direct",
    "pgscan_kswapd",
)


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


@dataclass(frozen=True, slots=True)
class HostPreflight:
    """One idle observation window used to qualify benchmark conditions."""

    before: HostSnapshot
    after: HostSnapshot
    filesystems: dict[str, str | None]
    problems: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "before": self.before.to_json(),
            "after": self.after.to_json(),
            "filesystems": self.filesystems,
            "problems": list(self.problems),
        }

    def summary(self) -> str:
        memory = self.before.memory_bytes
        available = memory.get("MemAvailable", 0)
        file_cache = (
            memory.get("Cached", 0) + memory.get("SReclaimable", 0) - memory.get("Shmem", 0)
        )
        pending_write = memory.get("Dirty", 0) + memory.get("Writeback", 0)
        refault, scan, _ = self._activity()
        filesystems = ", ".join(
            f"{path}={kind or 'unknown'}" for path, kind in self.filesystems.items()
        )
        return (
            "Benchmark preflight: "
            f"available={available / _MIB:.0f} MiB, "
            f"file_cache={file_cache / _MIB:.0f} MiB, "
            f"dirty+writeback={pending_write / _MIB:.1f} MiB, "
            f"refault={refault / _MIB:.1f} MiB/s, "
            f"scan={scan / _MIB:.1f} MiB/s, "
            f"filesystems=[{filesystems}]"
        )

    def _activity(self) -> tuple[int, int, int]:
        before = self.before.virtual_memory_events
        after = self.after.virtual_memory_events

        def changed(name: str) -> int:
            return max(0, after.get(name, 0) - before.get(name, 0))

        page_bytes = os.sysconf("SC_PAGE_SIZE")
        refault = changed("workingset_refault_file") * page_bytes
        scan = (changed("pgscan_direct") + changed("pgscan_kswapd")) * page_bytes
        swap = changed("pswpin") + changed("pswpout")
        return refault, scan, swap

    @classmethod
    def from_observations(
        cls,
        before: HostSnapshot,
        after: HostSnapshot,
        filesystems: dict[str, str | None],
    ) -> Self:
        memory = before.memory_bytes
        total = memory.get("MemTotal")
        available = memory.get("MemAvailable")
        pending_write = memory.get("Dirty", 0) + memory.get("Writeback", 0)
        provisional = cls(before, after, filesystems, ())
        refault, scan, swap = provisional._activity()
        problems = [
            f"benchmark path {path} is on RAM-backed {kind}"
            for path, kind in filesystems.items()
            if kind in {"tmpfs", "ramfs"}
        ]
        if (
            total is not None
            and available is not None
            and available < max(_MIB * 1024, total * 15 // 100)
        ):
            problems.append("available memory is too low")
        if total is not None and pending_write > max(_MIB * 256, total * 5 // 100):
            problems.append("too much dirty or writeback cache data")
        if refault > _MIB * 64:
            problems.append("file-cache refault activity is high")
        if scan > _MIB * 64:
            problems.append("memory reclaim is actively scanning pages")
        if swap:
            problems.append("swap I/O occurred during the preflight interval")
        return cls(before, after, filesystems, tuple(problems))


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


def _filesystem_type(path: Path) -> str | None:
    if shutil.which("findmnt") is None:
        return None
    try:
        completed = subprocess.run(
            ["findmnt", "-n", "-o", "FSTYPE", "--target", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    return completed.stdout.strip() or None


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


def capture_host_preflight(
    paths: tuple[Path, ...], *, interval_seconds: float = 1.0
) -> HostPreflight:
    """Observe an idle interval and identify conditions that distort timing."""
    filesystems = {str(path): _filesystem_type(path) for path in paths}
    before = capture_host_snapshot()
    time.sleep(interval_seconds)
    after = capture_host_snapshot()
    return HostPreflight.from_observations(before, after, filesystems)
