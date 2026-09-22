"""Build the public distribution from CPack archives and the Python wheel.

CMake and Python packaging own the component files. This command verifies and
combines those files without compiling or rebuilding either component.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import re
import shutil
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Distribution", "file_identity", "sha256_file"]

_VERSION_PATTERN = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 identity of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path, role: str) -> dict[str, object]:
    """Describe one file in a candidate manifest."""
    return {
        "name": path.name,
        "role": role,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _verify_checksum(archive: Path) -> None:
    checksum = archive.with_name(f"{archive.name}.sha256")
    fields = checksum.read_text(encoding="utf-8").split()
    if len(fields) != 2 or fields[1] != archive.name:
        raise ValueError(f"invalid checksum file: {checksum}")
    if fields[0] != sha256_file(archive):
        raise ValueError(f"checksum mismatch: {archive}")


def _verify_archive(archive: Path, required_paths: set[str]) -> None:
    with tarfile.open(archive, "r:gz") as contents:
        paths = {member.name.rstrip("/") for member in contents.getmembers()}
    missing = required_paths - paths
    if missing:
        raise ValueError(f"{archive.name} is missing: {', '.join(sorted(missing))}")


def _verify_wheel(wheel: Path, version: str) -> None:
    metadata_name = f"arm_yolo26n-{version}.dist-info/METADATA"
    with zipfile.ZipFile(wheel) as contents:
        try:
            metadata = contents.read(metadata_name).decode("utf-8")
        except KeyError as error:
            raise ValueError(f"{wheel.name} has no package metadata") from error
    if f"\nVersion: {version}\n" not in f"\n{metadata}":
        raise ValueError(f"{wheel.name} reports a different version")


def _write_tar(source: Path, output: Path) -> None:
    # Normalize ownership and timestamps so identical component files produce
    # identical distribution bytes in every build.
    with output.open("wb") as raw_output:
        with gzip.GzipFile(fileobj=raw_output, mode="wb", filename="", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for path in sorted(
                    source.rglob("*"), key=lambda item: item.relative_to(source).as_posix()
                ):
                    name = path.relative_to(source).as_posix()
                    info = archive.gettarinfo(str(path), arcname=name)
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    if info.isfile():
                        with path.open("rb") as content:
                            archive.addfile(info, content)
                    else:
                        archive.addfile(info)


def _write_checksum(path: Path) -> Path:
    checksum = path.with_name(f"{path.name}.sha256")
    checksum.write_text(f"{sha256_file(path)}  {path.name}\n", encoding="utf-8")
    return checksum


@dataclass(frozen=True, slots=True)
class _Components:
    directory: Path
    version: str

    @property
    def runtime(self) -> Path:
        return self.directory / f"arm-yolo26n-rpi5-{self.version}-runtime.tar.gz"

    @property
    def development(self) -> Path:
        return self.directory / f"arm-yolo26n-rpi5-{self.version}-development.tar.gz"

    @property
    def wheel(self) -> Path:
        return self.directory / f"arm_yolo26n-{self.version}-py3-none-any.whl"

    def verify(self) -> None:
        expected = {
            self.runtime.name,
            f"{self.runtime.name}.sha256",
            self.development.name,
            f"{self.development.name}.sha256",
            self.wheel.name,
        }
        actual = {path.name for path in self.directory.iterdir() if path.is_file()}
        if actual != expected:
            raise ValueError("build package directory does not contain the expected five files")
        _verify_checksum(self.runtime)
        _verify_checksum(self.development)
        _verify_archive(
            self.runtime,
            {
                f"lib/libarm-yolo26n.so.{self.version}",
                "lib/libarm-yolo26n.so.0",
                "share/doc/arm-yolo26n/LICENSE",
                "share/doc/arm-yolo26n/licenses/xnnpack/LICENSE",
            },
        )
        _verify_archive(
            self.development,
            {"include/arm_yolo26n.h", "lib/libarm-yolo26n.so"},
        )
        _verify_wheel(self.wheel, self.version)


@dataclass(frozen=True, slots=True)
class Distribution:
    """The four final files produced and retained by a successful main build."""

    directory: Path
    version: str

    @property
    def runtime(self) -> Path:
        return self.directory / f"arm-yolo26n-{self.version}-rpi5-trixie-arm64.tar.gz"

    @property
    def runtime_checksum(self) -> Path:
        return self.runtime.with_name(f"{self.runtime.name}.sha256")

    @property
    def development(self) -> Path:
        return self.directory / f"arm-yolo26n-{self.version}-rpi5-trixie-arm64-dev.tar.gz"

    @property
    def development_checksum(self) -> Path:
        return self.development.with_name(f"{self.development.name}.sha256")

    @property
    def files(self) -> tuple[Path, ...]:
        return (
            self.runtime,
            self.runtime_checksum,
            self.development,
            self.development_checksum,
        )

    def verify(self) -> None:
        """Reject a missing, damaged, or unexpectedly shaped distribution."""
        if not _VERSION_PATTERN.fullmatch(self.version):
            raise ValueError(f"invalid version: {self.version}")
        if {path.name for path in self.directory.iterdir() if path.is_file()} != {
            path.name for path in self.files
        }:
            raise ValueError("distribution directory does not contain the expected four files")
        _verify_checksum(self.runtime)
        _verify_checksum(self.development)
        _verify_archive(
            self.runtime,
            {
                f"lib/libarm-yolo26n.so.{self.version}",
                "lib/libarm-yolo26n.so.0",
                f"python/arm_yolo26n-{self.version}-py3-none-any.whl",
                "share/doc/arm-yolo26n/LICENSE",
                "share/doc/arm-yolo26n/licenses/xnnpack/LICENSE",
            },
        )
        _verify_archive(
            self.development,
            {"include/arm_yolo26n.h", "lib/libarm-yolo26n.so"},
        )

    @classmethod
    def build(cls, component_directory: Path, output_directory: Path, version: str) -> Distribution:
        """Combine verified component packages into the public distribution."""
        if not _VERSION_PATTERN.fullmatch(version):
            raise ValueError(f"invalid version: {version}")
        components = _Components(component_directory, version)
        components.verify()
        if output_directory.exists():
            raise ValueError(f"output already exists: {output_directory}")
        output_directory.mkdir(parents=True)
        distribution = cls(output_directory, version)

        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary)
            with tarfile.open(components.runtime, "r:gz") as runtime:
                runtime.extractall(staging, filter="data")
            wheel_directory = staging / "python"
            wheel_directory.mkdir()
            shutil.copyfile(components.wheel, wheel_directory / components.wheel.name)
            _write_tar(staging, distribution.runtime)

        # Preserve the CPack-produced development bytes; only its public name
        # changes when it enters the release distribution.
        shutil.copyfile(components.development, distribution.development)
        _write_checksum(distribution.runtime)
        _write_checksum(distribution.development)
        distribution.verify()
        return distribution


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    arguments = parser.parse_args()
    distribution = Distribution.build(
        arguments.input_dir.resolve(),
        arguments.output_dir.resolve(),
        arguments.version,
    )
    print("\n".join(str(path) for path in distribution.files))


if __name__ == "__main__":
    try:
        _main()
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as error:
        raise SystemExit(f"error: {error}") from None
