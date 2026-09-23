"""Verify final build packages and record their release-candidate evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path

_VERSION_PATTERN = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, role: str) -> dict[str, object]:
    return {
        "name": path.name,
        "role": role,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _verify_checksum(archive: Path) -> None:
    checksum = archive.with_name(f"{archive.name}.sha256")
    fields = checksum.read_text(encoding="utf-8").split()
    if len(fields) != 2 or fields[1] != archive.name:
        raise ValueError(f"invalid checksum file: {checksum}")
    if fields[0] != _sha256(archive):
        raise ValueError(f"checksum mismatch: {archive}")


def _verify_archive(archive: Path, required_paths: set[str]) -> None:
    with tarfile.open(archive, "r:gz") as contents:
        paths = {member.name.rstrip("/") for member in contents.getmembers()}
    missing = required_paths - paths
    if missing:
        raise ValueError(f"{archive.name} is missing: {', '.join(sorted(missing))}")


@dataclass(frozen=True, slots=True)
class _Evidence:
    package_directory: Path
    version: str
    revision: str
    build_run_url: str
    black_duck_run_url: str
    actor: str

    @property
    def runtime(self) -> Path:
        return self.package_directory / f"arm-yolo26n-{self.version}-rpi5-trixie-arm64.tar.gz"

    @property
    def runtime_checksum(self) -> Path:
        return self.runtime.with_name(f"{self.runtime.name}.sha256")

    @property
    def development(self) -> Path:
        name = f"arm-yolo26n-{self.version}-rpi5-trixie-arm64-dev.tar.gz"
        return self.package_directory / name

    @property
    def development_checksum(self) -> Path:
        return self.development.with_name(f"{self.development.name}.sha256")

    @property
    def package_files(self) -> tuple[Path, ...]:
        return (
            self.runtime,
            self.runtime_checksum,
            self.development,
            self.development_checksum,
        )

    def verify(self) -> None:
        if not _VERSION_PATTERN.fullmatch(self.version):
            raise ValueError(f"invalid version: {self.version}")
        if not _REVISION_PATTERN.fullmatch(self.revision):
            raise ValueError(f"invalid source revision: {self.revision}")
        actual = {path.name for path in self.package_directory.iterdir() if path.is_file()}
        expected = {path.name for path in self.package_files}
        if actual != expected:
            raise ValueError("build artifact does not contain the expected four package files")
        _verify_checksum(self.runtime)
        _verify_checksum(self.development)
        _verify_archive(
            self.runtime,
            {
                f"lib/libarm-yolo26n.so.{self.version}",
                "lib/libarm-yolo26n.so.0",
                f"python/arm_yolo26n-{self.version}-py3-none-any.whl",
                f"python/arm_yolo26n_converter-{self.version}-py3-none-any.whl",
                "share/doc/arm-yolo26n/LICENSE",
                "share/doc/arm-yolo26n/licenses/xnnpack/LICENSE",
            },
        )
        _verify_archive(
            self.development,
            {"include/arm_yolo26n.h", "lib/libarm-yolo26n.so"},
        )


def _write_evidence(evidence: _Evidence) -> tuple[Path, Path]:
    evidence.verify()
    name = f"arm-yolo26n-{evidence.version}-rpi5-trixie-arm64-candidate.json"
    manifest = evidence.package_directory / name
    checksum = manifest.with_name(f"{manifest.name}.sha256")
    roles = (
        "runnable-distribution",
        "runnable-distribution-checksum",
        "development-archive",
        "development-archive-checksum",
    )
    manifest.write_text(
        json.dumps(
            {
                "kind": "arm-yolo26n-release-candidate",
                "schema_version": 1,
                "project_version": evidence.version,
                "source_revision": evidence.revision,
                "created_by": evidence.actor,
                "build_run_url": evidence.build_run_url,
                "black_duck_run_url": evidence.black_duck_run_url,
                "files": [
                    _identity(path, role)
                    for path, role in zip(evidence.package_files, roles, strict=True)
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    checksum.write_text(f"{_sha256(manifest)}  {manifest.name}\n", encoding="utf-8")
    return manifest, checksum


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--build-run-url", required=True)
    parser.add_argument("--black-duck-run-url", required=True)
    parser.add_argument("--actor", required=True)
    arguments = parser.parse_args()
    outputs = _write_evidence(
        _Evidence(
            package_directory=arguments.package_dir.resolve(),
            version=arguments.version,
            revision=arguments.revision,
            build_run_url=arguments.build_run_url,
            black_duck_run_url=arguments.black_duck_run_url,
            actor=arguments.actor,
        )
    )
    print("\n".join(str(path) for path in outputs))


if __name__ == "__main__":
    try:
        _main()
    except (OSError, ValueError, tarfile.TarError) as error:
        raise SystemExit(f"error: {error}") from None
