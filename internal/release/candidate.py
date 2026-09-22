"""Assemble release-candidate files from one retained, tested main build.

The build workflow owns compilation, tests, and the CPack and wheel outputs.
This command verifies those exact files and combines them without compiling or
repackaging either component. Publication remains the workflow's responsibility.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

_VERSION_PATTERN = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True, slots=True)
class _Candidate:
    input_directory: Path
    output_directory: Path
    version: str
    revision: str
    build_run_url: str
    black_duck_run_url: str
    actor: str

    @property
    def runtime_archive(self) -> Path:
        return self.input_directory / f"arm-yolo26n-rpi5-{self.version}-runtime.tar.gz"

    @property
    def development_archive(self) -> Path:
        return self.input_directory / f"arm-yolo26n-rpi5-{self.version}-development.tar.gz"

    @property
    def wheel(self) -> Path:
        return self.input_directory / f"arm_yolo26n-{self.version}-py3-none-any.whl"


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


def _verify_cpack_checksum(archive: Path) -> None:
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
    # Normalize ownership and timestamps so retrying assembly from identical
    # build inputs produces identical distribution bytes.
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
    checksum.write_text(f"{_sha256(path)}  {path.name}\n", encoding="utf-8")
    return checksum


def _validate_inputs(candidate: _Candidate) -> None:
    if not _VERSION_PATTERN.fullmatch(candidate.version):
        raise ValueError(f"invalid version: {candidate.version}")
    if not _REVISION_PATTERN.fullmatch(candidate.revision):
        raise ValueError(f"invalid source revision: {candidate.revision}")

    # An exact set prevents an unrelated package accidentally uploaded by the
    # build job from becoming part of a release candidate.
    expected = {
        candidate.runtime_archive.name,
        f"{candidate.runtime_archive.name}.sha256",
        candidate.development_archive.name,
        f"{candidate.development_archive.name}.sha256",
        candidate.wheel.name,
    }
    actual = {path.name for path in candidate.input_directory.iterdir() if path.is_file()}
    if actual != expected:
        raise ValueError("retained build artifact does not contain the expected five files")

    _verify_cpack_checksum(candidate.runtime_archive)
    _verify_cpack_checksum(candidate.development_archive)
    _verify_archive(
        candidate.runtime_archive,
        {
            f"lib/libarm-yolo26n.so.{candidate.version}",
            "lib/libarm-yolo26n.so.0",
            "share/doc/arm-yolo26n/LICENSE",
            "share/doc/arm-yolo26n/licenses/xnnpack/LICENSE",
        },
    )
    _verify_archive(
        candidate.development_archive,
        {"include/arm_yolo26n.h", "lib/libarm-yolo26n.so"},
    )
    _verify_wheel(candidate.wheel, candidate.version)


def _assemble(candidate: _Candidate) -> tuple[Path, ...]:
    _validate_inputs(candidate)
    if candidate.output_directory.exists():
        raise ValueError(f"output already exists: {candidate.output_directory}")
    candidate.output_directory.mkdir(parents=True)

    release_name = f"arm-yolo26n-{candidate.version}-rpi5-trixie-arm64"
    runtime_output = candidate.output_directory / f"{release_name}.tar.gz"
    development_output = candidate.output_directory / f"{release_name}-dev.tar.gz"

    with tempfile.TemporaryDirectory() as temporary:
        staging = Path(temporary)
        with tarfile.open(candidate.runtime_archive, "r:gz") as runtime:
            runtime.extractall(staging, filter="data")
        wheel_directory = staging / "python"
        wheel_directory.mkdir()
        shutil.copyfile(candidate.wheel, wheel_directory / candidate.wheel.name)
        _write_tar(staging, runtime_output)

    # The development component needs only its public release name. Preserve
    # the CPack-produced bytes rather than giving release assembly a second
    # implementation of that archive.
    shutil.copyfile(candidate.development_archive, development_output)
    runtime_checksum = _write_checksum(runtime_output)
    development_checksum = _write_checksum(development_output)

    manifest = candidate.output_directory / f"{release_name}-candidate.json"
    manifest.write_text(
        json.dumps(
            {
                "kind": "arm-yolo26n-release-candidate",
                "schema_version": 1,
                "project_version": candidate.version,
                "source_revision": candidate.revision,
                "created_by": candidate.actor,
                "build_run_url": candidate.build_run_url,
                "black_duck_run_url": candidate.black_duck_run_url,
                "inputs": [
                    _identity(candidate.runtime_archive, "cpack-runtime"),
                    _identity(candidate.development_archive, "cpack-development"),
                    _identity(candidate.wheel, "python-wheel"),
                ],
                "outputs": [
                    _identity(runtime_output, "runnable-distribution"),
                    _identity(runtime_checksum, "runnable-distribution-checksum"),
                    _identity(development_output, "development-archive"),
                    _identity(development_checksum, "development-archive-checksum"),
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_checksum = _write_checksum(manifest)
    return (
        runtime_output,
        runtime_checksum,
        development_output,
        development_checksum,
        manifest,
        manifest_checksum,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--build-run-url", required=True)
    parser.add_argument("--black-duck-run-url", required=True)
    parser.add_argument("--actor", required=True)
    arguments = parser.parse_args()
    outputs = _assemble(
        _Candidate(
            input_directory=arguments.input_dir.resolve(),
            output_directory=arguments.output_dir.resolve(),
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
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as error:
        raise SystemExit(f"error: {error}") from None
