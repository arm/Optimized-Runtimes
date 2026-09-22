"""Verify final build packages and record their release-candidate evidence."""

from __future__ import annotations

import argparse
import json
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path

from .distribution import Distribution, file_identity, sha256_file

_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True, slots=True)
class _Evidence:
    distribution: Distribution
    revision: str
    build_run_url: str
    black_duck_run_url: str
    actor: str


def _write_evidence(evidence: _Evidence) -> tuple[Path, Path]:
    evidence.distribution.verify()
    if not _REVISION_PATTERN.fullmatch(evidence.revision):
        raise ValueError(f"invalid source revision: {evidence.revision}")

    name = f"arm-yolo26n-{evidence.distribution.version}-rpi5-trixie-arm64-candidate.json"
    manifest = evidence.distribution.directory / name
    checksum = manifest.with_name(f"{manifest.name}.sha256")
    if manifest.exists() or checksum.exists():
        raise ValueError("candidate evidence already exists")

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
                "project_version": evidence.distribution.version,
                "source_revision": evidence.revision,
                "created_by": evidence.actor,
                "build_run_url": evidence.build_run_url,
                "black_duck_run_url": evidence.black_duck_run_url,
                "files": [
                    file_identity(path, role)
                    for path, role in zip(evidence.distribution.files, roles, strict=True)
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    checksum.write_text(f"{sha256_file(manifest)}  {manifest.name}\n", encoding="utf-8")
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
            distribution=Distribution(arguments.package_dir.resolve(), arguments.version),
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
