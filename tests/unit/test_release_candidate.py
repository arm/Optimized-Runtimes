from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path


def _write_archive(path: Path, files: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def _write_checksum(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(f"{path.name}.sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")


def _final_packages(directory: Path) -> None:
    version = "0.1.0"
    runtime = directory / f"arm-yolo26n-{version}-rpi5-trixie-arm64.tar.gz"
    _write_archive(
        runtime,
        {
            f"lib/libarm-yolo26n.so.{version}": b"runtime",
            "lib/libarm-yolo26n.so.0": b"runtime",
            f"python/arm_yolo26n-{version}-py3-none-any.whl": b"wheel",
            f"python/arm_yolo26n_converter-{version}-py3-none-any.whl": b"converter",
            "share/doc/arm-yolo26n/LICENSE": b"project license",
            "share/doc/arm-yolo26n/licenses/xnnpack/LICENSE": b"dependency license",
        },
    )
    _write_checksum(runtime)

    development = directory / f"arm-yolo26n-{version}-rpi5-trixie-arm64-dev.tar.gz"
    _write_archive(
        development,
        {"include/arm_yolo26n.h": b"header", "lib/libarm-yolo26n.so": b"runtime"},
    )
    _write_checksum(development)


def test_build_packages_are_promoted_without_repackaging(tmp_path: Path) -> None:
    packages = tmp_path / "packages"
    packages.mkdir()
    _final_packages(packages)
    revision = "a" * 40
    package_identities = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in packages.iterdir()
    }

    subprocess.run(
        [
            sys.executable,
            "-m",
            "internal.release.candidate",
            "--package-dir",
            str(packages),
            "--version",
            "0.1.0",
            "--revision",
            revision,
            "--build-run-url",
            "https://example.test/build/1",
            "--black-duck-run-url",
            "https://example.test/black-duck/2",
            "--actor",
            "maintainer",
        ],
        check=True,
    )

    distribution = packages / "arm-yolo26n-0.1.0-rpi5-trixie-arm64.tar.gz"
    with tarfile.open(distribution, "r:gz") as archive:
        names = set(archive.getnames())
    assert "lib/libarm-yolo26n.so.0" in names
    assert "python/arm_yolo26n-0.1.0-py3-none-any.whl" in names
    assert "python/arm_yolo26n_converter-0.1.0-py3-none-any.whl" in names
    assert "share/doc/arm-yolo26n/licenses/xnnpack/LICENSE" in names

    assert package_identities == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in packages.iterdir()
        if path.name in package_identities
    }

    manifest_path = packages / "arm-yolo26n-0.1.0-rpi5-trixie-arm64-candidate.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_revision"] == revision
    assert {item["role"] for item in manifest["files"]} == {
        "runnable-distribution",
        "runnable-distribution-checksum",
        "development-archive",
        "development-archive-checksum",
    }
