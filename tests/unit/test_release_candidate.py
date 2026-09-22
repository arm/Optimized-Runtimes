from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tarfile
import zipfile
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


def _build_artifact(directory: Path) -> None:
    version = "0.1.0"
    runtime = directory / f"arm-yolo26n-rpi5-{version}-runtime.tar.gz"
    _write_archive(
        runtime,
        {
            f"lib/libarm-yolo26n.so.{version}": b"runtime",
            "lib/libarm-yolo26n.so.0": b"runtime",
            "share/doc/arm-yolo26n/LICENSE": b"project license",
            "share/doc/arm-yolo26n/licenses/xnnpack/LICENSE": b"dependency license",
        },
    )
    _write_checksum(runtime)

    development = directory / f"arm-yolo26n-rpi5-{version}-development.tar.gz"
    _write_archive(
        development,
        {"include/arm_yolo26n.h": b"header", "lib/libarm-yolo26n.so": b"runtime"},
    )
    _write_checksum(development)

    wheel = directory / f"arm_yolo26n-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as package:
        package.writestr(
            f"arm_yolo26n-{version}.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: arm-yolo26n\nVersion: 0.1.0\n",
        )


def test_candidate_command_assembles_the_tested_build_artifact(tmp_path: Path) -> None:
    inputs = tmp_path / "input"
    inputs.mkdir()
    _build_artifact(inputs)
    outputs = tmp_path / "output"
    revision = "a" * 40

    subprocess.run(
        [
            sys.executable,
            "-m",
            "internal.release.candidate",
            "--input-dir",
            str(inputs),
            "--output-dir",
            str(outputs),
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

    distribution = outputs / "arm-yolo26n-0.1.0-rpi5-trixie-arm64.tar.gz"
    with tarfile.open(distribution, "r:gz") as archive:
        names = set(archive.getnames())
    assert "lib/libarm-yolo26n.so.0" in names
    assert "python/arm_yolo26n-0.1.0-py3-none-any.whl" in names
    assert "share/doc/arm-yolo26n/licenses/xnnpack/LICENSE" in names

    manifest_path = outputs / "arm-yolo26n-0.1.0-rpi5-trixie-arm64-candidate.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_revision"] == revision
    assert [item["role"] for item in manifest["inputs"]] == [
        "cpack-runtime",
        "cpack-development",
        "python-wheel",
    ]
    assert {item["role"] for item in manifest["outputs"]} == {
        "runnable-distribution",
        "runnable-distribution-checksum",
        "development-archive",
        "development-archive-checksum",
    }
