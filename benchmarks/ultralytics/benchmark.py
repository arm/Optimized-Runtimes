#!/usr/bin/env python3
"""Run Ultralytics validation and retain its performance evidence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.reporting import record_benchmark


def _read_resolution(manifest_path: Path) -> int:
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping) or (
        document.get("kind"),
        document.get("version"),
    ) != ("yolo26n-exported-model", 1):
        raise RuntimeError("unsupported YOLO26n model manifest")
    model = document.get("model")
    shape = model.get("input_shape") if isinstance(model, Mapping) else None
    if (
        not isinstance(shape, list)
        or len(shape) != 4
        or shape[:2] != [1, 3]
        or shape[2] != shape[3]
        or shape[2] not in (320, 480, 640, 800, 960, 1280)
    ):
        raise RuntimeError("unsupported YOLO26n model input shape")
    return int(shape[2])


def _yolo_command(
    executable: Path,
    descriptor: Path,
    data: str,
    resolution: int,
    extra_arguments: Sequence[str],
) -> list[str]:
    return [
        str(executable),
        "val",
        f"model={descriptor.parent}",
        f"data={data}",
        "task=detect",
        f"imgsz={resolution}",
        "batch=1",
        "device=cpu",
        *extra_arguments,
    ]


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("yolo_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    runtime = args.runtime.expanduser().resolve()
    model = args.model.expanduser().resolve()
    manifest = model / "manifest.json"
    yolo = args.venv.expanduser().resolve() / "bin/yolo"
    if not runtime.is_file():
        raise FileNotFoundError(f"runtime library does not exist: {runtime}")
    if not model.is_dir():
        raise FileNotFoundError(f"converted model does not exist: {model}")
    if not yolo.is_file():
        raise FileNotFoundError("run make ultralytics-patch first")

    resolution = _read_resolution(manifest)
    extra_arguments = args.yolo_arguments
    if extra_arguments[:1] == ["--"]:
        extra_arguments = extra_arguments[1:]

    with TemporaryDirectory(prefix="arm-yolo26n-ultralytics-") as temporary:
        adapter = Path(temporary) / "arm_yolo26n_pi5_model"
        adapter.mkdir()
        descriptor = adapter / "arm-yolo26n-pi5.yaml"
        descriptor.write_text(
            json.dumps(
                {
                    "backend": "arm_yolo26n_pi5",
                    "runtime": str(runtime),
                    "model": str(model),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        benchmark_paths = [runtime, model]
        dataset = Path(args.data).expanduser()
        if dataset.exists():
            benchmark_paths.append(dataset.resolve())
        return_code = record_benchmark(
            args.output_dir.expanduser().resolve(),
            runtime,
            manifest,
            args.revision,
            _yolo_command(yolo, descriptor, args.data, resolution, extra_arguments),
            benchmark_paths,
        )

    print(f"benchmark evidence: {args.output_dir.expanduser().resolve()}")
    if return_code:
        raise SystemExit(return_code)


if __name__ == "__main__":
    try:
        _main()
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"error: {error}") from None
