"""Public checkpoint export and validation command behavior.

Pinned Ultralytics constructs the local checkpoint fixture. The real converter
reads, validates, lowers, and serializes it; no conversion stage is replaced.
"""

from __future__ import annotations

import hashlib
import pickle
import pickletools
import unittest
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.support import checkpoint, exported_model, read_json, run_module

type ArchiveTransform = Callable[[str, bytes], bytes | None]
type PickleTransform = Callable[[bytes], bytes]


def _rewrite_checkpoint(
    output: Path,
    transform: ArchiveTransform,
    additions: Mapping[str, bytes] | None = None,
) -> None:
    """Create a checkpoint variant without importing Torch or converter internals."""
    with (
        zipfile.ZipFile(checkpoint()) as source,
        zipfile.ZipFile(output, "w") as destination,
    ):
        for member in source.infolist():
            payload = transform(member.filename, source.read(member))
            if payload is not None:
                destination.writestr(member, payload)
        for name, payload in (additions or {}).items():
            destination.writestr(name, payload)


def _replace_data_pickle(transform: PickleTransform) -> ArchiveTransform:
    def replace(name: str, payload: bytes) -> bytes:
        return transform(payload) if name.endswith("/data.pkl") else payload

    return replace


def _fixed_data_pickle(replacement: bytes) -> ArchiveTransform:
    return _replace_data_pickle(lambda _payload: replacement)


def _replace_pickled_attribute(
    payload: bytes,
    name: str,
    expected_operation: str,
    expected_value: object,
    replacement: bytes,
    *,
    occurrence: int = 0,
) -> bytes:
    """Replace one fixed-width value associated with a named pickle attribute."""
    operations = list(pickletools.genops(payload))
    matches = [index for index, (_, value, _) in enumerate(operations) if value == name]
    key_index = matches[occurrence]
    value_index = key_index + 1
    while operations[value_index][0].name in {"BINPUT", "LONG_BINPUT", "MEMOIZE"}:
        value_index += 1
    operation, value, begin = operations[value_index]
    end = operations[value_index + 1][2]
    if (operation.name, value) != (expected_operation, expected_value):
        raise AssertionError(f"unexpected {name} encoding: {(operation.name, value)!r}")
    if len(replacement) != end - begin:
        raise AssertionError(f"replacement changes the encoded size of {name}")
    return payload[:begin] + replacement + payload[end:]


class ConverterCommandTests(unittest.TestCase):
    def assert_export_rejected(self, source: Path, diagnostic: str) -> None:
        output = source.with_name(source.stem + "-output")
        completed = run_module(
            "converter.export_yolo26n",
            "--checkpoint",
            source,
            "--resolution",
            320,
            "--output-dir",
            output,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("YOLO26n export rejected", completed.stderr)
        self.assertIn(diagnostic, completed.stderr)
        self.assertFalse(output.exists())

    def test_export_is_deterministic_and_describes_the_supported_model(self) -> None:
        first = exported_model(320)
        with TemporaryDirectory() as directory:
            second = Path(directory) / "export"
            completed = run_module(
                "converter.export_yolo26n",
                "--checkpoint",
                checkpoint(),
                "--resolution",
                320,
                "--output-dir",
                second,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                (first / "model.bin").read_bytes(), (second / "model.bin").read_bytes()
            )
            self.assertEqual(
                read_json(first / "manifest.json"), read_json(second / "manifest.json")
            )

        manifest = read_json(first / "manifest.json")
        model = manifest["model"]
        self.assertIsInstance(model, dict)
        assert isinstance(model, dict)
        self.assertEqual(model["input_shape"], [1, 3, 320, 320])
        self.assertEqual(model["output_shape"], [1, 84, 2100])
        self.assertEqual(len(model["class_names"]), 80)

    def test_every_documented_resolution_exports_its_public_shapes(self) -> None:
        for resolution in (320, 480, 640, 800, 960, 1280):
            with self.subTest(resolution=resolution):
                manifest = read_json(exported_model(resolution) / "manifest.json")
                model = manifest["model"]
                assert isinstance(model, dict)
                positions = sum((resolution // stride) ** 2 for stride in (8, 16, 32))
                self.assertEqual(model["input_shape"], [1, 3, resolution, resolution])
                self.assertEqual(model["output_shape"], [1, 84, positions])

    def test_validation_command_reports_the_exported_artifact(self) -> None:
        with TemporaryDirectory() as directory:
            receipt = Path(directory) / "validation.json"
            completed = run_module(
                "converter.validate",
                "--checkpoint",
                checkpoint(),
                "--resolution",
                320,
                "--output",
                receipt,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = read_json(receipt)

        payload = (exported_model(320) / "model.bin").read_bytes()
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(
            result["artifact"],
            {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
        )

    def test_invalid_inputs_are_rejected_without_publishing_output(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            malformed = root / "malformed.pt"
            malformed.write_bytes(b"not a checkpoint")
            cases = ((malformed, 320), (checkpoint(), 641))
            for index, (source, resolution) in enumerate(cases):
                with self.subTest(source=source.name, resolution=resolution):
                    output = root / f"rejected-{index}"
                    completed = run_module(
                        "converter.export_yolo26n",
                        "--checkpoint",
                        source,
                        "--resolution",
                        resolution,
                        "--output-dir",
                        output,
                    )
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn("YOLO26n export rejected", completed.stderr)
                    self.assertFalse(output.exists())

    def test_invalid_archive_contracts_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            # Setup: derive structurally invalid archives from the real checkpoint.
            missing_byteorder = root / "missing-byteorder.pt"
            _rewrite_checkpoint(
                missing_byteorder,
                lambda name, payload: None if name.endswith("/byteorder") else payload,
            )

            wrong_byteorder = root / "wrong-byteorder.pt"
            _rewrite_checkpoint(
                wrong_byteorder,
                lambda name, payload: b"big" if name.endswith("/byteorder") else payload,
            )

            ambiguous_payload = root / "ambiguous-payload.pt"
            _rewrite_checkpoint(
                ambiguous_payload,
                lambda _name, payload: payload,
                {"other/data.pkl": pickle.dumps(None, protocol=2)},
            )

            unsupported_pickle = root / "unsupported-pickle.pt"
            active_pickle = pickle.PROTO + b"\x02" + pickle.NONE + pickle.POP + pickle.STOP
            _rewrite_checkpoint(
                unsupported_pickle,
                _replace_data_pickle(lambda _payload: active_pickle),
            )

            truncated_storage = root / "truncated-storage.pt"
            storage_changed = False

            def truncate_first_storage(name: str, payload: bytes) -> bytes:
                nonlocal storage_changed
                parts = name.split("/")
                if not storage_changed and len(parts) == 3 and parts[-2] == "data":
                    storage_changed = True
                    return payload[:-1]
                return payload

            _rewrite_checkpoint(truncated_storage, truncate_first_storage)
            self.assertTrue(storage_changed)

            # Exercise and verify each distinct archive rejection through the command.
            cases = (
                (missing_byteorder, "requires one byteorder member"),
                (wrong_byteorder, "tensor storage must be little-endian"),
                (ambiguous_payload, "one unambiguous Torch data.pkl payload"),
                (unsupported_pickle, "unsupported pickle opcode: POP"),
                (truncated_storage, "storage extent mismatch"),
            )
            for source, diagnostic in cases:
                with self.subTest(checkpoint=source.name):
                    self.assert_export_rejected(source, diagnostic)

    def test_invalid_checkpoint_roots_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            # Each replacement is valid pickle but not the required checkpoint root.
            cases = (
                ("non-mapping", None, "checkpoint root must be a mapping"),
                ("missing-model", {"model": None}, "checkpoint model must be"),
            )
            for name, value, diagnostic in cases:
                with self.subTest(checkpoint=name):
                    source = root / f"{name}.pt"
                    replacement = pickle.dumps(value, protocol=2)
                    _rewrite_checkpoint(
                        source,
                        _fixed_data_pickle(replacement),
                    )
                    self.assert_export_rejected(source, diagnostic)

    def test_incompatible_model_contracts_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            # Setup: change one runtime-relevant model fact in each real checkpoint copy.
            wrong_model_kind = root / "wrong-model-kind.pt"

            def replace_model_kind(payload: bytes) -> bytes:
                expected = b"cultralytics.nn.tasks\nDetectionModel\n"
                replacement = b"cultralytics.nn.tasks\nDetectioXModel\n"
                if payload.count(expected) != 1 or len(expected) != len(replacement):
                    raise AssertionError("unexpected DetectionModel pickle encoding")
                return payload.replace(expected, replacement, 1)

            _rewrite_checkpoint(wrong_model_kind, _replace_data_pickle(replace_model_kind))

            wrong_routing = root / "wrong-routing.pt"
            _rewrite_checkpoint(
                wrong_routing,
                _replace_data_pickle(
                    lambda payload: _replace_pickled_attribute(
                        payload,
                        "i",
                        "BININT1",
                        0,
                        pickle.BININT1 + b"\x63",
                    )
                ),
            )

            wrong_kernel = root / "wrong-kernel.pt"
            _rewrite_checkpoint(
                wrong_kernel,
                _replace_data_pickle(
                    lambda payload: _replace_pickled_attribute(
                        payload,
                        "kernel_size",
                        "BININT1",
                        3,
                        pickle.BININT1 + b"\x05",
                    )
                ),
            )

            missing_residual = root / "missing-residual.pt"
            _rewrite_checkpoint(
                missing_residual,
                _replace_data_pickle(
                    lambda payload: _replace_pickled_attribute(
                        payload,
                        "add",
                        "NEWTRUE",
                        None,
                        pickle.NEWFALSE,
                    )
                ),
            )

            wrong_class_count = root / "wrong-class-count.pt"
            _rewrite_checkpoint(
                wrong_class_count,
                _replace_data_pickle(
                    lambda payload: _replace_pickled_attribute(
                        payload,
                        "nc",
                        "BININT1",
                        80,
                        pickle.BININT1 + b"\x4f",
                    )
                ),
            )

            nonfinite_tensor = root / "nonfinite-tensor.pt"
            tensor_changed = False

            def replace_first_tensor_value(name: str, payload: bytes) -> bytes:
                nonlocal tensor_changed
                parts = name.split("/")
                if not tensor_changed and len(parts) == 3 and parts[-2] == "data":
                    tensor_changed = True
                    return b"\x00\x7c" + payload[2:]
                return payload

            _rewrite_checkpoint(nonfinite_tensor, replace_first_tensor_value)
            self.assertTrue(tensor_changed)

            # Exercise and verify the fixed-model contract through the export command.
            cases = (
                (wrong_model_kind, "must be ultralytics.nn.tasks.DetectionModel"),
                (wrong_routing, "module 0 expected"),
                (wrong_kernel, "invalid tensor dtype/shape: model.0.conv.weight"),
                (missing_residual, "must retain its residual connection"),
                (wrong_class_count, "detection head must contain"),
                (nonfinite_tensor, "non-finite learned tensor"),
            )
            for source, diagnostic in cases:
                with self.subTest(checkpoint=source.name):
                    self.assert_export_rejected(source, diagnostic)

    def test_existing_output_is_left_unchanged(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "existing"
            output.mkdir()
            sentinel = output / "sentinel"
            sentinel.write_text("keep\n", encoding="utf-8")
            completed = run_module(
                "converter.export_yolo26n",
                "--checkpoint",
                checkpoint(),
                "--resolution",
                320,
                "--output-dir",
                output,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep\n")


if __name__ == "__main__":
    unittest.main()
