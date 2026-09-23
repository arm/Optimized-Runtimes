# Testing philosophy and remediation plan

## Philosophy

Test observable behavior through the public interfaces, not implementation
details. A test enters through the interface exposed to users, exercises the
real local implementation and dependencies, and continues to the boundary
where a remote system or unavailable target device begins. A substitute may
stand at that boundary; it must not replace repository-owned code merely to
make a test smaller or faster.

Implementation inspection and coverage reports are diagnostic tools. They can
identify an untested branch and prompt a missing behavioral scenario, but they
do not determine assertions. Refactoring private functions, file layout, call
order, or internal data structures without changing a supported behavior
should not break tests.

For this repository, the public interfaces are:

- the checkpoint export and validation commands;
- the CMake build, test, installation, and packaging commands;
- the C ABI declared in `arm_yolo26n.h`;
- the public objects exported by `arm_yolo26n`;
- the registered Ultralytics backend and benchmark command; and
- the command-line interfaces under `examples/`.

The converter, native runtime, Python wrapper, detector, OpenCV integration, and
Ultralytics integration are local implementation. Tests of their combined
behavior should use them rather than imitating them. Network downloads,
cameras, and operating-system hardware telemetry are examples of replaceable
boundaries. End-to-end tests execute the real runtime on Linux AArch64.
Raspberry Pi hardware behavior is not simulated to make camera, thermal, or
performance claims; those checks belong in hardware validation.

## Running the suites

`make lint` and the test targets prepare the hash-locked development
environment under `.venv/` when it is missing or outdated.
`make lint` runs formatting, static analysis, typing, import-boundary, and
syntax checks. Pytest discovers the portable, checkpoint-backed, and
end-to-end suites from `tests/unit`, `tests/converter`, and `tests/end_to_end`
respectively. `make test-unit` runs unit and command-interface tests with
NumPy and OpenCV but without the native runtime. `make model-check
CHECKPOINT=... RESOLUTION=...` validates one requested conversion, while
`make test-converter CHECKPOINT=...` runs the checkpoint-backed converter
behavior suite as a focused development command. Neither requires Torch or
Ultralytics.

`make test-end-to-end` first runs that converter behavior suite, then loads the
real native library and converted models on Linux AArch64 and runs inference
through the public NumPy runtime, detector, video, and camera-example
interfaces. `CHECKPOINT` supplies the converter fixture, `MODELS` must be a
colon-separated list containing exactly one converted model for each supported
resolution, and `RUNTIME` names the installed native library. The default
`MODELS` paths are prepared automatically; explicitly selected external paths
remain caller-provided artifacts. Only the physical camera boundary is
replaced.

Hardware validation runs separately on Raspberry Pi 5. It covers camera
operation, performance, thermals, throttling, and other properties of the
target device. `make ultralytics-benchmark` exercises the optional
Torch/Ultralytics adapter and records accuracy and performance evidence. A unit
or end-to-end functional pass is not hardware-validation evidence.

`make coverage CHECKPOINT=... MODELS=...` reruns the functional suites against
an instrumented first-party runtime and writes separate Python and native
Cobertura XML reports under `build/coverage`. It is a manual diagnostic: the
command has no coverage threshold and does not turn uncovered implementation
branches into test requirements.

## Test design rules

- Assert returned values, emitted artifacts, documented errors, process exit
  status, and externally visible resource behavior.
- Prefer one representative success path and table-driven rejection cases over
  many tests that repeat the same setup.
- Do not import private names, patch repository-owned collaborators, inspect
  private call arguments, or assert source text and internal object counts.
- Do not reproduce production calculations in a fake and then use that fake as
  the expected result. Use an independent reference implementation or a
  previously qualified artifact when numerical comparison is required.
- Keep exact byte or floating-point assertions only where exactness is part of
  an artifact, ABI, or numerical-accuracy contract.
- Test unsupported inputs because the product must reject them before unsafe or
  meaningless execution, not to accumulate speculative security cases.
- Keep functional correctness, target accuracy, and target performance as
  distinct gates with clearly stated environments.
- A test module begins with a short description of the public interface under
  test, the real dependencies exercised, and any external boundary replaced.
  Test names describe the user-visible scenario and result.
- In a test with substantial setup or several similar values, brief comments
  mark where setup ends, behavior is exercised, and results are checked. Omit
  these labels when the three phases are already obvious.

## Remediation checklist

Work through this list in order. Delete a test when its only purpose is covered
by static analysis or by a stronger public-interface test; do not preserve it
only to retain a test count.

### Shared test structure

- [x] Define shared command helpers for converter, model, release, and optional
  Ultralytics package behavior. Converter tests use a real caller-supplied
  checkpoint. Packaging tests use the smallest valid producer input so they do
  not acquire Torch merely to test CMake assembly.
- [x] Separate unit tests, end-to-end native execution, and Raspberry Pi
  hardware validation in the Make and CI entry points.
- [x] Document the interface and substituted external boundary at the top of
  every retained test module.
- [x] Consolidate repeated invalid-input cases into readable tables and remove
  assertions that merely repeat constants or production calculations.

### Compiler

- [x] Rewrite `tests/converter/test_converter.py` to invoke the export and
  validation commands. Retain observable acceptance, rejection, deterministic
  export, class metadata, and supported-resolution behavior.
- [x] Remove its direct tests of checkpoint-decoder stack operations,
  `parse_tensor_rebuild`, `_validation_result`, and other private parsing
  stages. Express required rejection through malformed or unsupported
  checkpoint inputs at the command boundary.
- [x] Collapse the many checkpoint mutation tests into a table of distinct
  supported and unsupported model behaviors, reporting the public diagnostic
  for each case.
- [x] Run compiler functional behavior explicitly with a real supported
  checkpoint. Routine portable checks do not synthesize checkpoints through
  Torch or Ultralytics merely to exercise the dependency-free compiler.
- [x] Replace converter tests of `_float32`,
  `_EffectiveTensor`, layout helpers, split helpers, and fusion call order with
  exported-artifact and reference-output tests. Retain only numerical cases
  that can detect a real exported-model or runtime result change.
- [x] Replace `tests/test_minimal_model_format.py` private slot and constant
  tests with public export determinism, learned-value independence, checkpoint
  tensor-order independence, and public rejection of damaged artifacts. Remove
  the monkeypatch of `RUNTIME_WEIGHT_NAMES`.

### Native runtime and model

- [x] Add end-to-end tests that load real exported models through the public
  Python API, execute them, and cover every supported resolution.
- [x] Exercise public rejection for incompatible tensors and damaged models.
  Assertions use documented errors and output behavior, not private validators.
- [x] Delete `buffer_validation_test.cpp` once its meaningful cases are covered
  through the C ABI.
- [x] Delete `fixed_memory_plan_test.cpp` once successful initialization and
  execution of every supported profile exercise the real plan. Use sanitizers
  as diagnostics rather than making private arena offsets a contract.
- [x] Replace `xnnpack_specialization_test.cpp` checks of XNNPACK private
  structures, operator counts, flags, and fallback construction with public
  runtime behavior and Pi performance evidence. Keep a
  narrowly labelled patch-maintenance arithmetic test only if it detects a
  failure that cannot be observed reliably at the runtime boundary.
- [x] Keep native build verification at the CMake and CTest command boundary.
  Do not reproduce CMake installation or packaging behavior in a Python test;
  cover the converter-to-runtime path in explicit converter and end-to-end
  workflows.
- [x] Replace the source-text and private-function assertions in
  `tests/test_runtime_source.py`: verify installed licenses through
  CMake packaging and invoke benchmark-context recording through its command
  interface. Do not retain help-only tests as substitutes for those behaviors.
- [x] Remove Python compilation checks from
  `tests/test_runtime_source.py`; the quality gate already owns syntax,
  formatting, typing, and lint analysis.

### Python runtime and detector

- [x] Keep portable admission behavior in `tests/unit/test_runtime_api.py` and run
  real runtime/model execution through the public `Yolo26nTensorSession` API
  in the end-to-end suite. Do not replace `ctypes.CDLL` with a Python imitation of the
  repository's native runtime.
- [x] Move preprocessing coverage to public detector results and remove calls
  to `_prepare_image_without_resize`. Cover BGR, RGB, BGRX, padded strides,
  resizing, and supported resolutions by their observable detections or
  reference tensors at the public boundary.
- [x] Retain public model-integrity, tensor shape/dtype, lifecycle, and native
  contract rejection without asserting whether the library loader was called.
- [x] Rewrite `tests/test_detector.py` to construct a real
  `Yolo26Detector`. Remove fake tensor sessions, patched local preprocessing,
  patched NMS, captured internal tensors, and assertions about internal
  dispatch.
- [x] Cover detector preprocessing, coordinate scaling, thresholds, class
  names, callback timing, invalid frames, and lifecycle through returned
  detections and documented errors. Prove that callback timing permits safe
  input reuse rather than merely recording that the callback ran. Merge
  duplicate constructor and metadata cases.
- [x] Exercise backend behavior through `make ultralytics-benchmark` with a
  real runtime and model produced by the public commands. Remove the fake
  tensor session, direct backend construction, and assertions about internal
  construction.
- [x] Reduce backend rejection cases to malformed or incompatible package
  inputs that the supported interface must reject; discard speculative path
  policing that is not a product requirement.
- [x] Remove the custom import blocker. Dependency manifests define the
  supported installation surface, and the Pi public-interface suite exercises
  both NumPy APIs without making framework-import internals a separate product
  behavior.

### Packaging and examples

- [x] Remove the obsolete Ultralytics package-assembly test with the packaged
  workspace. The public benchmark command creates its descriptor temporarily
  and proves the integration by completing Ultralytics validation.
- [x] Remove help-only example tests. Exercise the video and camera commands
  through their real behavior; only the unavailable camera API may be replaced
  at that external boundary.
- [x] Exercise the prerecorded-video example with the real detector on the
  end-to-end path. Keep a unit-level renderer test using real
  OpenCV and a documented detection-results fixture, and verify every output
  frame and a visible annotation; remove the fake detector and synthetic
  timing assertions from `tests/unit/test_video_example.py`.
- [x] Keep `tests/end_to_end/test_examples.py` substituting only the unavailable
  camera boundary. Use the real detector and assert observable
  capture, request release, result reporting, rendered preview, and shutdown
  behavior rather than internal pipeline ordering or fake call histories.
- [x] Delete the direct `tests/test_picamera2_overlay.py` implementation tests.
  Exercise the real overlay through the live camera example while substituting
  only the Picamera2 boundary.

### Completion

- [x] Re-read every retained assertion and confirm that a private refactor with
  unchanged public behavior would leave it passing.
- [x] Run the unit suite from a clean environment and report which product
  behavior it covers without counting end-to-end or hardware validation as passes.
- [x] Run real runtime/model functional behavior for all six resolutions on
  Raspberry Pi 5. Compare NumPy detector latency with the prior Torch/Ultralytics path,
  and rerun labeled accuracy measurement when preprocessing, filtering, or NMS
  changes.
