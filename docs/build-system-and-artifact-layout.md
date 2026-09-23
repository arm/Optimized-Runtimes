# Build system and artifact layout

## Purpose

The build system should expose the few workflows this purpose-built project
actually supports without giving the same lifecycle stage more than one owner.
The runtime is compiled once for Raspberry Pi 5. Checkpoints are converted
independently, and applications select one runtime and one converted model.

This proposal preserves two established architectural decisions:

- CMake owns native dependency acquisition, compilation, native verification,
  and installation. The top-level `make release` command packages the files
  installed by CMake with the converter and application API.
- Converting a checkpoint never invokes the native toolchain or changes the
  runtime library.

The top-level Makefile remains a discoverable command index. It may sequence
commands owned by other components, but it should not reimplement their
validation, packaging, or installation logic.

## Problems addressed

The existing commands work, but their artifact boundaries do not yet express
the architecture:

- `assemble_bundle.cmake` copies the same model-independent runtime into every
  resolution-specific model directory. This makes a deployment bundle look
  like the fundamental artifact even though the runtime and model have
  independent lifecycles.
- CMake installation and `package_runtime.cmake` formerly created two
  differently shaped native archives.
- `dist/` contains unpacked local model bundles and an internal Ultralytics
  workspace as well as release archives. A path under `dist/` therefore does
  not reliably mean that its contents can be published.
- `make examples` creates an environment, downloads inputs, converts a model,
  assembles a bundle, prepares a video, and runs inference. Setup and execution
  cannot be repeated or diagnosed independently.
- Example downloads and the example virtual environment live under `build/`,
  even though neither is native build output. Removing build products also
  removes reusable downloads and the prepared Python environment.
- Example setup always downloads a runtime release. It does not use a valid
  runtime already built from the same checkout.
- The Python API is made importable through a repository-wide `PYTHONPATH`.
  That hides the absence of an application installation contract.

## Lifecycle ownership

| Stage | Owner | Input | Output |
| --- | --- | --- | --- |
| Checkpoint conversion | `src/converter` | Supported checkpoint and resolution | Converted model directory |
| Native configuration and dependencies | CMake | `src/runtime` and pinned dependency revisions | CMake build tree |
| Native compilation and verification | CMake and CTest | Configured native build tree | Runtime shared library and test results |
| Native installation | CMake install rules | Built runtime | Prefix using the platform installation layout |
| Release packages | Top-level Makefile over CMake install rules and Python packaging metadata | Built runtime, converter, and application API | Publishable wheels under `dist/packages/` |
| Example preparation | `examples` workflow | Python package, checkpoint, media, and runtime source | Reusable example inputs and converted model |
| Example execution | Individual example | Runtime, converted model, and input media or camera | Results beside that example under `output/` |
| Ultralytics preparation | `benchmarks/ultralytics` | Pinned benchmark requirements and repository patch | Patched environment under `benchmarks/ultralytics/.venv` |
| Ultralytics benchmark | `benchmarks/ultralytics` | Runtime, converted model, dataset, and patched environment | Ultralytics validation results |
| Performance recording | `benchmarks/reporting` | Benchmark command and artifact identities | Evidence under `benchmarks/reporting/output` |
| Command discovery | Top-level Makefile | User-selected target and paths | Invocation of the owning component |

No new native build wrapper or dependency manager is required.

## Repository-local artifact layout

```text
downloads/
  checkpoints/
    yolo26n.pt
  media/
    test-video.mp4
  runtime/<release>/                Application and converter wheels and checksums

build/
  runtime/                         CMake cache, dependencies, objects, and tests

dist/
  runtime/                         CMake-installed source build for development
  models/<model>-<resolution>/     Locally converted model and manifest
  packages/                        Publishable wheels and checksums

.venv/                             Hash-locked repository check environment
examples/.venv/                    Example application environment
examples/video/output/             Video detections, rendered video, and logs
examples/picamera2/output/         Captured images and detection results
benchmarks/ultralytics/.venv/      Optional heavyweight benchmark environment
benchmarks/reporting/output/       Performance evidence for review
```

The meanings are strict:

- `downloads/` contains verified external files retained for reuse. The
  checkpoint is common to all examples. The test video is added only when the
  prerecorded-video example is requested. Downloaded release wheels and their
  published checksums remain here as reusable example inputs.
- `build/` contains CMake's private native build state. Nothing under `build/`
  is a supported application input.
- `dist/` contains final locally produced artifacts. A source build installs
  the runtime into `dist/runtime/`, while converted models live under
  `dist/models/`. Presence under `dist/` means the artifact is ready to use,
  not that this project has the right to redistribute a particular set of
  weights.
- `dist/packages/` contains the platform-specific application wheel, portable
  converter wheel, and their checksums. These are the files retained for a
  release candidate.
- Environments and user-visible results live with the workflow that owns them.
  A user running the video example finds its output under
  `examples/video/output/`, not in an unrelated build directory.

These generated and downloaded locations are ignored by Git. `make clean`
names and removes `build/` and `dist/`, while preserving downloads,
environments, and user-visible results. There is no single destructive command
that erases every generated location.

### Why anything remains under build

`build/` is limited to state CMake needs while producing and testing the native
runtime:

| Path | Purpose | Why it is transient |
| --- | --- | --- |
| `build/runtime/` | CMake cache, fetched native dependency sources, object files, linker map, the uninstalled shared library, and CTest state | These files contain machine-local configuration and can be recreated from source. The linked library must exist here before CMake can test and install it, but applications never load it from this path. |

`dist/runtime/` is the CMake-installed source build used by runtime developers
and internal benchmarks. Released applications load the same library layout
from the installed application wheel. The shared library in `build/runtime/`
is an unavoidable intermediate in the CMake lifecycle, not a supported runtime
installation.

## Runtime and model boundary

The execution inputs remain separate:

```text
runtime shared library + converted model directory -> one runtime session
```

A converted model directory contains `model.bin` and its manifest. It does not
contain a copy of the runtime. The application wheel contains the shared
library, project license, and native dependency notices. It does not contain
weights.

The Python session and detector interfaces therefore require the model
directory and locate their installed runtime automatically. Resolution and
class metadata continue to come from the model manifest. The Python layer
verifies the `model.bin` bytes against that manifest and passes the bytes to the
native ABI, which copies them and establishes whether the runtime can execute
them. An explicit runtime override remains available for source development
and qualification, not as an ordinary application requirement.

Removing the combined bundle also removes `assemble_bundle.cmake` and the
`yolo26n-bundle` target. Because the software is unreleased, no compatibility
bundle or forwarding command is needed.

## Native packaging

CMake defines the native target, verification tests, and runtime and
development install components. `make release` places CMake's verified runtime
library inside the platform-specific application wheel together with the
project and native-dependency licenses. The converter remains a separate
portable wheel. The project does not currently publish C or C++ development
files; a future Debian package can consume CMake's existing development
component when that interface has real users and a reviewed package contract.

The runtime uses a Debian-compatible library name and SONAME:

```text
shared library:        libarm-yolo26n.so.0
SONAME:                libarm-yolo26n.so.0
development symlink:   libarm-yolo26n.so
runtime package:       libarm-yolo26n-0
development package:   libarm-yolo26n-dev
Python distribution:   arm-yolo26n
Python import package:  arm_yolo26n
Debian Python package:  python3-arm-yolo26n
```

These names keep the YOLO26n-specific product boundary explicit and allow a
future Debian package to separate runtime and development files without
renaming the public library.

The library name and SONAME identify the supported API and ABI. They do not
encode the target machine: Debian package architecture and Multi-Arch metadata
carry that information. The application wheel's `linux_aarch64` platform tag
prevents installation on another architecture, while the documented Raspberry
Pi 5 boundary and runtime model checks define the narrower supported target.

## Runtime selection for examples and benchmarks

Release wheels install the runtime library inside `arm_yolo26n`; sessions and
detectors locate it automatically. `make examples` downloads and verifies the
application and converter wheels for the checkout's version, then installs
them into `examples/.venv`. A caller who only wants to run an example needs
neither CMake nor a compiler.

`make runtime` continues to build, test, and install a source build into
`dist/runtime/`. The Python API and example commands retain an explicit runtime
override for source development and qualification. The internal Ultralytics
benchmark also keeps its explicit runtime input so its report identifies the
exact library under measurement.

Checkpoint and media downloads follow the same rule: reuse a file only after
its expected digest has been verified, otherwise download it atomically and
verify it before publication into `downloads/`.

## Command surface

The project name already establishes the model family, so commands do not need
to repeat `yolo26n` everywhere.

| Command | Behavior |
| --- | --- |
| `make model [CHECKPOINT=...] RESOLUTION=...` | Download the pinned official checkpoint when omitted, then convert it into `dist/models/`; no native build |
| `make model-check [CHECKPOINT=...] RESOLUTION=...` | Download the pinned official checkpoint when omitted, then validate conversion behavior and write a receipt |
| `make runtime` | Configure, compile, run CTest, and install into `dist/runtime/` |
| `make release` | Produce the application and converter release wheels and checksums in `dist/packages/` |
| `make clean` | Remove reproducible artifacts under `build/` and `dist/` while preserving downloads, environments, and results |
| `make examples` | Prepare the example environment, verified inputs, runtime selection, and converted model; do not run an example |
| `make example-video` | Run prerecorded-video inference and write its detection document and rendered video |
| `make ultralytics-benchmark RUNTIME=... MODEL=... DATA=...` | Prepare the pinned Ultralytics environment when needed, then run validation and record evidence |
| `make lint` | Prepare the hash-locked development environment when needed, then run formatting, static analysis, type checking, import-boundary checks, and shell syntax checks |
| `make test-unit` | Run unit and public command-interface tests without the native runtime |
| `make test-converter CHECKPOINT=...` | Run checkpoint-backed converter behavior tests across the supported resolutions and rejection cases |
| `make test-end-to-end CHECKPOINT=... RUNTIME=... MODELS=...` | Run converter behavior, then the complete runtime and example behavior suite with the supported converted models on Linux AArch64 |

Make targets pass paths and options to the owning commands. Validation and
manifest construction remain in the converter, CMake, Python API, or benchmark
component that owns the relevant contract.

## Example workflow

`make examples` is setup. It may be rerun safely and performs only missing or
invalid work:

1. download and verify the release's application and converter wheels;
2. create or update `examples/.venv` and install those wheels normally;
3. obtain and verify the checkpoint in `downloads/`;
4. convert the checkpoint for the requested resolution when the expected model
   artifact is absent.

Make expresses the checkpoint, release wheels, and converted model as file
dependencies. The download rules use `curl` and `sha256sum`; model outputs are
produced by the converter command.
Caller-provided artifact paths are reused and have no automatic download rule.

It does not download the test video, run video inference, or open a camera.
`make example-video` obtains the verified video if it is absent, records
detections, and renders an annotated MP4 under `examples/video/output/`.
Picamera2 users therefore never download media they do not need. Camera actions
remain explicit because they have different resource and display requirements.

## Ultralytics workflow

The hidden preparation prerequisite of `make ultralytics-benchmark` creates
`benchmarks/ultralytics/.venv`, installs the pinned Ultralytics requirements,
and applies the repository's integration patch. Its stamp prevents repeated
installation until an environment input changes. It does not copy or package
a runtime or converted model.

Ultralytics accepts one `model` argument, while this runtime requires a native
library and a converted model directory. `make ultralytics-benchmark` therefore
creates a temporary model directory with the suffix registered by the patch and
writes one descriptor containing the caller's `RUNTIME` and `MODEL` paths. It
passes that temporary directory to Ultralytics, records the benchmark evidence,
and removes the directory when the command finishes. There is no persistent
Ultralytics workspace for a maintainer to enter or manage.

The patched environment remains isolated under
`benchmarks/ultralytics/.venv`; Torch and Ultralytics do not enter the example
or application dependency closure. Performance evidence remains beside its
recording workflow under `benchmarks/reporting/output/`.

## Implementation sequence

1. Change the Python runtime contract and tests to accept separate runtime and
   model locations, then remove bundle assembly.
2. Give the native library its reviewed Debian-compatible name and SONAME;
   split CMake runtime and development install components.
3. Use CMake install components as the native input to `make release` and
   remove the standalone runtime packaging script.
4. Add standard Python packaging for `arm_yolo26n` and remove repository-wide
   runtime reliance on `PYTHONPATH`.
5. Establish `downloads/`, `build/`, `dist/`, and workflow environment paths,
   then split example setup from execution.
6. Update the Ultralytics integration to consume separate runtime and model
   paths through a temporary descriptor; remove the packaged workspace.
7. Update public commands and documentation, delete obsolete commands rather
   than retaining aliases, and run clean source-build and release-runtime
   example checks.

Each step should preserve direct behavior tests through the public command or
API it changes. Performance must be measured after runtime-loading changes;
directory moves and packaging changes alone do not justify new inference
implementations.
