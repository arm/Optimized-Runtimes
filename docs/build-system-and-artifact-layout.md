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
| Release packages | Top-level Makefile over CMake install rules and Python packaging metadata | Native installation, converter, and application API | Publishable archives under `dist/packages/` |
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
  runtime/<release>/                Runtime archive and published checksum

build/
  runtime/                         CMake cache, dependencies, objects, and tests

dist/
  runtime/                         Installed or downloaded runtime distribution
  models/<model>-<resolution>/     Locally converted model and manifest
  packages/                        Component and final distribution packages

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
  prerecorded-video example is requested. A downloaded runtime archive and
  its published checksum remain here as the cached inputs used to establish
  `dist/runtime/`.
- `build/` contains CMake's private native build state. Nothing under `build/`
  is a supported application input.
- `dist/` contains final runnable artifacts. A source build installs the
  runtime into `dist/runtime/`; downloading a release populates that same
  directory. Locally converted models live under `dist/models/`. Presence
  under `dist/` means the artifact is ready to use, not that this project has
  the right to redistribute a particular set of weights.
- `dist/packages/` contains the converter and application API wheels plus the
  final runtime and development distributions. The build retains only the
  final files that can become a release candidate.
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

There is only one repository-local runtime used by applications:
`dist/runtime/`. A local source build and a verified release download are two
ways to establish that same directory. The shared library in `build/runtime/`
is an unavoidable intermediate in the CMake lifecycle, not a second supported
runtime installation.

## Runtime and model boundary

The public deployment inputs should be separate:

```text
runtime shared library + converted model directory -> one runtime session
```

A converted model directory contains `model.bin` and its manifest. It does not
contain a copy of the runtime. A runtime release contains the shared library,
project license, and native dependency notices. It does not contain weights.

The Python session and detector interfaces therefore accept a model
directory and a runtime-library location independently. Resolution and class
metadata continue to come from the model manifest. The Python layer verifies
the `model.bin` bytes against that manifest and passes the bytes to the native
ABI, which copies them and establishes whether the runtime can execute them.

The runtime location can later default to an installed system or Python-package
location. During this refactor it should remain explicit wherever implicit
selection would make the source of the executable code unclear.

Removing the combined bundle also removes `assemble_bundle.cmake` and the
`yolo26n-bundle` target. Because the software is unreleased, no compatibility
bundle or forwarding command is needed.

## Native packaging

CMake defines the native target, verification tests, and runtime and
development install components. `make release` stages those components using
`cmake --install` and creates the final archives once. The versioned shared
library and notices are runtime files, while the unversioned linker name and C
header are development files. A future Debian package can consume those same
install components without changing their ownership or layout.

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
carry that information, and release archives identify Raspberry Pi 5 in their
names and metadata. The Python distribution and import names likewise identify
the application API; native compatibility is enforced by the installed runtime
package and by the runtime's model checks.

## Runtime selection for examples and benchmarks

Preparation should establish `dist/runtime/` in this order:

1. keep an already valid runtime distribution in `dist/runtime/`; or
2. download, verify, and extract the configured release into `dist/runtime/`.

`make runtime` builds, tests, and installs a source build into that same
distribution directory through CMake. Example preparation does not inspect the
CMake build tree or perform native installation. A caller who only wants to run
an example needs neither CMake nor a compiler.

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
| `make release` | Produce the native, converter, and Python API release packages in `dist/packages/` |
| `make clean` | Remove reproducible artifacts under `build/` and `dist/` while preserving downloads, environments, and results |
| `make examples` | Prepare the example environment, verified inputs, runtime selection, and converted model; do not run an example |
| `make example-video` | Run prerecorded-video inference and write its detection document and rendered video |
| `make example-camera [CAMERA=...] [CAMERA_ARGS=...]` | Run display-independent Picamera2 detection with the prepared runtime and model |
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

1. create or update `examples/.venv` with the application dependencies;
2. make the repository's `arm_yolo26n` package importable through a normal Python
   installation rather than `PYTHONPATH`;
3. obtain and verify the checkpoint in `downloads/`;
4. select an existing local runtime or obtain a verified release runtime; and
5. convert the checkpoint for the requested resolution when the expected model
   artifact is absent or invalid.

Make expresses the checkpoint, runtime, and converted model as file
dependencies. The default checkpoint, runtime, and video rules use `curl`,
`sha256sum`, and `tar`; model outputs are produced by the converter command.
Caller-provided artifact paths are reused and have no automatic download rule.

It does not download the test video, run video inference, or open a camera.
`make example-video` obtains the verified video if it is absent, records
detections, and renders an annotated MP4 under `examples/video/output/`.
`make example-camera` explicitly opens Picamera2 for display-independent
detection, so it also works over SSH. Picamera2 users therefore never download
media they do not need, and camera actions remain explicit because they have
different resource requirements. The Picamera2 guide separately documents the
graphical preview and its desktop or DRM/KMS requirements.

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
