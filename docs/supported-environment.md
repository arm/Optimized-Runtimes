# Supported environment

## Product target

The runtime supports Raspberry Pi 5 running 64-bit Raspberry Pi OS Trixie. The
supported execution environment is Linux AArch64 with Armv8.2-A FP16
instructions and Python 3.13. Raspberry Pi OS kernel and system-package patch
updates remain supported; release evidence records their exact versions rather
than making one patch release part of the application interface.

The support claim applies to Raspberry Pi 5 across its memory capacities.
Available memory can still affect whether a selected resolution can be created
and can influence measured performance. Allocation failure must be reported to
the caller, and performance evidence records memory availability and pressure
for the measured session.

The project does not claim runtime support for Raspberry Pi 4, 32-bit operating
systems, Raspberry Pi OS Bookworm, other Linux distributions, other Arm boards,
or non-Arm hosts. A compatible machine may happen to execute the library, but
that does not extend the tested product boundary.

## Native runtime boundary

The native runtime is compiled for `armv8.2-a+fp16`. XNNPACK, pthreadpool,
cpuinfo, and FXdiv are pinned, patched where required, and linked into the
runtime. Applications do not install those libraries separately.

The released shared library dynamically uses only the AArch64 loader and the
following system libraries supplied by Raspberry Pi OS Trixie:

- `libc.so.6`;
- `libgcc_s.so.1`;
- `libm.so.6`; and
- `libstdc++.so.6`.

The CMake ABI verification test enforces this boundary. Running a released
runtime does not require CMake, a compiler, or the native dependency sources.

## Python and camera boundary

The application API supports Python 3.13. A later Python minor is added only
after its package dependencies and public behavior have been tested. NumPy and
headless OpenCV are application dependencies; Torch and Ultralytics belong only
to the optional benchmark integration.

Camera support is optional. The Picamera2 examples use `python3-picamera2`,
libcamera, and `rpicam-apps` from Raspberry Pi OS so the Python bindings match
the installed camera stack. Picamera2 is not installed from PyPI. Headless
capture requires a camera reported by `rpicam-hello --list-cameras`; live
preview also requires a working QtGL desktop or DRM console.

The first public release that claims camera support must identify the camera
module used for qualification and exercise both the headless and live examples
on it. Additional camera models recognized by Raspberry Pi OS may work, but are
not individually qualified unless release evidence names them.

## Build and release qualification

Release candidates are built in a controlled hosted Linux AArch64 Debian
Trixie environment. The build uses CMake 3.24 or newer, the GNU C, C++, and
assembler toolchain, and the repository's pinned native dependency sources and
patches. The runtime sources require C++17. This produces a cleaner and more
reproducible build environment than compiling release binaries on a development
Raspberry Pi.

A hosted build is not sufficient for release. Before publication, a Raspberry
Pi 5 matching this support contract must consume the exact candidate artifacts
produced by CI. Qualification records their SHA-256 identities and exercises:

- installation of the runtime and Python distribution;
- conversion and native execution at every supported resolution;
- the prerecorded-video example;
- the reviewed Ultralytics accuracy and performance workload; and
- the Picamera2 headless and live examples when the release claims camera
  support.

The qualification record includes the Pi model, Raspberry Pi OS and kernel,
Python and camera-stack package versions, runtime and model identities, and the
host context required to interpret performance. Publication copies the
qualified artifacts without rebuilding them and verifies that their digests
match the qualification record.

Pull-request CI remains responsible for linting, unit tests, native compilation,
all-resolution end-to-end execution, and packaging on hosted Arm hardware.
Physical Pi qualification owns device discovery, camera behavior, thermals,
throttling, memory conditions, accuracy, and performance. Passing one boundary
does not stand in for the other.
