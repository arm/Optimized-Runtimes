# Arm Optimized Runtimes

This repository contains Arm-optimized inference runtimes for supported models
and target platforms. Each runtime has its own directory with
its build, conversion, examples, benchmarks, tests, and release documentation.
Native source builds also use the shared top-level interface directory.

## Available runtimes

| Runtime | Target | Task | Source |
| --- | --- | --- | --- |
| YOLO26n | Raspberry Pi 5 | Object detection | [yolo26-raspberry-pi-5](yolo26-raspberry-pi-5/) |

Open a runtime directory for its prerequisites and commands. Run that runtime's
`make` targets from within its directory.

## Shared runtime interface

[`interfaces/loom/`](interfaces/loom/) owns the LOOM runtime C ABI v10 and
its conformance suite. YOLO26n exports it alongside its original C and Python
APIs. Native source builds require the repository checkout, including this
directory. Installed development
packages include the shared headers. See the interface README for the supported
host-memory contract and lifecycle.
