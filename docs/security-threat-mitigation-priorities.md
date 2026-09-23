# Security responsibilities and open work

## Purpose

This document records which security responsibilities belong to this project
and which belong to applications that use it. It replaces a broader threat
assessment that treated every caller-selected artifact as hostile input.

The project is a purpose-built YOLO26n runtime for Raspberry Pi 5. It is not a
sandbox for running models supplied by an untrusted party.

## Trust boundary

The caller authenticates and authorizes every checkpoint, converted model,
runtime library, Ultralytics descriptor, and benchmark candidate that it
selects. The caller must also keep those artifacts in storage that cannot be
replaced by an untrusted identity while they are being loaded or used.

The project remains responsible for three narrower boundaries:

1. The converter and runtime reject artifacts that do not satisfy the supported
   YOLO26n contract before attempting conversion or inference.
2. Given valid API pointers and lengths, parsers use checked arithmetic and do
   not access memory outside the supplied buffers while accepting or rejecting
   an artifact.
3. Project tooling identifies and verifies an official artifact when it selects
   or downloads that artifact on the caller's behalf. The release process binds
   published bytes to the build that was reviewed and qualified.

Artifact authentication and format validation are deliberately separate. A
manifest hash can establish that `manifest.json` and `model.bin` belong
together, but it does not establish who produced them. Conversely, an artifact
from a trusted source must still be rejected when its format is incompatible
with the runtime.

## Runtime and converter contract

The supported loading path validates the model container and dimensions before
constructing XNNPACK graphs. Graph construction then requires every learned
tensor to have the exact extent declared by the fixed topology. Inference is
not possible unless the complete execution contract is satisfied:

- model format and version;
- Raspberry Pi runtime profile and supported resolution;
- class count and all dimensions used to derive allocations;
- required weight slots, data types, layouts, alignment, offsets, and sizes;
- checked arithmetic for file ranges and runtime buffer sizes;
- exact input and output buffer sizes; and
- supported preprocessing dimensions and pixel format.

These checks describe what the implementation can execute. They are not an
attempt to infer whether the caller should trust the model.

The converter accepts a caller-trusted checkpoint. Its restricted decoder and
topology validation remain valuable because they provide a narrow, predictable
conversion path. The project does not promise bounded resource consumption for
an intentionally oversized checkpoint or model. Where the fixed format implies
an exact size, the implementation should reject impossible sizes before copying
or allocating instead of introducing an unrelated policy ceiling.

The C API is non-reentrant for a single runtime instance. The caller must not
overlap operations on one instance and must not destroy it while any operation
is in progress. Independent instances may be used concurrently. Invalid
pointers, stale handles, and violations of those lifetime requirements are
caller errors rather than inputs the runtime can validate reliably.

## Open project work

The following work remains within the project's security boundary, in priority
order.

### Completed current repository work

The caller-owned model-buffer change and compatibility audit have been
requalified on Raspberry Pi 5. Public creation rejects unsupported dimensions
and malformed weight inventories. Valid models initialize and produce output
identical to the accepted reference runtime at 320, 480, 640, 800, 960, and
1280 pixels.

COCO128 results remain precision `0.738`, recall `0.538`, mAP50 `0.645`, and
mAP50-95 `0.479`. The recorded 640-pixel run measured `38.3 ms/image` inference
against the accepted `38.4 ms/image` baseline. The CPU remained at 2.4 GHz with
no current throttling, major page faults, or swap activity during the run; the
filesystem cache was approximately 6.58 GB.

### Public repository and release process

1. Run the documented Black Duck closures and require review of relevant
   findings before publication.
2. When the release-candidate workflow is implemented, record the build
   environment used for its published artifacts.
3. Publish version-specific application and converter wheels whose digests are
   bound to the qualified build. Official download tooling must verify those
   identities before installation or reuse.

The repository-level vulnerability-reporting path, supported-version policy,
and Arm PSIRT routing are recorded in [`SECURITY.md`](../SECURITY.md).
The current build workflow pins its container and action by immutable digest,
retains read-only repository permission, and does not persist checkout
credentials. Release environment recording remains with the not-yet-implemented
release-candidate workflow because pull-request builds publish no artifacts.

The supported Python environments have separate, hash-locked dependency
closures for applications, package builds, repository development, and the
optional Ultralytics benchmark. Package builds use their committed build lock
instead of downloading build requirements in an isolated environment.
Picamera2 remains an operating-system package supplied by Raspberry Pi OS and
is not represented as a Python package download.

## Responsibilities assigned to callers

The following findings do not require defensive runtime machinery:

| Responsibility | Owner |
| --- | --- |
| Decide whether a checkpoint, model, runtime library, or descriptor is authorized | Application or conversion operator |
| Protect selected artifacts from replacement between selection and use | Deployment operator |
| Synchronize calls and destruction for one native runtime instance | Direct C caller |
| Isolate a benchmark candidate that has not already been trusted | Benchmark operator |
| Protect retained local benchmark and detection results | Owner of the evidence |

An explicitly supplied alternative download URL also belongs to the caller.
The project remains responsible for the default official source selected by its
own tooling.

## Reclassified findings

The following previously proposed controls are not current implementation work:

- Generic ZIP, pickle, tensor, model, manifest, and class-name budgets are not
  required for caller-trusted artifacts. Structural and arithmetic validation
  remains required.
- A global native-handle registry and synchronization around every C call are
  not required. The public API instead has an exclusive-use lifetime contract.
- Caller-supplied Ultralytics descriptors are trusted configuration. Supported
  project workflows remain responsible for generating correct private
  descriptors.
- Local benchmark and detection documents do not require signatures merely to
  be consumed by the operator that created them. Release decisions may require
  stronger custody through the release workflow.
- Detector candidate truncation is not a security control and must not change
  valid detector results. Configuration checks remain ordinary public API
  validation.
- Absolute diagnostic paths and presentation of trusted class names are
  usability concerns unless a concrete external disclosure boundary is
  introduced.

## Unresolved media boundary

Recorded video and camera input are operational inputs rather than model
artifacts. Before adding security limits, the project must decide whether an
application is expected to process media supplied by an untrusted party.
Camera-request cleanup and correct resource ownership remain correctness
requirements either way. Streaming large videos may also be a worthwhile
usability improvement without being classified as a security mitigation.
