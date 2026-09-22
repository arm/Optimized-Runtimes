# Release publishing workflow

## Purpose

The public repository will build release candidates in hosted Arm CI, qualify
those exact files on a supported Raspberry Pi 5, and publish them without a
second build. This preserves the clean CI build environment while making
physical-device validation a release requirement.

Candidate construction is implemented by the manually dispatched
`release-candidate.yml` workflow. Publication remains a separate human action
after Raspberry Pi qualification.

## Published files

The release page presents one primary download for the supported platform:

```text
arm-yolo26n-<version>-rpi5-trixie-arm64.tar.gz
```

This is the prebuilt distribution for running YOLO26n. It contains the native
runtime installation and the Python wheel. The release notes link to it as
**Download for Raspberry Pi 5** and give the extraction and Python installation
commands. A Python application needs no other release asset.

Repository examples obtain this distribution automatically. The public
implementation must derive a version-specific release URL from the checkout
rather than using `releases/latest`, so an older tag continues to obtain its
own runtime.

| File | Owner | Contents |
| --- | --- | --- |
| `arm-yolo26n-<version>-rpi5-trixie-arm64.tar.gz` and checksum | Main build assembly from CMake and Python outputs | Ready-to-run native installation, Python wheel, project license, and third-party licenses |
| `arm-yolo26n-<version>-rpi5-trixie-arm64-dev.tar.gz` and checksum | CMake and CPack | Optional C header and unversioned linker name for C or C++ development |
| `arm-yolo26n-<version>-rpi5-qualification.tar.gz` and checksum | Raspberry Pi qualification | Review evidence: test report, candidate identities, accuracy and performance logs, and recorded host context |

GitHub's archive for the release tag is the complete project source
distribution for users who want to inspect or build the project. GitHub labels
those automatic assets **Source code (zip)** and **Source code (tar.gz)**. A
separate Python source distribution is unnecessary unless the Python package
is later published through an index such as PyPI.

The release contains no checkpoint or converted model. Qualification downloads
the official checkpoint from its publisher, verifies its known digest, and
keeps it out of the published files in accordance with
[`yolo26n-raspberry-pi-publishing-strategy.md`](yolo26n-raspberry-pi-publishing-strategy.md).

## Required pre-merge and post-merge checks

The public repository requires the normal build and test workflow on pull
requests. Applicable contributions containing AI-generated code also require
the maintainer-approved Black Duck snippet scan described in
[`black-duck-scanning-scheme.md`](black-duck-scanning-scheme.md).

After merge, the trusted full Black Duck scan records the runtime, examples,
build, and Ultralytics dependency closures. A release candidate must come from
a `main` commit whose scan completed successfully and whose BOM and notice
decisions have been reviewed. The release workflow references that result; it
does not repeat an unchanged scan.

## Candidate construction

A separate, manually triggered GitHub Actions workflow operates on the current
`main` commit and reads its version from `VERSION`. The ordinary `main` build
remains the only build: it has already run the tests, built the Python wheel,
had CMake and CPack produce the runtime and development components, and
assembled the final distribution. Candidate construction:

1. verifies the current `main` commit and its version;
2. requires a successful build and successful Black Duck run for that exact
   commit, plus the maintainer's confirmation that the Black Duck result was
   reviewed;
3. downloads the retained `main-build-<commit>` artifact without rebuilding;
4. verifies the final distribution checksums and expected native, Python, and
   license files;
5. adds a manifest identifying the source commit, workflow runs, and unchanged
   distribution files; and
6. creates one draft GitHub release targeting that commit.

The build's release-assembly step does not introduce another native or Python
packager. It combines the files produced by those owners into the single
platform distribution users need. The candidate workflow only verifies and
annotates those retained bytes. It uses the `release-candidate` GitHub
environment so repository administrators can add required reviewers before
granting permission to create a draft release.

To construct a candidate, open **Actions → Release candidate → Run workflow**,
select `main`, and confirm Black Duck review only after inspecting that
commit's successful scan. Candidate runs are serialized so two maintainers
cannot race to create the same release. The workflow rejects malformed
versions, missing successful runs, existing release tags, and unexpected or
damaged build packages.

This follows [`arm/topo`'s release workflow](https://github.com/arm/topo/blob/main/.github/workflows/release.yaml)
in using a manual release from current `main`, serializing release runs, and
handing artifacts to a later publishing stage by producer-run identity. Topo
rebuilds with GoReleaser because it produces tagged binaries for several
platforms and then delegates signing and Artifactory publication. This project
instead promotes its already-tested, fixed-platform `main` packages and creates
a draft because Raspberry Pi qualification must occur before publication.

Successful builds triggered by a push to `main` retain the final runtime and
development distributions and their checksums for 30 days under an artifact
named `main-build-<commit>`. Candidate construction verifies those files,
records their source commit and workflow evidence, and promotes the unchanged
bytes into the draft release used by qualification.

## Raspberry Pi qualification

Qualification is performed by a human on a Raspberry Pi 5 matching
[`supported-environment.md`](supported-environment.md); it does not require a
self-hosted GitHub runner. The maintainer downloads the candidate with the
GitHub interface or `gh release download` and verifies its manifest and
checksums. Draft releases are visible to maintainers but are not public
releases.

The Pi is freshly restarted before qualification. After startup activity has
settled, qualification runs without unrelated user workloads competing for
CPU, memory, or I/O. It uses a clean staging directory and Python virtual
environment, installs only the downloaded candidate files, and never builds
the native runtime from the repository checkout.

The qualification procedure exercises:

- installation of the candidate runtime and Python wheel;
- checkpoint conversion and native execution at every supported resolution;
- the prerecorded-video example;
- the reviewed Ultralytics accuracy and performance workload; and
- both Picamera2 examples when the release claims camera support.

Performance measurement runs through the existing benchmark-reporting harness.
Its benchmark log, artifact identities, and before-and-after host context are
retained rather than reproduced by release-specific measurement code.

Qualification produces a normal test report and one evidence directory. The
report identifies the candidate manifest, source commit, Pi environment,
commands and outcomes, camera model when applicable, and the accuracy and
performance evidence. A changed candidate digest invalidates the report and
requires qualification of the new file.

## Review and publication

Performance acceptance is a human review decision. The workflow records the
measurements and environment but defines no automatic regression threshold. A
maintainer decides whether to publish, investigate, or repeat the session.

After functional, accuracy, performance, camera, license, and Black Duck
evidence have been reviewed, a maintainer:

1. verifies that the draft release targets the qualified source commit;
2. verifies its existing candidate checksums against the qualification report;
3. attaches the qualification evidence to that draft; and
4. publishes the draft without rebuilding, replacing, or re-uploading a
   candidate artifact.

Publication may later be automated behind a protected GitHub release
environment, but it must continue to consume the qualified candidate rather
than invoking another build.

If candidate construction, Black Duck review, Pi qualification, or human
performance review is incomplete, no release is published. An unavailable
camera also prevents a release from claiming qualified camera support.
