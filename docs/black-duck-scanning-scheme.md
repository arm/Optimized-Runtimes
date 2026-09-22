# Black Duck scanning scheme

Black Duck reporting is split by dependency closure so that a reviewer can tell
which dependencies are shipped in the native runtime, needed by an application,
used to build the project, or installed only for Ultralytics benchmarking. A
single aggregate BOM obscures these distinctions.

## Project and version structure

All closures remain versions of the existing `Loom` project. For a source
revision `<revision>`, create these versions:

| Version | Question answered |
| --- | --- |
| `<revision>-runtime` | What third-party code is incorporated into the native runtime library? |
| `<revision>-examples` | What must be installed to run the repository examples? |
| `<revision>-build` | What is required to produce the runtime and model artifacts? |
| `<revision>-ultralytics` | What must be installed to run the Ultralytics benchmark? |

The examples, build, and Ultralytics versions reference the runtime version as
a Black Duck subproject. The native dependency inventory therefore has one
owner and is not copied into each parent BOM.

Separate Black Duck projects are unnecessary unless these closures later gain
independent release lifecycles, ownership, or policy requirements.

The runtime and examples versions have `EXTERNAL` distribution because they
describe software intended for users. The build and Ultralytics versions have
`INTERNAL` distribution because those environments remain within the project
and are not delivered to users.

## Black Duck treatment rationale

The following decisions apply Black Duck's own concepts to the repository's
dependency boundaries:

| Decision | Justification and consequence |
| --- | --- |
| Use a separate version for each closure | Black Duck aggregates all scans mapped to one project version into one BOM. Separate versions preserve the distinction between shipped runtime code, application dependencies, build tools, and benchmark dependencies. See [Scanning best practices](https://docs.blackduck.com/r/blackduck/latest/black-duck-documentation/scanning-best-practices.html). |
| Reference the runtime as a subproject | Black Duck subprojects represent one project version inside another and propagate updates from the subproject to its parents. This gives the native runtime one authoritative BOM without copying its component inventory into every closure. See [Managing subprojects](https://docs.blackduck.com/r/blackduck/latest/black-duck-documentation/managing-subprojects.html). |
| Mark build and Ultralytics versions `INTERNAL` | Black Duck defines internal distribution as software that remains within the company and does not reach customers. It also excludes internal versions from project-level risk calculations, so this setting is used deliberately only for the two internal environments. See [Creating a project version](https://docs.blackduck.com/r/blackduck/latest/black-duck-documentation/creating-a-new-version-of-a-project.html). |
| Mark build tools `Dev. Tool / Excluded` | Black Duck recommends this usage for development tools discovered by a scan but not distributed with the released product. Their licenses no longer contribute to the version's license-risk calculation, while their security and operational risks remain available. See [Excluding a component from a BOM](https://docs.blackduck.com/r/blackduck/latest/black-duck-documentation/excluding-a-component-from-a-bom.html). |
| Exclude build tools from the build Notices report | Notices inclusion is independent of component usage and all components are included by default. Each build tool is therefore explicitly excluded. Runtime release notices are generated from the runtime version, which never contains the tools. See [Excluding a component or subproject from the Notices File report](https://docs.blackduck.com/r/blackduck/latest/black-duck-documentation/excluding-a-component-or-subproject-from-the-notices-file-report.html). |
| Classify native dependencies as statically linked | The final link command incorporates XNNPACK, pthreadpool, cpuinfo, and FXdiv into `libarm-yolo26n.so.0`. Usage is corrected from Black Duck's default rather than allowing the default to misdescribe how the released library is constructed. Black Duck documents `Statically Linked` as a supported component usage in [Excluding a component from a BOM](https://docs.blackduck.com/r/blackduck/latest/black-duck-documentation/excluding-a-component-from-a-bom.html). |

Black Duck configuration expresses how components are used; it does not decide
the repository's product boundaries. The runtime, examples, build, and
Ultralytics closures come from the actual artifacts and workflows maintained by
this repository. The Black Duck settings record those boundaries rather than
creating them.

The repository root `LICENSE` is the source of truth for the Apache-2.0 license
of project-owned code. Runtime archives include that
file as `LICENSE`; dependency licenses remain separate under `licenses/`.
Black Duck project-version metadata records Apache-2.0 from this repository
declaration rather than treating an inferred component license as the project's
license.

## Closure contents

### Runtime

The runtime closure contains the dependency sources acquired by CMake and
incorporated into `libarm-yolo26n.so.0`:

- XNNPACK
- pthreadpool
- cpuinfo
- FXdiv

These components are classified as statically linked. The scan target is the
clean CMake-fetched source tree, not the complete build directory. Scanning the
complete build directory would add generated files and duplicate historical
dependency trees without describing the runtime more accurately.

### Examples

The examples closure contains:

- the runtime version as a subproject;
- NumPy; and
- OpenCV.

Its package-manager scan uses only the complete, hash-locked
`requirements.txt`. It must not discover the development or Ultralytics locks
by walking the repository.

### Build

The build closure contains the runtime version as a subproject plus the tools
strictly required to produce and verify the runtime and model artifacts. The
qualified Raspberry Pi 5 build environment for revision `be2bbfe1` contains:

| Tool | Qualified version | Debian package revision | Repository use |
| --- | --- | --- | --- |
| CMake | 3.31.6 | `cmake` 3.31.6-2 | Dependency acquisition, native build, tests, installation, bundling, and packaging. The build requires CMake 3.24 or newer. |
| GCC and G++ | 14.2.0 | `gcc-14-aarch64-linux-gnu` and `g++-14-aarch64-linux-gnu` 14.2.0-19 | Linux AArch64 C and C++17 compilation and linking. |
| GNU Make | 4.4.1 | `make` 4.4.1-2 | CMake build backend. |
| GNU Patch | 2.8 | `patch` 2.8-2 | Applies the repository-owned dependency patches. |
| GNU Binutils | 2.44 | `binutils` 2.44-3 | Supplies `readelf` and `nm` for native-library verification. |
| GNU C Library | 2.41 | `libc6` 2.41-12+rpt1 | Provides the C library and Linux AArch64 ABI used by the native toolchain. |
| Python | 3.13.5 | `python3.13` 3.13.5-2 | Runs the checkpoint exporter, which otherwise uses only the Python standard library. |

Build tools are classified as development tools rather than shipped runtime
components. Linters, test runners, and other repository quality tools are not
build dependencies merely because a developer environment installs them.

Black Duck cannot infer system tools such as CMake or the compiler from the
CMake-fetched source tree. They are therefore entered from this explicit,
reviewed inventory, which is represented by Black Duck component identities in
`internal/blackduck/build-tools.json`. Each tool uses the
`Dev. Tool / Excluded` usage and is excluded from the build version's Notices
report. Pip and the hash-locked Python build requirement are discovered by the
package scan and receive the same treatment. This retains security and
operational information without treating build tools as code distributed with
Loom.

Runtime release attribution is generated from the runtime version, where these
tools do not appear. Attribution inclusion is controlled separately from
component usage, so it is set explicitly rather than assumed from the
`Dev. Tool / Excluded` classification.

### Ultralytics benchmark

The Ultralytics closure contains:

- the runtime version as a subproject; and
- the direct and transitive dependencies selected by
  `benchmarks/ultralytics/requirements.txt`, including Torch and
  Ultralytics.

This is a validation environment, not an application runtime dependency list.
Its size and license profile should not be attributed to the runtime or
examples.

## Scan rules

Each closure has its own code location named with both the closure and source
revision. A scan is mapped only to its corresponding version.

Every scan sets the project-version license from the repository declaration:

```text
--detect.project.version.license=Apache License 2.0
--detect.project.version.update=true
```

The update flag is required for an existing version; without it, Detect applies
the license only when it creates a version. This is project metadata, not a
license override for a third-party component.

Python scans must:

- name the exact requirements file or files;
- run against an environment containing only that closure's packages; and
- use the native package inspector rather than accepting a lower-accuracy
  requirements-file parse.

The isolated environment is important because the native inspector can follow
optional dependency metadata for any matching packages already installed in
the environment. A shared development environment can therefore contaminate a
benchmark BOM with unrelated tools.

The native signature scan uses:

- the source tree produced by a clean CMake dependency fetch;
- source upload disabled;
- snippet matching disabled; and
- license search enabled.

Component usage and false-positive decisions are reviewed after ingestion.
They are not inferred from Black Duck's default `DYNAMICALLY_LINKED` usage.

## Trial at `be2bbfe1`

The initial trial created:

- `be2bbfe1-runtime`
- `be2bbfe1-examples`
- `be2bbfe1-build`
- `be2bbfe1-ultralytics`

`be2bbfe1-runtime` and `be2bbfe1-examples` use external distribution.
`be2bbfe1-build` and `be2bbfe1-ultralytics` use internal distribution.

The existing `initial-018ab2b` aggregate version remains available for
comparison. It should not receive further closure-specific scans while this
scheme is evaluated.

The trial demonstrated that the subproject structure works and exposed issues
that the aggregate BOM concealed:

- Black Duck did not map the installed `+cpu` Torch and Torchvision packages
  into the Ultralytics BOM.
- A signature match identified `ML-Embedded-Eval-Kit` from files belonging to
  XNNPACK, cpuinfo, and pthreadpool. It was ignored as a false positive because
  that project is not a dependency of this repository.

These are issues to resolve in the individual closures. They are not reasons to
return to a combined BOM.

## GitHub Actions automation

`.github/workflows/blackduck.yml` runs the four closure scans after every push
to `main`. A maintainer can also dispatch it for `main`; an explicit job guard
prevents a credentialed run against another ref. The jobs use the `blackduck`
GitHub environment, which must restrict deployment branches to `main`.

Configure the environment with:

- `BLACKDUCK_TOKEN`, an environment secret authorized to update the existing
  project;
- `BLACKDUCK_URL`, a variable containing the Black Duck server URL;
- `BLACKDUCK_PROJECT_ID`, a variable containing the stable project UUID; and
- optionally `BLACKDUCK_PROJECT_NAME`, if the project is renamed from `Loom`.

Each scan uses the first eight characters of the merge revision in both the
version and code-location names. The runtime scan configures CMake to fetch its
four native dependencies, scans only their source directories, enables license
search, and disables source upload and snippet matching. Each Python scan
stages exactly one hash-locked requirements file, installs it into an isolated
environment, and requires Black Duck's high-accuracy PIP native inspector.

After all scans finish, `internal/blackduck/reconcile.py` establishes the three
runtime subproject relationships, adds the reviewed system build tools,
applies component usage and Notices settings, and verifies that the resulting
BOMs still match their intended closures. It reports the known Torch and
Torchvision mapping gap in the GitHub job summary.

Two scanning paths will serve different purposes.

### Full repository scan after merge

A full Black Duck scan runs automatically after changes are merged to `main`.
It provides ongoing dependency and software-composition-analysis coverage and
supplies the evidence used for release and SBOM work.

The Black Duck token is stored in the branch-restricted `blackduck` GitHub
environment. The workflow has no pull-request trigger, so code from an
untrusted fork cannot receive the token.

The full scan will preserve the dependency-closure scheme described in this
document. Runtime, examples, build, and Ultralytics dependencies will remain
separate Black Duck versions, with the runtime reused as a subproject by the
other closures.

### Planned pre-merge snippet scan for AI-generated code

The post-merge workflow does not implement snippet review. This remains a
separate workflow because it has a different trigger, credential boundary, and
review result.

A contribution containing AI-generated code will require a separate Black Duck
snippet scan before merge. A maintainer will explicitly trigger or approve the
scan so that an external contributor cannot obtain Black Duck credentials.

The workflow will:

1. compare the pull request with its base commit;
2. extract only newly added source lines;
3. run Black Duck snippet matching against that content;
4. fail when relevant snippet matches require review; and
5. publish enough evidence for the maintainer to assess the matches.

A successful snippet scan will be a prerequisite for merging an applicable
pull request. The workflow must execute trusted workflow logic from the base
repository. Contributor-controlled GitHub Actions definitions, build scripts,
or other executable project code must not run in a context containing Black
Duck credentials.

This design keeps the public repository as the source of truth and avoids an
internal mirror maintained only for compliance scanning. It also keeps
full-project dependency analysis separate from targeted snippet checking of
newly introduced AI-generated code.

## Evaluation

The scheme is successful when each BOM answers its stated question without
unrelated components, the three parent closures reuse the runtime subproject,
and the closures together account for the dependencies that were previously
indistinguishable in the aggregate scan. After that review, the aggregate
version can either be archived or retained with an explicit label as a
historical development-environment view.
