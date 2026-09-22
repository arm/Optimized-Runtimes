# Post-refactor follow-up

PR #21 is primarily a code-quality refactor of the YOLO26n compiler, native
runtime, and tests. It is not the final repository structure, validation
policy, packaging, documentation, or example experience. Complete the work
below after merging it.

## Performance evidence

The ownership decisions and implementation checklist are recorded in
[`validation-refactor-plan.md`](validation-refactor-plan.md).

- [x] Run the context recorder around real Ultralytics benchmarks and confirm the
  before-and-after host, memory, cache, fault, and swap observations are useful
  during review.
- [x] Retain each Ultralytics log, runtime and model identities, source revision, and host
  context as one self-contained evidence directory.
- [x] Present the current and reference Ultralytics results with their corresponding
  context files. Keep acceptance a human decision unless a concrete automated
  policy is agreed later.

## Repository and build layout

The proposed lifecycle ownership, artifact directories, and command surface are
recorded in
[`build-system-and-artifact-layout.md`](build-system-and-artifact-layout.md).

- [x] Name the converter and native implementation area `runtime`; the project is
  purpose-built and no longer represents a temporary or reduced implementation.
- [x] Move the Ultralytics backend and patch out of the current top-level `runtime`
  directory and into an Ultralytics-specific integration area before taking
  that name for the main runtime.
- [x] Move each generic `scripts` command next to the component or workflow that
  owns it. Keep a top-level script only when it genuinely coordinates multiple
  components.
- [x] Review the Make, CMake, export, installation, packaging, and release targets
  as one build system. Remove duplicate orchestration and make ownership of
  each lifecycle stage explicit.
- [x] Review the complete repository layout after those moves, including generated
  build state, downloaded inputs, runnable artifacts, publication archives,
  tests, examples, and internal tools.
- [x] Update imports, static-analysis configuration, commands, and tests as part of
  each move rather than leaving compatibility packages or forwarding wrappers.

### Makefile ownership

- [x] Remove the unused `runtime-install` wrapper. Local use is served by
  `make runtime`, and CMake remains the direct installation interface until the
  project has a concrete Debian installation workflow.
- [x] Make CMake and CTest own execution of the model-backed native API test.
- [x] Have the end-to-end and coverage targets invoke that CTest-owned behavior
  instead of executing the native test binary themselves.
- [x] Separate validation of one requested checkpoint and resolution from the
  converter's development test suite.
- [x] Express example preparation as Make dependencies on the checkpoint,
  runtime, and converted-model outputs. Keep downloading in narrow file rules
  and invoke the converter directly rather than adding another orchestrator.
- [x] Organize the remaining Make variables, targets, and help text by workflow
  after misplaced implementation logic has been removed.

## Python APIs, dependencies, and examples

- [x] Decide how applications install the public `arm_yolo26n` APIs. The final
  user workflow should not depend on setting `PYTHONPATH` to a repository
  checkout.
- [x] Re-audit dependency groups for the tensor API, image detector, prerecorded
  video example, Picamera2 examples, development tools, and optional
  Ultralytics integration. Keep Torch and Ultralytics isolated to that optional
  integration, and evaluate `opencv-python-headless` for the image and video
  paths.
- [x] Ensure examples can consume either an existing locally built runtime or a
  verified release runtime without compiling native code or installing CMake.
- [x] Separate example preparation from execution. Setup should obtain and verify
  dependencies and inputs; running video or camera workloads should remain an
  explicit action.
- [x] Finalize the runtime/model artifact boundary and their local directory layout
  during the broader build-system review. Do not copy the model-independent
  runtime into a resolution-specific model directory merely for convenience.
- [x] Recheck the prerecorded-video example from a clean Raspberry Pi
  installation using an existing runtime and no native toolchain.
- Recheck headless camera capture and live preview with a connected camera.

## Publication and supported environment

- Before publishing the repository, confirm that the project's SA COCO128
  dataset is publicly accessible and permitted for this use, then make it the
  default Ultralytics benchmark dataset in place of `coco128.yaml`.

- [x] Define the supported Raspberry Pi model, operating-system release,
  architecture, Python version, camera stack, and native dependency boundary in
  [`supported-environment.md`](supported-environment.md).
- [x] Define the release artifacts and a repeatable publishing workflow for the
  model-independent runtime, checksums, notices, Python installation, and
  performance and accuracy evidence in
  [`release-publishing-workflow.md`](release-publishing-workflow.md).
- [x] Choose a Debian-compatible shared-library name, SONAME, and package layout
  so the runtime can later be integrated into versioned runtime and development
  packages without another public-library rename.
- [x] Keep checkpoints and converted weights out of project releases. Retain the
  verified local download-and-export path, and require an explicit future
  policy change before redistributing any weight artifact.
- When moving to the public repository, run the full build and test workflow on
  pull requests and make it a required check for `main`. Require the tested
  change to be current with `main` and prevent direct pushes so the merged pull
  request is the test record. Do not rerun unchanged tests after merge. Use a
  merge queue and its `merge_group` check if concurrent merges require testing
  the prospective combined result.

## Documentation and final verification

- Investigate making Python and native coverage reporting fast enough to run
  regularly in CI. Reuse the existing functional test execution, avoid repeated
  checkpoint conversion, and cache the native dependency build where that does
  not compromise the instrumented first-party runtime. Keep Python and native
  results separate and diagnostic; adopting a required coverage threshold is a
  separate policy decision.
- The user will perform another editing pass over every README. Reconcile the
  remaining design documents afterward so they describe the resulting system
  rather than preserving superseded plans.
- After the structural, dependency, packaging, and example changes, rerun the
  repository quality gates, all supported-resolution functional checks,
  accuracy evaluation, clean build, and reviewed Pi performance results with
  their recorded host context.
- Perform a final review from a new user's perspective: clone, install, obtain
  a runtime and checkpoint, export a model, and run each example using only the
  documented public interfaces.
