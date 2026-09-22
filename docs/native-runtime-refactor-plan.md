# YOLO26n native runtime remediation

This record applies the compiler-refactor principles to the model-independent
native runtime and captures the evidence used to close the remediation.

## Baseline

- Source commit: `9038fbd5`
- Native SHA-256: `8e48d898de082780b6ebadaffda0e04cb942a80be5911a2621651b30f7bd453b`
- Stock exported-model SHA-256:
  `8d5a617332cdd1bab070f6d60a02e4cb1cd00a5da10381469d2cd613988f895b`
- Established 640 output SHA-256:
  `c7011eb86934dd93569663ad6864da003d2a4fbcf500ece606b40d7ff00aa70b`
- Qualified Pi median inference: p50 `36.1991555 ms`, p95 `36.9467805 ms`
- Clean-build reproducibility receipt: passed with identical native and bundle
  trees.

## Implementation

- [x] Replace the generic 43-step schedule interpreter with named phase
  execution and typed operations.
- [x] Derive tensor counts and storage extents from the semantic tensor
  inventory; retain the qualified 640/80 placement unchanged.
- [x] Replace generic graph-node interpretation with named graph construction
  and typed XNNPACK operation helpers.
- [x] Preserve the complete qualified value-declaration order where it affects
  XNNPACK workspace planning, while replacing anonymous numeric flags, weight
  slots, and sentinels with named concepts.
- [x] Parse the model header and slot directory through named records and
  focused validation stages.
- [x] Remove learned-payload hashes and the runtime SHA implementation; retain
  only structure needed to execute safely.
- [x] Derive classification width instead of serializing it and remove other
  redundant model fields.
- [x] Consolidate supported resolutions, tensor counts, thread count, status
  values, alignment, and related runtime constants.
- [x] Document the installed C ABI's ownership, buffer, tensor-layout, and
  non-reentrancy contracts; make unsupported buffers rejectable.
- [x] Explain the private XNNPACK projection invariants and replace positional
  projection configuration with named facts.
- [x] Split dependency patches by independent responsibility and document each
  performance or accuracy-sensitive specialization.
- [x] Add enforced native formatting, compiler warnings, and applicable static
  analysis without creating a second build system.

## Non-regression gates

- [x] Every supported resolution builds and passes linked positive and negative
  validation.
- [x] Established numerical output hashes remain exact.
- [x] Stock accuracy remains within the established result: the candidate and
  baseline both produced precision `0.738`, recall `0.538`, mAP50 `0.645`, and
  mAP50-95 `0.479` on COCO128.
- [x] Two clean builds reproduce the same native and bundle identities.
- [x] CTest verifies the installed ABI and dynamic dependency boundary.
- [x] Repository tests and Python quality gates pass.
- [x] Five eligible Pi sessions show no measurable p50 or p95 regression
  against either recorded baseline. Candidate median p50/p95 are
  `36.204`/`36.837 ms`; the best baseline is `36.199`/`36.947 ms`. The
  `0.005 ms` p50 difference is well inside the combined `0.081 ms`
  between-session standard deviation, while p95 improves by `0.110 ms`. The
  candidate improves both metrics by more than 4% against the immediately
  preceding qualified remediation step.
- [x] The complete native diff is re-read for junior-engineer comprehension and
  all review threads are reconciled.

## Reopened maintainability audit

The initial audit established behavioral correctness and performance, but it
did not establish that the runtime was understandable or maintainable. Work
through this list in order; do not close an item merely because tests pass.

### Fixed topology

- [x] Introduce the topology once: define P2, P3, P4, and P5 as the stride-4,
  stride-8, stride-16, and stride-32 feature levels, and explain the roles of
  the backbone, neck, detection head, and nineteen XNNPACK partitions.
- [x] Document `fixed_graph.h`: the `GraphId` ordering/index invariant,
  `GraphContext` ownership and lifetime, and `BuildFixedGraphs` failure
  contract.
- [x] Replace raw positional value operands and external binding IDs in the
  fixed-graph fragments with semantic names. Preserve the qualified XNNPACK
  declaration order, but make that order an implementation constraint rather
  than the graph's human-facing vocabulary.
- [x] Organize each fixed-graph fragment around recognizable YOLO26n modules
  and transformations. Explain non-obvious layout changes, residual paths,
  attention steps, and partition boundaries without introducing a general
  graph IR or returning to generated native source.
- [x] Make the builder's value-ID behavior explicit, including XNNPACK's
  reservation of external IDs and the distinction between a declaration
  position and the XNNPACK value ID it resolves to.
- [x] Document the weight-slot fragment's source of truth and its ordering
  relationship with the exporter. Ensure maintainers can trace a semantic
  weight name from the compiler contract to a runtime slot.
- [x] Decide whether each `.inc` file is genuinely a private textual fragment.
  Rename or restructure files whose extension currently implies ownership or
  compilation behavior they do not have.

### Schedule and memory

- [x] Document `fixed_schedule.h`: group the tensor inventory by lifecycle,
  explain `ScheduleState`, and state the initialization and execution
  contracts.
- [x] Orient `fixed_schedule.cpp` around its responsibilities: shape sizing,
  input/output layout bridges, activation resolution, partition binding, and
  the initialization, backbone, neck, pyramid, and detection phases.
- [x] Explain why each non-obvious copy, split, transpose, and FP16/FP32
  conversion exists. Prefer semantic helper names where a comment would only
  translate opaque mechanics.
- [x] Explain that the 640/80 arena offsets deliberately overlap tensors whose
  lifetimes do not overlap, why that placement matters to Pi performance, and
  why other supported shapes use the fallback placement.
- [x] Make the memory plan a clearly owned private component rather than a
  header-shaped include fragment, and validate its bounds and live-range
  aliasing assumptions for every supported resolution.
- [x] Document `buffer_validation.h` as production runtime input validation:
  preprocessing and inference reject overlapping buffers because neither path
  supports in-place operation. Add focused boundary and overflow tests if the
  existing negative validation does not cover them.
- [x] Re-audit the remaining native sources for the same standard, especially
  model loading, preprocessing, projection optimization, and their headers.
  Add explanations only for contracts and non-obvious decisions; rename or
  restructure opaque code instead of narrating it line by line.

### Dependency patches

- [x] Replace the fused-concatenate recognizer's `delegate-1`, node-number,
  value-number, and shape fingerprint with an explicit semantic request from
  the fixed graph. Remove the obsolete framework-specific explanation.
- [x] Put the concatenate specialization in the patch whose stated
  responsibility actually owns it; keep API pruning, operator selection, and
  arithmetic kernels separate.
- [x] State whether the concatenate specialization is intentionally limited to
  the 640 input shape and document the behavior for every other supported
  resolution.
- [x] Remove unused selection/invocation counters or make them part of a
  deliberate internal activation test. Do not ship unconsumed instrumentation.
- [x] Rewrite convolution-to-fully-connected flag forwarding using a named
  masked value and explain that only the private fused-GEMM flag crosses that
  lowering boundary.
- [x] Establish a reproducible source of truth for the SiLU kernels. Generated
  files need the project-owned template inputs and exact generation command;
  hand-maintained assembly needs an explicit ownership statement.
- [x] Document the fused assembly's algorithm, constants, register/stack
  contract, stock-versus-SiLU entry paths, and tail handling at the level
  needed to modify it safely.
- [x] Update the patches README so its responsibility table and maintenance
  instructions describe the resulting patch contents precisely.
- [x] Add focused tests that prove semantic specialization selection, flag
  propagation, safe fallback, and exact arithmetic rather than relying only on
  end-to-end output and latency measurements.

### Requalification and completion

- [x] Run formatting, warnings, static analysis, native tests, and linked
  positive and negative validation after each coherent remediation slice.
- [x] Re-run exact outputs at all supported resolutions and clean-build
  reproducibility after topology, schedule, memory, or patch changes.
- [x] Re-run accuracy whenever arithmetic can change, and Pi latency whenever
  dispatch, kernels, allocation, declaration order, or execution order can
  change. Compare both with upstream minimal and with the preceding refactor
  commit so regressions within the refactor are visible.
- [x] Perform a final line-by-line maintainer review and reconcile every PR
  thread before marking this audit complete.

## Reopened-audit evidence

- Repository quality checks pass: 113 Python tests pass with 18 expected
  skips; native formatting, compiler warnings, Ruff, Mypy, Vulture, and the
  import-linter are clean. All four native CTests pass.
- Linked positive and negative validation passes at 320, 480, 640, 800, 960,
  and 1280. The established output SHA-256 values remain exact at every
  resolution.
- Two clean builds reproduce native SHA-256
  `bcb575f87f7f69c24794382f61aa77231793f3c5160ca35b7db7bc40dcab21ac`
  and identical bundle trees.
- COCO128 accuracy is unchanged: precision `0.738`, recall `0.538`, mAP50
  `0.645`, and mAP50-95 `0.479`.
- Five eligible Pi 5 sessions pass the hardware protocol. Candidate session
  p50 averages `37.734 ms`; the immediately preceding refactor artifact
  averages `37.825 ms`. The comparison receipt selects medians of
  `37.650 ms` and `37.817 ms`, respectively, a `0.44%` candidate improvement.
  Candidate and preceding p95 samples contain scheduler-interruption outliers;
  their comparison passes the recorded variance test and does not establish a
  p95 regression.
- The recorded upstream-minimal run measured `37.704 ms` p50 under comparable
  frequency and temperature conditions. The candidate's `37.650 ms` median is
  `0.14%` faster, a difference small enough to treat as equivalent rather than
  an improvement claim. An older `36.199 ms` refactor receipt was captured in
  a different thermal state and is retained as historical evidence, not used
  to attribute a code regression in this audit.
- The complete native diff was re-read after qualification. All 116 PR review
  threads are resolved; no remaining thread or source finding requires a code
  change.
