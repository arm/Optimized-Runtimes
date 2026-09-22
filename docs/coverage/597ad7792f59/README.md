# Coverage record for 597ad7792f59

This directory preserves the first complete Python and native coverage run for
source commit `597ad7792f59`. It is a project-history record, not a release
artifact or a coverage acceptance policy.

The reports were produced on 2026-09-21 with:

- Debian 13 (Trixie), Linux AArch64;
- Python 3.13.5;
- GCC 14.2.0; and
- the supported YOLO26n checkpoint with SHA-256
  `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.

`make coverage PYTHON=.venv/bin/python` passed its native ABI check and all 21
Python functional tests. Python line coverage was 87.0%, branch coverage was
62.9%, and Coverage.py's combined total was 82.0%. The first-party native
runtime had 87% line coverage and 75% branch coverage.
Fetched dependencies and assembly kernels are not part of the native report.
The Cobertura XML files retain per-file line and branch data for machine
processing.

- [Python coverage](python.xml)
- [Native coverage](native.xml)

## Converter branch audit

The report identified 118 unexercised converter branch outcomes:

- 30 in checkpoint archive decoding and intake;
- 54 in model validation and its typed contracts; and
- 34 in lowering, serialization, and command-exit invariants.

The existing public tests demonstrated rejection of a non-checkpoint file and
an unsupported resolution, but not of a structurally invalid Torch archive or
a parseable checkpoint incompatible with the fixed runtime. Follow-up tests now
exercise representative archive structure, pickle, storage, checkpoint-root,
model-kind, routing, convolution, residual, detection-head, and learned-value
rejections through the export command. Internal post-validation invariants are
not assigned tests solely to increase branch coverage.
