# YOLO26n Raspberry Pi publishing strategy

## Product boundary

Publish one purpose-built YOLO26n runtime for Raspberry Pi 5, not a general
compiler or inference framework. The native library owns the fixed YOLO26n
topology and is built independently of model weights. Exporting a supported
checkpoint produces `model.bin` and its metadata. Applications select that
converted model and the independently installed native runtime.

The supported static square profiles are 320, 480, 640, 800, 960, and 1280.
Class count and names come from the checkpoint. These choices do not create a
general dynamic-shape or arbitrary-model interface: the exporter rejects any
topology or tensor contract the runtime cannot execute.

## Initial publication

The initial GitHub publication should contain:

- source and pinned dependencies needed to reproduce the converter and one
  model-independent `libarm-yolo26n.so.0`;
- a model-independent runtime archive with its checksum and required notices;
- the direct Python tensor and detector APIs;
- a prerecorded-video example for measuring model throughput on a repeatable
  user-supplied workload;
- bounded Picamera2 examples for headless capture and live preview;
- the Ultralytics accuracy/benchmark integration; and
- functional test results, Ultralytics accuracy results, and Raspberry Pi 5
  performance reports.

The 80-class 640 profile has the established accuracy and Raspberry Pi 5
performance baseline. Other supported profiles and fine-tunes must not inherit
those claims; publish evidence only for each artifact and profile actually
measured.

Project releases do not redistribute the stock checkpoint, user-supplied
checkpoints, or converted `model.bin` files. `make examples` downloads the
official checkpoint from its publisher, verifies its known digest, converts it
locally, and uses it with the released model-independent runtime. Fine-tuned
users likewise supply and convert checkpoints under their own authority.

Availability from an upstream download location is not treated as permission
for this project to republish the bytes in another form. Any future exception
requires an explicit, reviewed change to this policy based on established
redistribution rights.

## Examples and validation

A prerecorded-video example should be the clearest performance demonstration.
It lets users rerun the same frames across artifacts and adapt the workflow to
their own footage. The repository can use that same program for authenticated
internal validation; those roles reinforce each other as long as its reported
model latency remains separate from video decoding and rendering time.

Picamera2 examples then demonstrate live camera integration. Camera scheduling,
display, and overlays remain application concerns. The detector owns model
preparation, execution, postprocessing, and lifecycle.

The Ultralytics backend supports familiar validation and prediction workflows
without defining the detector API.

## Release standard

Describe the result as a Raspberry Pi 5 engineering preview. A release must
report accuracy and distinguish native inference latency, prerecorded-video
throughput, and end-to-end camera throughput. It must preserve raw measurements,
identify ineligible thermal sessions, and avoid extending evidence from one
profile to another.

CI builds the release candidate in its controlled Arm Trixie environment. The
candidate is published only after a Raspberry Pi 5 has qualified those exact
artifact bytes and recorded their digests; qualification on the Pi does not
rebuild them. The complete platform and qualification boundary is defined in
[`supported-environment.md`](supported-environment.md).

The implementation workflow is documented in
[`../src/runtime/README.md`](../src/runtime/README.md), and fine-tuned
checkpoint policy is recorded in
[`fine-tuned-yolo26-runtime-strategy.md`](fine-tuned-yolo26-runtime-strategy.md).
The candidate build, Raspberry Pi qualification, Black Duck gates, and manual
publication process are defined in
[`release-publishing-workflow.md`](release-publishing-workflow.md).
