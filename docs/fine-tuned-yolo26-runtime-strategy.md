# Fine-tuned YOLO26n runtime strategy

## Supported boundary

Fine-tuning commonly changes learned values and often replaces the original 80
classes. The YOLO26n exporter therefore accepts class count, class names, and
learned tensors from the checkpoint while requiring the exact supported
YOLO26n detection topology.

Each export selects one of six static square input resolutions: 320, 480, 640,
800, 960, or 1280. The exporter validates the architecture, fuses batch
normalization, lowers tensors into the runtime's semantic weight slots, and
writes `model.bin` plus metadata. It never compiles native code. The same
model-independent `libarm-yolo26n.so.0` initializes its nineteen fixed graphs
from any accepted export.

Checkpoints and converted models remain user-supplied local inputs. Project
releases distribute the model-independent runtime and conversion tooling, not
stock or fine-tuned weight artifacts. Accuracy and performance evidence may
identify a checkpoint by digest without including its bytes.

This is intentionally not support for arbitrary YOLO variants, topology
changes, tasks other than detection, quantization modes, dynamic shapes, or
runtime-selectable thread counts. Unsupported inputs are rejected during
conversion or session construction before inference.

## Why variable classes matter

A review performed during the original research found 24 documented YOLO26n
detection fine-tunes among 94 public repositories. Twenty-two replaced the
application's class list; two retained the original list; none kept all 80 and
added more. A broader review of YOLO11n and YOLOv8n repositories showed the
same general pattern. Variable replacement class lists are therefore part of
the narrow product requirement rather than an attempt to generalize the
runtime.

## Qualification policy

Acceptance by the exporter proves structural compatibility, not accuracy or
performance. Every checkpoint/profile combination for which the project
publishes accuracy, performance, or support claims needs:

1. deterministic conversion and model validation;
2. exact raw-output and decoded-detection parity against its reference path;
3. accuracy evaluation on the checkpoint's representative validation data;
4. Raspberry Pi 5 latency, initialization, memory, and thermal evidence; and
5. correct class metadata and rejection behavior through the packaged Python
   and Ultralytics paths.

The stock 80-class 640 profile carries the established baseline. Other
resolutions or class counts must not inherit it without their own evidence.
