# Native dependency patches

The runtime builds immutable upstream XNNPACK and pthreadpool revisions, then
applies the patches in this directory from CMake. They patches are intended 
to include only functionality required to enable YOLO26n inference on 
Raspberry Pi 5 using Linux AArch64 and four worker threads. They are not intended 
to remain useful to arbitrary XNNPACK or pthreadpool consumers.

## Patch responsibilities

| Patch | Responsibility |
| --- | --- |
| [`xnnpack-graph-api-pruning.patch`](xnnpack-graph-api-pruning.patch) | Removes graph entry points and implementations that the program cannot reach. It contains no new behavior. |
| [`xnnpack-yolo26-graph-extensions.patch`](xnnpack-yolo26-graph-extensions.patch) | Adds SiLU and flags for the three explicit requests through which our fixed graph selects the new implementations. This is the only functional change to XNNPACK's graph layer. |
| [`xnnpack-operator-pruning.patch`](xnnpack-operator-pruning.patch) | Removes operator dispatch and microparameter implementations that the Pi 5 runtime cannot reach. It contains no new behavior. |
| [`xnnpack-yolo26-operator-extensions.patch`](xnnpack-yolo26-operator-extensions.patch) | Implements the tagged concatenate, registers standalone SiLU, and connects tagged convolution operators to the fused GEMM or indirect-GEMM paths. It selects implementations but does not contain arithmetic kernels. |
| [`xnnpack-yolo26-kernels.patch`](xnnpack-yolo26-kernels.patch) | Adds the Arm64 FP16 convolution-plus-SiLU assembly and standalone FP16/FP32 SiLU kernels. |
| [`pthreadpool-spin-wait.patch`](pthreadpool-spin-wait.patch) | Extends the worker spin-wait period used by the fixed four-thread runtime. This reduces wake-up latency between short operators. |

The five XNNPACK patches apply in the table's order, the exact application commands and dependency revisions live in [`../CMakeLists.txt`](../CMakeLists.txt).

## Specialized concatenate

`BuildP2ToP3` explicitly marks the model.2 concatenate with
`XNN_FLAG_YOLO26_FUSED_C2F_CONCATENATE`. XNNPACK accepts that request only for
three batch-one NHWC FP16 inputs with 16 channels each and a batch-one NHWC
FP16 output with 48 channels. It validates matching spatial dimensions and
fails runtime construction instead of silently selecting the specialization
for a different contract.

The operator parallelizes over spatial rows and performs the three channel
copies together. Its spatial extents are derived during reshape, so the same
tagged operation is used at 320, 480, 640, 800, 960, and 1280. Every untagged
concatenate retains XNNPACK's ordinary per-input copy path.

## Custom kernels

The added `f16-vsilu` and `f32-vsilu` files are deliberate
hand-maintained kernel derivatives. There is no project-supported
generator command for them. Do not run an upstream generator and assume the
result retains the fused SiLU behavior; changes must be made in the patch and
reviewed as arithmetic changes.

The algorithm, constants, registers, stack frame, stock and fused entry paths,
and tail handling are documented beside the assembly that implements them in
`xnnpack-yolo26-kernels.patch`.

## Maintenance rules

- Keep each change with the layer that owns it. Do not put assembly in the
  operator-extension patch or functional changes in either pruning patch.
- Keep implementation contracts beside the changed code in the patch. This
  README is not a substitute for source comments.
- Do not restore general-purpose operators or architectures without a concrete
  runtime requirement. This build deliberately supports only its fixed target.
- Update the pinned dependency revision and its patches together. A patch that
  no longer applies is a configuration failure requiring review, not something
  CMake should silently bypass.
- Treat dispatch, kernel, and pthreadpool changes as performance-sensitive.
  Record new Raspberry Pi 5 performance evidence and review it against the
  relevant reference result.
- Treat graph/API changes as topology-sensitive. Re-run the native and public
  runtime behavior tests at every supported resolution. Use the Ultralytics
  harness when arithmetic or execution changes require renewed accuracy
  evidence.

Upstream dependency revisions are pinned in [`../CMakeLists.txt`](../CMakeLists.txt),
and their deployment notices are retained under [`../licenses/`](../licenses/).
