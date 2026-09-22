# Ultralytics benchmark

This optional integration presents the YOLO26n runtime and a converted model to
Ultralytics for accuracy and performance evaluation. It is the only repository area that
requires Torch or Ultralytics.

- `arm_yolo26n_pi5_backend.py` adapts the native tensor session to the
  Ultralytics backend interface.
- `requirements.txt` pins the optional benchmark environment.
- `patches/` registers that backend with the pinned Ultralytics release.
- `patch.sh` installs the pinned benchmark environment and applies the patch.
- `benchmark.py` creates the temporary descriptor required by Ultralytics,
  runs the caller-selected validation dataset, and records the result.

The top-level Make targets remain the supported command interface. Application
code uses `arm_yolo26n` directly and does not depend on this benchmark tooling.

```sh
make ultralytics-patch
make ultralytics-benchmark \
  RUNTIME="$PWD/dist/runtime/lib/libarm-yolo26n.so.0" \
  MODEL="$PWD/dist/models/yolo26n-640" \
  DATA=coco128.yaml
```

The temporary model descriptor exists only for the benchmark process. The
runtime and converted model are not copied into a benchmark workspace.
