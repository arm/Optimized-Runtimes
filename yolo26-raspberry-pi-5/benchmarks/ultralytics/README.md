# Ultralytics benchmark

This optional integration runs the YOLO26n runtime through the Ultralytics
validation harness for accuracy and performance comparison with other
runtimes.

## Usage

Run these commands from the `yolo26-raspberry-pi-5/` runtime directory.
Install the [project prerequisites](../../README.md#prerequisites) first.

Run the benchmark:

```sh
make ultralytics-benchmark
```

The target prepares any missing default runtime, converted model, and pinned
benchmark environment.

Select another Ultralytics dataset configuration with `DATA`:

```sh
make ultralytics-benchmark DATA=/path/to/dataset.yaml
```

Override `RUNTIME` or `MODEL` to measure existing artifacts:

```sh
make ultralytics-benchmark \
  RUNTIME=/path/to/libarm-yolo26n.so.0 \
  MODEL=/path/to/converted-model
```

Validation defaults to `workers=0` (no data-loader subprocesses) and
`plots=False` (no validation plots). These settings do not change the native
runtime's four-worker inference configuration. Override either setting through
`YOLO_ARGS`, for example:

```sh
make ultralytics-benchmark YOLO_ARGS="workers=2 plots=True"
```

Each invocation:

1. obtains the default runtime and converts the pinned YOLO26n checkpoint when
   those artifacts are missing;
2. creates or reuses the pinned Ultralytics virtual environment;
3. applies the integration patch and installs this runtime's backend;
4. runs validation and records the result.

See [Recording Ultralytics benchmark evidence](../reporting/README.md) for the
output layout.

## Layout

- `arm_yolo26n_pi5_backend.py` adapts the runtime to the Ultralytics backend interface.
- `patches/` contains the patch that registers the backend with the pinned Ultralytics release.
- `patch.sh` prepares the virtual environment and applies the integration patch.
- `benchmark.py` creates the temporary model descriptor and runs the recorded benchmark.
