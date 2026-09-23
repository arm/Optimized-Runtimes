# Release process

This is the authoritative runbook for creating, qualifying, and publishing an
Arm Runtime for YOLO26n release. Follow it in order. A release promotes one
build from `main`, qualifies those exact files on a Raspberry Pi 5, and
publishes them without rebuilding or replacing them.

## Release boundary

The release publishes:

| File | Contents |
| --- | --- |
| `arm_yolo26n-<version>-py3-none-linux_aarch64.whl` | Application API, runtime library, project license, and dependency licenses |
| `arm_yolo26n-<version>-py3-none-linux_aarch64.whl.sha256` | Application wheel identity |
| `arm_yolo26n_converter-<version>-py3-none-any.whl` | Portable checkpoint converter |
| `arm_yolo26n_converter-<version>-py3-none-any.whl.sha256` | Converter wheel identity |

GitHub supplies source archives for the release tag. The project does not
publish a prebuilt development archive, checkpoint, or converted model.
Developers build from the tagged source.

The supported Pi, operating system, Python, library, and camera boundaries are
defined in [`supported-environment.md`](supported-environment.md). This
runbook defines how to demonstrate that one candidate satisfies them.

## 1. Prepare `main`

Before dispatching a candidate:

1. Set `VERSION` to the intended `major.minor.patch` release number.
2. Merge all release changes to `main`.
3. Confirm that the **Build** workflow passed for the current `main` commit.
4. Confirm that the **Black Duck** workflow passed for the same commit.
5. Review the four `main-*` Black Duck BOMs and their policy results.
6. Complete any applicable pre-merge Black Duck snippet review described in
   [`black-duck-scanning-scheme.md`](black-duck-scanning-scheme.md).

The successful `main` build is the only release build. It runs linting, unit
tests, checkpoint conversion, all-resolution end-to-end tests, and
`make release`, then retains its outputs for 30 days. Candidate creation must
reuse the wheels from that artifact.

## 2. Create the draft candidate

Open **Actions → Create release candidate → Run workflow**, select `main`, and
confirm Black Duck review. The workflow:

1. reads the version and source revision from `main`;
2. requires successful Build and Black Duck runs for that exact revision;
3. downloads the retained build artifact without rebuilding it;
4. verifies both wheels, their checksums, and the application wheel's expected
   runtime and license contents; and
5. creates a draft `v<version>` release containing those four files.

Do not publish the draft. Record its source revision, Build run, and Black Duck
run from the generated release notes.

## 3. Stage the candidate on the Pi

Use a Raspberry Pi 5 running 64-bit Raspberry Pi OS Trixie. Install the system
camera packages before qualification if the release claims camera support:

```bash
sudo apt update
sudo apt install --yes python3-picamera2 python3-venv rpicam-apps
```

Choose a persistent, empty qualification directory. Do not use `/tmp`, because
the Pi is restarted later in the procedure.

```bash
set -euo pipefail
repository=arm/Arm-Runtime-for-YOLO26-on-Raspberry-Pi-5
version=0.1.0                         # Set the candidate version.
release="v$version"
qualification="$HOME/arm-yolo26n-$version-qualification"

test ! -e "$qualification"
mkdir -p "$qualification"/{downloads,evidence,logs,models}

revision=$(gh release view "$release" --repo "$repository" \
  --json isDraft,targetCommitish \
  --jq 'select(.isDraft).targetCommitish')
test -n "$revision"

git clone "https://github.com/$repository.git" "$qualification/source"
git -C "$qualification/source" checkout --detach "$revision"
test "$(cat "$qualification/source/VERSION")" = "$version"

gh release download "$release" --repo "$repository" \
  --dir "$qualification/downloads"
test "$(find "$qualification/downloads" -maxdepth 1 -type f | wc -l)" -eq 4

cd "$qualification/downloads"
sha256sum --check "arm_yolo26n-$version-py3-none-linux_aarch64.whl.sha256"
sha256sum --check "arm_yolo26n_converter-$version-py3-none-any.whl.sha256"
```

Record SHA-256 values for both wheels. Every later step must use these staged
files. A changed digest invalidates the qualification and requires a new
candidate.

```bash
cd "$qualification"
sha256sum \
  downloads/*.whl \
  > evidence/candidate-sha256.txt
```

## 4. Install the candidate application environment

Create the environment on persistent storage. `--system-site-packages` exposes
the Raspberry Pi OS Picamera2 installation without installing a mismatched
PyPI camera stack.

```bash
cd "$qualification"
python3 -m venv --system-site-packages application-venv

PIP_CONFIG_FILE=/dev/null PIP_NO_CACHE_DIR=1 \
  application-venv/bin/python -m pip install --require-hashes \
  --requirement source/requirements.txt

PIP_CONFIG_FILE=/dev/null PIP_NO_CACHE_DIR=1 \
  application-venv/bin/python -m pip install --no-deps \
  "downloads/arm_yolo26n-$version-py3-none-linux_aarch64.whl" \
  "downloads/arm_yolo26n_converter-$version-py3-none-any.whl"
```

Verify that both distributions report the candidate version and import from
`application-venv`, not the repository checkout.

```bash
application-venv/bin/python - <<'PY'
from importlib.metadata import version
import arm_yolo26n
import converter

print("arm-yolo26n", version("arm-yolo26n"), arm_yolo26n.__file__)
print("arm-yolo26n-converter", version("arm-yolo26n-converter"), converter.__file__)
PY
```

Locate the installed runtime and add its digest to the candidate identities:

```bash
runtime=$(application-venv/bin/python - <<'PY'
from pathlib import Path
import arm_yolo26n

print(Path(arm_yolo26n.__file__).with_name("libarm-yolo26n.so.0"))
PY
)
test -f "$runtime"
sha256sum "$runtime" >> evidence/candidate-sha256.txt
```

Confirm that it is an AArch64 shared object with the expected SONAME and only
the supported system-library dependencies:

```bash
readelf -h "$runtime"
readelf -d "$runtime" | grep -E 'SONAME|NEEDED'
```

## 5. Convert and execute every supported resolution

Download the official checkpoint and verify its pinned identity:

```bash
cd "$qualification"
curl --fail --location --show-error \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt \
  --output downloads/yolo26n.pt.tmp
echo '9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef  downloads/yolo26n.pt.tmp' \
  | sha256sum --check
mv downloads/yolo26n.pt.tmp downloads/yolo26n.pt
```

Use the installed candidate converter for every supported resolution:

```bash
for resolution in 320 480 640 800 960 1280; do
  application-venv/bin/python -m converter.export_yolo26n \
    --checkpoint downloads/yolo26n.pt \
    --resolution "$resolution" \
    --output-dir "models/yolo26n-$resolution" \
    2>&1 | tee "logs/convert-$resolution.log"
done
```

Use the installed candidate API and its bundled runtime to execute each model.
The check verifies the public tensor shapes and finite output, not internal
implementation details.

```bash
MODELS="$qualification/models" \
  application-venv/bin/python - <<'PY' 2>&1 \
  | tee logs/runtime-all-resolutions.log
import os
from pathlib import Path
import numpy as np
from arm_yolo26n import Yolo26nTensorSession

models = Path(os.environ["MODELS"])
for resolution in (320, 480, 640, 800, 960, 1280):
    with Yolo26nTensorSession(model=models / f"yolo26n-{resolution}") as session:
        output = session.execute(np.zeros(session.input_shape, dtype=np.float32))
        assert tuple(output.shape) == session.output_shape
        assert np.isfinite(output).all()
        print(resolution, session.input_shape, session.output_shape, "pass")
PY
```

## 6. Run the prerecorded-video example

Download and verify the public MediaPipe test video, then process 200 frames
through the candidate runtime and API:

```bash
curl --fail --location --show-error \
  https://raw.githubusercontent.com/google-ai-edge/mediapipe/0ad5a71bcdff3d756dc5b07f93765aaeb4152538/mediapipe/examples/desktop/object_detection/test_video.mp4 \
  --output downloads/test-video.mp4.tmp
echo '710831289c00251c86eafb0b0d11cf6190fda02f95bcfaa5037511eabb5c9de8  downloads/test-video.mp4.tmp' \
  | sha256sum --check
mv downloads/test-video.mp4.tmp downloads/test-video.mp4

mkdir -p evidence/video
PYTHONPATH="$qualification/source" \
  application-venv/bin/python -m examples.video.detect \
  --model "$qualification/models/yolo26n-640" \
  --video "$qualification/downloads/test-video.mp4" \
  --output "$qualification/evidence/video/detections.json" \
  --frames 200 \
  2>&1 | tee logs/video-example.log
```

Record the frame count and input-video identity from `detections.json`.
Throughput observed before the clean restart is functional evidence only, not
the release performance result.

## 7. Prepare the Ultralytics environment

The benchmark has its own pinned environment. The current
`ultralytics-patch` implementation builds the application API from the source
checkout while installing its backend. That source-built wheel must not stand
in for the candidate wheel, so reinstall the exact candidate wheel afterward.

```bash
benchmark_venv="$qualification/ultralytics-venv"
PIP_NO_CACHE_DIR=1 \
  "$qualification/source/benchmarks/ultralytics/patch.sh" "$benchmark_venv"

PIP_CONFIG_FILE=/dev/null PIP_NO_CACHE_DIR=1 \
  "$benchmark_venv/bin/python" -m pip install \
  --force-reinstall --no-deps \
  "$qualification/downloads/arm_yolo26n-$version-py3-none-linux_aarch64.whl"
```

Verify `arm-yolo26n` and Ultralytics versions and import locations before
continuing. A future implementation should let `ultralytics-patch` accept the
candidate wheel directly; until then this reinstall is required.

## 8. Restart and inspect the Pi

Complete all downloads and environment installation before the restart. Attach
the camera, restart the Pi, allow startup activity to settle, and avoid
unrelated workloads during performance measurement.

```bash
sudo reboot
```

After reconnecting, record:

```bash
set -euo pipefail
repository=arm/Arm-Runtime-for-YOLO26-on-Raspberry-Pi-5
version=0.1.0                         # Set the candidate version.
release="v$version"
qualification="$HOME/arm-yolo26n-$version-qualification"
revision=$(git -C "$qualification/source" rev-parse HEAD)
benchmark_venv="$qualification/ultralytics-venv"
runtime=$(application-venv/bin/python - <<'PY'
from pathlib import Path
import arm_yolo26n

print(Path(arm_yolo26n.__file__).with_name("libarm-yolo26n.so.0"))
PY
)

uptime -p
cat /proc/loadavg
free -h
df -h / /tmp
taskset -pc $$
vcgencmd get_throttled
vcgencmd measure_temp
rpicam-hello --list-cameras
cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
cat /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq
```

Do not continue if the host is currently throttled, using swap, under material
memory or disk pressure, or competing with unrelated workloads. Record
historical throttling flags and decide whether another restart or hardware
investigation is required. Do not clear the Linux page cache manually; the
benchmark recorder captures cache, fault, pressure, and swap context for human
interpretation.

## 9. Record accuracy and performance

The accepted comparison uses COCO128, batch 1, CPU execution, `workers=0`, and
`plots=False`. Create the Ultralytics configuration directory before running
to prevent a fallback to unrelated temporary state.

Temporarily select the `performance` governor at 2.4 GHz to match the accepted
baseline, then restore the prior governor after the command:

```bash
cd "$qualification"
governor=/sys/devices/system/cpu/cpufreq/policy0/scaling_governor
original_governor=$(cat "$governor")
restore_governor() {
  printf '%s' "$original_governor" | sudo tee "$governor" >/dev/null
}
trap restore_governor EXIT
printf performance | sudo tee "$governor" >/dev/null

mkdir -p ultralytics-config/Ultralytics
PYTHONPATH="$qualification/source" \
YOLO_CONFIG_DIR="$qualification/ultralytics-config" \
  application-venv/bin/python -m benchmarks.ultralytics.benchmark \
  --runtime "$runtime" \
  --model "$qualification/models/yolo26n-640" \
  --data coco128.yaml \
  --venv "$qualification/ultralytics-venv" \
  --output-dir "$qualification/evidence/ultralytics-coco128-qualified" \
  --revision "$revision" \
  -- workers=0 plots=False

restore_governor
trap - EXIT
```

Compare precision, recall, mAP50, mAP50-95, and stage timings with
[`../benchmarks/reporting/baseline.md`](../benchmarks/reporting/baseline.md).
The evidence directory retains the benchmark log, runtime and model identities,
peak RSS, and before/after host context. Performance acceptance is a human
review decision; this runbook intentionally defines no automatic percentage
threshold. Do not repeat a completed run merely to select a more favorable
number. Repeat only when the evidence identifies an invalid session.

## 10. Qualify the physical camera

The first release claiming camera support must record the camera module and
pass both examples. First confirm camera discovery:

```bash
rpicam-hello --list-cameras | tee logs/camera-discovery.log
```

Run bounded headless capture:

```bash
PYTHONPATH="$qualification/source" \
  application-venv/bin/python -m examples.picamera2.detect \
  --model "$qualification/models/yolo26n-640" \
  --camera 0 \
  --frames 100 \
  2>&1 | tee logs/picamera2-headless.log
```

For a Wayland Raspberry Pi desktop reached from another session, use the full
documented Qt environment. Omitting `DISPLAY` can make Qt select an unusable
backend even when the Wayland socket exists.

```bash
DISPLAY=:0 \
WAYLAND_DISPLAY=wayland-0 \
XDG_RUNTIME_DIR="/run/user/$(id -u)" \
PYTHONPATH="$qualification/source" \
  application-venv/bin/python -m examples.picamera2.live \
  --model "$qualification/models/yolo26n-640" \
  --camera 0 \
  --preview qtgl \
  --frames 100 \
  2>&1 | tee logs/picamera2-live.log
```

Use `--preview drm` instead only when qualifying a direct DRM console rather
than the desktop QtGL path.

## 11. Record the evidence

Write `evidence/qualification-report.md` as a normal test report. It must
identify:

- release version, source revision, candidate asset names, and SHA-256 values;
- Pi model, OS, kernel, Python, camera, and camera-stack package versions;
- each command or public behavior checked and its pass, fail, or blocked result;
- all-resolution conversion and execution results;
- video-example result;
- accuracy metrics and performance timings;
- the human performance decision and its rationale;
- benchmark host context, including thermals, throttling, frequency, memory,
  filesystem cache, pressure, faults, and swap; and
- any failed attempts or process corrections needed to interpret the session.

Retain the report and supporting evidence with the project's internal release
records. The qualification report is not a release asset.

## 12. Freeze the Black Duck release BOMs

Before publication, confirm that each `main-*` Black Duck version is mapped to
the candidate revision. Clone component data from:

- `main-runtime` to `<version>-runtime`;
- `main-examples` to `<version>-examples`;
- `main-build` to `<version>-build`; and
- `main-ultralytics` to `<version>-ultralytics`.

Review the cloned usages, ignored matches, manually inventoried tools,
subproject relationships, Notices settings, licenses, and policy results. The
clones preserve the reviewed release BOM; do not rescan or reconstruct it with
a repository-specific reconciliation tool.

## 13. Publish

Before publication, verify:

1. the draft targets the qualified source revision;
2. its original application and converter wheels and checksums still match the
   qualification report;
3. the qualification report and supporting evidence have been retained in the
   project's internal release records;
4. the four cloned Black Duck versions have been reviewed;
5. accuracy, performance, functional, and camera results are accepted; and
6. the release notes identify the supported Pi environment and state that
   qualification passed.

Publish the existing draft without rebuilding, replacing, or re-uploading its
candidate assets. GitHub creates the release tag and source archives at this
point. After publication, verify the tag target, all four attached files, both
checksums, and the user-facing installation commands.

If any candidate digest changes, qualification is invalid. If any required
check is incomplete or rejected, leave the release as a draft.
