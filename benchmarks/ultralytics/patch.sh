#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
venv=${1:-"$repo/benchmarks/ultralytics/.venv"}

if [[ $(uname -m) != aarch64 ]]; then
  printf 'error: the optimized runtime requires aarch64\n' >&2
  exit 1
fi
if [[ ! -r /etc/os-release ]] || ! grep -q '^VERSION_CODENAME=trixie$' /etc/os-release; then
  printf 'error: the validated workflow requires Debian/Raspberry Pi OS trixie\n' >&2
  exit 1
fi
if [[ ! -r /proc/device-tree/model ]] || ! grep -q 'Raspberry Pi 5' /proc/device-tree/model; then
  printf 'error: the optimized runtime is validated on Raspberry Pi 5\n' >&2
  exit 1
fi

python3 -m venv "$venv"
"$venv/bin/python" -m pip install --upgrade pip
"$venv/bin/python" -m pip install -r "$repo/benchmarks/ultralytics/requirements.txt"
"$venv/bin/python" -m pip install "$repo"

site=$("$venv/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)
install -m 0644 "$repo/benchmarks/ultralytics/arm_yolo26n_pi5_backend.py" \
  "$site/ultralytics/nn/backends/arm_yolo26n_pi5.py"

integration_patch="$repo/benchmarks/ultralytics/patches/ultralytics-8.4.108-arm-yolo26-pi5.patch"
if patch --dry-run --forward -p1 -d "$site" < "$integration_patch" >/dev/null; then
  patch --forward -p1 -d "$site" < "$integration_patch"
elif ! patch --dry-run --reverse -p1 -d "$site" < "$integration_patch" >/dev/null; then
  printf 'error: Ultralytics 8.4.108 integration patch does not apply cleanly\n' >&2
  exit 1
fi

"$venv/bin/python" - <<'PY'
import ultralytics

assert ultralytics.__version__ == "8.4.108", ultralytics.__version__
from arm_yolo26n import Yolo26nTensorSession
from ultralytics.nn.backends import ArmYolo26nPi5Backend

assert Yolo26nTensorSession is not None
assert ArmYolo26nPi5Backend is not None
print("installed ArmYolo26nPi5Backend")
PY
