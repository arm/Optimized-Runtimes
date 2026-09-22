# Python dependency locks

The `.in` files declare direct dependencies for four supported environments:

| Input | Generated lock | Purpose |
| --- | --- | --- |
| `runtime.in` | `requirements.txt` | Python API and examples |
| `build.in` | `requirements-build.txt` | Building the Python package |
| `dev.in` | `requirements-dev.txt` | Repository checks and tests |
| `ultralytics.in` | `benchmarks/ultralytics/requirements.txt` | Optional Ultralytics benchmark |

`lock.in` owns pip-tools itself and generates `requirements-lock.txt`. Lock
generation therefore does not add packaging tools to an application or
development environment.

Every generated lock contains the complete resolved closure and the SHA-256
hashes published for each pinned distribution. Supported setup commands use
pip's `--require-hashes`. Package builds install the separate build lock and
use `--no-build-isolation`, so their build requirements are not downloaded
implicitly.

To update dependencies, install `requirements-lock.txt` into a dedicated
Python 3.13 environment. Then run `make requirements-lock` with
`LOCK_COMPILER` set to that environment's `pip-compile`, and review both
version and hash changes. Normal builds install the committed locks; they do
not regenerate them.
