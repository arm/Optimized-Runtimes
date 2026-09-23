# Python dependency locks

Each `.in` file declares the direct dependencies for one supported environment
and is stored beside the generated `.txt` lock that consumes it:

| Input | Generated lock | Purpose |
| --- | --- | --- |
| `requirements.in` | `requirements.txt` | Python API and examples |
| `requirements-build.in` | `requirements-build.txt` | Building the Python package |
| `requirements-dev.in` | `requirements-dev.txt` | Repository checks and tests |
| `benchmarks/ultralytics/requirements.in` | `benchmarks/ultralytics/requirements.txt` | Optional Ultralytics benchmark |

`requirements-lock.in` owns pip-tools itself and generates
`requirements-lock.txt`. Lock generation therefore does not add packaging
tools to an application or development environment.

Every generated lock contains the complete resolved closure and the SHA-256
hashes published for each pinned distribution. Supported setup commands use
pip's `--require-hashes`. Package builds install the separate build lock and
use `--no-build-isolation`, so their build requirements are not downloaded
implicitly.

Repository lint and test targets install `requirements-dev.txt` into `.venv/`
through a hidden Make prerequisite. The environment is recreated only when it
is absent or the committed development lock changes.

To update dependencies, install `requirements-lock.txt` into a dedicated
Python 3.13 environment. Then run `make requirements-lock` with
`LOCK_COMPILER` set to that environment's `pip-compile`, and review both
version and hash changes. Normal builds install the committed locks; they do
not regenerate them.
