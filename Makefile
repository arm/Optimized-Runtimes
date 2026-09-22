SHELL := /bin/bash
.DEFAULT_GOAL := help

# Shared commands and repository paths
ROOT := $(CURDIR)
PYTHON ?= python3
CLANG_FORMAT ?= clang-format
LOCK_COMPILER ?= pip-compile
TEST_PYTHONPATH := $(ROOT)/src:$(ROOT)/src/runtime/api/python:$(ROOT)
VERSION := $(strip $(file <$(ROOT)/VERSION))

# Checkpoint conversion
DEFAULT_CHECKPOINT := $(ROOT)/downloads/checkpoints/yolo26n.pt
CHECKPOINT ?= $(DEFAULT_CHECKPOINT)
CHECKPOINT_URL := https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt
CHECKPOINT_SHA256 := 9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef
RESOLUTION ?= 640
MODEL ?= $(ROOT)/dist/models/yolo26n-$(RESOLUTION)
MODEL_VALIDATION ?= $(ROOT)/dist/models/validation/yolo26n-$(RESOLUTION).json

# Native runtime
RUNTIME_BUILD ?= $(ROOT)/build/runtime
RUNTIME_DIST ?= $(ROOT)/dist/runtime
DEFAULT_RUNTIME := $(RUNTIME_DIST)/lib/libarm-yolo26n.so.0
RUNTIME ?= $(DEFAULT_RUNTIME)
PACKAGE_DIST ?= $(ROOT)/dist/packages
RUNTIME_RELEASE ?= v$(VERSION)
RUNTIME_ARCHIVE := arm-yolo26n-rpi5-$(VERSION)-runtime.tar.gz
RUNTIME_URL ?= https://github.com/Arm-Debug/yolo-raspi-research/releases/download/$(RUNTIME_RELEASE)/$(RUNTIME_ARCHIVE)
RUNTIME_DOWNLOAD := $(ROOT)/downloads/runtime/$(RUNTIME_RELEASE)/$(RUNTIME_ARCHIVE)
RUNTIME_CHECKSUM := $(RUNTIME_DOWNLOAD).sha256

# Examples
EXAMPLES_VENV ?= $(ROOT)/examples/.venv
EXAMPLES_PYTHON := $(EXAMPLES_VENV)/bin/python
DEFAULT_VIDEO := $(ROOT)/downloads/media/test-video.mp4
VIDEO ?= $(DEFAULT_VIDEO)
VIDEO_URL := https://raw.githubusercontent.com/google-ai-edge/mediapipe/0ad5a71bcdff3d756dc5b07f93765aaeb4152538/mediapipe/examples/desktop/object_detection/test_video.mp4
VIDEO_SHA256 := 710831289c00251c86eafb0b0d11cf6190fda02f95bcfaa5037511eabb5c9de8
VIDEO_OUTPUT ?= $(ROOT)/examples/video/output/detections.json

# Ultralytics benchmark
ULTRALYTICS_VENV ?= $(ROOT)/benchmarks/ultralytics/.venv
DATA ?= coco128.yaml
SOURCE_REVISION ?= $(shell git rev-parse HEAD)
BENCHMARK_OUTPUT ?= $(ROOT)/benchmarks/reporting/output/current

# Development checks
MODELS ?= $(ROOT)/dist/models/yolo26n-320:$(ROOT)/dist/models/yolo26n-480:$(ROOT)/dist/models/yolo26n-640:$(ROOT)/dist/models/yolo26n-800:$(ROOT)/dist/models/yolo26n-960:$(ROOT)/dist/models/yolo26n-1280
MODEL_DIRECTORIES := $(subst :, ,$(MODELS))
MODEL_ARTIFACTS := $(foreach model,$(MODEL_DIRECTORIES),\
	$(model)/manifest.json $(model)/model.bin)
# The C API test needs one valid model to create a session. The Python
# end-to-end tests exercise every model in MODELS.
NATIVE_API_TEST_MODEL = $(firstword $(MODEL_DIRECTORIES))/model.bin
COVERAGE_ROOT ?= $(ROOT)/build/coverage
COVERAGE_RUNTIME_BUILD ?= $(COVERAGE_ROOT)/runtime
COVERAGE_RUNTIME ?= $(COVERAGE_RUNTIME_BUILD)/libarm-yolo26n.so
PYTHON_COVERAGE_REPORT := $(COVERAGE_ROOT)/python/coverage.xml
NATIVE_COVERAGE_REPORT := $(COVERAGE_ROOT)/native/coverage.xml

.PHONY: help
help:
	@printf '%s\n' \
	  'YOLO26n Raspberry Pi 5 workflows:' \
	  '' \
	  'Build and conversion:' \
	  '  make model [CHECKPOINT=/path/to/yolo26n.pt] RESOLUTION=640' \
	  '      Convert a checkpoint under dist/models/. Without CHECKPOINT, download' \
	  '      and verify the pinned official YOLO26n checkpoint.' \
	  '' \
	  '  make model-check [CHECKPOINT=/path/to/yolo26n.pt] RESOLUTION=640' \
	  '      Validate conversion and write a receipt. Without CHECKPOINT, download' \
	  '      and verify the pinned official YOLO26n checkpoint.' \
	  '' \
	  '  make runtime' \
	  '      Configure, compile, test, and write the native runtime to dist/runtime/.' \
	  '  make runtime-configure' \
	  '      Configure the native build and fetch its pinned dependencies.' \
	  '' \
	  '  make runtime-package' \
	  '      Produce CPack runtime and development archives under dist/packages/.' \
	  '' \
	  '  make requirements-lock' \
	  '      Regenerate the hash-locked Python dependency closures.' \
	  '' \
	  'Examples and benchmarks:' \
	  '  make examples' \
	  '      Install dependencies in examples/.venv and prepare a runtime and model.' \
	  '' \
	  '  make example-video' \
	  '      Run the prerecorded-video example and write examples/video/output/.' \
	  '' \
	  '  make ultralytics-patch' \
	  '      Install pinned Ultralytics and apply the benchmark integration patch.' \
	  '' \
	  '  make ultralytics-benchmark RUNTIME=... MODEL=... DATA=...' \
	  '      Run Ultralytics validation for the selected runtime, model, and dataset.' \
	  '' \
	  'Development checks:' \
	  '  make lint' \
	  '      Run source formatting, static analysis, typing, and syntax checks.' \
	  '' \
	  '  make test-unit' \
	  '      Run unit and command-interface tests without the native runtime.' \
	  '' \
	  '  make test-converter CHECKPOINT=/path/to/yolo26n.pt' \
	  '      Run checkpoint-backed converter behavior tests.' \
	  '' \
	  '  make test-end-to-end CHECKPOINT=... RUNTIME=... MODELS=/model/320:/model/480:...' \
	  '      Run converter, native runtime, and example behavior tests.' \
	  '' \
	  '  make coverage CHECKPOINT=... MODELS=/model/320:/model/480:...' \
	  '      Generate diagnostic Python and native coverage reports.'

# Python dependencies
.PHONY: requirements-lock
requirements-lock:
	PIP_CONFIG_FILE=/dev/null CUSTOM_COMPILE_COMMAND='make requirements-lock' "$(LOCK_COMPILER)" \
	  --generate-hashes --allow-unsafe --strip-extras \
	  --output-file requirements.txt requirements.in
	PIP_CONFIG_FILE=/dev/null CUSTOM_COMPILE_COMMAND='make requirements-lock' "$(LOCK_COMPILER)" \
	  --generate-hashes --allow-unsafe --strip-extras \
	  --output-file requirements-build.txt requirements-build.in
	PIP_CONFIG_FILE=/dev/null CUSTOM_COMPILE_COMMAND='make requirements-lock' "$(LOCK_COMPILER)" \
	  --generate-hashes --allow-unsafe --strip-extras \
	  --output-file requirements-dev.txt requirements-dev.in
	PIP_CONFIG_FILE=/dev/null CUSTOM_COMPILE_COMMAND='make requirements-lock' "$(LOCK_COMPILER)" \
	  --generate-hashes --allow-unsafe --strip-extras \
	  --output-file requirements-lock.txt requirements-lock.in
	PIP_CONFIG_FILE=/dev/null CUSTOM_COMPILE_COMMAND='make requirements-lock' "$(LOCK_COMPILER)" \
	  --generate-hashes --allow-unsafe --strip-extras \
	  --output-file benchmarks/ultralytics/requirements.txt \
	  benchmarks/ultralytics/requirements.in

# Checkpoint conversion
.PHONY: model model-check
model: $(CHECKPOINT)
	PYTHONPATH="$(ROOT)/src" "$(PYTHON)" -m converter.export_yolo26n \
	  --checkpoint "$(CHECKPOINT)" \
	  --resolution "$(RESOLUTION)" \
	  --output-dir "$(MODEL)"

model-check: $(CHECKPOINT)
	PYTHONPATH="$(ROOT)/src" "$(PYTHON)" -m converter.validate \
	  --checkpoint "$(CHECKPOINT)" \
	  --resolution "$(RESOLUTION)" \
	  --output "$(MODEL_VALIDATION)"

# Native runtime
.PHONY: runtime-configure runtime runtime-package
runtime-configure:
	cmake -S src/runtime -B "$(RUNTIME_BUILD)" \
	  -DCMAKE_BUILD_TYPE=Release

runtime: runtime-configure
	cmake --build "$(RUNTIME_BUILD)" \
	  --target arm_yolo26n yolo26n_runtime_api_test --parallel
	ctest --test-dir "$(RUNTIME_BUILD)" --output-on-failure
	cmake --install "$(RUNTIME_BUILD)" --prefix "$(RUNTIME_DIST)" --component runtime

runtime-package: runtime
	mkdir -p "$(PACKAGE_DIST)"
	cpack --config "$(RUNTIME_BUILD)/CPackConfig.cmake" -B "$(PACKAGE_DIST)"

# Examples
.PHONY: examples example-video
examples: $(RUNTIME) $(MODEL)/manifest.json $(MODEL)/model.bin
	@# Picamera2 is installed by Raspberry Pi OS, so camera examples must see system packages.
	"$(PYTHON)" -m venv --system-site-packages "$(EXAMPLES_VENV)"
	PIP_CONFIG_FILE=/dev/null "$(EXAMPLES_PYTHON)" -m pip install --require-hashes \
	  --requirement requirements-build.txt --requirement requirements.txt
	PIP_CONFIG_FILE=/dev/null "$(EXAMPLES_PYTHON)" -m pip install \
	  --no-deps --no-build-isolation "$(ROOT)"

$(DEFAULT_CHECKPOINT):
	mkdir -p "$(@D)"
	curl --fail --location --show-error "$(CHECKPOINT_URL)" --output "$@.tmp"
	echo "$(CHECKPOINT_SHA256)  $@.tmp" | sha256sum --check
	mv "$@.tmp" "$@"

# A source build and a release download establish the same runtime path. Attach
# the download rules only when no usable source-built distribution is present.
ifneq ($(shell test -f "$(DEFAULT_RUNTIME)" && printf present),present)
$(RUNTIME_DOWNLOAD):
	mkdir -p "$(@D)"
	curl --fail --location --show-error "$(RUNTIME_URL)" --output "$@.tmp"
	mv "$@.tmp" "$@"

$(RUNTIME_CHECKSUM):
	mkdir -p "$(@D)"
	curl --fail --location --show-error "$(RUNTIME_URL).sha256" --output "$@.tmp"
	mv "$@.tmp" "$@"

$(DEFAULT_RUNTIME): $(RUNTIME_DOWNLOAD) $(RUNTIME_CHECKSUM)
	@test ! -e "$(RUNTIME_DIST)" || { \
	  printf 'runtime distribution is incomplete: %s\n' "$(RUNTIME_DIST)" >&2; exit 2; \
	}
	@cd "$(dir $(RUNTIME_DOWNLOAD))" && sha256sum --check "$(notdir $(RUNTIME_CHECKSUM))"
	@staging="$$(mktemp -d --tmpdir="$(dir $(RUNTIME_DIST))" .runtime.XXXXXX)"; \
	  trap 'rm -rf -- "$$staging"' EXIT; \
	  tar --extract --gzip --file "$(RUNTIME_DOWNLOAD)" --directory "$$staging"; \
	  test -f "$$staging/lib/libarm-yolo26n.so.0"; \
	  mv "$$staging" "$(RUNTIME_DIST)"
endif

# One converter invocation publishes these two model artifacts together.
$(MODEL)/manifest.json $(MODEL)/model.bin &: $(CHECKPOINT)
	PYTHONPATH="$(ROOT)/src" "$(PYTHON)" -m converter.export_yolo26n \
	  --checkpoint "$(CHECKPOINT)" --resolution "$(RESOLUTION)" --output-dir "$(MODEL)"

$(DEFAULT_VIDEO):
	mkdir -p "$(@D)"
	curl --fail --location --show-error "$(VIDEO_URL)" --output "$@.tmp"
	echo "$(VIDEO_SHA256)  $@.tmp" | sha256sum --check
	mv "$@.tmp" "$@"

example-video: examples $(VIDEO)
	"$(EXAMPLES_PYTHON)" -m examples.video.detect \
	  --runtime "$(RUNTIME)" --model "$(MODEL)" --video "$(VIDEO)" \
	  --output "$(VIDEO_OUTPUT)" --frames 200

# Ultralytics benchmark
.PHONY: ultralytics-patch ultralytics-benchmark
ultralytics-patch:
	./benchmarks/ultralytics/patch.sh "$(ULTRALYTICS_VENV)"

ultralytics-benchmark:
	"$(PYTHON)" -m benchmarks.ultralytics.benchmark \
	  --runtime "$(RUNTIME)" \
	  --model "$(MODEL)" \
	  --data "$(DATA)" \
	  --venv "$(ULTRALYTICS_VENV)" \
	  --output-dir "$(BENCHMARK_OUTPUT)" \
	  --revision "$(SOURCE_REVISION)" \
	  -- $(YOLO_ARGS)

# Development checks
.PHONY: lint test-unit test-converter test-end-to-end coverage
lint:
	@# clang-format verifies the C and C++ source formatting.
	@find src/runtime -type f \
	  \( -name '*.c' -o -name '*.cpp' -o -name '*.h' -o -name '*.inc' \) \
	  -exec "$(CLANG_FORMAT)" --dry-run --Werror {} +
	@# Ruff's formatter verifies the Python source formatting.
	"$(PYTHON)" -m ruff format --check src/converter src/runtime/api/python/arm_yolo26n \
	  benchmarks examples internal/blackduck tests
	@# Ruff's linter detects Python correctness, style, and import problems.
	"$(PYTHON)" -m ruff check src/converter src/runtime/api/python/arm_yolo26n \
	  benchmarks examples internal/blackduck tests
	@# Import Linter enforces the converter's declared dependency direction.
	PYTHONPATH="$(ROOT)/src" "$(PYTHON)" -c \
	  'from importlinter.cli import lint_imports; raise SystemExit(lint_imports(no_cache=True))'
	@# Mypy checks consistency between the declared Python types and their uses.
	"$(PYTHON)" -m mypy
	@# Vulture reports Python functions, classes, and variables that are never used.
	"$(PYTHON)" -m vulture
	@# Bash parses every benchmark shell script without executing it.
	@find benchmarks/ultralytics -maxdepth 1 -type f -name '*.sh' -exec bash -n {} +
	@# Python's JSON parser verifies the checked-in JSON documents.
	@shopt -s nullglob; for document in src/converter/*.json benchmarks/reporting/*.json \
	    internal/blackduck/*.json; do \
	  "$(PYTHON)" -m json.tool "$$document" >/dev/null; \
	done
	@# Git detects whitespace errors introduced by the current change.
	git diff --check

test-unit:
	PYTHONPATH="$(TEST_PYTHONPATH)" "$(PYTHON)" -m pytest tests/unit

test-converter: $(CHECKPOINT)
	PYTHONPATH="$(TEST_PYTHONPATH)" YOLO26N_CHECKPOINT="$(CHECKPOINT)" \
	  "$(PYTHON)" -m pytest tests/converter

test-end-to-end: $(MODEL_ARTIFACTS) test-converter runtime
	YOLO26N_RUNTIME_API_TEST_MODEL="$(NATIVE_API_TEST_MODEL)" \
	  ctest --test-dir "$(RUNTIME_BUILD)" --output-on-failure \
	  --tests-regex '^yolo26n_runtime_api_behavior$$'
	PYTHONPATH="$(TEST_PYTHONPATH)" \
	  YOLO26N_E2E_RUNTIME="$(RUNTIME)" YOLO26N_E2E_MODELS="$(MODELS)" \
	  "$(PYTHON)" -m pytest tests/end_to_end

coverage: $(CHECKPOINT) $(MODEL_ARTIFACTS)
	@# Build the runtime with coverage instrumentation and discard stale counters.
	cmake -S src/runtime -B "$(COVERAGE_RUNTIME_BUILD)" \
	  -DCMAKE_BUILD_TYPE=Debug -DYOLO26N_ENABLE_COVERAGE=ON
	cmake --build "$(COVERAGE_RUNTIME_BUILD)" \
	  --target arm_yolo26n yolo26n_runtime_api_test --parallel
	find "$(COVERAGE_RUNTIME_BUILD)/CMakeFiles/arm_yolo26n.dir" -name '*.gcda' -delete
	YOLO26N_RUNTIME_API_TEST_MODEL="$(NATIVE_API_TEST_MODEL)" \
	  ctest --test-dir "$(COVERAGE_RUNTIME_BUILD)" --output-on-failure
	@# Run the public Python behavior suites against the instrumented runtime.
	mkdir -p "$(dir $(PYTHON_COVERAGE_REPORT))" "$(dir $(NATIVE_COVERAGE_REPORT))"
	"$(PYTHON)" -m coverage erase
	PYTHONPATH="$(TEST_PYTHONPATH)" \
	  YOLO26N_CHECKPOINT="$(CHECKPOINT)" \
	  YOLO26N_E2E_RUNTIME="$(COVERAGE_RUNTIME)" \
	  YOLO26N_E2E_MODELS="$(MODELS)" \
	  "$(PYTHON)" -m coverage run -m pytest tests/unit tests/converter tests/end_to_end
	@# Write one machine-readable report for each implementation language.
	"$(PYTHON)" -m coverage combine
	"$(PYTHON)" -m coverage report
	"$(PYTHON)" -m coverage xml -o "$(PYTHON_COVERAGE_REPORT)"
	"$(PYTHON)" -m gcovr --root "$(ROOT)" --filter "$(ROOT)/src/runtime/" \
	  --exclude-unreachable-branches --exclude-throw-branches \
	  "$(COVERAGE_RUNTIME_BUILD)/CMakeFiles/arm_yolo26n.dir" \
	  --txt - --cobertura "$(NATIVE_COVERAGE_REPORT)"
