#!/usr/bin/env python3
"""Complete and verify the dependency-closure model after Detect finishes.

Detect creates the four revision-specific versions and populates their
discovered components. It cannot establish the runtime subproject links, add
the reviewed system build tools, or apply the repository's component usage and
Notices decisions. This command performs those operations through Black Duck's
API and fails if the resulting BOMs no longer match the documented closures.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterable, Mapping
from pathlib import Path

from .api import BlackDuck, BlackDuckError

_PROJECT_ID_ENVIRONMENT_VARIABLE = "BLACKDUCK_PROJECT_ID"
_TOKEN_ENVIRONMENT_VARIABLE = "BLACKDUCK_TOKEN"
_URL_ENVIRONMENT_VARIABLE = "BLACKDUCK_URL"
_RUNTIME_COMPONENTS = {"fxdiv", "google/XNNPACK", "libcpuinfo0", "pthreadpool"}
_RUNTIME_FALSE_POSITIVES = {"ML-Embedded-Eval-Kit"}
_EXAMPLE_COMPONENTS = {"NumPy", "opencv-python"}
_VERSION_DESCRIPTIONS = {
    "runtime": (
        "Dependencies incorporated into the compiled Arm YOLO26n runtime library "
        "distributed to users."
    ),
    "examples": (
        "Dependencies required to run the Arm YOLO26n examples, including the "
        "runtime subproject and application-facing Python packages."
    ),
    "build": (
        "Internal tools required to build, convert, verify, and package Arm "
        "YOLO26n artifacts; these tools are not distributed with the runtime."
    ),
    "ultralytics": (
        "Internal dependencies required to run the Ultralytics accuracy and "
        "performance benchmark; these are not application or runtime dependencies."
    ),
}
_VERSION_DISTRIBUTIONS = {
    "runtime": "EXTERNAL",
    "examples": "EXTERNAL",
    "build": "INTERNAL",
    "ultralytics": "INTERNAL",
}


def _require_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise BlackDuckError(f"{name} is not configured")
    return value


def _version_path(project_id: str, version_id: str) -> str:
    return f"/api/projects/{project_id}/versions/{version_id}"


def _find_versions(
    client: BlackDuck, project_id: str, revision: str
) -> dict[str, dict[str, object]]:
    wanted = {f"{revision}-{closure}": closure for closure in _VERSION_DESCRIPTIONS}
    versions = client.collection(f"/api/projects/{project_id}/versions")
    found: dict[str, dict[str, object]] = {}
    for item in versions:
        name = item.get("versionName")
        if isinstance(name, str) and name in wanted:
            found[wanted[name]] = item
    missing = sorted(set(_VERSION_DESCRIPTIONS) - set(found))
    if missing:
        raise BlackDuckError(f"missing Black Duck versions: {', '.join(missing)}")
    return found


def _resource_id(resource: Mapping[str, object]) -> str:
    metadata = resource.get("_meta")
    if not isinstance(metadata, dict):
        raise BlackDuckError("Black Duck resource has no metadata")
    href = metadata.get("href")
    if not isinstance(href, str) or not href:
        raise BlackDuckError("Black Duck resource has no URL")
    return href.rstrip("/").rsplit("/", 1)[-1]


def _wait_for_bom(client: BlackDuck, project_id: str, version_id: str) -> None:
    path = f"{_version_path(project_id, version_id)}/bom-status"
    for _ in range(60):
        status = client.request("GET", path).get("status")
        if status == "UP_TO_DATE":
            return
        if status in {"ERROR", "FAILED"}:
            raise BlackDuckError(f"BOM processing failed for version {version_id}")
        time.sleep(10)
    raise BlackDuckError(f"BOM processing timed out for version {version_id}")


def _components(client: BlackDuck, project_id: str, version_id: str) -> list[dict[str, object]]:
    return client.collection(f"{_version_path(project_id, version_id)}/components")


def _update_version(
    client: BlackDuck,
    project_id: str,
    version: Mapping[str, object],
    closure: str,
) -> None:
    distribution = _VERSION_DISTRIBUTIONS[closure]
    license_data = version.get("license")
    if not isinstance(license_data, dict):
        raise BlackDuckError(f"{closure} version has no project license")
    version_id = _resource_id(version)
    client.request(
        "PUT",
        _version_path(project_id, version_id),
        {
            "versionName": version["versionName"],
            "phase": version["phase"],
            "distribution": distribution,
            "license": license_data,
            "releaseComments": _VERSION_DESCRIPTIONS[closure],
        },
    )


def _ensure_runtime_subproject(
    client: BlackDuck,
    project_id: str,
    runtime_version_id: str,
    parent_version_id: str,
) -> None:
    client.request(
        "POST",
        f"{_version_path(project_id, parent_version_id)}/components",
        {
            "component": f"{client.base_url}/api/components/{project_id}",
            "componentVersion": (
                f"{client.base_url}/api/components/{project_id}/versions/{runtime_version_id}"
            ),
            "usages": ["DYNAMICALLY_LINKED"],
        },
        expected_statuses=(200, 201, 412),
    )


def _load_build_tools() -> list[dict[str, str]]:
    path = Path(__file__).with_name("build-tools.json")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BlackDuckError(f"cannot read build-tool inventory: {error}") from error
    if not isinstance(document, list) or any(not isinstance(item, dict) for item in document):
        raise BlackDuckError("invalid build-tool inventory")
    required = {"name", "version", "component_id", "component_version_id"}
    for item in document:
        if set(item) != required or any(not isinstance(value, str) for value in item.values()):
            raise BlackDuckError("invalid build-tool inventory entry")
    return document


def _ensure_build_tools(client: BlackDuck, project_id: str, build_version_id: str) -> set[str]:
    component_ids: set[str] = set()
    collection_path = f"{_version_path(project_id, build_version_id)}/components"
    for tool in _load_build_tools():
        component_id = tool["component_id"]
        component_ids.add(component_id)
        client.request(
            "POST",
            collection_path,
            {
                "component": f"{client.base_url}/api/components/{component_id}",
                "componentVersion": (
                    f"{client.base_url}/api/components/{component_id}/versions/"
                    f"{tool['component_version_id']}"
                ),
                "usages": ["DEV_TOOL_EXCLUDED"],
                "inAttributionReport": False,
            },
            expected_statuses=(200, 201, 412),
        )
    return component_ids


def _update_component(
    client: BlackDuck, component: Mapping[str, object], payload: Mapping[str, object]
) -> None:
    metadata = component.get("_meta")
    href = metadata.get("href") if isinstance(metadata, dict) else None
    if not isinstance(href, str):
        raise BlackDuckError(f"component {component.get('componentName')!r} has no URL")
    client.request("PUT", href, payload)


def _reconcile_runtime(client: BlackDuck, project_id: str, version_id: str) -> None:
    components = _components(client, project_id, version_id)
    for component in components:
        if component.get("componentName") in _RUNTIME_FALSE_POSITIVES:
            _update_component(client, component, {"ignored": True})
    _wait_for_bom(client, project_id, version_id)
    active = [
        component
        for component in _components(client, project_id, version_id)
        if not component.get("ignored")
    ]
    names = {component.get("componentName") for component in active}
    if names != _RUNTIME_COMPONENTS:
        raise BlackDuckError(
            "runtime BOM components do not match the four native dependencies: "
            f"{sorted(str(name) for name in names)}"
        )
    for component in active:
        _update_component(
            client,
            component,
            {"usages": ["STATICALLY_LINKED"], "inAttributionReport": True},
        )


def _reconcile_build(
    client: BlackDuck,
    project_id: str,
    version_id: str,
    expected_tool_ids: set[str],
) -> None:
    components = _components(client, project_id, version_id)
    present_tool_ids: set[str] = set()
    for component in components:
        if component.get("componentType") == "SUB_PROJECT":
            continue
        component_url = component.get("component")
        if not isinstance(component_url, str):
            raise BlackDuckError("build BOM component has no component URL")
        component_id = component_url.rstrip("/").rsplit("/", 1)[-1]
        if component_id in expected_tool_ids:
            present_tool_ids.add(component_id)
        _update_component(
            client,
            component,
            {"usages": ["DEV_TOOL_EXCLUDED"], "inAttributionReport": False},
        )
    missing = expected_tool_ids - present_tool_ids
    if missing:
        raise BlackDuckError(f"build BOM is missing {len(missing)} reviewed system tools")


def _verify_subproject(
    components: Iterable[Mapping[str, object]], project_id: str, closure: str
) -> None:
    subprojects = [
        component for component in components if component.get("componentType") == "SUB_PROJECT"
    ]
    expected_suffix = f"/api/components/{project_id}"
    if len(subprojects) != 1 or not str(subprojects[0].get("component", "")).endswith(
        expected_suffix
    ):
        raise BlackDuckError(f"{closure} BOM does not contain exactly one runtime subproject")


def _verify_examples(components: Iterable[Mapping[str, object]]) -> None:
    names = {
        component.get("componentName")
        for component in components
        if component.get("componentType") != "SUB_PROJECT"
    }
    if names != _EXAMPLE_COMPONENTS:
        raise BlackDuckError(
            f"unexpected examples BOM components: {sorted(str(name) for name in names)}"
        )


def _torch_mapping_status(
    components: Iterable[Mapping[str, object]],
) -> tuple[bool, bool]:
    identifiers: set[str] = set()
    for component in components:
        external_ids = component.get("inputExternalIds")
        if isinstance(external_ids, list):
            identifiers.update(str(identifier).lower() for identifier in external_ids)
    names = {str(component.get("componentName", "")).lower() for component in components}
    torch = "torch" in names or any(identifier.startswith("torch/") for identifier in identifiers)
    torchvision = "torchvision" in names or any(
        identifier.startswith("torchvision/") for identifier in identifiers
    )
    return torch, torchvision


def _write_summary(
    client: BlackDuck,
    project_id: str,
    versions: Mapping[str, Mapping[str, object]],
    components: Mapping[str, list[dict[str, object]]],
) -> None:
    lines = [
        "## Black Duck dependency closures",
        "",
        "| Closure | BOM entries | Policy status |",
        "| --- | ---: | --- |",
    ]
    for closure in _VERSION_DESCRIPTIONS:
        version_id = _resource_id(versions[closure])
        policy = client.request(
            "GET", f"{_version_path(project_id, version_id)}/policy-status"
        ).get("overallStatus", "UNKNOWN")
        lines.append(f"| {closure} | {len(components[closure])} | {policy} |")
    torch, torchvision = _torch_mapping_status(components["ultralytics"])
    if not torch or not torchvision:
        missing = [
            name for name, found in (("Torch", torch), ("Torchvision", torchvision)) if not found
        ]
        message = "Black Duck did not map the installed +cpu packages: " + ", ".join(missing)
        print(f"::warning::{message}")
        lines.extend(["", f"> **Coverage gap:** {message}"])
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as summary:
            summary.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="revision prefix used by the scan")
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    if not args.revision or any(character not in "0123456789abcdef" for character in args.revision):
        raise BlackDuckError("revision must be a lowercase hexadecimal Git revision")
    project_id = _require_environment(_PROJECT_ID_ENVIRONMENT_VARIABLE)
    client = BlackDuck(
        _require_environment(_URL_ENVIRONMENT_VARIABLE),
        _require_environment(_TOKEN_ENVIRONMENT_VARIABLE),
    )
    versions = _find_versions(client, project_id, args.revision)
    for closure, version in versions.items():
        _wait_for_bom(client, project_id, _resource_id(version))
        _update_version(client, project_id, version, closure)

    runtime_id = _resource_id(versions["runtime"])
    for closure in ("examples", "build", "ultralytics"):
        _ensure_runtime_subproject(client, project_id, runtime_id, _resource_id(versions[closure]))
    build_id = _resource_id(versions["build"])
    expected_tool_ids = _ensure_build_tools(client, project_id, build_id)

    for version in versions.values():
        _wait_for_bom(client, project_id, _resource_id(version))
    _reconcile_runtime(client, project_id, runtime_id)
    _reconcile_build(client, project_id, build_id, expected_tool_ids)

    for version in versions.values():
        _wait_for_bom(client, project_id, _resource_id(version))
    closure_components = {
        closure: _components(client, project_id, _resource_id(version))
        for closure, version in versions.items()
    }
    _verify_examples(closure_components["examples"])
    for closure in ("examples", "build", "ultralytics"):
        _verify_subproject(closure_components[closure], project_id, closure)
    _write_summary(client, project_id, versions, closure_components)


if __name__ == "__main__":
    try:
        _main()
    except BlackDuckError as error:
        raise SystemExit(f"error: {error}") from None
