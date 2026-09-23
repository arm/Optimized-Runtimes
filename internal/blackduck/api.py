"""Provide the HTTP boundary used to reconcile Black Duck scan results.

The Black Duck scan action runs Detect, but it does not expose the API
operations needed to link project versions or update component metadata.
``reconcile`` uses this module for authentication, requests, and collection
pagination; dependency policy remains in ``reconcile.py``.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping


class BlackDuckError(RuntimeError):
    """Report an invalid response or dependency closure."""


class BlackDuck:
    """Perform the authenticated API operations that reconciliation needs."""

    def __init__(self, base_url: str, token: str) -> None:
        if not base_url.startswith("https://"):
            raise BlackDuckError("BLACKDUCK_URL must use HTTPS")
        self._base_url = base_url.rstrip("/")
        self._bearer_token = self._authenticate(token)

    @property
    def base_url(self) -> str:
        return self._base_url

    def _authenticate(self, token: str) -> str:
        request = urllib.request.Request(
            f"{self._base_url}/api/tokens/authenticate",
            method="POST",
            headers={
                "Accept": "application/vnd.blackducksoftware.user-4+json",
                "Authorization": f"token {token}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                document = json.load(response)
        except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as error:
            raise BlackDuckError(f"Black Duck authentication failed: {error}") from error
        bearer_token = document.get("bearerToken")
        if not isinstance(bearer_token, str) or not bearer_token:
            raise BlackDuckError("Black Duck authentication returned no bearer token")
        return bearer_token

    def request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, object] | None = None,
        *,
        expected_statuses: Iterable[int] = (200,),
    ) -> dict[str, object]:
        url = path if path.startswith("https://") else f"{self._base_url}{path}"
        data = json.dumps(payload).encode() if payload is not None else None
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._bearer_token}",
        }
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, method=method, headers=headers)
        allowed = set(expected_statuses)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read()
                status = response.status
        except urllib.error.HTTPError as error:
            body = error.read()
            status = error.code
            if status not in allowed:
                message = body.decode(errors="replace")
                raise BlackDuckError(
                    f"Black Duck {method} {url} returned HTTP {status}: {message}"
                ) from error
        except OSError as error:
            raise BlackDuckError(f"Black Duck {method} {url} failed: {error}") from error
        if status not in allowed:
            raise BlackDuckError(f"Black Duck {method} {url} returned unexpected HTTP {status}")
        if not body:
            return {}
        try:
            document: object = json.loads(body)
        except json.JSONDecodeError as error:
            raise BlackDuckError(f"Black Duck {method} {url} returned invalid JSON") from error
        if not isinstance(document, dict) or any(not isinstance(key, str) for key in document):
            raise BlackDuckError(f"Black Duck {method} {url} returned a non-object document")
        return {str(key): value for key, value in document.items()}

    def collection(self, path: str) -> list[dict[str, object]]:
        """Read every page from a Black Duck collection."""
        items: list[dict[str, object]] = []
        offset = 0
        while True:
            separator = "&" if "?" in path else "?"
            document = self.request("GET", f"{path}{separator}limit=100&offset={offset}")
            page = document.get("items")
            total = document.get("totalCount")
            if (
                not isinstance(page, list)
                or not isinstance(total, int)
                or any(
                    not isinstance(item, dict) or any(not isinstance(key, str) for key in item)
                    for item in page
                )
            ):
                raise BlackDuckError(f"invalid Black Duck collection: {path}")
            items.extend({str(key): value for key, value in item.items()} for item in page)
            if len(items) >= total:
                return items
            offset = len(items)
