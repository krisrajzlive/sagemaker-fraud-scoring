"""Minimal HTTP client for a SageMaker-style ``/invocations`` endpoint (standard library only)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class EndpointError(RuntimeError):
    """The endpoint was unreachable (``status`` is None) or answered with an error status."""

    def __init__(self, status: int | None, detail: str) -> None:
        self.status = status
        self.detail = detail
        super().__init__(f"{status}: {detail}" if status else detail)


def _error_detail(body: bytes) -> str:
    """The SageMaker toolkit appends a traceback to error bodies; keep only the message line."""
    first_line = body.decode("utf-8", errors="replace").strip().splitlines()[:1]
    if not first_line:
        return "empty error response"
    try:
        parsed = json.loads(first_line[0])
    except ValueError:
        return first_line[0]
    return str(parsed.get("error", parsed)) if isinstance(parsed, dict) else first_line[0]


def ping(base_url: str, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/ping", timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def invoke(
    base_url: str, instances: list[dict[str, Any]], *, timeout: float = 30.0
) -> dict[str, Any]:
    """POST transactions to ``<base_url>/invocations`` and return the parsed response."""
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/invocations",
        data=json.dumps({"instances": instances}).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise EndpointError(exc.code, _error_detail(exc.read())) from exc
    except (urllib.error.URLError, OSError) as exc:
        raise EndpointError(None, f"cannot reach {base_url}: {exc}") from exc
