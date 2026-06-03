from __future__ import annotations

import json
from typing import Any, Optional
from urllib import error, request


class RecorderHTTPClient:
    def __init__(
        self, base_url: str = "http://127.0.0.1:18080", timeout: float = 5.0
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        return self._request_json("GET", "/health")

    def status(self) -> dict[str, Any]:
        return self._request_json("GET", "/status")

    def start_recording(
        self,
        topic: Optional[str] = None,
        folder: Optional[str] = None,
        base_name: Optional[str] = None,
    ) -> dict[str, Any]:
        payload = {"topic": topic, "folder": folder, "base_name": base_name}
        cleaned = {k: v for k, v in payload.items() if v is not None}
        return self._request_json("POST", "/start", cleaned)

    def stop_recording(self) -> dict[str, Any]:
        return self._request_json("POST", "/stop", timeout=30.0)

    def _request_json(
        self,
        method: str,
        path: str,
        body: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        payload = None
        headers: dict[str, str] = {}
        if body is not None:
            payload = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = request.Request(url=url, method=method, data=payload, headers=headers)
        try:
            with request.urlopen(
                req, timeout=timeout if timeout is not None else self.timeout
            ) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"Recorder API {method} {path} failed ({exc.code}): {detail}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"Recorder API {method} {path} unreachable: {exc}"
            ) from exc
