"""HTTP reads restricted to an operator-selected origin."""

from collections.abc import Mapping
from urllib.parse import urlsplit

import httpx

from shared.errors import ToolError


def validate_origin(origin: str, allow_local: bool = False) -> str:
    try:
        parts = urlsplit(origin)
        port = parts.port
    except ValueError as exc:
        raise ToolError("Origin has an invalid host or port.") from exc
    if any(char.isspace() or ord(char) < 32 for char in origin) or port == 0:
        raise ToolError("Origin contains invalid characters or port.")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ToolError("Origin must not contain credentials, query parameters, or fragments.")
    if not parts.hostname or parts.path not in ("", "/"):
        raise ToolError("Configure an origin, without a path.")
    local = parts.hostname in {"localhost", "127.0.0.1", "::1"}
    if parts.scheme != "https" and not (allow_local and local and parts.scheme == "http"):
        raise ToolError("HTTPS is required, except explicitly enabled loopback fixtures.")
    return origin.rstrip("/")


class OriginClient:
    def __init__(
        self,
        origin: str,
        *,
        allow_local: bool = False,
        max_bytes: int = 500000,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.origin = validate_origin(origin, allow_local)
        self.max_bytes = max_bytes
        self.client = httpx.Client(
            timeout=10,
            follow_redirects=False,
            trust_env=False,
            transport=transport,
        )

    def close(self) -> None:
        self.client.close()

    def get(
        self,
        path: str,
        params: Mapping[str, str | int] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> tuple[bytes, httpx.Headers]:
        if (
            not path.startswith("/")
            or path.startswith("//")
            or any(char.isspace() or ord(char) < 32 for char in path)
            or any(mark in path for mark in ("\\", "?", "#", "%", ".."))
        ):
            raise ToolError("Only literal paths within the configured origin are allowed.")
        try:
            with self.client.stream(
                "GET",
                self.origin + path,
                params=params,
                headers=headers,
            ) as response:
                if response.status_code == 429:
                    raise ToolError("API rate limited the request; retry later.")
                if response.status_code != 200:
                    raise ToolError(f"HTTP {response.status_code}; redirects are not followed.")
                chunks = bytearray()
                for chunk in response.iter_bytes(chunk_size=8192):
                    chunks.extend(chunk)
                    if len(chunks) > self.max_bytes:
                        raise ToolError("HTTP response exceeds the byte budget.")
                return bytes(chunks), response.headers
        except (httpx.HTTPError, httpx.InvalidURL) as exc:
            raise ToolError("HTTP request failed or timed out.") from exc
