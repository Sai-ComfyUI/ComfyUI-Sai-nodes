"""Versioned protocol primitives for the Moonland browser bridge."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import secrets
import threading
import time
from typing import Any, Callable, Mapping

from .contracts import (
    CONTRACT_SCHEMA_VERSION,
    MAX_CONTRACT_BYTES,
    MoonlandContractError,
    MoonlandUrl,
    canonicalize_moonland_url,
    normalize_resolved_contract,
)


PROTOCOL_VERSION = 1
PAIRING_TTL_SECONDS = 300
PAIRING_CODE_DIGITS = 6
_BRIDGE_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{8,128}$")


class BridgeProtocolError(ValueError):
    """Raised when a bridge message does not match the public protocol."""


@dataclass(frozen=True)
class MoonlandStatus:
    authenticated: bool
    page_url: str | None = None
    workspace_id: str | None = None
    workspace_slug: str | None = None


@dataclass(frozen=True)
class BridgeHello:
    bridge_id: str
    extension_version: str
    moonland: MoonlandStatus


@dataclass(frozen=True)
class BridgeConnection:
    bridge_id: str
    extension_version: str
    moonland: MoonlandStatus
    connected_at: float
    last_seen_at: float

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["protocol_version"] = PROTOCOL_VERSION
        return value


def _optional_string(value: Any, *, field_name: str, maximum_length: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or len(value) > maximum_length:
        raise BridgeProtocolError(f"{field_name} must be a non-empty string")
    return value


def parse_hello(message: Mapping[str, Any]) -> BridgeHello:
    """Parse and sanitize the initial extension handshake."""

    if message.get("type") != "hello":
        raise BridgeProtocolError("first message must be a hello")
    if message.get("protocolVersion") != PROTOCOL_VERSION:
        raise BridgeProtocolError("unsupported protocol version")
    bridge_id = message.get("bridgeId")
    if not isinstance(bridge_id, str) or not _BRIDGE_ID_PATTERN.fullmatch(bridge_id):
        raise BridgeProtocolError("bridgeId has an invalid format")
    extension_version = message.get("extensionVersion")
    if not isinstance(extension_version, str) or not extension_version or len(extension_version) > 32:
        raise BridgeProtocolError("extensionVersion has an invalid format")
    moonland = message.get("moonland")
    if not isinstance(moonland, Mapping):
        raise BridgeProtocolError("moonland status is required")
    authenticated = moonland.get("authenticated")
    if not isinstance(authenticated, bool):
        raise BridgeProtocolError("moonland.authenticated must be boolean")
    return BridgeHello(
        bridge_id=bridge_id,
        extension_version=extension_version,
        moonland=MoonlandStatus(
            authenticated=authenticated,
            page_url=_optional_string(moonland.get("pageUrl"), field_name="moonland.pageUrl", maximum_length=2048),
            workspace_id=_optional_string(moonland.get("workspaceId"), field_name="moonland.workspaceId", maximum_length=128),
            workspace_slug=_optional_string(moonland.get("workspaceSlug"), field_name="moonland.workspaceSlug", maximum_length=128),
        ),
    )


class BridgeRegistry:
    """In-memory pairing and connection state with no credential persistence."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic, pairing_ttl_seconds: int = PAIRING_TTL_SECONDS) -> None:
        self._clock = clock
        self._pairing_ttl_seconds = pairing_ttl_seconds
        self._pairings: dict[str, float] = {}
        self._connections: dict[str, BridgeConnection] = {}
        self._lock = threading.Lock()

    def issue_pairing_code(self) -> tuple[str, int]:
        with self._lock:
            self._remove_expired_pairings_locked()
            while True:
                code = f"{secrets.randbelow(10**PAIRING_CODE_DIGITS):0{PAIRING_CODE_DIGITS}d}"
                if code not in self._pairings:
                    break
            self._pairings[code] = self._clock() + self._pairing_ttl_seconds
        return code, self._pairing_ttl_seconds

    def consume_pairing_code(self, code: str) -> bool:
        with self._lock:
            self._remove_expired_pairings_locked()
            expiry = self._pairings.pop(code, None)
            return expiry is not None and expiry >= self._clock()

    def connect(self, hello: BridgeHello) -> BridgeConnection:
        now = self._clock()
        connection = BridgeConnection(hello.bridge_id, hello.extension_version, hello.moonland, now, now)
        with self._lock:
            self._connections[hello.bridge_id] = connection
        return connection

    def heartbeat(self, hello: BridgeHello) -> BridgeConnection:
        now = self._clock()
        with self._lock:
            previous = self._connections.get(hello.bridge_id)
            connection = BridgeConnection(
                hello.bridge_id,
                hello.extension_version,
                hello.moonland,
                previous.connected_at if previous else now,
                now,
            )
            self._connections[hello.bridge_id] = connection
        return connection

    def disconnect(self, bridge_id: str) -> None:
        with self._lock:
            self._connections.pop(bridge_id, None)

    def status(self) -> dict[str, Any]:
        with self._lock:
            connections = [item.public_dict() for item in sorted(self._connections.values(), key=lambda item: item.connected_at)]
        return {"protocol_version": PROTOCOL_VERSION, "connected": bool(connections), "connections": connections}

    def _remove_expired_pairings_locked(self) -> None:
        now = self._clock()
        for code in [code for code, expiry in self._pairings.items() if expiry < now]:
            self._pairings.pop(code, None)


__all__ = [
    "BridgeConnection",
    "BridgeHello",
    "BridgeProtocolError",
    "BridgeRegistry",
    "CONTRACT_SCHEMA_VERSION",
    "MAX_CONTRACT_BYTES",
    "MoonlandContractError",
    "MoonlandStatus",
    "MoonlandUrl",
    "PAIRING_TTL_SECONDS",
    "PROTOCOL_VERSION",
    "canonicalize_moonland_url",
    "normalize_resolved_contract",
    "parse_hello",
]
