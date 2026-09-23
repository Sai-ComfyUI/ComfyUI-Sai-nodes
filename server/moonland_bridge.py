"""In-memory WebSocket bridge between ComfyUI and the Moonland extension."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from aiohttp import WSMsgType, web
from server import PromptServer

from ..moonland_bridge_protocol import BridgeHello, BridgeProtocolError, BridgeRegistry, PROTOCOL_VERSION, parse_hello


PAIRING_ROUTE = "/sai/moonland-bridge/pairing"
STATUS_ROUTE = "/sai/moonland-bridge/status"
WEBSOCKET_ROUTE = "/sai/moonland-bridge/ws"
_REGISTERED = False
_REGISTRY = BridgeRegistry()
_ACTIVE_SOCKETS: dict[str, web.WebSocketResponse] = {}
_PENDING_ACTIONS: dict[str, tuple[str, asyncio.Future[dict[str, Any]]]] = {}
_ALLOWED_ACTIONS = {"resource.ensureImage", "tool.resolveContract"}
_SERVER_LOOP: asyncio.AbstractEventLoop | None = None


class MoonlandBridgeActionError(RuntimeError):
    """Raised when an allowlisted browser-side Moonland action fails."""


def get_primary_connection() -> dict[str, Any]:
    """Return the first authenticated Bridge connection's public metadata."""

    for connection in _REGISTRY.status()["connections"]:
        if connection["moonland"]["authenticated"] and connection["bridge_id"] in _ACTIVE_SOCKETS:
            return connection
    raise MoonlandBridgeActionError(
        "No authenticated Moonland Bridge is connected. Open Moonland, then reconnect the Bridge."
    )


async def _request_bridge_action_on_server_loop(
    action: str,
    payload: dict[str, Any],
    *,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Ask the connected extension to perform one explicitly allowlisted action."""

    if action not in _ALLOWED_ACTIONS:
        raise MoonlandBridgeActionError(f"Unsupported Moonland Bridge action: {action}")
    connection = get_primary_connection()
    bridge_id = connection["bridge_id"]
    websocket = _ACTIVE_SOCKETS[bridge_id]
    request_id = str(uuid.uuid4())
    future = asyncio.get_running_loop().create_future()
    _PENDING_ACTIONS[request_id] = (bridge_id, future)
    try:
        await websocket.send_json(
            {
                "type": "actionRequest",
                "protocolVersion": PROTOCOL_VERSION,
                "requestId": request_id,
                "action": action,
                "payload": payload,
            }
        )
        return await asyncio.wait_for(future, timeout=timeout_seconds)
    except asyncio.TimeoutError as error:
        raise MoonlandBridgeActionError(
            f"Moonland Bridge action timed out after {timeout_seconds:g} seconds."
        ) from error
    finally:
        _PENDING_ACTIONS.pop(request_id, None)


async def request_bridge_action(
    action: str,
    payload: dict[str, Any],
    *,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Dispatch an action on the aiohttp loop even when a node uses another loop."""

    server_loop = _SERVER_LOOP
    if server_loop is None or server_loop.is_closed():
        raise MoonlandBridgeActionError("Moonland Bridge server loop is unavailable.")
    current_loop = asyncio.get_running_loop()
    if current_loop is server_loop:
        return await _request_bridge_action_on_server_loop(
            action,
            payload,
            timeout_seconds=timeout_seconds,
        )
    concurrent_future = asyncio.run_coroutine_threadsafe(
        _request_bridge_action_on_server_loop(
            action,
            payload,
            timeout_seconds=timeout_seconds,
        ),
        server_loop,
    )
    return await asyncio.wrap_future(concurrent_future)


def _resolve_action_result(bridge_id: str, message: dict[str, Any]) -> None:
    if message.get("protocolVersion") != PROTOCOL_VERSION:
        raise BridgeProtocolError("unsupported actionResult protocol version")
    request_id = message.get("requestId")
    if not isinstance(request_id, str):
        raise BridgeProtocolError("actionResult.requestId must be a string")
    pending = _PENDING_ACTIONS.get(request_id)
    if pending is None:
        return
    expected_bridge_id, future = pending
    if expected_bridge_id != bridge_id:
        raise BridgeProtocolError("actionResult came from the wrong bridge")
    ok = message.get("ok")
    if not isinstance(ok, bool):
        raise BridgeProtocolError("actionResult.ok must be boolean")
    if future.done():
        return
    if ok:
        result = message.get("result")
        if not isinstance(result, dict):
            raise BridgeProtocolError("successful actionResult.result must be an object")
        future.set_result(result)
    else:
        error_message = message.get("error")
        if not isinstance(error_message, str) or not error_message:
            error_message = "Moonland Bridge action failed."
        future.set_exception(MoonlandBridgeActionError(error_message))


def _fail_pending_actions(bridge_id: str) -> None:
    for request_id, (pending_bridge_id, future) in list(_PENDING_ACTIONS.items()):
        if pending_bridge_id != bridge_id:
            continue
        _PENDING_ACTIONS.pop(request_id, None)
        if not future.done():
            future.set_exception(MoonlandBridgeActionError("Moonland Bridge disconnected during the action."))


async def handle_pairing(_request: web.Request) -> web.Response:
    code, expires_in = _REGISTRY.issue_pairing_code()
    return web.json_response({"protocol_version": PROTOCOL_VERSION, "pairing_code": code, "expires_in_seconds": expires_in})


async def handle_status(_request: web.Request) -> web.Response:
    return web.json_response(_REGISTRY.status())


def _hello_from_heartbeat(message: dict[str, Any]) -> BridgeHello:
    return parse_hello({**message, "type": "hello"})


async def handle_websocket(request: web.Request) -> web.WebSocketResponse:
    global _SERVER_LOOP
    _SERVER_LOOP = asyncio.get_running_loop()
    websocket = web.WebSocketResponse(heartbeat=30)
    await websocket.prepare(request)
    bridge_id: str | None = None
    try:
        first = await asyncio.wait_for(websocket.receive(), timeout=10)
        if first.type != WSMsgType.TEXT:
            raise BridgeProtocolError("hello message must be JSON text")
        message = json.loads(first.data)
        hello = parse_hello(message)
        pairing_code = message.get("pairingCode")
        if not isinstance(pairing_code, str) or not _REGISTRY.consume_pairing_code(pairing_code):
            raise BridgeProtocolError("invalid or expired pairing code")
        bridge_id = hello.bridge_id
        _REGISTRY.connect(hello)
        previous_socket = _ACTIVE_SOCKETS.get(bridge_id)
        if previous_socket is not None and previous_socket is not websocket:
            await previous_socket.close(code=1000, message=b"Bridge reconnected")
        _ACTIVE_SOCKETS[bridge_id] = websocket
        await websocket.send_json({"type": "helloAck", "protocolVersion": PROTOCOL_VERSION, "bridgeId": bridge_id})
        async for incoming in websocket:
            if incoming.type == WSMsgType.TEXT:
                message = json.loads(incoming.data)
                message_type = message.get("type")
                if message_type == "heartbeat":
                    heartbeat = _hello_from_heartbeat(message)
                    if heartbeat.bridge_id != bridge_id:
                        raise BridgeProtocolError("bridgeId changed during the session")
                    _REGISTRY.heartbeat(heartbeat)
                    await websocket.send_json({"type": "heartbeatAck", "protocolVersion": PROTOCOL_VERSION})
                elif message_type == "actionResult":
                    _resolve_action_result(bridge_id, message)
                else:
                    raise BridgeProtocolError("unsupported bridge message type")
            elif incoming.type in {WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR}:
                break
    except (asyncio.TimeoutError, BridgeProtocolError, json.JSONDecodeError) as error:
        await websocket.close(code=1008, message=str(error).encode("utf-8")[:120])
    finally:
        if bridge_id is not None:
            if _ACTIVE_SOCKETS.get(bridge_id) is websocket:
                _ACTIVE_SOCKETS.pop(bridge_id, None)
                _fail_pending_actions(bridge_id)
                _REGISTRY.disconnect(bridge_id)
    return websocket


def register_moonland_bridge_routes() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    routes = PromptServer.instance.routes
    routes.post(PAIRING_ROUTE)(handle_pairing)
    routes.get(STATUS_ROUTE)(handle_status)
    routes.get(WEBSOCKET_ROUTE)(handle_websocket)
    _REGISTERED = True


__all__ = [
    "MoonlandBridgeActionError",
    "PAIRING_ROUTE",
    "STATUS_ROUTE",
    "WEBSOCKET_ROUTE",
    "get_primary_connection",
    "handle_pairing",
    "handle_status",
    "handle_websocket",
    "register_moonland_bridge_routes",
    "request_bridge_action",
]
