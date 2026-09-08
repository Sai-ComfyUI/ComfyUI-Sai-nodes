"""Local-only ComfyUI restart endpoint for the project development service."""

from __future__ import annotations

import ipaddress
import os
import threading

from aiohttp import web
from server import PromptServer


RESTART_ROUTE = "/sai/restart"
RESTART_EXIT_CODE = 75
_REGISTERED = False


def is_loopback_address(address: str | None) -> bool:
    """Return whether an aiohttp peer address belongs to this machine."""

    if not address:
        return False
    host = address.split("%", 1)[0]
    try:
        parsed = ipaddress.ip_address(host)
        if parsed.is_loopback:
            return True
        mapped = getattr(parsed, "ipv4_mapped", None)
        return bool(mapped and mapped.is_loopback)
    except ValueError:
        return host.lower() == "localhost"


def schedule_restart(delay_seconds: float = 0.75) -> None:
    """Exit with the code handled by the project-owned supervising launcher."""

    timer = threading.Timer(delay_seconds, lambda: os._exit(RESTART_EXIT_CODE))
    timer.daemon = True
    timer.start()


async def handle_restart(request: web.Request) -> web.Response:
    """Accept an explicit local restart request from the ComfyUI frontend."""

    if not is_loopback_address(request.remote):
        raise web.HTTPForbidden(
            text="ComfyUI restart is only available from localhost."
        )

    schedule_restart()
    return web.json_response(
        {
            "status": "restarting",
            "message": "ComfyUI is restarting. The page will reload when ready.",
        }
    )


def register_restart_route() -> None:
    """Register the endpoint once during V3 extension load."""

    global _REGISTERED
    if _REGISTERED:
        return
    PromptServer.instance.routes.post(RESTART_ROUTE)(handle_restart)
    _REGISTERED = True


__all__ = [
    "RESTART_ROUTE",
    "RESTART_EXIT_CODE",
    "handle_restart",
    "is_loopback_address",
    "register_restart_route",
    "schedule_restart",
]
