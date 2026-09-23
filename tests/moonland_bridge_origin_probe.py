"""Manual integration probe for ComfyUI origin protection and page-origin bridge."""

from __future__ import annotations

import asyncio
import json
import os

from aiohttp import ClientSession, WSServerHandshakeError


BASE_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:9527")
WEBSOCKET_URL = BASE_URL.replace("http", "ws", 1) + "/sai/moonland-bridge/ws"


async def main() -> None:
    async with ClientSession() as session:
        extension_origin_status = None
        try:
            await session.ws_connect(WEBSOCKET_URL, origin="chrome-extension://origin-probe")
        except WSServerHandshakeError as error:
            extension_origin_status = error.status

        pairing = await session.post(BASE_URL + "/sai/moonland-bridge/pairing")
        pairing_data = await pairing.json()
        websocket = await session.ws_connect(WEBSOCKET_URL, origin=BASE_URL)
        await websocket.send_json(
            {
                "type": "hello",
                "protocolVersion": 1,
                "bridgeId": "bridge-origin-probe",
                "extensionVersion": "0.2.0",
                "pairingCode": pairing_data["pairing_code"],
                "moonland": {"authenticated": True, "workspaceSlug": "purple-bright-coyote"},
            }
        )
        acknowledgement = await websocket.receive_json()
        await websocket.close()

    result = {
        "extension_origin_rejected": extension_origin_status == 403,
        "page_origin_connected": acknowledgement.get("type") == "helloAck",
    }
    print(json.dumps(result, indent=2))
    if not all(result.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
