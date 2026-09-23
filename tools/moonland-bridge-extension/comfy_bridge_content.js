(() => {
    if (globalThis.__saiMoonlandBridgeInstalled) return;
    globalThis.__saiMoonlandBridgeInstalled = true;

    const HEARTBEAT_INTERVAL_MS = 20000;
    const CONNECT_TIMEOUT_MS = 10000;
    let socket = null;
    let heartbeatTimer = null;
    let bridgeId = null;

    function websocketUrl() {
        const url = new URL("/sai/moonland-bridge/ws", window.location.origin);
        url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
        return url.toString();
    }

    function stopSocket() {
        if (heartbeatTimer !== null) window.clearInterval(heartbeatTimer);
        heartbeatTimer = null;
        if (socket) socket.close();
        socket = null;
    }

    async function protocolMessage(type, extensionVersion, pairingCode = null) {
        const payload = await chrome.runtime.sendMessage({ type: "bridge.payload" });
        const message = {
            type,
            protocolVersion: payload.protocolVersion,
            bridgeId,
            extensionVersion,
            moonland: payload.moonland,
        };
        if (type === "hello") message.pairingCode = pairingCode;
        return message;
    }

    function notifyConnectionState(connected, message) {
        try {
            chrome.runtime.sendMessage({ type: "bridge.connectionState", connected, message }).catch(() => {});
        } catch {
            // The page may outlive an unpacked extension reload.
        }
    }

    async function runBridgeAction(nextSocket, request) {
        let response;
        try {
            const delivery = await chrome.runtime.sendMessage({
                type: "bridge.action",
                action: request.action,
                payload: request.payload,
            });
            if (!delivery?.ok) throw new Error(delivery?.error || "Moonland action failed.");
            response = {
                type: "actionResult",
                protocolVersion: request.protocolVersion,
                requestId: request.requestId,
                ok: true,
                result: delivery.result,
            };
        } catch (error) {
            response = {
                type: "actionResult",
                protocolVersion: request.protocolVersion,
                requestId: request.requestId,
                ok: false,
                error: String(error?.message || error || "Moonland action failed."),
            };
        }
        if (nextSocket.readyState === WebSocket.OPEN) {
            nextSocket.send(JSON.stringify(response));
        }
    }

    async function connect(message) {
        stopSocket();
        bridgeId = crypto.randomUUID();
        const targetUrl = websocketUrl();
        return new Promise((resolve, reject) => {
            const nextSocket = new WebSocket(targetUrl);
            socket = nextSocket;
            let settled = false;
            let timeout = null;
            const fail = (reason) => {
                if (timeout !== null) window.clearTimeout(timeout);
                notifyConnectionState(false, reason);
                if (!settled) {
                    settled = true;
                    reject(new Error(reason));
                }
            };
            timeout = window.setTimeout(() => fail(`Timed out connecting to ${targetUrl}.`), CONNECT_TIMEOUT_MS);
            nextSocket.addEventListener("open", async () => {
                try {
                    nextSocket.send(JSON.stringify(await protocolMessage("hello", message.extensionVersion, message.pairingCode)));
                } catch (error) {
                    fail(error.message);
                }
            });
            nextSocket.addEventListener("message", (event) => {
                let response;
                try {
                    response = JSON.parse(event.data);
                } catch {
                    fail("Bridge returned an invalid response.");
                    return;
                }
                if (response.type === "actionRequest") {
                    runBridgeAction(nextSocket, response);
                    return;
                }
                if (response.type !== "helloAck") return;
                if (timeout !== null) window.clearTimeout(timeout);
                notifyConnectionState(true, "Bridge connected");
                heartbeatTimer = window.setInterval(async () => {
                    if (nextSocket.readyState !== WebSocket.OPEN) return;
                    try {
                        nextSocket.send(JSON.stringify(await protocolMessage("heartbeat", message.extensionVersion)));
                    } catch {
                        stopSocket();
                    }
                }, HEARTBEAT_INTERVAL_MS);
                if (!settled) {
                    settled = true;
                    resolve({ connected: true, message: "Bridge connected" });
                }
            });
            nextSocket.addEventListener("close", (event) => {
                const reason = event.reason || `Bridge closed with code ${event.code}.`;
                if (socket === nextSocket) socket = null;
                if (!settled) fail(reason);
                else notifyConnectionState(false, reason);
            });
            nextSocket.addEventListener("error", () => {
                // The close event supplies the actionable status and reason.
            });
        });
    }

    chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
        if (message.type !== "moonlandBridge.connect") return false;
        connect(message)
            .then((status) => sendResponse({ ok: true, status }))
            .catch((error) => sendResponse({ ok: false, error: error.message }));
        return true;
    });
})();
