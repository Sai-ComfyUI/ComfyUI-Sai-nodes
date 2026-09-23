const baseUrl = process.env.COMFY_URL ?? "http://127.0.0.1:9527";
const websocketUrl = baseUrl.replace(/^http/, "ws") + "/sai/moonland-bridge/ws";

const pairing = await fetch(`${baseUrl}/sai/moonland-bridge/pairing`, { method: "POST" }).then((response) => response.json());
const socket = new WebSocket(websocketUrl);
const bridgeId = "bridge-e2e-0001";
const moonland = {
    authenticated: true,
    pageUrl: "https://moonland.ai/w/purple-bright-coyote/lab?t=test",
    workspaceId: "cmthzpmu3015h01s66hnj3wr1",
    workspaceSlug: "purple-bright-coyote",
};

await new Promise((resolve, reject) => {
    socket.addEventListener("open", () => {
        socket.send(JSON.stringify({
            type: "hello",
            protocolVersion: 1,
            bridgeId,
            extensionVersion: "0.1.0",
            pairingCode: pairing.pairing_code,
            moonland,
        }));
    });
    socket.addEventListener("message", (event) => {
        const message = JSON.parse(event.data);
        if (message.type === "helloAck") resolve();
    });
    socket.addEventListener("error", reject);
});

const connected = await fetch(`${baseUrl}/sai/moonland-bridge/status`).then((response) => response.json());
socket.send(JSON.stringify({ type: "heartbeat", protocolVersion: 1, bridgeId, extensionVersion: "0.1.0", moonland }));
await new Promise((resolve) => setTimeout(resolve, 100));
socket.close();
await new Promise((resolve) => setTimeout(resolve, 100));
const disconnected = await fetch(`${baseUrl}/sai/moonland-bridge/status`).then((response) => response.json());

const result = {
    pairingCodeInUrl: websocketUrl.includes(pairing.pairing_code),
    connected: connected.connected,
    workspaceSlug: connected.connections.at(0)?.moonland.workspace_slug,
    disconnected: !disconnected.connected,
};
console.log(JSON.stringify(result, null, 2));

if (result.pairingCodeInUrl || !result.connected || result.workspaceSlug !== "purple-bright-coyote" || !result.disconnected) {
    process.exitCode = 1;
}
