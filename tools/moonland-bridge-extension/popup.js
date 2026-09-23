const DEFAULT_COMFY_URL = "http://127.0.0.1:9527";
const comfyUrlInput = document.querySelector("#comfy-url");
const pairingCodeInput = document.querySelector("#pairing-code");
const connectButton = document.querySelector("#connect");
const moonlandStatus = document.querySelector("#moonland-status");
const bridgeStatus = document.querySelector("#bridge-status");

function normalizeComfyUrl(value) {
    const url = new URL(value);
    if (!["http:", "https:"].includes(url.protocol)) throw new Error("ComfyUI URL must use http or https.");
    url.pathname = "";
    url.search = "";
    url.hash = "";
    return url.toString().replace(/\/$/, "");
}

function hostPermissionPattern(comfyUrl) {
    const url = new URL(comfyUrl);
    return `${url.protocol}//${url.hostname}/*`;
}

async function preflightFromPopup(comfyUrl) {
    let response;
    try {
        response = await fetch(`${comfyUrl}/sai/moonland-bridge/status`, {
            cache: "no-store",
        });
    } catch (error) {
        throw new Error(`Cannot reach ${comfyUrl}. Allow Local Network Access if Chrome asks, then verify the host and port. (${error.message})`);
    }
    if (!response.ok) throw new Error(`ComfyUI replied ${response.status}; Moonland Bridge routes are not available.`);
}

async function renderStatus() {
    const status = await chrome.runtime.sendMessage({ type: "bridge.status" });
    bridgeStatus.textContent = status.message;
    moonlandStatus.textContent = status.moonland.authenticated
        ? `Moonland signed in${status.moonland.workspaceSlug ? ` - ${status.moonland.workspaceSlug}` : ""}`
        : "Open a signed-in Moonland tab in this browser";
}

connectButton.addEventListener("click", async () => {
    connectButton.disabled = true;
    bridgeStatus.textContent = "Checking ComfyUI...";
    try {
        const comfyUrl = normalizeComfyUrl(comfyUrlInput.value);
        const pairingCode = pairingCodeInput.value.trim();
        if (!/^\d{6}$/.test(pairingCode)) throw new Error("Enter the 6-digit pairing code.");
        const originPattern = hostPermissionPattern(comfyUrl);
        const granted = await chrome.permissions.request({ origins: [originPattern] });
        if (!granted) throw new Error("Permission to connect to this ComfyUI address was denied.");
        await preflightFromPopup(comfyUrl);
        await chrome.storage.local.set({ comfyUrl });
        const result = await chrome.runtime.sendMessage({ type: "bridge.connect", comfyUrl, pairingCode });
        if (!result.ok) throw new Error(result.error);
        pairingCodeInput.value = "";
        await renderStatus();
    } catch (error) {
        bridgeStatus.textContent = error instanceof Error ? error.message : String(error);
    } finally {
        connectButton.disabled = false;
    }
});

chrome.storage.local.get({ comfyUrl: DEFAULT_COMFY_URL }).then(({ comfyUrl }) => {
    comfyUrlInput.value = comfyUrl === "http://127.0.0.1:8188" ? DEFAULT_COMFY_URL : comfyUrl;
});
void renderStatus();
