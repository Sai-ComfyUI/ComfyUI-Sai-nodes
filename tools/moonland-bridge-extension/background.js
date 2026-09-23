const PROTOCOL_VERSION = 1;
let currentStatus = { authenticated: false, pageUrl: null, workspaceId: null, workspaceSlug: null };
let connectionState = { connected: false, message: "Not connected" };
const moonlandStatusByTab = new Map();

function sanitizeMoonlandStatus(status) {
    return {
        authenticated: Boolean(status?.authenticated),
        pageUrl: status?.pageUrl ?? null,
        workspaceId: status?.workspaceId ?? null,
        workspaceSlug: status?.workspaceSlug ?? null,
    };
}

function refreshCurrentMoonlandStatus() {
    const statuses = Array.from(moonlandStatusByTab.values());
    currentStatus = statuses.sort((left, right) => {
        const score = (status) => Number(status.authenticated) * 2 + Number(Boolean(status.workspaceId));
        return score(right) - score(left);
    })[0] ?? { authenticated: false, pageUrl: null, workspaceId: null, workspaceSlug: null };
}

async function refreshMoonlandTabs() {
    const tabs = await chrome.tabs.query({ url: "https://moonland.ai/*" });
    if (tabs.length === 0) throw new Error("Open a signed-in Moonland tab in this browser, then try again.");
    let delivered = 0;
    for (const tab of tabs) {
        if (tab.id === undefined) continue;
        try {
            await chrome.tabs.sendMessage(tab.id, { type: "moonland.refresh" });
            delivered += 1;
        } catch {
            // Existing tabs keep the previous content-script context after an
            // unpacked extension reload. Reloading the page injects this version.
        }
    }
    if (delivered === 0) throw new Error("Reload the Moonland tab after reloading the extension, then try again.");
}

async function findComfyTab(comfyUrl) {
    const wantedOrigin = new URL(comfyUrl).origin;
    const tabs = await chrome.tabs.query({ url: ["http://*/*", "https://*/*"] });
    return tabs.find((tab) => {
        try {
            return tab.id !== undefined && new URL(tab.url).origin === wantedOrigin;
        } catch {
            return false;
        }
    });
}

async function connectBridge(comfyUrl, pairingCode) {
    await refreshMoonlandTabs();
    const comfyTab = await findComfyTab(comfyUrl);
    if (!comfyTab) throw new Error(`Open ${comfyUrl} in this browser, then try again.`);

    await chrome.scripting.executeScript({
        target: { tabId: comfyTab.id },
        files: ["comfy_bridge_content.js"],
    });
    const result = await chrome.tabs.sendMessage(comfyTab.id, {
        type: "moonlandBridge.connect",
        pairingCode,
        protocolVersion: PROTOCOL_VERSION,
        extensionVersion: chrome.runtime.getManifest().version,
    });
    if (!result?.ok) throw new Error(result?.error || "The ComfyUI page bridge did not respond.");
    connectionState = result.status;
    return connectionState;
}

async function runMoonlandAction(action, payload) {
    const tabs = await chrome.tabs.query({ url: "https://moonland.ai/*" });
    if (tabs.length === 0) throw new Error("Open a signed-in Moonland tab, then try again.");
    let actionTabs = tabs;
    if (action === "tool.resolveContract") {
        let requested;
        try {
            requested = new URL(payload?.url);
        } catch {
            throw new Error("INVALID_URL: Moonland resolver URL is invalid.");
        }
        actionTabs = tabs.filter((tab) => {
            const status = moonlandStatusByTab.get(tab.id);
            if (!status?.authenticated || typeof tab.url !== "string") return false;
            try {
                const open = new URL(tab.url);
                if (requested.pathname.startsWith("/generation/")) return open.pathname === requested.pathname;
                if (requested.pathname.endsWith("/gen")) {
                    return open.pathname.replace(/\/$/, "") === requested.pathname.replace(/\/$/, "")
                        && open.searchParams.get("g") === requested.searchParams.get("g");
                }
                return open.pathname === requested.pathname && open.searchParams.get("t") === requested.searchParams.get("t");
            } catch {
                return false;
            }
        });
        if (actionTabs.length === 0) {
            throw new Error("TAB_NOT_FOUND: Open the exact Moonland Lab topic or generation URL, then try again.");
        }
    }
    let lastError = null;
    for (const tab of actionTabs) {
        if (tab.id === undefined) continue;
        try {
            const response = await chrome.tabs.sendMessage(tab.id, {
                type: "moonland.action",
                action,
                payload,
            });
            if (response?.ok) return response.result;
            lastError = new Error(response?.error || "Moonland tab rejected the action.");
        } catch (error) {
            lastError = error;
        }
    }
    throw lastError || new Error("No Moonland tab could perform the action. Reload the Moonland tab and reconnect.");
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "moonland.status") {
        if (sender.tab?.id !== undefined) {
            moonlandStatusByTab.set(sender.tab.id, sanitizeMoonlandStatus(message.status));
            refreshCurrentMoonlandStatus();
        }
        sendResponse({ ok: true });
        return false;
    }
    if (message.type === "bridge.payload") {
        sendResponse({ protocolVersion: PROTOCOL_VERSION, moonland: currentStatus });
        return false;
    }
    if (message.type === "bridge.connectionState") {
        connectionState = {
            connected: Boolean(message.connected),
            message: String(message.message || (message.connected ? "Bridge connected" : "Not connected")),
        };
        sendResponse({ ok: true });
        return false;
    }
    if (message.type === "bridge.status") {
        sendResponse({ ...connectionState, moonland: currentStatus });
        return false;
    }
    if (message.type === "bridge.connect") {
        connectBridge(message.comfyUrl, message.pairingCode)
            .then((status) => sendResponse({ ok: true, status }))
            .catch((error) => sendResponse({ ok: false, error: error.message }));
        return true;
    }
    if (message.type === "bridge.action") {
        runMoonlandAction(message.action, message.payload)
            .then((result) => sendResponse({ ok: true, result }))
            .catch((error) => sendResponse({ ok: false, error: String(error?.message || error) }));
        return true;
    }
    return false;
});

chrome.tabs.onRemoved.addListener((tabId) => {
    moonlandStatusByTab.delete(tabId);
    refreshCurrentMoonlandStatus();
});
