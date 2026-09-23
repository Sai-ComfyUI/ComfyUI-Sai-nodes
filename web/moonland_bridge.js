import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const STATUS_COMMAND_ID = "sai.moonlandBridgeStatus";
const PAIR_COMMAND_ID = "sai.moonlandBridgePair";
const STATUS_ROUTE = "/sai/moonland-bridge/status";
const PAIRING_ROUTE = "/sai/moonland-bridge/pairing";

function toast(severity, summary, detail, life = 5000) {
    app.extensionManager.toast.add({ severity, summary, detail, life });
}

async function fetchJson(path, options = {}) {
    const response = await api.fetchApi(path, { cache: "no-store", ...options });
    if (!response.ok) throw new Error(await response.text());
    return response.json();
}

async function checkBridgeStatus() {
    try {
        const status = await fetchJson(STATUS_ROUTE);
        if (!status.connected) {
            toast("warn", "Moonland Bridge not connected", "Create a pairing code, then connect from the Moonland Bridge browser extension.", 8000);
            return;
        }
        const moonland = status.connections[0].moonland;
        toast(
            moonland.authenticated ? "success" : "warn",
            moonland.authenticated ? "Moonland connected" : "Bridge connected; Moonland not signed in",
            moonland.workspace_slug ? `Workspace: ${moonland.workspace_slug}` : "Keep a signed-in Moonland tab open.",
            8000,
        );
    } catch (error) {
        toast("error", "Cannot check Moonland Bridge", error instanceof Error ? error.message : String(error), 10000);
    }
}

async function copyText(text) {
    if (navigator.clipboard?.writeText) {
        try {
            await navigator.clipboard.writeText(text);
            return true;
        } catch {
            // Fall through for browsers that deny async clipboard access.
        }
    }
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.append(textarea);
    textarea.select();
    let copied = false;
    try {
        copied = document.execCommand("copy");
    } finally {
        textarea.remove();
    }
    return copied;
}

async function createPairingCode() {
    try {
        const pairing = await fetchJson(PAIRING_ROUTE, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
        const copied = await copyText(pairing.pairing_code);
        if (copied) {
            toast(
                "success",
                "Pairing code copied",
                `${pairing.pairing_code} is ready to paste into Moonland Bridge. It expires in ${Math.ceil(pairing.expires_in_seconds / 60)} minutes and works once.`,
                10000,
            );
            return;
        }
        await app.extensionManager.dialog.confirm({
            title: "Moonland Bridge pairing — copy required",
            message: `Clipboard access was unavailable. Copy this code manually:\n\n${pairing.pairing_code}\n\nComfyUI URL: ${window.location.origin}\nThe code expires in ${Math.ceil(pairing.expires_in_seconds / 60)} minutes and works once.`,
        });
    } catch (error) {
        toast("error", "Cannot create Moonland Bridge pairing", error instanceof Error ? error.message : String(error), 10000);
    }
}

app.registerExtension({
    name: "sai.moonlandBridge",
    commands: [
        { id: STATUS_COMMAND_ID, label: "Check Moonland Bridge", icon: "pi pi-link", function: checkBridgeStatus },
        { id: PAIR_COMMAND_ID, label: "Pair Moonland Bridge", icon: "pi pi-key", function: createPairingCode },
    ],
    menuCommands: [{ path: ["Sai", "Moonland Bridge"], commands: [STATUS_COMMAND_ID, PAIR_COMMAND_ID] }],
    actionBarButtons: [{ icon: "pi pi-link", label: "Moonland", tooltip: "Check Moonland Bridge connection", onClick: checkBridgeStatus }],
});
